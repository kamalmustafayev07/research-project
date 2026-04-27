# app.py
"""
Streamlit UI for Agent-Enhanced GraphRAG
Uses same pipeline logic as run_pipeline.py

Features:
- User asks short natural question
- Query goes through:
    1. Decomposer
    2. Retriever
    3. Passage Reranker (outputs/models/)
    4. Reasoner
    5. Critic
- Final answer shown
- Evidence chain visualization
- Retrieved passages
- Agent traces
- Performance metrics
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import SETTINGS
from src.agents.entity_linker import build_retrieval_entity_context
from src.agents.hop_safety import (
    EntityType,
    HopMemory,
    HopRecord,
    build_retrieval_query as build_safe_retrieval_query,
    get_reasoner_prompt as get_safe_reasoner_prompt,
    verify_hop_answer,
)
from src.graph.retrieval_query_builder import build_retrieval_query_pack, resolve_placeholders
from src.pipeline import AgentEnhancedGraphRAG
from src.utils.corpus_builder import build_unified_corpus, corpus_paths, DEFAULT_SPLITS
from src.evaluation.hotpotqa_eval import heuristic_explainability_score
from src.utils.llm import ollama_healthcheck

# ==========================================================
# CONFIG
# ==========================================================

DEFAULT_MODEL = "qwen2.5:7b"

# Load the complete train + validation splits of all three datasets.
# None = no cap (download everything available).
# The passages JSON and FAISS index are cached to disk so this only
# happens once; subsequent restarts load from cache in seconds.
CORPUS_MAX_PER_SPLIT: int | None = None
CORPUS_SPLITS = DEFAULT_SPLITS  # ["train", "validation"]


# ==========================================================
# SETTINGS
# ==========================================================

def apply_settings(
    model_name: str,
    temperature: float,
    loops: int,
    llm_backend: str = "ollama",
):
    root = Path(__file__).resolve().parent

    SETTINGS.paths.root = root
    SETTINGS.model.llm_backend = llm_backend
    SETTINGS.model.ollama_model = model_name
    SETTINGS.model.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    SETTINGS.model.temperature = temperature
    SETTINGS.model.max_new_tokens = 384

    SETTINGS.retrieval.max_retrieval_loops = loops
    SETTINGS.retrieval.use_reranker = True

    SETTINGS.paths.output_models.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD PIPELINE + CORPUS (cached together so the FAISS index
# is built once per session and reused across queries)
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_pipeline_and_corpus(
    model_name: str,
    temperature: float,
    loops: int,
    llm_backend: str,
):
    """Initialise pipeline and pre-load the FAISS index over the full corpus.

    Both objects are cached by ``@st.cache_resource`` so re-runs of the
    Streamlit script reuse the same pipeline and index without re-encoding
    passages on every user interaction.

    On first run the three datasets are downloaded from HuggingFace and saved
    to disk; the FAISS index is built from the GPU-encoded embeddings and also
    persisted to disk.  Subsequent restarts skip both steps and are fast.
    """
    apply_settings(model_name, temperature, loops, llm_backend)
    pipeline = AgentEnhancedGraphRAG()

    # Build (or load from disk) the unified passage corpus.
    corpus = build_unified_corpus(
        max_per_split=CORPUS_MAX_PER_SPLIT,
        splits=CORPUS_SPLITS,
    )

    # Resolve the persistent FAISS index path that matches this corpus version.
    _, faiss_path = corpus_paths(CORPUS_MAX_PER_SPLIT, CORPUS_SPLITS)

    # Load or build the FAISS index; save it to disk for future restarts.
    pipeline.retriever.retriever.preload_corpus(corpus, index_path=faiss_path)

    return pipeline, corpus


# ==========================================================
# RUN WITH TRACES
# ==========================================================

_BAD_PIVOT_MARKERS = (
    "insufficient", "not enough", "cannot", "unable", "no evidence", "i don",
)


def _extract_hop1_pivot(
    llm: Any,
    sub_questions: list[str],
    passages: list[dict[str, Any]],
) -> str | None:
    """Ask the LLM to answer the first sub-question using only the retrieved passages.

    Uses structured JSON output so the answer is parsed reliably regardless of
    how verbose the LLM's preamble is.  Returns a clean entity string suitable
    as a retry pivot, or None if the answer cannot be found.
    """
    if not sub_questions or not passages:
        return None
    hop1_q = sub_questions[0]
    context = "\n".join(
        f"- [{p.get('title', '')}]: {p.get('text', '')[:300]}"
        for p in passages[:5]
    )
    prompt = (
        "You are a precise entity extractor.\n"
        "ROLE RULE: If the question asks 'Who is the [ROLE] of X?' (e.g. founder, director, "
        "author, creator, composer), return ONLY the PERSON who holds that role. "
        "Do NOT return a location, product, organisation, or date — even if those appear "
        "prominently in the passage.\n"
        "PLACE RULE: If the question asks for a country or place and the passage uses a "
        "nationality adjective (e.g. 'British', 'French', 'American'), return the COUNTRY NAME "
        "instead (e.g. 'United Kingdom', 'France', 'United States').\n\n"
        f"Question: {hop1_q}\n\n"
        f"Passages:\n{context}\n\n"
        "Return STRICT JSON only:\n"
        '{"entity": "<answer, 1-5 words>", "found": true}\n'
        "If the answer is not in the passages:\n"
        '{"entity": "", "found": false}\n\n'
        "JSON:"
    )
    try:
        response = llm.generate(prompt, max_new_tokens=80)
        parsed = llm.extract_json(response.text)
        if not isinstance(parsed, dict) or not parsed.get("found"):
            return None
        entity = str(parsed.get("entity") or "").strip().strip("\"'.,;:()")
        if not entity or len(entity.split()) > 8:
            return None
        if any(w in entity.lower() for w in _BAD_PIVOT_MARKERS):
            return None
        return entity
    except Exception:
        return None


def _pivot_match(answer: str, pivot: str) -> bool:
    """Return True when the answer is effectively the same string as the pivot.

    Used to detect when the MULTI-HOP PARTIAL ANSWER rule in the reasoner prompt
    incorrectly returns the already-known intermediate entity instead of the final
    answer.  We compare normalised forms (lowercase, punctuation stripped).
    """
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", (s or "").lower().strip())
    a, p = norm(answer), norm(pivot)
    return bool(a and p and (a == p or (len(p) > 4 and p in a) or (len(a) > 4 and a in p)))


def _resolve_hop_placeholders(sub_question: str, prev_answer: str) -> str:
    """Replace ALL <answer of hop k> placeholders with the most recent known answer.

    In a chain of N hops, each loop resolves one sub-question.  The answer from
    the previous loop is substituted into the current sub-question so the LLM
    receives a concrete, answerable question rather than an abstract placeholder.
    """
    return re.sub(
        r"<answer of hop \d+>",
        prev_answer,
        sub_question,
        flags=re.IGNORECASE,
    )


def _build_retrieval_query(
    question: str,
    decomposition: Any,
    ref_anchors: list[dict],
) -> str:
    """Build the initial (loop-0) retrieval query string."""
    query = question
    if decomposition.sub_questions:
        resolved_sqs = [
            resolve_placeholders(sq, ref_anchors)
            for sq in decomposition.sub_questions[:2]
        ]
        query += " ; " + " ; ".join(resolved_sqs)
    extra: list[str] = []
    for row in decomposition.sub_question_entities[:3]:
        extra.extend(row.get("resolved_entities") or [])
    if extra:
        query += " ; " + " ; ".join(str(x) for x in extra[:4])
    return query


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def _infer_entity_type(surface: str, relation: str = "") -> EntityType:
    s = (surface or "").strip()
    low = s.lower()
    rel = (relation or "").lower()
    if not s:
        return "Unknown"
    if any(k in low for k in ("university", "college", "institute", "school", "company", "inc", "ltd")):
        return "Organization"
    if any(k in low for k in ("city", "country", "state", "province", "kingdom", "republic")):
        return "Location"
    if any(k in rel for k in ("author_of", "directed_by", "created_by", "composed_by", "invented_by")):
        return "Work"
    if any(k in rel for k in ("studied_at", "worked_at", "taught_at", "born_in")):
        return "Person"
    if len(s.split()) >= 2 and any(ch.isupper() for ch in s[:1]):
        return "Person"
    return "Unknown"


def _subject_type_from_relation(relation: str) -> EntityType:
    rel = (relation or "").lower()
    if rel in {"author_of", "directed_by", "created_by", "composed_by", "invented_by", "designed_by", "produced_by"}:
        return "Work"
    if rel in {"studied_at", "born_in", "worked_at", "taught_at", "related_to"}:
        return "Person"
    if rel in {"founded_by"}:
        return "Organization"
    if rel in {"capital_of", "located_in"}:
        return "Location"
    return "Unknown"


def _upsert_hop_record(
    hop_memory: HopMemory,
    subquestion_text: str,
    relation: str,
    subject_text: str,
    subject_type: EntityType,
    role: str = "subject",
) -> HopRecord:
    target = _normalize_text(subquestion_text)
    for hop in hop_memory.hops:
        if _normalize_text(hop.subquestion_text) == target:
            return hop
    return hop_memory.add_hop(
        subquestion_text=subquestion_text,
        main_entity_text=subject_text,
        main_entity_type=subject_type,
        relation=relation or "related_to",
        role=role,
    )


def _passage_key(passage: dict[str, Any]) -> str:
    """Stable key for per-hop retry exclusion in the Streamlit loop."""
    pid = str(passage.get("passage_id") or "").strip()
    if pid:
        return f"id:{pid}"
    title = str(passage.get("title") or "").strip().lower()
    head = re.sub(r"\s+", " ", str(passage.get("text") or "")[:120]).strip().lower()
    if title or head:
        return f"th:{title}|{head}"
    return ""


def run_pipeline(
    pipeline: AgentEnhancedGraphRAG,
    question: str,
    contexts: list[dict[str, Any]],
):
    traces = []
    latency: dict[str, float] = {}

    total_start = perf_counter()

    # --------------------------------------------------
    # 1. ENTITY LINKING  (once per question)
    # --------------------------------------------------
    t_link = perf_counter()
    linked = pipeline.entity_linker.link(question)
    latency["entity_linker"] = perf_counter() - t_link
    traces.append({"agent": "entity_linker", "output": linked.as_dict()})

    # --------------------------------------------------
    # 2. DECOMPOSER  (once per question)
    # --------------------------------------------------
    t0 = perf_counter()
    decomposition = pipeline.decomposer.run(question, entity_linking=linked)
    latency["decomposer"] = perf_counter() - t0
    traces.append({"agent": "decomposer", "output": asdict(decomposition)})

    base_entity_context = build_retrieval_entity_context(
        linked, decomposition.sub_question_entities, decomposition.relation_sequence
    )
    ref_anchors = linked.as_dict().get("referential_anchors", [])
    decomp_dict = asdict(decomposition)

    # --------------------------------------------------
    # 3-5. RETRIEVE → REASON → CRITIC with retry loop
    #
    # On the first pass the pivot entity is whatever the entity linker
    # found (e.g. "the Hobbit").  If the critic rejects and the reasoner
    # produced a short named-entity answer (e.g. "J. R. R. Tolkien"), that
    # answer becomes the new pivot for the next retrieval pass so that
    # hop-2 queries (e.g. "Tolkien university") can now be generated.
    # --------------------------------------------------
    max_loops = SETTINGS.retrieval.max_retrieval_loops
    n_sub = len(decomposition.sub_questions)

    selected: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    reasoned = None
    judged = None
    # prefound_pivot: entity resolved in the current intermediate hop, to be used
    # as the retrieval pivot for the NEXT loop.
    prefound_pivot: str | None = None
    # current_pivot: the most recently confirmed intermediate entity, used to
    # substitute <answer of hop k> placeholders into subsequent sub-questions.
    current_pivot: str | None = None
    # _pivot_from_intermediate: True when current_pivot was set from a genuine
    # intermediate hop result (via intermediate_hop_mode).  The PIVOT MATCH GUARD
    # only fires in this case.  When the pivot was set from a REJECTED final loop
    # answer (to re-use as retrieval seed), the guard must NOT fire — that
    # rejected answer might actually be the correct final answer (e.g. "Thomas
    # Keneally" correctly returned twice but critic-rejected on the first pass).
    _pivot_from_intermediate: bool = False
    t_ret = t_rsn = t_crt = 0.0
    hop_memory = HopMemory(original_question=question)
    hop_seen_passage_keys: dict[int, list[str]] = {}
    hop_attempts: dict[int, int] = {}

    for loop_idx in range(max_loops):
        is_retry = loop_idx >= 1

        # Determine the active hop semantics for this loop before retrieval.
        intermediate_hop_mode = (
            loop_idx < n_sub - 1
            and loop_idx < max_loops - 1
            and n_sub >= 2
        )
        relation_seq = decomp_dict.get("relation_sequence") or []
        rel_idx = min(loop_idx, max(0, len(relation_seq) - 1)) if relation_seq else -1
        active_relation = relation_seq[rel_idx] if rel_idx >= 0 else "related_to"

        if intermediate_hop_mode:
            active_subquestion = decomposition.sub_questions[loop_idx]
        elif decomposition.sub_questions:
            active_subquestion = decomposition.sub_questions[min(loop_idx, n_sub - 1)]
        else:
            active_subquestion = question

        if loop_idx == 0 and ref_anchors:
            active_subquestion = resolve_placeholders(active_subquestion, ref_anchors) or active_subquestion
        if current_pivot:
            active_subquestion = _resolve_hop_placeholders(active_subquestion, current_pivot)

        subject_text = ""
        subject_role = "subject"
        if current_pivot:
            subject_text = current_pivot
        elif loop_idx == 0 and ref_anchors:
            anchor = ref_anchors[0]
            anchor_work = str(anchor.get("work") or "").strip()
            anchor_role = str(anchor.get("role") or "subject").strip()
            if anchor_work:
                subject_text = anchor_work
                subject_role = f"{anchor_role}_context"
        if not subject_text:
            canonicals = list(base_entity_context.get("canonical_entities") or [])
            if canonicals:
                subject_text = str(canonicals[0]).strip()
        if not subject_text:
            subject_text = question.strip()

        subject_type = _subject_type_from_relation(active_relation)
        if subject_type == "Unknown":
            subject_type = _infer_entity_type(subject_text, active_relation)

        active_hop = _upsert_hop_record(
            hop_memory=hop_memory,
            subquestion_text=active_subquestion,
            relation=active_relation,
            subject_text=subject_text,
            subject_type=subject_type,
            role=subject_role,
        )
        guarded_query = build_safe_retrieval_query(active_hop, hop_memory)

        # ---- build entity context for this loop ----
        ec: dict[str, Any] = dict(base_entity_context)
        # Critical: retrieval hints must track the ACTIVE hop relation only.
        # Carrying prior-hop relations (e.g. author_of into studied_at) causes
        # retries to keep pulling work-centric passages and miss education facts.
        ec["relation_hints"] = [active_relation] if active_relation else []

        active_hop_idx = int(active_hop.hop_index)
        attempt_no = int(hop_attempts.get(active_hop_idx, 0))
        seen_for_hop = list(hop_seen_passage_keys.get(active_hop_idx, []))
        if seen_for_hop:
            ec["exclude_passage_keys"] = seen_for_hop
        if attempt_no > 0:
            ec["retry_depth"] = attempt_no

        if is_retry:
            # Determine the intermediate pivot for this retry pass.
            # Priority: (1) verified prefound_pivot from intermediate mode,
            # (2) already-verified current_pivot,
            # (3) extractor fallback that passes hop verification.
            if prefound_pivot is not None:
                intermediate = prefound_pivot
                is_valid_pivot = True
                current_pivot = prefound_pivot   # remember for sub-question resolution
                _pivot_from_intermediate = True  # genuine intermediate hop result
                prefound_pivot = None
            elif current_pivot:
                intermediate = current_pivot
                is_valid_pivot = True
            elif decomposition.sub_questions and selected:
                intermediate = ""
                is_valid_pivot = False
                hop1 = _extract_hop1_pivot(pipeline.llm, decomposition.sub_questions, selected)
                if hop1:
                    first_subq = resolve_placeholders(decomposition.sub_questions[0], ref_anchors)
                    first_relation = relation_seq[0] if relation_seq else "related_to"
                    first_subject = subject_text
                    first_type = _subject_type_from_relation(first_relation)
                    if first_type == "Unknown":
                        first_type = _infer_entity_type(first_subject, first_relation)
                    first_hop = _upsert_hop_record(
                        hop_memory=hop_memory,
                        subquestion_text=first_subq,
                        relation=first_relation,
                        subject_text=first_subject,
                        subject_type=first_type,
                        role="subject",
                    )
                    hop_check = verify_hop_answer(hop1, first_hop.subquestion_text, hop_memory)
                    traces.append({
                        "agent": "hop_verifier",
                        "loop": loop_idx + 1,
                        "stage": "retry_pivot_fallback",
                        "approved": hop_check.approved,
                        "explanation": hop_check.explanation,
                        "recovery_action": hop_check.recovery_action,
                        "candidate_answer": hop1,
                    })
                    if hop_check.approved:
                        intermediate = hop1
                        is_valid_pivot = True
                        current_pivot = hop1
                        _pivot_from_intermediate = True
                        if first_hop.answer is None:
                            hop_memory.record_answer(first_hop.hop_index, hop1, 0.55)
            else:
                intermediate = ""
                is_valid_pivot = False

            ec["semantic_blend"] = SETTINGS.retrieval.retrieval_semantic_blend_retry
            if is_valid_pivot:
                ec["canonical_entities"] = [intermediate]
                ec["match_strings"] = [intermediate]
                ec["retry_pivot"] = intermediate
                query = f"{intermediate} ; {question}"
            else:
                ec["relation_hints"] = [active_relation] if active_relation else []
                cans = list(ec.get("canonical_entities") or [])
                query = (
                    " ; ".join(cans[:6]) + " ; " + question if cans else question
                )
        else:
            query = _build_retrieval_query(question, decomposition, ref_anchors)

        # Anchor retrieval to full hop memory to prevent subject drift.
        query = f"{guarded_query}\n\n{query}" if query else guarded_query

        # ---- build query variants pack ----
        ec_arg: dict[str, Any] | None = None
        if ec.get("match_strings"):
            ec_arg = ec
            variants, q_dbg = build_retrieval_query_pack(question, decomp_dict, ec_arg)
            if guarded_query not in variants:
                variants = [guarded_query, *variants]
            ec_arg["retrieval_queries"] = variants
            ec_arg["retrieval_query_debug"] = q_dbg

        # ---- RETRIEVER ----
        t1 = perf_counter()
        retrieval = pipeline.retriever.run(query, contexts, entity_context=ec_arg)
        t_ret += perf_counter() - t1

        selected = retrieval.selected_passages
        chain = retrieval.evidence_chain

        updated_seen = list(seen_for_hop)
        for p in selected:
            k = _passage_key(p)
            if k and k not in updated_seen:
                updated_seen.append(k)
        hop_seen_passage_keys[active_hop_idx] = updated_seen[-120:]
        hop_attempts[active_hop_idx] = attempt_no + 1

        traces.append({
            "agent": "retriever",
            "loop": loop_idx + 1,
            "query": query,
            "retrieval_queries": (ec_arg or {}).get("retrieval_queries"),
            "retrieval_query_debug": (ec_arg or {}).get("retrieval_query_debug"),
            "hop_memory": hop_memory.chain_lines(),
            "selected_passages": selected[:5],
            "graph_stats": retrieval.graph_stats,
        })

        ec_for_agents = ec_arg if ec_arg is not None else base_entity_context

        if intermediate_hop_mode:
            hop_q = active_subquestion
            guard_prompt = get_safe_reasoner_prompt(active_hop.subquestion_text, hop_memory)

            t2 = perf_counter()
            reasoned = pipeline.reasoner.run(
                hop_q,
                evidence_chain=chain,
                selected_passages=selected,
                entity_context=ec_for_agents,
                decomposition=decomp_dict,
                guard_prompt=guard_prompt,
            )
            t_rsn += perf_counter() - t2
            traces.append({
                "agent": "reasoner",
                "loop": loop_idx + 1,
                "thoughts": reasoned.thoughts,
                "answer": reasoned.answer,
                "confidence": reasoned.confidence,
            })

            intermediate = (reasoned.answer or "").strip()
            is_valid_hop = (
                intermediate
                and 1 <= len(intermediate.split()) <= 8
                and not any(w in intermediate.lower() for w in _BAD_PIVOT_MARKERS)
            )

            # Determine whether this hop expects a place-name answer.
            # For place-type relations the LLM often returns a nationality adjective
            # ("British") instead of a proper country name ("United Kingdom") because
            # the passage only contains the adjective form.  Running the focused JSON
            # extractor — which carries an explicit nationality-normalization instruction —
            # reliably fixes this for any nationality in any language.
            hop_relation = active_relation
            place_type_hop = hop_relation in {"born_in", "located_in", "capital_of"}

            if place_type_hop or not is_valid_hop:
                # For place-type hops: always normalize via the JSON extractor.
                # For other hops: only fall back when the reasoner answer is invalid.
                hop_fb = _extract_hop1_pivot(pipeline.llm, [hop_q], selected)
                if hop_fb:
                    intermediate = hop_fb
                    is_valid_hop = True

            hop_check = verify_hop_answer(intermediate, active_hop.subquestion_text, hop_memory)
            traces.append({
                "agent": "hop_verifier",
                "loop": loop_idx + 1,
                "stage": "intermediate",
                "approved": hop_check.approved,
                "explanation": hop_check.explanation,
                "recovery_action": hop_check.recovery_action,
                "candidate_answer": intermediate,
            })

            if is_valid_hop and hop_check.approved:
                prefound_pivot = intermediate
                current_pivot = intermediate
                _pivot_from_intermediate = True  # will be the intermediate pivot next loop
                hop_memory.record_answer(active_hop.hop_index, intermediate, float(reasoned.confidence))

            continue  # ALWAYS proceed to the next loop in intermediate mode
        # --------------------------------------------------------------------------

        # ---- REASONER (final or non-multi-hop) ----
        # The REASONER always receives the ORIGINAL question in the final loop.
        # This ensures retrieval query and reasoning question stay aligned, and
        # avoids pivot-substituted forms confusing the MULTI-HOP PARTIAL ANSWER
        # rule (e.g. "Bill Gates was born in which city?" might misfire because
        # "Bill Gates" is recognised as an intermediate entity by the LLM).
        guard_prompt = get_safe_reasoner_prompt(active_hop.subquestion_text, hop_memory)
        t2 = perf_counter()
        reasoned = pipeline.reasoner.run(
            question,
            evidence_chain=chain,
            selected_passages=selected,
            entity_context=ec_for_agents,
            decomposition=decomp_dict,
            guard_prompt=guard_prompt,
        )
        t_rsn += perf_counter() - t2
        traces.append({
            "agent": "reasoner",
            "loop": loop_idx + 1,
            "thoughts": reasoned.thoughts,
            "answer": reasoned.answer,
            "confidence": reasoned.confidence,
        })

        # ---- PIVOT MATCH GUARD -----------------------------------------------
        # Detect when the MULTI-HOP PARTIAL ANSWER rule fires incorrectly in the
        # final loop: the reasoner returns the already-known intermediate entity
        # (current_pivot) instead of the final answer.  In that case, force
        # "Not enough information" — the final answer is genuinely absent.
        # Example: pivot="Christopher Nolan", question="...studied where?",
        #          bad answer="Christopher Nolan" → override to "Not enough information"
        #
        # CRITICAL: the guard fires ONLY when current_pivot came from a genuine
        # intermediate hop (_pivot_from_intermediate=True).  If the pivot was
        # set from a REJECTED final-loop answer (e.g. "Thomas Keneally" was
        # returned correctly but the critic rejected it, then it became the
        # retrieval seed for the next loop), we must NOT override — that answer
        # may be the correct final answer returned again.
        if (
            current_pivot
            and _pivot_from_intermediate
            and reasoned is not None
            and not (reasoned.answer or "").lower().startswith("not enough")
            and _pivot_match(reasoned.answer, current_pivot)
        ):
            from types import SimpleNamespace as _NS
            reasoned = _NS(
                thoughts=["Final answer not found; intermediate entity already resolved"],
                answer="Not enough information in retrieved passages.",
                confidence=0.15,
            )
        # -----------------------------------------------------------------------

        # ---- CAPITAL CITY FALLBACK -----------------------------------------------
        # When the reasoner returns "Not enough information" but we know the country
        # (current_pivot) and the relation chain includes capital_of, ask the LLM
        # a direct single-fact question to get the capital city from world knowledge.
        # This is universal: it works for any country, not just specific ones.
        # -------------------------------------------------------------------------
        _rel_seq = decomp_dict.get("relation_sequence") or []
        _capital_fallback_used = False
        if (
            (reasoned.answer or "").lower().startswith("not enough")
            and current_pivot
            and "capital_of" in _rel_seq
        ):
            _cap_prompt = (
                f"What is the capital city of {current_pivot}?\n"
                "Return ONLY the city name, nothing else:"
            )
            try:
                _cap_resp = pipeline.llm.generate(_cap_prompt, max_new_tokens=15)
                _cap_text = (_cap_resp.text or "").strip().splitlines()[0].strip()
                _cap_text = _cap_text.strip(".,;:()\"'")
                if _cap_text and 1 <= len(_cap_text.split()) <= 4 and "not" not in _cap_text.lower():
                    from types import SimpleNamespace as _NS
                    reasoned = _NS(
                        thoughts=[f"Capital of {current_pivot} resolved from world knowledge"],
                        answer=_cap_text,
                        confidence=0.85,
                    )
                    _capital_fallback_used = True
            except Exception:
                pass
        # -------------------------------------------------------------------------

        # ---- HOP VERIFIER (final answer gate) -----------------------------------
        if not _capital_fallback_used and not (reasoned.answer or "").lower().startswith("not enough"):
            final_hop_check = verify_hop_answer(reasoned.answer, active_hop.subquestion_text, hop_memory)
            traces.append({
                "agent": "hop_verifier",
                "loop": loop_idx + 1,
                "stage": "final",
                "approved": final_hop_check.approved,
                "explanation": final_hop_check.explanation,
                "recovery_action": final_hop_check.recovery_action,
                "candidate_answer": reasoned.answer,
            })
            if final_hop_check.approved:
                hop_memory.record_answer(active_hop.hop_index, reasoned.answer, float(reasoned.confidence))
            else:
                from types import SimpleNamespace as _NS
                reasoned = _NS(
                    thoughts=["Hop verifier rejected answer due to entity drift or relation mismatch"],
                    answer="Not enough information in retrieved passages.",
                    confidence=min(float(reasoned.confidence), 0.2),
                )
        # -------------------------------------------------------------------------

        # ---- CRITIC ----
        t3 = perf_counter()
        if _capital_fallback_used:
            # The answer was resolved from world knowledge (capital city lookup).
            # The critic cannot verify it against retrieved passages, so we
            # accept it directly rather than having the critic incorrectly reject it.
            from types import SimpleNamespace as _NS
            judged = _NS(
                approved=True,
                critique=(
                    f"Capital city of {current_pivot} resolved from world knowledge."
                ),
                confidence=0.85,
            )
            t_crt += perf_counter() - t3
            traces.append({
                "agent": "critic",
                "loop": loop_idx + 1,
                "approved": judged.approved,
                "critique": judged.critique,
                "confidence": judged.confidence,
            })
            break
        # The CRITIC receives a simpler, directly verifiable question.
        # Substituting the resolved pivot into the original question via the
        # referential-anchor surface (e.g. "the founder of Microsoft" → "Bill
        # Gates") gives the critic a single-hop question it can evaluate against
        # the current-loop passages WITHOUT demanding the full multi-hop chain.
        critic_query = question
        if current_pivot and ref_anchors and n_sub >= 2:
            for anchor in ref_anchors:
                surface = (anchor.get("surface") or "").strip()
                if surface and surface.lower() in question.lower():
                    candidate = re.sub(
                        re.escape(surface), current_pivot, question, flags=re.IGNORECASE
                    )
                    if "<" not in candidate and candidate.strip():
                        critic_query = candidate
                    break
        judged = pipeline.critic.run(
            query=critic_query,
            answer=reasoned.answer,
            confidence=reasoned.confidence,
            evidence_count=len(chain),
            evidence_chain=chain,
            selected_passages=selected,
            entity_context=ec_for_agents,
            decomposition=decomp_dict,
        )
        t_crt += perf_counter() - t3
        traces.append({
            "agent": "critic",
            "loop": loop_idx + 1,
            "approved": judged.approved,
            "critique": judged.critique,
            "confidence": judged.confidence,
        })

        if judged.approved or loop_idx >= max_loops - 1:
            break

    latency["retriever+rereanker"] = t_ret
    latency["reasoner"] = t_rsn
    latency["critic"] = t_crt

    # --------------------------------------------------
    # FINALIZE
    # --------------------------------------------------
    answer = pipeline._finalize_answer(
        question=question,
        raw_answer=reasoned.answer,  # type: ignore[union-attr]
        evidence_chain=chain,
        selected_passages=selected,
    )

    total = perf_counter() - total_start

    explainability = heuristic_explainability_score(chain, answer)

    return {
        "answer": answer,
        "confidence": judged.confidence,  # type: ignore[union-attr]
        "approved": judged.approved,  # type: ignore[union-attr]
        "critique": judged.critique,  # type: ignore[union-attr]
        "traces": traces,
        "chain": chain,
        "selected": selected,
        "latency": latency,
        "total": total,
        "explainability": explainability,
        "sub_questions": [
            resolve_placeholders(sq, ref_anchors)
            for sq in decomposition.sub_questions
        ],
    }


# ==========================================================
# UI helpers
# ==========================================================

def _format_latency(seconds: float) -> str:
    """Human-readable latency: always show milliseconds; show seconds when >= 1 s."""
    ms = float(seconds) * 1000.0
    if seconds >= 1.0:
        return f"{ms:.1f} ms ({seconds:.3f} s)"
    return f"{ms:.1f} ms"


def main():
    st.set_page_config(
        page_title="Agent GraphRAG",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Agent-Enhanced GraphRAG")
    st.caption("4 Agents + Passage Reranker + Evidence Chain")

    # ------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------
    st.sidebar.header("Settings")

    model_name = st.sidebar.selectbox(
        "Model",
        [DEFAULT_MODEL, "qwen2.5:14b", "llama3.2:3b"]
    )

    temperature = st.sidebar.slider(
        "Temperature",
        0.0, 1.0, 0.1, 0.05
    )

    loops = st.sidebar.slider(
        "Max Retrieval Loops",
        1, 5, 3
    )

    llm_backend = st.sidebar.radio(
        "LLM backend",
        ("ollama", "mock"),
        format_func=lambda x: "Ollama (local)" if x == "ollama" else "Mock — no Ollama (smoke test)",
        help="Use Mock to exercise the UI when Ollama is not installed or not running.",
    )

    apply_settings(model_name, temperature, loops, llm_backend)
    if llm_backend == "ollama":
        ok, err = ollama_healthcheck()
        if not ok:
            st.sidebar.warning("Ollama not reachable")
            st.sidebar.caption(err)
            st.warning(
                "Ollama does not appear to be running. Install and start the Ollama app, "
                "or select **Mock — no Ollama** in the sidebar to try the pipeline without a real LLM. "
                "Remote server: set environment variable `OLLAMA_HOST` (see https://ollama.com/download )."
            )

    # ------------------------------------------------------
    # LOAD
    # ------------------------------------------------------
    with st.spinner(
        "Loading corpus + models … "
        "(first run: downloads all 3 datasets + builds FAISS index — takes a few minutes)"
    ):
        pipeline, contexts = get_pipeline_and_corpus(
            model_name, temperature, loops, llm_backend
        )

    st.success(f"Loaded {len(contexts):,} passages from 3 datasets (train + validation)")

    # ------------------------------------------------------
    # INPUT
    # ------------------------------------------------------
    question = st.text_input(
        "Ask short question",
        placeholder="director of dunkirk studied where?"
    )

    if st.button("Get Answer", use_container_width=True):

        if not question.strip():
            st.warning("Enter question.")
            st.stop()

        with st.spinner("Running 4-agent pipeline..."):
            try:
                result = run_pipeline(pipeline, question, contexts)
            except ConnectionError as exc:
                st.error(
                    "Could not connect to Ollama. Start the Ollama application (it listens on "
                    "port 11434 by default), or choose **Mock — no Ollama** under LLM backend in the sidebar."
                )
                st.caption(str(exc))
                st.stop()

        # --------------------------------------------------
        # TABS
        # --------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "Answer",
            "Evidence Chain",
            "Retrieved Passages",
            "Agents / Metrics"
        ])

        # ==================================================
        # ANSWER
        # ==================================================
        with tab1:
            st.subheader("Final Answer")
            st.markdown(f"## {result['answer']}")

            c1, c2, c3 = st.columns(3)

            c1.metric("Confidence", f"{result['confidence']:.3f}")
            expl = float(result["explainability"])
            c2.metric(
                "Explainability (0–100)",
                f"{expl * 100:.1f}",
                help="Scaled from the 0–1 heuristic used in evaluation (×100).",
            )
            c3.metric("Approved", "YES" if result["approved"] else "NO")

            st.info(result["critique"])

            if result["sub_questions"]:
                st.markdown("### Multi-hop decomposition")
                for i, q in enumerate(result["sub_questions"], 1):
                    st.markdown(f"- Hop {i}: {q}")

        # ==================================================
        # EVIDENCE CHAIN
        # ==================================================
        with tab2:
            st.subheader("Evidence Chain")

            for i, hop in enumerate(result["chain"], 1):
                with st.expander(f"Hop {i}"):

                    st.write("From:", hop.get("node_from"))
                    st.write("To:", hop.get("node_to"))
                    st.write("Relation:", hop.get("relation"))
                    st.write("Score:", round(float(hop.get("score", 0)), 3))
                    st.write(hop.get("text", ""))

        # ==================================================
        # PASSAGES
        # ==================================================
        with tab3:
            st.subheader("Top Retrieved Passages")

            for i, p in enumerate(result["selected"][:10], 1):
                with st.expander(f"Passage {i}"):

                    st.write("Dataset:", p.get("dataset"))
                    st.write("Title:", p.get("title"))

                    st.write(
                        "Final Score:",
                        round(float(p.get("score", 0)), 3)
                    )

                    if "rerank_score" in p:
                        st.write(
                            "Reranker:",
                            round(float(p["rerank_score"]), 3)
                        )

                    st.write(p.get("text", ""))

        # ==================================================
        # TRACES / METRICS
        # ==================================================
        with tab4:
            st.subheader("Agent Traces")

            for tr in result["traces"]:
                loop_label = f" — loop {tr['loop']}" if "loop" in tr else ""
                with st.expander(tr["agent"].upper() + loop_label):
                    st.json(tr)

            if result.get("debug_metrics"):
                st.subheader("Benchmark debug metrics")
                st.json(result["debug_metrics"])

            st.subheader("Latency")

            lat = result["latency"]
            order = [
                ("Entity linker", "entity_linker"),
                ("Decomposer", "decomposer"),
                ("Retriever + reranker", "retriever+rereanker"),
                ("Reasoner", "reasoner"),
                ("Critic", "critic"),
            ]
            labels = [lbl for lbl, key in order if key in lat]
            cols = st.columns(len(labels) or 1)
            col_idx = 0
            for label, key in order:
                if key not in lat:
                    continue
                cols[col_idx].metric(label, _format_latency(lat[key]))
                col_idx += 1

            st.metric("Total wall time", _format_latency(result["total"]))


if __name__ == "__main__":
    main()