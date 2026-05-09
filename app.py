# app.py - Streamlit UI (same agents as run_pipeline)

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

DEFAULT_MODEL = "qwen2.5:7b"

# full train+val unless capped; corpus + FAISS cached on disk after first build
CORPUS_MAX_PER_SPLIT: int | None = None
CORPUS_SPLITS = DEFAULT_SPLITS

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


@st.cache_resource(show_spinner=False)
def get_pipeline_and_corpus(
    model_name: str,
    temperature: float,
    loops: int,
    llm_backend: str,
):
    """Build/load pipeline + unified corpus + FAISS once per Streamlit session."""
    apply_settings(model_name, temperature, loops, llm_backend)
    pipeline = AgentEnhancedGraphRAG()

    corpus = build_unified_corpus(
        max_per_split=CORPUS_MAX_PER_SPLIT,
        splits=CORPUS_SPLITS,
    )

    # index path tied to corpus fingerprint
    _, faiss_path = corpus_paths(CORPUS_MAX_PER_SPLIT, CORPUS_SPLITS)

    pipeline.retriever.retriever.preload_corpus(corpus, index_path=faiss_path)

    return pipeline, corpus


_BAD_PIVOT_MARKERS = (
    "insufficient", "not enough", "cannot", "unable", "no evidence", "i don",
)


def _extract_hop1_pivot(
    llm: Any,
    sub_questions: list[str],
    passages: list[dict[str, Any]],
) -> str | None:
    """First sub-q answer from passages via strict JSON; None if not found."""
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
    """True if answer is the same entity as pivot (normalised) - catches bogus 'final' = intermediate."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", (s or "").lower().strip())
    a, p = norm(answer), norm(pivot)
    return bool(a and p and (a == p or (len(p) > 4 and p in a) or (len(a) > 4 and a in p)))


def _resolve_hop_placeholders(sub_question: str, prev_answer: str) -> str:
    """Swap <answer of hop k> placeholders for the latest resolved span."""
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

    t_link = perf_counter()
    linked = pipeline.entity_linker.link(question)
    latency["entity_linker"] = perf_counter() - t_link
    traces.append({"agent": "entity_linker", "output": linked.as_dict()})

    t0 = perf_counter()
    decomposition = pipeline.decomposer.run(question, entity_linking=linked)
    latency["decomposer"] = perf_counter() - t0
    traces.append({"agent": "decomposer", "output": asdict(decomposition)})

    base_entity_context = build_retrieval_entity_context(
        linked, decomposition.sub_question_entities, decomposition.relation_sequence
    )
    ref_anchors = linked.as_dict().get("referential_anchors", [])
    decomp_dict = asdict(decomposition)

    # retrieve → reason → critic; rejected short entities become next loop's retrieval pivot
    max_loops = SETTINGS.retrieval.max_retrieval_loops
    n_sub = len(decomposition.sub_questions)

    selected: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    reasoned = None
    judged = None
    prefound_pivot: str | None = None  # verified intermediate, seeds next loop retrieval
    current_pivot: str | None = None  # for placeholder fill in later subqs
    # pivot-match guard only if pivot came from intermediate hop, not from failed final + retry seed
    _pivot_from_intermediate: bool = False
    t_ret = t_rsn = t_crt = 0.0
    hop_memory = HopMemory(original_question=question)
    hop_seen_passage_keys: dict[int, list[str]] = {}
    hop_attempts: dict[int, int] = {}

    for loop_idx in range(max_loops):
        is_retry = loop_idx >= 1

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

        ec: dict[str, Any] = dict(base_entity_context)
        # one relation hint - mixing prior hops pulls wrong passage type on retry
        ec["relation_hints"] = [active_relation] if active_relation else []

        active_hop_idx = int(active_hop.hop_index)
        attempt_no = int(hop_attempts.get(active_hop_idx, 0))
        seen_for_hop = list(hop_seen_passage_keys.get(active_hop_idx, []))
        if seen_for_hop:
            ec["exclude_passage_keys"] = seen_for_hop
        if attempt_no > 0:
            ec["retry_depth"] = attempt_no

        if is_retry:
            # pivot: prefound → current_pivot → verified hop1 extract
            if prefound_pivot is not None:
                intermediate = prefound_pivot
                is_valid_pivot = True
                current_pivot = prefound_pivot
                _pivot_from_intermediate = True
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

        query = f"{guarded_query}\n\n{query}" if query else guarded_query

        ec_arg: dict[str, Any] | None = None
        if ec.get("match_strings"):
            ec_arg = ec
            variants, q_dbg = build_retrieval_query_pack(question, decomp_dict, ec_arg)
            if guarded_query not in variants:
                variants = [guarded_query, *variants]
            ec_arg["retrieval_queries"] = variants
            ec_arg["retrieval_query_debug"] = q_dbg

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

            # place hops: LLM often returns nationality adjective; JSON extractor normalises to country
            hop_relation = active_relation
            place_type_hop = hop_relation in {"born_in", "located_in", "capital_of"}

            if place_type_hop or not is_valid_hop:
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
                _pivot_from_intermediate = True
                hop_memory.record_answer(active_hop.hop_index, intermediate, float(reasoned.confidence))

            continue  # intermediate mode: always next loop
        # final loop: reason on original question (not pivot-substituted text) so partial-answer heuristics don't misfire
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

        # if final answer == intermediate pivot (and pivot was from a real intermediate hop), abstain
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
        # capital in corpus often missing - one-shot world-knowledge ask when chain has capital_of
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
        t3 = perf_counter()
        if _capital_fallback_used:
            # capital fallback isn't in passages - skip critic
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
        # critic: substitute referential surface with pivot so check is single-hop vs current passages
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


def _format_latency(seconds: float) -> str:
    """ms always; add seconds if >= 1s."""
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

    with st.spinner(
        "Loading corpus + models … "
        "(first run: downloads all 3 datasets + builds FAISS index — takes a few minutes)"
    ):
        pipeline, contexts = get_pipeline_and_corpus(
            model_name, temperature, loops, llm_backend
        )

    st.success(f"Loaded {len(contexts):,} passages from 3 datasets (train + validation)")

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

        tab1, tab2, tab3, tab4 = st.tabs([
            "Answer",
            "Evidence Chain",
            "Retrieved Passages",
            "Agents / Metrics"
        ])

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

        with tab2:
            st.subheader("Evidence Chain")

            for i, hop in enumerate(result["chain"], 1):
                with st.expander(f"Hop {i}"):

                    st.write("From:", hop.get("node_from"))
                    st.write("To:", hop.get("node_to"))
                    st.write("Relation:", hop.get("relation"))
                    st.write("Score:", round(float(hop.get("score", 0)), 3))
                    st.write(hop.get("text", ""))

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