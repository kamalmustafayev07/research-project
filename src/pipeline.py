"""LangGraph pipeline for Agent-Enhanced GraphRAG with baseline methods."""

from __future__ import annotations

from dataclasses import asdict
import re
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.critic import CriticAgent
from src.agents.entity_linker import EntityLinker, build_retrieval_entity_context, entity_linking_result_from_dict
from src.graph.retrieval_query_builder import build_retrieval_query_pack, resolve_placeholders
from src.agents.graph_retriever import GraphRetrieverAgent
from src.agents.hop_safety import (
    EntityType,
    HopMemory,
    build_reasoner_guard_prompt,
    expected_answer_granularity,
    expected_answer_type,
    hop_memory_from_dict,
    hop_memory_to_dict,
    infer_entity_type,
)
from src.agents.query_decomposer import QueryDecomposerAgent
from src.agents.react_reasoner import ReActReasonerAgent
from src.agents.hop_execution_context import attach_hop_execution_context
from src.agents.relation_scorer import (
    generate_relation_prototype,
    prototype_from_dict,
    prototype_to_dict,
)
from src.config import SETTINGS
from src.utils.llm import LLMClient


class GraphRAGState(TypedDict, total=False):
    """Typed state for LangGraph execution."""

    question: str
    context_passages: list[dict[str, Any]]
    entity_linking: dict[str, Any]
    entity_context: dict[str, Any]
    decomposition: dict[str, Any]
    evidence_chain: list[dict[str, Any]]
    selected_passages: list[dict[str, Any]]
    graph_stats: dict[str, int]
    thoughts: list[str]
    answer: str
    confidence: float
    critique: str
    approved: bool
    retrieval_loops: int
    max_retrieval_loops: int
    latency_breakdown: dict[str, float]
    # Persistent multi-hop state machine (serialised HopMemory).
    # Tracks the *active* hop independently of the retrieval-loop counter so a
    # rejected hop is retried without advancing past it.
    hop_memory: dict[str, Any]
    current_hop_index: int  # 1-based; the locked active hop being solved
    hop_attempts: int        # attempts spent on the current hop
    max_hop_attempts: int    # per-hop attempt budget
    final_answer: str        # set only when the last hop is approved
    # Cache of LLM-generated relation prototypes keyed by hop_index (as str).
    # Lets the retriever score passages with hop-aware cues conditioned on
    # prior-hop answers, replacing the static relation-keyword table.
    relation_prototypes: dict[str, Any]
    # Per-hop memory of passage keys already used in previous attempts.
    # Used to avoid retry stagnation (same top passages every loop).
    hop_seen_passage_keys: dict[str, list[str]]


class AgentEnhancedGraphRAG:
    """Production pipeline with four agents and critic-to-retriever feedback."""

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.entity_linker = EntityLinker()
        self.decomposer = QueryDecomposerAgent(self.llm)
        self.retriever = GraphRetrieverAgent()
        self.reasoner = ReActReasonerAgent(self.llm)
        self.critic = CriticAgent(self.llm)
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(GraphRAGState)
        builder.add_node("entity_linker", self._entity_link)
        builder.add_node("decomposer", self._decompose)
        builder.add_node("retriever", self._retrieve)
        builder.add_node("reasoner", self._reason)
        builder.add_node("critic", self._critic)
        # Hop-state transition nodes — these make hop progression explicit
        # rather than implicit in the loop counter.
        builder.add_node("advance_hop", self._advance_hop)
        builder.add_node("retry_same_hop", self._on_retry_same_hop)
        builder.add_node("finalize", self._finalize_hop_memory)

        builder.add_edge(START, "entity_linker")
        builder.add_edge("entity_linker", "decomposer")
        builder.add_edge("decomposer", "retriever")
        builder.add_edge("retriever", "reasoner")
        builder.add_edge("reasoner", "critic")
        builder.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "advance": "advance_hop",
                "retry": "retry_same_hop",
                "done": "finalize",
            },
        )
        # advance_hop / retry_same_hop both lead back to retrieval for the
        # next attempt (with updated state).
        builder.add_edge("advance_hop", "retriever")
        builder.add_edge("retry_same_hop", "retriever")
        builder.add_edge("finalize", END)
        return builder.compile()

    @staticmethod
    def _updated_latency(state: GraphRAGState, key: str, elapsed: float) -> dict[str, float]:
        totals = dict(state.get("latency_breakdown", {}))
        totals[key] = float(totals.get(key, 0.0)) + float(elapsed)
        return totals

    def _entity_link(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()
        linked = self.entity_linker.link(state["question"])
        elapsed = perf_counter() - start
        return {
            "entity_linking": linked.as_dict(),
            "latency_breakdown": self._updated_latency(state, "entity_linker", elapsed),
        }

    def _decompose(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()
        link = entity_linking_result_from_dict(state.get("entity_linking"))
        result = self.decomposer.run(state["question"], entity_linking=link)
        entity_context: dict[str, Any] = {}
        if link is not None:
            entity_context = build_retrieval_entity_context(link, result.sub_question_entities, result.relation_sequence)

        # Build the persistent hop-state machine. Each hop is created with an
        # explicit subject anchor; for hops k>1 the subject is initially a
        # placeholder that gets resolved when hop k-1 is approved. This is the
        # data structure that prevents loop-counter regression — the "active
        # hop" is now an explicit field, not derived from retrieval_loops.
        hop_memory = self._build_initial_hop_memory(
            question=state["question"],
            decomposition=asdict(result),
            entity_context=entity_context,
        )

        elapsed = perf_counter() - start
        return {
            "decomposition": asdict(result),
            "entity_context": entity_context,
            "hop_memory": hop_memory_to_dict(hop_memory),
            "current_hop_index": 1,
            "hop_attempts": 0,
            "max_hop_attempts": int(state.get("max_hop_attempts") or 2),
            "hop_seen_passage_keys": {},
            "latency_breakdown": self._updated_latency(state, "decomposer", elapsed),
        }

    @staticmethod
    def _build_initial_hop_memory(
        question: str,
        decomposition: dict[str, Any],
        entity_context: dict[str, Any] | None,
    ) -> HopMemory:
        """Create a HopMemory aligned with the decomposer's plan.

        Hop 1's subject anchor is the first canonical entity (typically the
        referential work or the question's leading proper noun). Later hops
        receive a placeholder subject ``<answer of hop k-1>`` that the routing
        logic resolves once the prior hop is approved. No entity-, dataset-,
        or example-specific logic is involved: the structure is built purely
        from the generic decomposer output and the entity context.
        """
        sub_questions = list(decomposition.get("sub_questions") or [])
        relation_sequence = list(decomposition.get("relation_sequence") or [])
        ec = entity_context or {}

        # Hop 1 subject: the first canonical entity (or the leading work
        # extracted by the entity linker). Empty string is acceptable — the
        # verifier only uses subject text for equality regression checks.
        canonical = list(ec.get("canonical_entities") or [])
        anchors = list(ec.get("referential_anchors") or [])
        hop1_subject = ""
        if anchors:
            hop1_subject = str(anchors[0].get("work") or "").strip()
        if not hop1_subject and canonical:
            hop1_subject = str(canonical[0]).strip()
        # Type-classify hop 1's subject heuristically; falls back to Unknown.
        hop1_subject_type: EntityType = infer_entity_type(hop1_subject)

        memory = HopMemory(original_question=question)
        for idx, sq in enumerate(sub_questions):
            relation = relation_sequence[idx] if idx < len(relation_sequence) else "related_to"
            if idx == 0:
                memory.add_hop(
                    subquestion_text=sq,
                    main_entity_text=hop1_subject,
                    main_entity_type=hop1_subject_type,
                    relation=relation,
                    role="subject",
                )
            else:
                memory.add_hop(
                    subquestion_text=sq,
                    main_entity_text=f"<answer of hop {idx}>",
                    main_entity_type="Unknown",
                    relation=relation,
                    role="subject",
                )
        if not memory.hops:
            # Single-hop fallback: treat the whole question as one hop.
            memory.add_hop(
                subquestion_text=question,
                main_entity_text=hop1_subject,
                main_entity_type=hop1_subject_type,
                relation="related_to",
                role="subject",
            )
        return memory

    def _retrieve(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()
        query = state["question"]
        decomposition = state.get("decomposition", {})
        sub_questions = decomposition.get("sub_questions", [])
        sub_plans = decomposition.get("sub_question_entities") or []
        extra_bits: list[str] = []
        for row in sub_plans[:3]:
            for e in row.get("resolved_entities") or []:
                if str(e).strip():
                    extra_bits.append(str(e).strip())
        head = [query, *sub_questions[:2], *extra_bits[:4]]
        retrieval_query = " ; ".join([h for h in head if h])

        # Determine the active hop and its pivot from the persistent state
        # machine. The pivot is the *previous approved hop's answer* — never
        # a rejected candidate or a free-floating state.answer — which is
        # what guarantees retrieval stays on-mission for the active hop.
        memory = self._hop_memory(state)
        active_idx = self._current_hop_index(state)
        pivot_entity = ""
        if memory is not None and active_idx > 1 and len(memory.hops) >= active_idx - 1:
            prior = memory.hops[active_idx - 2]
            if prior.answer:
                pivot_entity = prior.answer.strip()

        seen_by_hop = dict(state.get("hop_seen_passage_keys") or {})
        hop_seen_key = str(active_idx)
        seen_for_hop = list(seen_by_hop.get(hop_seen_key) or [])

        # LLM-generated relation prototype for the active hop. Replaces the
        # static keyword table with hop-aware cues conditioned on prior-hop
        # answers. Cached on state by hop_index so retries reuse it.
        prototype_payload = self._ensure_relation_prototype(state, memory, active_idx)
        prototype_cache = dict(state.get("relation_prototypes") or {})
        if prototype_payload is not None:
            prototype_cache[str(active_idx)] = prototype_to_dict(prototype_payload)

        raw_ec = state.get("entity_context") or {}
        entity_context: dict[str, Any] | None = None
        resolved_hop_question = ""
        if raw_ec.get("match_strings"):
            entity_context = dict(raw_ec)
            if prototype_payload is not None:
                # Surface the prototype to scoring/reasoning consumers.
                entity_context["relation_prototype"] = prototype_to_dict(prototype_payload)
                # Also expose a flat cue list so the reasoner's focus-sentence
                # extractor can pull cue-bearing sentences from truncated
                # passage tails.
                entity_context["relation_prototype_cues"] = list(
                    prototype_payload.cue_phrases
                )
            if pivot_entity:
                # Hop k>1: pivot retrieval on the prior approved answer.
                entity_context["semantic_blend"] = SETTINGS.retrieval.retrieval_semantic_blend_retry
                entity_context["canonical_entities"] = [pivot_entity]
                entity_context["match_strings"] = [pivot_entity]
                entity_context["retry_pivot"] = pivot_entity
                retrieval_query = f"{pivot_entity} ; {state['question']}"
                entity_context["retrieval_fallback"] = (
                    entity_context.get("retrieval_fallback") or ""
                ) + f";hop{active_idx}:pivot={pivot_entity[:40]}"
            elif int(state.get("hop_attempts") or 0) >= 1:
                # Hop 1 retry — broaden retrieval slightly.
                entity_context["semantic_blend"] = SETTINGS.retrieval.retrieval_semantic_blend_retry
                entity_context["retrieval_fallback"] = (
                    entity_context.get("retrieval_fallback") or ""
                ) + ";hop1:retry"

            retry_depth = int(state.get("hop_attempts") or 0)
            if retry_depth > 0:
                entity_context["retry_depth"] = retry_depth
            if seen_for_hop:
                entity_context["exclude_passage_keys"] = seen_for_hop

            # Shared hop contract: same objective, subject, type, granularity, and
            # prior chain for retriever scoring, reasoner prompts, and critic checks.
            active_resolve = self._resolve_active_hop_subject(state)
            if active_resolve is not None:
                mem_r, hop_r = active_resolve
                # Critical: relation hints must follow the ACTIVE hop only.
                # Mixing prior-hop relations into retries (e.g. author_of during
                # a studied_at hop) keeps retrieval stuck on work-centric passages.
                entity_context["relation_hints"] = [hop_r.relation] if hop_r.relation else []
                rsq = self._resolved_subquestion(mem_r, hop_r)
                resolved_hop_question = rsq
                entity_context = attach_hop_execution_context(
                    entity_context, mem_r, hop_r, rsq
                )

            feedback_terms: list[str] = []
            if retry_depth > 0:
                critique = str(state.get("critique") or "")
                feedback_terms = self._extract_retry_feedback_terms(
                    f"{critique} {resolved_hop_question}"
                )
                if feedback_terms:
                    entity_context["retry_feedback_terms"] = feedback_terms

            variants, q_dbg = build_retrieval_query_pack(
                state["question"],
                decomposition,
                entity_context,
            )

            if feedback_terms:
                seeds: list[str] = []
                if pivot_entity:
                    seeds.append(pivot_entity)
                if resolved_hop_question:
                    seeds.append(resolved_hop_question)
                seeds.append(state["question"])
                for seed in seeds[:3]:
                    for term in feedback_terms[:5]:
                        variants.append(f"{seed} {term}"[:420])

            # Deduplicate while preserving order after dynamic retry expansion.
            deduped: list[str] = []
            seen_q: set[str] = set()
            for q in variants:
                t = str(q or "").strip()
                if not t:
                    continue
                k = t.lower()
                if k in seen_q:
                    continue
                seen_q.add(k)
                deduped.append(t)
            variants = deduped[:32]

            # Fold the prototype's focus_query in as an additional retrieval
            # variant (deduplicated). It carries the LLM's best guess at a
            # hop-aware search query and is conditioned on prior hops.
            if prototype_payload is not None and prototype_payload.focus_query:
                fq = prototype_payload.focus_query.strip()
                if fq and fq not in variants:
                    variants.append(fq)
            entity_context["retrieval_queries"] = variants
            entity_context["retrieval_query_debug"] = q_dbg
        # If no match_strings, keep retrieval_query from head (question + sub-questions + entities); do not drop hops.

        output = self.retriever.run(retrieval_query, state["context_passages"], entity_context=entity_context)

        updated_seen = list(seen_for_hop)
        for p in output.selected_passages:
            key = self._passage_key(p)
            if key and key not in updated_seen:
                updated_seen.append(key)
        seen_by_hop[hop_seen_key] = updated_seen[-120:]

        loops = state.get("retrieval_loops", 0) + 1
        elapsed = perf_counter() - start
        return {
            "evidence_chain": output.evidence_chain,
            "selected_passages": output.selected_passages,
            "graph_stats": output.graph_stats,
            "retrieval_loops": loops,
            "entity_context": entity_context if entity_context is not None else state.get("entity_context"),
            "relation_prototypes": prototype_cache,
            "hop_seen_passage_keys": seen_by_hop,
            "latency_breakdown": self._updated_latency(state, "retriever", elapsed),
        }

    def _ensure_relation_prototype(
        self,
        state: GraphRAGState,
        memory: HopMemory | None,
        active_idx: int,
    ):
        """Get-or-generate the LLM relation prototype for the active hop.

        Caching strategy:
        - One LLM call per (hop_index) regardless of retry attempts. The
          prototype is conditioned on the prior-hop chain, which is stable for
          the lifetime of a hop, so retries always reuse the same prototype.
        - On any failure (LLM unavailable, malformed JSON), an empty prototype
          is returned and the legacy keyword table seamlessly takes over.
        """
        if memory is None or not memory.hops:
            return None
        if active_idx > len(memory.hops):
            return None
        cache = state.get("relation_prototypes") or {}
        cached = cache.get(str(active_idx))
        if cached:
            return prototype_from_dict(cached)
        active = self._resolve_active_hop_subject(state)
        if active is None:
            return None
        memory_resolved, hop = active
        resolved_subq = self._resolved_subquestion(memory_resolved, hop)
        return generate_relation_prototype(
            llm=self.llm,
            hop=hop,
            prior_hops=memory_resolved.previous_hops(hop.hop_index),
            original_question=memory_resolved.original_question,
            resolved_subquestion=resolved_subq,
            expected_type=expected_answer_type(hop.relation, hop.subquestion_text),
            expected_granularity=expected_answer_granularity(
                hop.relation, hop.subquestion_text
            ),
        )

    # ------------------------------------------------------------------
    # Hop-state accessors — every node reads the *active* hop from the
    # persistent HopMemory state, never re-derives it from a loop counter.
    # ------------------------------------------------------------------

    def _hop_memory(self, state: GraphRAGState) -> HopMemory | None:
        return hop_memory_from_dict(state.get("hop_memory"))

    def _current_hop_index(self, state: GraphRAGState) -> int:
        idx = int(state.get("current_hop_index") or 1)
        return max(1, idx)

    def _is_intermediate_hop_mode(self, state: GraphRAGState) -> bool:
        """Backwards-compat shim — true when the active hop is not the last hop.

        Retained for any external callers; the new code paths read the active
        hop from ``hop_memory`` directly via ``_active_hop``.
        """
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            return False
        return self._current_hop_index(state) < len(memory.hops)

    def _active_hop(self, state: GraphRAGState):
        """Return the locked active HopRecord, or None if state isn't ready."""
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            return None
        idx = self._current_hop_index(state)
        if idx > len(memory.hops):
            idx = len(memory.hops)
        return memory.get_hop(idx)

    @staticmethod
    def _substitute_placeholders(text: str, memory: HopMemory) -> str:
        """Replace ``<answer of hop N>`` placeholders with prior approved answers."""
        def _replace(match: re.Match) -> str:
            try:
                idx = int(match.group(1))
            except (TypeError, ValueError):
                return match.group(0)
            if 1 <= idx <= len(memory.hops):
                ans = memory.hops[idx - 1].answer
                if ans:
                    return ans
            return match.group(0)

        return re.sub(
            r"<\s*answer\s+of\s+hop\s+(\d+)\s*>",
            _replace,
            text or "",
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _passage_key(passage: dict[str, Any]) -> str:
        pid = str(passage.get("passage_id") or "").strip()
        if pid:
            return f"id:{pid}"
        title = str(passage.get("title") or "").strip().lower()
        head = re.sub(r"\s+", " ", str(passage.get("text") or "")[:120]).strip().lower()
        if title or head:
            return f"th:{title}|{head}"
        return ""

    @staticmethod
    def _extract_retry_feedback_terms(text: str, max_terms: int = 6) -> list[str]:
        """Extract lightweight lexical cues for retry query diversification."""
        stop = {
            "the", "and", "for", "with", "from", "into", "that", "this", "more", "need",
            "needs", "needed", "passage", "passages", "retrieved", "retrieve", "retrieval",
            "evidence", "answer", "question", "information", "about", "which", "what", "where",
            "when", "who", "does", "did", "have", "has", "was", "were", "not", "enough",
            "provide", "provides", "direct", "directly", "relevant", "stronger", "supporting",
        }
        toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", str(text or "").lower())
        out: list[str] = []
        for t in toks:
            if t in stop:
                continue
            if t.isdigit():
                continue
            if t not in out:
                out.append(t)
            if len(out) >= max_terms:
                break
        return out

    def _resolve_active_hop_subject(self, state: GraphRAGState) -> tuple[HopMemory, Any] | None:
        """Return (memory, hop) where the active hop has its subject resolved.

        If the active hop's ``main_entity.text`` is a placeholder, it is
        replaced with the prior hop's approved answer in-place. Returns None
        if the state is not yet initialised.
        """
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            return None
        idx = self._current_hop_index(state)
        if idx > len(memory.hops):
            return None
        hop = memory.get_hop(idx)

        # If the subject is still a placeholder, substitute it now using the
        # prior hop's approved answer. This is the moment the retriever and
        # reasoner gain a concrete subject anchor for hop k>1.
        if hop.main_entity.text and "<answer of hop" in hop.main_entity.text.lower():
            resolved = self._substitute_placeholders(hop.main_entity.text, memory).strip()
            if resolved and "<answer of hop" not in resolved.lower():
                from src.agents.hop_safety import EntityState
                hop.main_entity = EntityState(
                    text=resolved,
                    entity_type=infer_entity_type(resolved),
                    role=hop.main_entity.role or "subject",
                )
        return memory, hop

    def _resolved_subquestion(self, memory: HopMemory, hop) -> str:
        """Return the active hop's sub-question with all placeholders resolved."""
        return self._substitute_placeholders(hop.subquestion_text, memory)

    def _reason(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()

        # Resolve the active hop and lock the reasoner to its sub-question.
        # The active hop comes from HopMemory, NOT from the loop counter, so
        # a rejected final hop is retried with the same sub-question instead
        # of regressing to the original multi-hop question.
        question = state["question"]
        guard_prompt: str | None = None
        active = self._resolve_active_hop_subject(state)
        if active is not None:
            memory, hop = active
            ref_anchors = (state.get("entity_linking") or {}).get("referential_anchors") or []
            # Resolve placeholders using prior approved answers from HopMemory
            # first (authoritative), then fall back to descriptive replacement
            # via the entity-linker referential anchors for any unfilled slots.
            hop_q = self._resolved_subquestion(memory, hop)
            hop_q = resolve_placeholders(hop_q, ref_anchors) or hop_q
            question = hop_q

            # Guard prompt: encodes the active sub-question, expected answer
            # type, and prior-hop subjects/answers as forbidden outputs. This
            # is a SOFT defence; the verifier in _critic enforces it as a HARD
            # check.
            guard_prompt = build_reasoner_guard_prompt(hop, memory)

        output = self.reasoner.run(
            question,
            state.get("evidence_chain", []),
            selected_passages=state.get("selected_passages", []),
            entity_context=state.get("entity_context"),
            decomposition=state.get("decomposition"),
            guard_prompt=guard_prompt,
        )
        elapsed = perf_counter() - start
        return {
            "thoughts": output.thoughts,
            "answer": output.answer,
            "confidence": output.confidence,
            "latency_breakdown": self._updated_latency(state, "reasoner", elapsed),
        }

    def _critic(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()
        chain = state.get("evidence_chain", [])

        # The critic always evaluates the *active hop's* resolved sub-question
        # — never the original multi-hop question — and ships the active hop
        # record (subject, relation, expected answer type) to the critic so it
        # can do typed regression-aware validation. This is what stops
        # "Oxford" from being rejected as ambiguous and "J. R. R. Tolkien"
        # from sneaking through as a regression.
        query = state["question"]
        active_hop_payload: dict[str, Any] | None = None
        active = self._resolve_active_hop_subject(state)
        if active is not None:
            memory, hop = active
            query = self._resolved_subquestion(memory, hop) or query
            active_hop_payload = {
                "hop_index": hop.hop_index,
                "subquestion_text": hop.subquestion_text,
                "resolved_subquestion": query,
                "subject_text": hop.main_entity.text,
                "subject_type": hop.main_entity.entity_type,
                "relation": hop.relation,
                "expected_answer_type": expected_answer_type(
                    hop.relation, hop.subquestion_text
                ),
                "expected_answer_granularity": expected_answer_granularity(
                    hop.relation, hop.subquestion_text
                ),
                "prior_subjects": [
                    p.main_entity.text for p in memory.previous_hops(hop.hop_index)
                ],
                "prior_answers": [
                    p.answer or "" for p in memory.previous_hops(hop.hop_index)
                ],
            }

        output = self.critic.run(
            query=query,
            answer=state.get("answer", ""),
            confidence=state.get("confidence", 0.0),
            evidence_count=len(chain),
            evidence_chain=chain,
            selected_passages=state.get("selected_passages", []),
            entity_context=state.get("entity_context"),
            decomposition=state.get("decomposition"),
            active_hop=active_hop_payload,
        )
        elapsed = perf_counter() - start
        return {
            "approved": output.approved,
            "critique": output.critique,
            "confidence": output.confidence,
            "latency_breakdown": self._updated_latency(state, "critic", elapsed),
        }

    def _route_after_critic(self, state: GraphRAGState) -> str:
        """Decide what to do after the critic's verdict on the active hop.

        Routing semantics — driven by HopMemory, not the loop counter:

        - approved AND active hop is the LAST hop:  "done"
          (the answer is the final answer; the advance node will record it.)
        - approved AND more hops remain:           "advance"
          (record the answer, increment current_hop_index, retry retrieval.)
        - rejected AND retries remaining:          "retry"
          (same hop, same subject, fresh retrieval/reasoning attempt.)
        - rejected AND budget exhausted:           "done"
          (graceful termination with the best partial state.)
        """
        approved = bool(state.get("approved", False))
        loops = int(state.get("retrieval_loops", 0) or 0)
        max_loops = int(
            state.get("max_retrieval_loops") or SETTINGS.retrieval.max_retrieval_loops
        )
        attempts = int(state.get("hop_attempts") or 0)
        max_attempts = int(state.get("max_hop_attempts") or 2)
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            # Graph not initialised with hop_memory (e.g. from baselines) —
            # fall back to the legacy approval/loop-budget routing.
            if approved or loops >= max_loops:
                return "done"
            return "retry"

        idx = self._current_hop_index(state)
        is_last_hop = idx >= len(memory.hops)
        if approved:
            return "done" if is_last_hop else "advance"
        # Not approved — retry the same hop while we still have a budget.
        if attempts + 1 < max_attempts and loops < max_loops:
            return "retry"
        return "done"

    def _advance_hop(self, state: GraphRAGState) -> GraphRAGState:
        """Persist the approved hop's answer and advance to the next hop.

        This is its own LangGraph node so the state update is atomic with
        respect to the retrieval node that follows.
        """
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            return {}
        idx = self._current_hop_index(state)
        if idx > len(memory.hops):
            return {}
        # Record the approved answer + confidence on the current hop.
        memory.record_answer(
            hop_index=idx,
            answer=str(state.get("answer") or ""),
            confidence=float(state.get("confidence") or 0.0),
        )
        # Advance to the next hop and reset per-hop attempt counter.
        next_idx = min(idx + 1, len(memory.hops))
        return {
            "hop_memory": hop_memory_to_dict(memory),
            "current_hop_index": next_idx,
            "hop_attempts": 0,
        }

    def _on_retry_same_hop(self, state: GraphRAGState) -> GraphRAGState:
        """Increment the per-hop attempt counter without advancing the hop."""
        return {"hop_attempts": int(state.get("hop_attempts") or 0) + 1}

    def _finalize_hop_memory(self, state: GraphRAGState) -> GraphRAGState:
        """At graph end, record the final approved answer in HopMemory."""
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            return {}
        idx = self._current_hop_index(state)
        if idx <= len(memory.hops) and bool(state.get("approved", False)):
            memory.record_answer(
                hop_index=idx,
                answer=str(state.get("answer") or ""),
                confidence=float(state.get("confidence") or 0.0),
            )
        last_answer = (memory.hops[-1].answer or "") if memory.hops else ""
        return {
            "hop_memory": hop_memory_to_dict(memory),
            "final_answer": last_answer or str(state.get("answer") or ""),
        }

    def invoke(self, question: str, context_passages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run the full 4-agent workflow and return answer plus evidence."""
        start_total = perf_counter()
        # Total iteration budget needs to be roughly hops * per-hop attempts so
        # the new state machine can both *advance* between hops and *retry* a
        # rejected hop within its attempt budget.
        max_hop_attempts = 2
        max_loops = max(
            int(SETTINGS.retrieval.max_retrieval_loops),
            max_hop_attempts * 3,
        )
        result = self.graph.invoke(
            {
                "question": question,
                "context_passages": context_passages,
                "retrieval_loops": 0,
                "max_retrieval_loops": max_loops,
                "max_hop_attempts": max_hop_attempts,
                "hop_attempts": 0,
                "current_hop_index": 1,
                "latency_breakdown": {
                    "entity_linker": 0.0,
                    "decomposer": 0.0,
                    "retriever": 0.0,
                    "reasoner": 0.0,
                    "critic": 0.0,
                },
            }
        )
        total_latency = perf_counter() - start_total
        result_breakdown = dict(result.get("latency_breakdown", {}))
        latency_breakdown = {
            "entity_linker": float(result_breakdown.get("entity_linker", 0.0)),
            "decomposer": float(result_breakdown.get("decomposer", 0.0)),
            "retriever": float(result_breakdown.get("retriever", 0.0)),
            "reasoner": float(result_breakdown.get("reasoner", 0.0)),
            "critic": float(result_breakdown.get("critic", 0.0)),
        }
        # Prefer the final-hop answer from HopMemory when the state machine
        # finished cleanly; otherwise fall back to the last reasoner output.
        raw_answer = result.get("final_answer") or result.get("answer", "")
        finalized_answer = self._finalize_answer(
            question=question,
            raw_answer=raw_answer,
            evidence_chain=result.get("evidence_chain", []),
            selected_passages=result.get("selected_passages", []),
        )
        return {
            "answer": finalized_answer,
            "confidence": result.get("confidence", 0.0),
            "evidence_chain": result.get("evidence_chain", []),
            "thoughts": result.get("thoughts", []),
            "graph_stats": result.get("graph_stats", {}),
            "critique": result.get("critique", ""),
            "retrieval_loops": result.get("retrieval_loops", 0),
            "hop_memory": result.get("hop_memory", {}),
            "current_hop_index": result.get("current_hop_index", 1),
            "latency_total": total_latency,
            "latency_breakdown": latency_breakdown,
        }

    def train_retriever_reranker(
        self,
        train_examples: list[dict[str, Any]],
        validation_examples: list[dict[str, Any]] | None = None,
        test_examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Train retrieval reranker on labeled supporting-fact examples."""
        return self.retriever.fit_reranker(
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
        )

    def has_trained_reranker(self) -> bool:
        """Return whether retriever has a trained reranker ready."""
        return self.retriever.has_trained_reranker()

    def _extract_hop1_pivot(
        self,
        sub_questions: list[str],
        passages: list[dict[str, Any]],
    ) -> str | None:
        """Ask the LLM to answer the first sub-question using only the retrieved passages.

        Uses structured JSON output so the answer is parsed reliably regardless of
        how verbose the LLM's preamble is.  Returns a clean entity string for use
        as the retry pivot, or None.
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
            "prominently in the passage.\n\n"
            f"Question: {hop1_q}\n\n"
            f"Passages:\n{context}\n\n"
            "Return STRICT JSON only:\n"
            '{"entity": "<answer, 1-5 words>", "found": true}\n'
            "If the answer is not in the passages:\n"
            '{"entity": "", "found": false}\n\n'
            "JSON:"
        )
        _bad = ("insufficient", "not enough", "cannot", "unable", "no evidence", "i don")
        try:
            response = self.llm.generate(prompt, max_new_tokens=80)
            parsed = self.llm.extract_json(response.text)
            if not isinstance(parsed, dict) or not parsed.get("found"):
                return None
            entity = str(parsed.get("entity") or "").strip().strip("\"'.,;:()")
            if not entity or len(entity.split()) > 8:
                return None
            if any(w in entity.lower() for w in _bad):
                return None
            return entity
        except Exception:
            return None

    def _finalize_answer(
        self,
        question: str,
        raw_answer: Any,
        evidence_chain: list[dict[str, Any]],
        selected_passages: list[dict[str, Any]],
    ) -> str:
        text = str(raw_answer or "").strip()
        if not text:
            return self._fallback_answer(question, evidence_chain, selected_passages)

        parsed = self.llm.extract_json(text)
        if isinstance(parsed, dict):
            for key in ["answer", "final_answer", "response"]:
                candidate = parsed.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate.strip()
                    break

        text = text.replace("```json", "").replace("```", "").strip()
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if first_line:
            text = first_line

        text = re.sub(r"^(answer\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(
            r"^(the answer is|it is|based on the evidence[,\s]*)",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = text.strip(" .\"'")

        if self._is_insufficient_answer(text):
            fallback = self._fallback_answer(question, evidence_chain, selected_passages)
            return fallback or text

        options = re.findall(r"\b([A-Z][A-Za-z0-9'\- ]{1,40}?)\s+or\s+([A-Z][A-Za-z0-9'\- ]{1,40}?)\?", question)
        if options:
            left, right = options[0]
            low = text.lower()
            if left.lower() in low:
                return left.strip()
            if right.lower() in low:
                return right.strip()

        if len(text.split()) > 12:
            text = re.split(r"[\.;]", text, maxsplit=1)[0].strip()

        if not text or text == "{}":
            return self._fallback_answer(question, evidence_chain, selected_passages)
        return text

    @staticmethod
    def _is_insufficient_answer(text: str) -> bool:
        low = str(text or "").strip().lower()
        return (
            "not enough information" in low
            or "insufficient" in low
            or "cannot determine" in low
            or "unable to determine" in low
        )

    def _fallback_answer(
        self,
        question: str,
        evidence_chain: list[dict[str, Any]],
        selected_passages: list[dict[str, Any]],
    ) -> str:
        question_low = question.lower()
        snippets: list[str] = []
        for p in selected_passages[:6]:
            txt = str(p.get("text", "")).strip()
            if txt:
                snippets.append(txt)
        if not snippets:
            for h in evidence_chain[:8]:
                txt = str(h.get("text", "")).strip()
                if txt:
                    snippets.append(txt)
        best_text = "\n".join(snippets)

        if not best_text:
            if selected_passages:
                title = str(selected_passages[0].get("title") or "").strip()
                if title:
                    return title
            q_ent = re.search(r"\b([A-Z][A-Za-z0-9'\-]+(?:\s+[A-Z][A-Za-z0-9'\-]+){0,4})\b", question)
            return q_ent.group(1).strip() if q_ent else "Unknown"

        if any(k in question_low for k in ("university", "college", "school", "alma mater", "attend", "studied")):
            org_pat = re.compile(
                r"\b((?:University|College|Institute|School|Academy)\s+of\s+[A-Z][A-Za-z&.'\- ]+|"
                r"[A-Z][A-Za-z&.'\- ]+\s+(?:University|College|Institute|School|Academy|Oxford|Cambridge))\b"
            )
            org_match = org_pat.search(best_text)
            if org_match:
                return org_match.group(1).strip(" .;,")

        if question_low.startswith("who"):
            by_match = re.search(
                r"\b(?:by|written by|authored by|directed by|founded by|created by|composed by)\s+"
                r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,4})",
                best_text,
            )
            if by_match:
                return by_match.group(1).strip(" .;,")

        if question_low.startswith("when") or "what year" in question_low:
            year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", best_text)
            if year_match:
                return year_match.group(1)

        name_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", best_text)
        if name_match:
            return name_match.group(1)

        if selected_passages:
            top_title = str(selected_passages[0].get("title") or "").strip()
            if top_title:
                return top_title
        return "Unknown"

    def _estimate_confidence(self, answer: str, scores: list[float], evidence_count: int) -> float:
        """Estimate confidence from retrieval/evidence quality and answer completeness."""
        normalized_scores = [max(0.0, min(1.0, (s + 1.0) / 2.0)) for s in scores]
        score_signal = (sum(normalized_scores) / len(normalized_scores)) if normalized_scores else 0.0

        token_count = len(answer.split()) if answer else 0
        answer_signal = min(1.0, token_count / 12.0)
        evidence_signal = min(1.0, evidence_count / 5.0)

        combined = 0.55 * score_signal + 0.25 * answer_signal + 0.20 * evidence_signal
        return round(max(0.0, min(1.0, combined)), 4)

    def dense_rag_baseline(self, question: str, context_passages: list[dict[str, Any]]) -> dict[str, Any]:
        """Baseline B1: dense retrieval + single-pass answer generation."""
        retrieved = self.retriever.retriever.retrieve(question, context_passages)
        top_context = "\n".join(
            [f"{p['title']}: {p['text'][:400]}" for p in retrieved.selected_passages[:4]]
        )
        prompt = (
            "Answer the question only using provided passages. Return only a short answer span (1-8 words). "
            "No explanation.\n"
            f"Question: {question}\nPassages:\n{top_context}"
        )
        response = self.llm.generate(prompt)
        finalized_answer = self._finalize_answer(
            question=question,
            raw_answer=response.text,
            evidence_chain=[],
            selected_passages=retrieved.selected_passages,
        )
        passage_scores = [float(p.get("score", 0.0)) for p in retrieved.selected_passages[:4]]
        confidence = self._estimate_confidence(
            answer=finalized_answer,
            scores=passage_scores,
            evidence_count=len(retrieved.selected_passages[:4]),
        )
        return {
            "answer": finalized_answer,
            "confidence": confidence,
            "evidence_chain": [
                {"source": p["title"], "text": p["text"], "score": p.get("score", 0.0)}
                for p in retrieved.selected_passages[:4]
            ],
        }

    def basic_graphrag_baseline(self, question: str, context_passages: list[dict[str, Any]]) -> dict[str, Any]:
        """Baseline B2: graph retrieval without decomposition/reasoner/critic loop."""
        retrieved = self.retriever.run(question, context_passages)
        evidence = "\n".join(
            [
                f"{hop.get('node_from', '')} -{hop.get('relation', '')}-> {hop.get('node_to', '')} ({hop.get('source', '')})"
                for hop in retrieved.evidence_chain[:8]
            ]
        )
        prompt = (
            "Answer the question from graph evidence. Return only a short answer span (1-8 words). "
            "No explanation.\n"
            f"Question: {question}\nEvidence:\n{evidence}"
        )
        response = self.llm.generate(prompt)
        finalized_answer = self._finalize_answer(
            question=question,
            raw_answer=response.text,
            evidence_chain=retrieved.evidence_chain,
            selected_passages=retrieved.selected_passages,
        )
        evidence_scores = [float(hop.get("score", 0.0)) for hop in retrieved.evidence_chain[:8]]
        confidence = self._estimate_confidence(
            answer=finalized_answer,
            scores=evidence_scores,
            evidence_count=len(retrieved.evidence_chain[:8]),
        )
        return {
            "answer": finalized_answer,
            "confidence": confidence,
            "evidence_chain": retrieved.evidence_chain,
        }
