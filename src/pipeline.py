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
from src.utils.helpers import normalize_text
from src.utils.llm import LLMClient


class GraphRAGState(TypedDict, total=False):
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
    hop_memory: dict[str, Any]  # serialised HopMemory; active hop != retrieval loop idx
    current_hop_index: int
    hop_attempts: int
    max_hop_attempts: int
    final_answer: str
    relation_prototypes: dict[str, Any]  # per-hop LLM relation cues (cached across retries)
    hop_seen_passage_keys: dict[str, list[str]]  # de-stagnate retries


class AgentEnhancedGraphRAG:
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
        """HopMemory from decomposer output; hop>1 subjects start as placeholders until prior hop approved."""
        sub_questions = list(decomposition.get("sub_questions") or [])
        relation_sequence = list(decomposition.get("relation_sequence") or [])
        ec = entity_context or {}

        canonical = list(ec.get("canonical_entities") or [])
        anchors = list(ec.get("referential_anchors") or [])
        hop1_subject = ""
        if anchors:
            hop1_subject = str(anchors[0].get("work") or "").strip()
        if not hop1_subject and canonical:
            hop1_subject = str(canonical[0]).strip()
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
                entity_context["relation_prototype"] = prototype_to_dict(prototype_payload)
                entity_context["relation_prototype_cues"] = list(
                    prototype_payload.cue_phrases
                )
            if pivot_entity:
                entity_context["semantic_blend"] = SETTINGS.retrieval.retrieval_semantic_blend_retry
                entity_context["canonical_entities"] = [pivot_entity]
                entity_context["match_strings"] = [pivot_entity]
                entity_context["retry_pivot"] = pivot_entity
                retrieval_query = f"{pivot_entity} ; {state['question']}"
                entity_context["retrieval_fallback"] = (
                    entity_context.get("retrieval_fallback") or ""
                ) + f";hop{active_idx}:pivot={pivot_entity[:40]}"
            elif int(state.get("hop_attempts") or 0) >= 1:
                entity_context["semantic_blend"] = SETTINGS.retrieval.retrieval_semantic_blend_retry
                entity_context["retrieval_fallback"] = (
                    entity_context.get("retrieval_fallback") or ""
                ) + ";hop1:retry"

            retry_depth = int(state.get("hop_attempts") or 0)
            if retry_depth > 0:
                entity_context["retry_depth"] = retry_depth
            if seen_for_hop:
                entity_context["exclude_passage_keys"] = seen_for_hop

            active_resolve = self._resolve_active_hop_subject(state)
            if active_resolve is not None:
                mem_r, hop_r = active_resolve
                # active hop's relation only — stale hints skew passage type
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

            if prototype_payload is not None and prototype_payload.focus_query:
                fq = prototype_payload.focus_query.strip()
                if fq and fq not in variants:
                    variants.append(fq)
            entity_context["retrieval_queries"] = variants
            entity_context["retrieval_query_debug"] = q_dbg
        # no match_strings: still use head-built retrieval_query

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
        """One prototype per hop_index (stable for retries); on failure scorer falls back to keyword table."""
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

    def _hop_memory(self, state: GraphRAGState) -> HopMemory | None:
        return hop_memory_from_dict(state.get("hop_memory"))

    def _current_hop_index(self, state: GraphRAGState) -> int:
        idx = int(state.get("current_hop_index") or 1)
        return max(1, idx)

    def _is_intermediate_hop_mode(self, state: GraphRAGState) -> bool:
        """True if more hops after current (compat shim)."""
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
        """(memory, hop) with placeholder subject filled from prior hop; None if not ready."""
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            return None
        idx = self._current_hop_index(state)
        if idx > len(memory.hops):
            return None
        hop = memory.get_hop(idx)

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
        """Sub-q text with placeholders swapped."""
        return self._substitute_placeholders(hop.subquestion_text, memory)

    def _reason(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()

        question = state["question"]
        guard_prompt: str | None = None
        active = self._resolve_active_hop_subject(state)
        if active is not None:
            memory, hop = active
            ref_anchors = (state.get("entity_linking") or {}).get("referential_anchors") or []
            hop_q = self._resolved_subquestion(memory, hop)
            hop_q = resolve_placeholders(hop_q, ref_anchors) or hop_q
            question = hop_q

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
        """Route: done | advance | retry from critic + hop memory (not loop counter alone)."""
        approved = bool(state.get("approved", False))
        loops = int(state.get("retrieval_loops", 0) or 0)
        max_loops = int(
            state.get("max_retrieval_loops") or SETTINGS.retrieval.max_retrieval_loops
        )
        attempts = int(state.get("hop_attempts") or 0)
        max_attempts = int(state.get("max_hop_attempts") or 2)
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            if approved or loops >= max_loops:
                return "done"
            return "retry"

        idx = self._current_hop_index(state)
        is_last_hop = idx >= len(memory.hops)
        if approved:
            return "done" if is_last_hop else "advance"
        if attempts + 1 < max_attempts and loops < max_loops:
            return "retry"
        return "done"

    def _advance_hop(self, state: GraphRAGState) -> GraphRAGState:
        """Record hop answer and bump current_hop_index."""
        memory = self._hop_memory(state)
        if memory is None or not memory.hops:
            return {}
        idx = self._current_hop_index(state)
        if idx > len(memory.hops):
            return {}
        memory.record_answer(
            hop_index=idx,
            answer=str(state.get("answer") or ""),
            confidence=float(state.get("confidence") or 0.0),
        )
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
        max_hop_attempts = 2
        # max_loops must cover hop advances and same-hop retries
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
        """First sub-q from passages via JSON; pivot string or None."""
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

    @staticmethod
    def _binary_choice_tail_options(question: str) -> list[str]:
        """Trailing `..., A or B?` pair; needs comma before `A` to avoid false splits."""
        q = (question or "").strip()
        if " or " not in q.lower():
            return []
        m = re.search(r",\s*([^,?]{1,200}?)\s+or\s+([^?]{1,200}?)\s*\?\s*$", q, re.IGNORECASE | re.DOTALL)
        if not m:
            return []
        a, b = m.group(1).strip(), m.group(2).strip()
        for ch in '"\'':
            a, b = a.strip(ch), b.strip(ch)
        if not a or not b:
            return []
        return [a, b]

    @staticmethod
    def _comparative_pick_binary_choice(
        question: str,
        options: list[str],
        selected_passages: list[dict[str, Any]],
        evidence_chain: list[dict[str, Any]],
    ) -> str | None:
        """When the question compares two named options in time, pick using years in evidence."""
        if len(options) != 2:
            return None
        qlow = (question or "").lower()
        want_earlier = any(
            w in qlow for w in ("first", "earliest", "before the other", "before ", "earlier", "older")
        )
        want_later = any(
            w in qlow for w in ("last", "latest", "most recent", "after the other", "after ", "later", "newer", "younger")
        )
        if not want_earlier and not want_later:
            return None

        parts: list[str] = []
        for p in (selected_passages or [])[:14]:
            t = str(p.get("text", "") or "").strip()
            if t:
                parts.append(t)
        if not parts:
            for h in (evidence_chain or [])[:14]:
                t = str(h.get("text", "") or "").strip()
                if t:
                    parts.append(t)
        blob = "\n".join(parts)
        if not blob.strip():
            return None

        def min_year_for(entity: str) -> int | None:
            el = (entity or "").lower().strip()
            if len(el) < 2:
                return None
            years: list[int] = []
            for sent in re.split(r"(?<=[.!?])\s+", blob):
                if el not in sent.lower():
                    continue
                for y in re.findall(r"\b((?:19|20)\d{2})\b", sent):
                    years.append(int(y))
            return min(years) if years else None

        y0, y1 = min_year_for(options[0]), min_year_for(options[1])
        if y0 is not None and y1 is not None:
            if want_earlier and not want_later:
                return options[0] if y0 <= y1 else options[1]
            if want_later and not want_earlier:
                return options[0] if y0 >= y1 else options[1]
            if want_earlier and want_later:
                return options[0] if y0 <= y1 else options[1]
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

        binary_opts = self._binary_choice_tail_options(question)
        if len(binary_opts) == 2:
            nt_pred = normalize_text(text)
            left, right = binary_opts[0].strip(), binary_opts[1].strip()
            for choice in (left, right):
                if nt_pred == normalize_text(choice):
                    return choice
            low = text.lower()
            if left.lower() in low and right.lower() not in low:
                return left
            if right.lower() in low and left.lower() not in low:
                return right
            picked = self._comparative_pick_binary_choice(
                question, binary_opts, selected_passages, evidence_chain
            )
            if picked:
                return picked.strip()

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
