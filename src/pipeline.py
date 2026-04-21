"""LangGraph pipeline for Agent-Enhanced GraphRAG with baseline methods."""

from __future__ import annotations

from dataclasses import asdict
import re
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.critic import CriticAgent
from src.agents.graph_retriever import GraphRetrieverAgent
from src.agents.query_decomposer import QueryDecomposerAgent
from src.agents.react_reasoner import ReActReasonerAgent
from src.config import SETTINGS
from src.utils.llm import LLMClient


class GraphRAGState(TypedDict, total=False):
    """Typed state for LangGraph execution."""

    question: str
    context_passages: list[dict[str, Any]]
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


class AgentEnhancedGraphRAG:
    """Production pipeline with four agents and critic-to-retriever feedback."""

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.decomposer = QueryDecomposerAgent(self.llm)
        self.retriever = GraphRetrieverAgent()
        self.reasoner = ReActReasonerAgent(self.llm)
        self.critic = CriticAgent(self.llm)
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(GraphRAGState)
        builder.add_node("decomposer", self._decompose)
        builder.add_node("retriever", self._retrieve)
        builder.add_node("reasoner", self._reason)
        builder.add_node("critic", self._critic)

        builder.add_edge(START, "decomposer")
        builder.add_edge("decomposer", "retriever")
        builder.add_edge("retriever", "reasoner")
        builder.add_edge("reasoner", "critic")
        builder.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "retry": "retriever",
                "done": END,
            },
        )
        return builder.compile()

    @staticmethod
    def _updated_latency(state: GraphRAGState, key: str, elapsed: float) -> dict[str, float]:
        totals = dict(state.get("latency_breakdown", {}))
        totals[key] = float(totals.get(key, 0.0)) + float(elapsed)
        return totals

    def _decompose(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()
        result = self.decomposer.run(state["question"])
        elapsed = perf_counter() - start
        return {
            "decomposition": asdict(result),
            "latency_breakdown": self._updated_latency(state, "decomposer", elapsed),
        }

    def _retrieve(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()
        query = state["question"]
        decomposition = state.get("decomposition", {})
        sub_questions = decomposition.get("sub_questions", [])
        retrieval_query = " ; ".join([query] + sub_questions[:2]) if sub_questions else query

        output = self.retriever.run(retrieval_query, state["context_passages"])
        loops = state.get("retrieval_loops", 0) + 1
        elapsed = perf_counter() - start
        return {
            "evidence_chain": output.evidence_chain,
            "selected_passages": output.selected_passages,
            "graph_stats": output.graph_stats,
            "retrieval_loops": loops,
            "latency_breakdown": self._updated_latency(state, "retriever", elapsed),
        }

    def _reason(self, state: GraphRAGState) -> GraphRAGState:
        start = perf_counter()
        output = self.reasoner.run(
            state["question"],
            state.get("evidence_chain", []),
            selected_passages=state.get("selected_passages", []),
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
        output = self.critic.run(
            query=state["question"],
            answer=state.get("answer", ""),
            confidence=state.get("confidence", 0.0),
            evidence_count=len(state.get("evidence_chain", [])),
        )
        elapsed = perf_counter() - start
        return {
            "approved": output.approved,
            "critique": output.critique,
            "confidence": output.confidence,
            "latency_breakdown": self._updated_latency(state, "critic", elapsed),
        }

    def _route_after_critic(self, state: GraphRAGState) -> str:
        approved = state.get("approved", False)
        loops = state.get("retrieval_loops", 1)
        max_loops = state.get("max_retrieval_loops", SETTINGS.retrieval.max_retrieval_loops)
        if approved or loops >= max_loops:
            return "done"
        return "retry"

    def invoke(self, question: str, context_passages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run the full 4-agent workflow and return answer plus evidence."""
        start_total = perf_counter()
        result = self.graph.invoke(
            {
                "question": question,
                "context_passages": context_passages,
                "retrieval_loops": 0,
                "max_retrieval_loops": SETTINGS.retrieval.max_retrieval_loops,
                "latency_breakdown": {
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
            "decomposer": float(result_breakdown.get("decomposer", 0.0)),
            "retriever": float(result_breakdown.get("retriever", 0.0)),
            "reasoner": float(result_breakdown.get("reasoner", 0.0)),
            "critic": float(result_breakdown.get("critic", 0.0)),
        }
        finalized_answer = self._finalize_answer(
            question=question,
            raw_answer=result.get("answer", ""),
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

    def _fallback_answer(
        self,
        question: str,
        evidence_chain: list[dict[str, Any]],
        selected_passages: list[dict[str, Any]],
    ) -> str:
        question_low = question.lower()
        best_text = ""

        if selected_passages:
            best_text = str(selected_passages[0].get("text", ""))
        elif evidence_chain:
            best_text = str(evidence_chain[0].get("text", ""))

        if not best_text:
            return ""

        if question_low.startswith("when") or "what year" in question_low:
            year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", best_text)
            if year_match:
                return year_match.group(1)

        name_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", best_text)
        if name_match:
            return name_match.group(1)

        return ""

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
