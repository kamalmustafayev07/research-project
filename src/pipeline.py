"""LangGraph pipeline for Agent-Enhanced GraphRAG with baseline methods."""

from __future__ import annotations

from dataclasses import asdict
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

    def _decompose(self, state: GraphRAGState) -> GraphRAGState:
        result = self.decomposer.run(state["question"])
        return {"decomposition": asdict(result)}

    def _retrieve(self, state: GraphRAGState) -> GraphRAGState:
        query = state["question"]
        decomposition = state.get("decomposition", {})
        sub_questions = decomposition.get("sub_questions", [])
        retrieval_query = " ; ".join([query] + sub_questions[:2]) if sub_questions else query

        output = self.retriever.run(retrieval_query, state["context_passages"])
        loops = state.get("retrieval_loops", 0) + 1
        return {
            "evidence_chain": output.evidence_chain,
            "selected_passages": output.selected_passages,
            "graph_stats": output.graph_stats,
            "retrieval_loops": loops,
        }

    def _reason(self, state: GraphRAGState) -> GraphRAGState:
        output = self.reasoner.run(state["question"], state.get("evidence_chain", []))
        return {
            "thoughts": output.thoughts,
            "answer": output.answer,
            "confidence": output.confidence,
        }

    def _critic(self, state: GraphRAGState) -> GraphRAGState:
        output = self.critic.run(
            query=state["question"],
            answer=state.get("answer", ""),
            confidence=state.get("confidence", 0.0),
            evidence_count=len(state.get("evidence_chain", [])),
        )
        return {
            "approved": output.approved,
            "critique": output.critique,
            "confidence": output.confidence,
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
        result = self.graph.invoke(
            {
                "question": question,
                "context_passages": context_passages,
                "retrieval_loops": 0,
                "max_retrieval_loops": SETTINGS.retrieval.max_retrieval_loops,
            }
        )
        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "evidence_chain": result.get("evidence_chain", []),
            "thoughts": result.get("thoughts", []),
            "graph_stats": result.get("graph_stats", {}),
            "critique": result.get("critique", ""),
            "retrieval_loops": result.get("retrieval_loops", 0),
        }

    def dense_rag_baseline(self, question: str, context_passages: list[dict[str, Any]]) -> dict[str, Any]:
        """Baseline B1: dense retrieval + single-pass answer generation."""
        retrieved = self.retriever.retriever.retrieve(question, context_passages)
        top_context = "\n".join(
            [f"{p['title']}: {p['text'][:400]}" for p in retrieved.selected_passages[:4]]
        )
        prompt = (
            "Answer the question only using provided passages. Return concise answer.\n"
            f"Question: {question}\nPassages:\n{top_context}"
        )
        response = self.llm.generate(prompt)
        return {
            "answer": response.text.strip(),
            "confidence": 0.55,
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
            "Answer the question from graph evidence. Return concise answer.\n"
            f"Question: {question}\nEvidence:\n{evidence}"
        )
        response = self.llm.generate(prompt)
        return {
            "answer": response.text.strip(),
            "confidence": 0.62,
            "evidence_chain": retrieved.evidence_chain,
        }
