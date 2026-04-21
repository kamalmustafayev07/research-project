"""Graph retriever agent wrapping dynamic KG + hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.graph.retriever import HybridGraphRetriever


@dataclass(slots=True)
class GraphRetrieverOutput:
    """Graph retrieval output passed to reasoner."""

    evidence_chain: list[dict[str, Any]]
    selected_passages: list[dict[str, Any]]
    graph_stats: dict[str, int]


class GraphRetrieverAgent:
    """Retrieve evidence chain from context passages."""

    def __init__(self, retriever: HybridGraphRetriever | None = None) -> None:
        self.retriever = retriever or HybridGraphRetriever()

    def run(self, query: str, context_passages: list[dict[str, Any]]) -> GraphRetrieverOutput:
        """Execute retrieval pipeline for a given query and passage pool."""
        output = self.retriever.retrieve(query=query, passages=context_passages)
        return GraphRetrieverOutput(
            evidence_chain=output.evidence_chain,
            selected_passages=output.selected_passages,
            graph_stats=output.graph_stats,
        )

    def fit_reranker(
        self,
        train_examples: list[dict[str, Any]],
        validation_examples: list[dict[str, Any]] | None = None,
        test_examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Train the optional passage reranker used by retrieval."""
        return self.retriever.fit_reranker(
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
        )

    def has_trained_reranker(self) -> bool:
        """Return whether a trained reranker is available."""
        return self.retriever.has_trained_reranker()
