"""Facade re-exports for entity-aware passage reranking (implementation in ``src.graph``)."""

from __future__ import annotations

from src.graph.entity_scoring import (
    demote_passages_without_entity_overlap,
    hybrid_recall_first_score,
    passage_entity_overlap_score,
)
from src.graph.passage_reranker import PassageReranker, RerankerTrainMetrics

__all__ = [
    "PassageReranker",
    "RerankerTrainMetrics",
    "hybrid_recall_first_score",
    "demote_passages_without_entity_overlap",
    "passage_entity_overlap_score",
]
