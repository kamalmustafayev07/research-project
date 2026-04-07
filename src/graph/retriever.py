"""Embedding-guided retrieval over passages and dynamic graph traversal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from src.config import SETTINGS
from src.graph.knowledge_graph import DynamicKnowledgeGraph
from src.graph.passage_reranker import PassageReranker
from src.utils.embeddings import EmbeddingEncoder


@dataclass(slots=True)
class RetrievalOutput:
    """Retriever output for downstream reasoning."""

    evidence_chain: list[dict[str, Any]]
    selected_passages: list[dict[str, Any]]
    graph_stats: dict[str, int]


class HybridGraphRetriever:
    """Dense retrieval + dynamic KG + embedding-guided BFS."""

    def __init__(self, encoder: EmbeddingEncoder | None = None) -> None:
        self.encoder = encoder or EmbeddingEncoder()
        self.kg_builder = DynamicKnowledgeGraph()
        self.reranker = PassageReranker(self.encoder) if SETTINGS.retrieval.use_reranker else None

    def _build_faiss(self, passages: list[dict[str, Any]]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
        texts = [f"{p.get('title', '')}: {p.get('text', '')}" for p in passages]
        vectors = self.encoder.encode(texts, normalize=True)
        if vectors.size == 0:
            raise ValueError("No passage vectors could be generated.")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index, vectors

    def _retrieve_passages(self, query: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        index, _ = self._build_faiss(passages)
        query_vector = self.encoder.encode([query], normalize=True)
        scores, indices = index.search(query_vector, min(SETTINGS.retrieval.top_k_passages, len(passages)))

        selected: list[dict[str, Any]] = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            item = dict(passages[int(idx)])
            item["score"] = float(scores[0][rank])
            selected.append(item)

        if self.reranker is not None and self.reranker.is_trained:
            selected = self.reranker.rerank(query=query, passages=selected)
        return selected

    @staticmethod
    def _fallback_evidence(query: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fallback: list[dict[str, Any]] = []
        for idx, passage in enumerate(passages[: min(3, len(passages))]):
            title = str(passage.get("title", "unknown"))
            text = str(passage.get("text", ""))
            fallback.append(
                {
                    "hop": 1,
                    "source": title,
                    "node_from": title,
                    "node_to": "answer_candidate",
                    "relation": "supports",
                    "text": text,
                    "score": float(passage.get("score", 0.0)),
                    "query": query,
                    "fallback": True,
                }
            )
        return fallback

    def _bfs_evidence(
        self,
        query: str,
        graph_passages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        build_result = self.kg_builder.build(graph_passages)
        graph = build_result.graph

        if graph.number_of_edges() == 0:
            return []

        query_vector = self.encoder.encode([query], normalize=True)[0]
        node_queue: deque[tuple[str, int]] = deque()
        seen: set[str] = set()
        evidence: list[dict[str, Any]] = []

        # Seed from entities found in the query; fallback to top-degree nodes.
        query_entities = self.kg_builder.extract_entities(query)
        if query_entities:
            for entity in query_entities:
                if entity in graph:
                    node_queue.append((entity, 0))
        else:
            for node, _ in sorted(graph.degree, key=lambda x: x[1], reverse=True)[:3]:
                node_queue.append((node, 0))

        while node_queue:
            node, depth = node_queue.popleft()
            if node in seen or depth > SETTINGS.retrieval.max_hops:
                continue
            seen.add(node)

            for nbr in graph.successors(node):
                edge_data = graph.get_edge_data(node, nbr) or {}
                relation = edge_data.get("relation", "related_to")
                source_title = edge_data.get("source_title", "unknown")
                edge_text = f"{node} {relation} {nbr}. {edge_data.get('text', '')[:280]}"

                edge_vec = self.encoder.encode([edge_text], normalize=True)[0]
                score = float(np.dot(query_vector, edge_vec))
                evidence.append(
                    {
                        "hop": depth + 1,
                        "source": source_title,
                        "node_from": node,
                        "node_to": nbr,
                        "relation": relation,
                        "text": edge_data.get("text", ""),
                        "score": score,
                    }
                )
                if nbr not in seen and depth + 1 <= SETTINGS.retrieval.max_hops:
                    node_queue.append((nbr, depth + 1))

        evidence.sort(key=lambda x: x["score"], reverse=True)
        return evidence[: SETTINGS.retrieval.top_k_edges]

    def retrieve(self, query: str, passages: list[dict[str, Any]]) -> RetrievalOutput:
        """Run full hybrid retrieval and return an evidence chain."""
        selected_passages = self._retrieve_passages(query, passages)
        evidence_chain = self._bfs_evidence(query, selected_passages)
        if not evidence_chain:
            evidence_chain = self._fallback_evidence(query, selected_passages)

        graph_stats = {
            "nodes": self.kg_builder.graph.number_of_nodes(),
            "edges": self.kg_builder.graph.number_of_edges(),
            "fallback_evidence": 1 if evidence_chain and evidence_chain[0].get("fallback") else 0,
        }
        return RetrievalOutput(
            evidence_chain=evidence_chain,
            selected_passages=selected_passages,
            graph_stats=graph_stats,
        )

    def fit_reranker(
        self,
        train_examples: list[dict[str, Any]],
        validation_examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.reranker is None:
            return {"enabled": False, "trained": False}

        metrics = self.reranker.fit(
            train_examples=train_examples,
            validation_examples=validation_examples,
            max_examples=SETTINGS.retrieval.reranker_max_train_examples,
            negatives_per_positive=SETTINGS.retrieval.reranker_negatives_per_positive,
        )
        return {
            "enabled": True,
            "trained": metrics.trained,
            "train_pairs": metrics.train_pairs,
            "positive_rate": metrics.positive_rate,
            "validation_accuracy": metrics.validation_accuracy,
            "model_path": str(SETTINGS.paths.reranker_model),
        }

    def has_trained_reranker(self) -> bool:
        return bool(self.reranker is not None and self.reranker.is_trained)
