"""Embedding-guided retrieval over passages and dynamic graph traversal."""

from __future__ import annotations

from collections import deque
import logging
import re
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from src.config import SETTINGS
from src.graph.entity_scoring import (
    edge_entity_score,
    hybrid_recall_first_score,
    passage_entity_overlap_score,
)
from src.graph.knowledge_graph import DynamicKnowledgeGraph
from src.graph.passage_reranker import PassageReranker
from src.utils.embeddings import EmbeddingEncoder


@dataclass(slots=True)
class RetrievalOutput:
    """Retriever output for downstream reasoning."""

    evidence_chain: list[dict[str, Any]]
    selected_passages: list[dict[str, Any]]
    graph_stats: dict[str, int]


logger = logging.getLogger(__name__)


class HybridGraphRetriever:
    """Dense retrieval + dynamic KG + embedding-guided BFS."""

    def __init__(self, encoder: EmbeddingEncoder | None = None) -> None:
        self.encoder = encoder or EmbeddingEncoder()
        self.kg_builder = DynamicKnowledgeGraph(encoder=self.encoder)
        self.reranker = PassageReranker(self.encoder) if SETTINGS.retrieval.use_reranker else None
        # Pre-built index populated by preload_corpus() for fast per-query search.
        self._cached_passages: list[dict[str, Any]] | None = None
        self._cached_index: faiss.IndexFlatIP | None = None

    def preload_corpus(
        self,
        passages: list[dict[str, Any]],
        index_path: "Path | str | None" = None,
    ) -> None:
        """Pre-build (or load from disk) a FAISS index for the full corpus.

        Call this once at startup with the entire passage pool.  Subsequent
        calls to ``retrieve()`` reuse the index instead of re-encoding all
        passages on every query, reducing per-query latency from O(corpus)
        to O(queries).

        Parameters
        ----------
        passages:
            The full list of passage dicts that form the search corpus.
        index_path:
            Optional path to a ``.faiss`` file.  If the file exists the index
            is loaded from disk (fast).  Otherwise the index is built from
            ``passages`` and saved there for future restarts.
        """
        if not passages:
            return

        from pathlib import Path as _Path

        idx_path = _Path(index_path) if index_path else None

        # Fast path: load pre-built index from disk.
        if idx_path and idx_path.exists():
            logger.info("Loading FAISS index from %s …", idx_path)
            self._cached_index = faiss.read_index(str(idx_path))
            self._cached_passages = passages
            logger.info("FAISS index loaded (%d vectors).", self._cached_index.ntotal)
            return

        logger.info("Pre-building FAISS index for %d passages …", len(passages))
        self._cached_passages = passages
        self._cached_index, _ = self._build_faiss(passages)
        logger.info("FAISS index ready (%d vectors).", len(passages))

        if idx_path:
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._cached_index, str(idx_path))
            logger.info("FAISS index saved to %s.", idx_path)

    @staticmethod
    def _graph_nodes_matching_entities(graph: Any, match_strings: list[str]) -> list[str]:
        if not match_strings:
            return []
        hits: list[str] = []
        for node in graph.nodes:
            folded = re.sub(r"[^a-z0-9]", "", str(node).lower())
            for m in match_strings:
                mf = re.sub(r"[^a-z0-9]", "", m.lower())
                if len(mf) >= 3 and (mf in folded or folded in mf):
                    hits.append(str(node))
                    break
        return list(dict.fromkeys(hits))

    def _build_faiss(self, passages: list[dict[str, Any]]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
        texts = [f"{p.get('title', '')}: {p.get('text', '')}" for p in passages]
        vectors = self.encoder.encode(texts, normalize=True)
        if vectors.size == 0:
            raise ValueError("No passage vectors could be generated.")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index, vectors

    def _merge_dense_by_max(
        self,
        passages: list[dict[str, Any]],
        query_texts: list[str],
        pool_k: int,
    ) -> dict[int, float]:
        """OR-merge: per-passage max inner product across query variants (recall-first)."""
        if not passages or not query_texts:
            return {}
        # Use pre-built index when the caller passes the cached corpus object.
        if self._cached_index is not None and passages is self._cached_passages:
            index = self._cached_index
        else:
            index, _ = self._build_faiss(passages)
        n = len(passages)
        k = min(pool_k, n)
        vecs = self.encoder.encode(list(query_texts), normalize=True)
        merged: dict[int, float] = {}
        for row in range(vecs.shape[0]):
            qv = vecs[row : row + 1]
            scores, indices = index.search(qv, k)
            for col in range(indices.shape[1]):
                idx = int(indices[0, col])
                if idx < 0:
                    continue
                s = float(scores[0, col])
                merged[idx] = max(merged.get(idx, -1e9), s)
        return merged

    def _retrieve_passages(
        self,
        query: str,
        passages: list[dict[str, Any]],
        entity_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recall-first hybrid retrieval: multiple OR-style query embeddings merged by max score,
        then soft hybrid re-ranking (dense + entity + relation). No hard filter that empties output.

        When ``preload_corpus()`` has been called, the search is performed over
        the full pre-indexed corpus instead of the ``passages`` argument, giving
        much broader coverage without paying the encoding cost on every query.
        """
        # Prefer the pre-loaded corpus for maximum coverage.
        actual_passages = self._cached_passages if self._cached_passages is not None else passages
        n = len(actual_passages)
        if n == 0:
            return []
        mult = max(1, int(SETTINGS.retrieval.retrieval_dense_pool_multiplier))
        pool_k = min(max(SETTINGS.retrieval.top_k_passages * mult, SETTINGS.retrieval.top_k_passages), n)
        ec = dict(entity_context) if entity_context else {}

        query_texts = list(ec.get("retrieval_queries") or [])
        if not query_texts:
            query_texts = [q for q in (query or "").split(";") if q.strip()] or [query]
        query_texts = [t.strip() for t in query_texts if t and str(t).strip()][:20]
        if not query_texts:
            query_texts = [str(query or "query")]

        merged = self._merge_dense_by_max(actual_passages, query_texts, pool_k)
        if not merged and n > 0:
            ec["retrieval_fallback"] = (ec.get("retrieval_fallback") or "") + ";empty_merge→question_only"
            merged = self._merge_dense_by_max(actual_passages, [str(query)[:500]], pool_k)

        if not merged and n > 0:
            cans = list(ec.get("canonical_entities") or ec.get("match_strings") or [])
            if cans:
                ec["retrieval_fallback"] = (ec.get("retrieval_fallback") or "") + ";canonical_only"
                merged = self._merge_dense_by_max(actual_passages, [cans[0]], pool_k)

        if not merged and n > 0:
            merged = {0: 1.0}

        take_n = min(max(SETTINGS.retrieval.top_k_passages * 3, SETTINGS.retrieval.top_k_passages), n)
        ranked_idx = sorted(merged.keys(), key=lambda i: merged[i], reverse=True)[:take_n]

        selected: list[dict[str, Any]] = []
        for idx in ranked_idx:
            item = dict(actual_passages[int(idx)])
            item["score"] = float(merged[int(idx)])
            item["retrieval_or_merge"] = True
            selected.append(item)

        primary_q = query_texts[0] if query_texts else query
        if self.reranker is not None and self.reranker.is_trained:
            selected = self.reranker.rerank(
                query=primary_q,
                passages=selected,
                entity_context=ec,
            )
        else:
            selected = self._entity_score_and_sort_non_learned(primary_q, selected, ec)

        return selected[: SETTINGS.retrieval.top_k_passages]

    def _entity_score_and_sort_non_learned(
        self,
        query: str,
        passages: list[dict[str, Any]],
        entity_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not entity_context or not entity_context.get("match_strings"):
            return sorted(passages, key=lambda x: float(x.get("score", 0.0)), reverse=True)

        rescored: list[dict[str, Any]] = []
        for p in passages:
            item = dict(p)
            base = float(item.get("score", 0.0))
            item["dense_score"] = base
            item["entity_score"] = hybrid_recall_first_score(item, query, entity_context, base)
            item["score"] = item["entity_score"]
            rescored.append(item)
        rescored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return rescored

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
        entity_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        build_result = self.kg_builder.build(graph_passages)
        graph = build_result.graph

        if graph.number_of_edges() == 0:
            return []

        query_vector = self.encoder.encode([query], normalize=True)[0]
        node_queue: deque[tuple[str, int]] = deque()
        seen: set[str] = set()
        evidence: list[dict[str, Any]] = []

        match_strings = list((entity_context or {}).get("match_strings") or [])
        linked_nodes = self._graph_nodes_matching_entities(graph, match_strings)

        # Seed from resolved entities first, then query NER, then high-degree nodes.
        for node in linked_nodes:
            if node in graph:
                node_queue.append((node, 0))
        query_entities = self.kg_builder.extract_entities(query)
        if query_entities:
            for entity in query_entities:
                if entity in graph:
                    node_queue.append((entity, 0))
        if not node_queue:
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
                sem = float(np.dot(query_vector, edge_vec))
                edge_row = {
                    "hop": depth + 1,
                    "source": source_title,
                    "node_from": node,
                    "node_to": nbr,
                    "relation": relation,
                    "edge_semantic": edge_data.get("edge_semantic", "entity_linking"),
                    "provenance": edge_data.get("provenance", []),
                    "text": edge_data.get("text", ""),
                    "score": sem,
                }
                ent_w = float(SETTINGS.retrieval.entity_combined_score_weight)
                e_ent = edge_entity_score(edge_row, entity_context)
                if entity_context and match_strings:
                    edge_row["score"] = (1.0 - ent_w) * sem + ent_w * (e_ent * 2.0 - 1.0)
                else:
                    edge_row["score"] = sem
                evidence.append(edge_row)
                if nbr not in seen and depth + 1 <= SETTINGS.retrieval.max_hops:
                    node_queue.append((nbr, depth + 1))

        for e in evidence:
            e["entity_edge_match"] = edge_entity_score(e, entity_context) if (entity_context and match_strings) else 0.0
        evidence.sort(
            key=lambda x: float(x["score"]) + 0.12 * float(x.get("entity_edge_match", 0.0)),
            reverse=True,
        )
        return evidence[: SETTINGS.retrieval.top_k_edges]

    def retrieve(
        self,
        query: str,
        passages: list[dict[str, Any]],
        entity_context: dict[str, Any] | None = None,
    ) -> RetrievalOutput:
        """Run full hybrid retrieval and return an evidence chain."""
        selected_passages = self._retrieve_passages(query, passages, entity_context=entity_context)
        evidence_chain = self._bfs_evidence(query, selected_passages, entity_context=entity_context)
        if not evidence_chain and selected_passages:
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
        test_examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.reranker is None:
            return {"enabled": False, "trained": False}

        metrics = self.reranker.fit(
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            max_examples=SETTINGS.retrieval.reranker_max_train_examples,
            negatives_per_positive=SETTINGS.retrieval.reranker_negatives_per_positive,
            epochs=SETTINGS.retrieval.reranker_epochs,
        )
        return {
            "enabled": True,
            "trained": metrics.trained,
            "train_pairs": metrics.train_pairs,
            "validation_pairs": metrics.validation_pairs,
            "test_pairs": metrics.test_pairs,
            "positive_rate": metrics.positive_rate,
            "train_loss": metrics.train_loss,
            "train_accuracy": metrics.train_accuracy,
            "validation_loss": metrics.validation_loss,
            "validation_accuracy": metrics.validation_accuracy,
            "test_loss": metrics.test_loss,
            "test_accuracy": metrics.test_accuracy,
            "test_confusion_matrix": metrics.test_confusion_matrix,
            "confusion_matrix_labels": ["negative", "positive"],
            "epochs": SETTINGS.retrieval.reranker_epochs,
            "history": metrics.history,
            "model_path": str(SETTINGS.paths.reranker_model),
        }

    def has_trained_reranker(self) -> bool:
        return bool(self.reranker is not None and self.reranker.is_trained)
