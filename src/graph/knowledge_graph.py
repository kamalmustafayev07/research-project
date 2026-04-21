"""Dynamic knowledge graph builder using lightweight NER and relation extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(slots=True)
class GraphBuildResult:
    """Output container for dynamic KG construction."""

    graph: nx.DiGraph
    node_count: int
    edge_count: int


class DynamicKnowledgeGraph:
    """Build an in-memory graph from retrieved passages at query time."""

    def __init__(
        self,
        encoder: Any | None = None,
        similarity_threshold: float = 0.68,
        max_similarity_edges_per_node: int = 2,
    ) -> None:
        self.graph = nx.DiGraph()
        self._nlp = self._load_spacy()
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.max_similarity_edges_per_node = max_similarity_edges_per_node

    @staticmethod
    def _load_spacy() -> Any:
        try:
            import spacy

            return spacy.load("en_core_web_sm")
        except Exception:
            return None

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities using spaCy, with regex fallback."""
        if self._nlp is not None:
            doc = self._nlp(text)
            blocked_labels = {"DATE", "TIME", "MONEY", "PERCENT", "CARDINAL", "ORDINAL", "QUANTITY"}
            entities = [
                ent.text.strip()
                for ent in doc.ents
                if len(ent.text.strip()) > 2 and ent.label_ not in blocked_labels
            ]
            if entities:
                return list(dict.fromkeys(entities))

        # Fallback: consecutive title-cased tokens.
        pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        entities = [match.group(1).strip() for match in pattern.finditer(text)]
        entities = [ent for ent in entities if not re.fullmatch(r"\d+", ent)]
        return list(dict.fromkeys(entities))

    def extract_relations(self, text: str, entities: list[str]) -> list[tuple[str, str, str]]:
        """Extract simple relation triples from nearby entity mentions."""
        triples: list[tuple[str, str, str]] = []
        if len(entities) < 2:
            return triples

        sentence_units: list[str] = []
        if self._nlp is not None:
            doc = self._nlp(text)
            sentence_units = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentence_units:
            sentence_units = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]

        seen: set[tuple[str, str, str]] = set()
        for sentence in sentence_units:
            sentence_entities = [entity for entity in entities if entity in sentence]
            if len(sentence_entities) < 2:
                continue

            relation = self._relation_from_sentence(sentence)
            for idx in range(len(sentence_entities) - 1):
                src = sentence_entities[idx]
                dst = sentence_entities[idx + 1]
                triple = (src, relation, dst)
                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)
        return triples

    @staticmethod
    def _relation_from_sentence(sentence: str) -> str:
        low = sentence.lower()
        if "born in" in low or "birthplace" in low or "born at" in low:
            return "born_in"
        if "founded" in low or "established" in low:
            return "founded"
        if "directed by" in low or "director" in low:
            return "directed_by"
        if "starring" in low or "played by" in low or "stars" in low:
            return "played_by"
        if "located in" in low or "based in" in low or "lies in" in low:
            return "located_in"
        return "related_to"

    @staticmethod
    def _passage_provenance(passage: dict[str, Any], fallback_title: str) -> dict[str, Any]:
        return {
            "title": str(passage.get("title", fallback_title)),
            "passage_id": str(passage.get("passage_id", fallback_title)),
            "source_type": str(passage.get("source_type", "passage")),
        }

    def _add_node_with_provenance(self, entity: str, provenance: dict[str, Any]) -> None:
        if entity in self.graph:
            node_data = self.graph.nodes[entity]
            history = list(node_data.get("provenance", []))
            if provenance not in history:
                history.append(provenance)
            source_titles = sorted({str(item.get("title", "")) for item in history if item.get("title")})
            self.graph.nodes[entity]["provenance"] = history
            self.graph.nodes[entity]["source_titles"] = source_titles
            return

        self.graph.add_node(
            entity,
            label=entity,
            provenance=[provenance],
            source_titles=[str(provenance.get("title", ""))],
        )

    def _upsert_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        edge_semantic: str,
        passage: dict[str, Any],
        text: str,
        score: float,
    ) -> None:
        provenance = self._passage_provenance(passage, fallback_title="unknown")
        if self.graph.has_edge(src, dst):
            edge = self.graph[src][dst]
            history = list(edge.get("provenance", []))
            if provenance not in history:
                history.append(provenance)

            semantics = list(edge.get("edge_semantics", []))
            if edge_semantic not in semantics:
                semantics.append(edge_semantic)

            relation_counts = dict(edge.get("relation_counts", {}))
            relation_counts[relation] = int(relation_counts.get(relation, 0)) + 1

            edge["provenance"] = history
            edge["edge_semantics"] = semantics
            edge["relation_counts"] = relation_counts
            edge["score"] = max(float(edge.get("score", 0.0)), score)
            if text and not edge.get("text"):
                edge["text"] = text
            if not edge.get("source_title"):
                edge["source_title"] = provenance["title"]
            return

        self.graph.add_edge(
            src,
            dst,
            relation=relation,
            edge_semantic=edge_semantic,
            edge_semantics=[edge_semantic],
            relation_counts={relation: 1},
            source_title=provenance["title"],
            provenance=[provenance],
            text=text,
            score=score,
        )

    def _add_cooccurrence_edges(self, entities: list[str], passage: dict[str, Any], text: str) -> None:
        if len(entities) < 2:
            return

        for idx, src in enumerate(entities[:-1]):
            for dst in entities[idx + 1 :]:
                self._upsert_edge(
                    src=src,
                    dst=dst,
                    relation="co_occurs_with",
                    edge_semantic="co_occurrence",
                    passage=passage,
                    text=text,
                    score=float(passage.get("score", 0.0)),
                )

    def _add_embedding_similarity_edges(self) -> None:
        if self.encoder is None:
            return

        nodes = list(self.graph.nodes)
        if len(nodes) < 2:
            return

        embeddings = self.encoder.encode(nodes, normalize=True)
        if embeddings.size == 0:
            return

        similarity = np.dot(embeddings, embeddings.T)
        semantic_passage = {
            "title": "semantic_similarity",
            "passage_id": "semantic_global",
            "source_type": "embedding_similarity",
        }

        for src_idx, src in enumerate(nodes):
            ranked = np.argsort(similarity[src_idx])[::-1]
            added = 0
            for dst_idx in ranked:
                if src_idx == int(dst_idx):
                    continue
                score = float(similarity[src_idx, int(dst_idx)])
                if score < self.similarity_threshold:
                    break

                dst = nodes[int(dst_idx)]
                self._upsert_edge(
                    src=src,
                    dst=dst,
                    relation="semantic_similar",
                    edge_semantic="embedding_similarity",
                    passage=semantic_passage,
                    text="",
                    score=score,
                )
                added += 1
                if added >= self.max_similarity_edges_per_node:
                    break

    def build(self, passages: list[dict[str, Any]]) -> GraphBuildResult:
        """Build graph from a list of passages."""
        self.graph.clear()

        for passage in passages:
            title = passage.get("title", "untitled")
            text = passage.get("text", "")
            entities = self.extract_entities(text)
            provenance = self._passage_provenance(passage, fallback_title=str(title))

            for entity in entities:
                self._add_node_with_provenance(entity, provenance)

            for src, rel, dst in self.extract_relations(text, entities):
                self._upsert_edge(
                    src=src,
                    dst=dst,
                    relation=rel,
                    edge_semantic="entity_linking",
                    passage=passage,
                    text=text,
                    score=float(passage.get("score", 0.0)),
                )

            self._add_cooccurrence_edges(entities=entities, passage=passage, text=text)

        self._add_embedding_similarity_edges()

        return GraphBuildResult(
            graph=self.graph,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
        )
