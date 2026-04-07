"""Dynamic knowledge graph builder using lightweight NER and relation extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass(slots=True)
class GraphBuildResult:
    """Output container for dynamic KG construction."""

    graph: nx.DiGraph
    node_count: int
    edge_count: int


class DynamicKnowledgeGraph:
    """Build an in-memory graph from retrieved passages at query time."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._nlp = self._load_spacy()

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

    def build(self, passages: list[dict[str, Any]]) -> GraphBuildResult:
        """Build graph from a list of passages."""
        self.graph.clear()

        for passage in passages:
            title = passage.get("title", "untitled")
            text = passage.get("text", "")
            entities = self.extract_entities(text)

            for entity in entities:
                self.graph.add_node(entity, label=entity, source_title=title)

            for src, rel, dst in self.extract_relations(text, entities):
                self.graph.add_edge(
                    src,
                    dst,
                    relation=rel,
                    source_title=title,
                    text=text,
                    score=float(passage.get("score", 0.0)),
                )

        return GraphBuildResult(
            graph=self.graph,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
        )
