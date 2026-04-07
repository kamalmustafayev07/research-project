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
            entities = [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2]
            if entities:
                return list(dict.fromkeys(entities))

        # Fallback: consecutive title-cased tokens.
        pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        entities = [match.group(1).strip() for match in pattern.finditer(text)]
        return list(dict.fromkeys(entities))

    def extract_relations(self, text: str, entities: list[str]) -> list[tuple[str, str, str]]:
        """Extract simple relation triples from nearby entity mentions."""
        triples: list[tuple[str, str, str]] = []
        if len(entities) < 2:
            return triples

        for idx in range(len(entities) - 1):
            src = entities[idx]
            dst = entities[idx + 1]
            relation = "related_to"
            sentence_window = text[:500]
            low = sentence_window.lower()
            if "born" in low:
                relation = "born_in"
            elif "located" in low or "in " in low:
                relation = "located_in"
            elif "founded" in low:
                relation = "founded"
            triples.append((src, relation, dst))
        return triples

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
