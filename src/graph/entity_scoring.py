"""Entity-aware scoring utilities for passages and graph edges.

Final ranking is entity-first: dense similarity is at most a small tie-breaker
after hard overlap filtering.
"""

from __future__ import annotations

import re
from typing import Any

from src.config import SETTINGS

_RELATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "author_of": ("author", "wrote", "written", "novel", "book"),
    "studied_at": (
        "studied",
        "attended",
        "graduated",
        "university",
        "college",
        "read ",
        "ba ",
        "ma ",
        "degree",
        "oxford",
    ),
    "born_in": ("born", "birth", "b.", "native "),
    "directed_by": ("directed", "director", "film", "movie"),
    "located_in": ("located", "based in", "city", "country", "situat"),
}

_GENERIC_BIO_SIGNALS = (
    "governor",
    "politician",
    "served as",
    "member of parliament",
    "president of",
    "chief justice",
    "mayor",
    "campaign",
)


def _blob(passage: dict[str, Any]) -> str:
    return f"{passage.get('title', '')} {passage.get('text', '')}"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _contains_folded(haystack_folded: str, needle: str) -> bool:
    n = re.sub(r"[^a-z0-9]", "", needle.lower())
    if len(n) < 3:
        return False
    return n in haystack_folded


def passage_entity_overlap_score(passage: dict[str, Any], match_strings: list[str]) -> float:
    """1.0 if any query entity string matches title+text; 0.0 if none."""
    if not match_strings:
        return 0.5
    blob = _blob(passage)
    folded = re.sub(r"[^a-z0-9]", "", blob.lower())
    for m in match_strings:
        if _contains_folded(folded, m):
            return 1.0
    return 0.0


def exact_entity_match_score(passage: dict[str, Any], match_strings: list[str]) -> float:
    return passage_entity_overlap_score(passage, match_strings)


def relation_consistency_score(passage: dict[str, Any], relation_hints: list[str]) -> float:
    if not relation_hints:
        return 0.5
    low = _blob(passage).lower()
    hits = 0
    for rel in relation_hints:
        kws = _RELATION_KEYWORDS.get(rel, ())
        if any(k in low for k in kws):
            hits += 1
    if hits == 0:
        return 0.0
    return min(1.0, 0.35 + 0.32 * hits)


def title_overlap_score(passage: dict[str, Any], query: str) -> float:
    qt = _tokenize(query)
    tt = _tokenize(str(passage.get("title", "")))
    if not qt:
        return 0.0
    return len(qt & tt) / max(1, len(qt))


def entity_mismatch_indicator(exact: float) -> float:
    return 1.0 if exact < 0.5 else 0.0


def normalize_dense_score(dense: float) -> float:
    return max(0.0, min(1.0, (float(dense) + 1.0) / 2.0))


def hybrid_recall_first_score(
    passage: dict[str, Any],
    query: str,
    entity_context: dict[str, Any] | None,
    base_semantic: float,
) -> float:
    """
    Precision refinement after recall: dense + entity bonus + relation/title boosts.
    No single signal zeroes a passage (recall-first policy).
    """
    d = normalize_dense_score(base_semantic)
    if not entity_context or not entity_context.get("match_strings"):
        return float(base_semantic)

    ms = list(entity_context.get("match_strings") or [])
    rel_h = list(entity_context.get("relation_hints") or [])
    ent = passage_entity_overlap_score(passage, ms)
    rel = relation_consistency_score(passage, rel_h) if rel_h else 0.45
    if rel_h and rel < 0.12:
        rel = 0.18
    title = title_overlap_score(passage, query)

    w_d = float(entity_context.get("w_dense", 0.40))
    w_e = float(entity_context.get("w_entity", 0.32))
    w_r = float(entity_context.get("w_relation", 0.16))
    w_t = float(entity_context.get("w_title", 0.12))
    return w_d * d + w_e * ent + w_r * rel + w_t * title


def final_entity_grounded_score(
    passage: dict[str, Any],
    query: str,
    entity_context: dict[str, Any] | None,
    base_semantic: float,
) -> float:
    """
    Entity-first score (not normalized to [0,1]):

    2.0 * exact_entity_match
  + 1.5 * relation_match_score
  + 1.0 * title_overlap
  - 3.0 * entity_mismatch_penalty

    Small dense tie-break only when overlap exists (via semantic_blend in context).
    """
    if not entity_context or not entity_context.get("match_strings"):
        return float(base_semantic)

    match_strings = list(entity_context.get("match_strings") or [])
    rel_hints = list(entity_context.get("relation_hints") or [])

    exact = exact_entity_match_score(passage, match_strings)
    rel = relation_consistency_score(passage, rel_hints)
    title = title_overlap_score(passage, query)
    mismatch = entity_mismatch_indicator(exact)

    base = 2.0 * exact + 1.5 * rel + 1.0 * title - 3.0 * mismatch

    blend = float(
        entity_context.get(
            "semantic_blend",
            SETTINGS.retrieval.retrieval_semantic_blend,
        )
    )
    dense = max(-1.0, min(1.0, float(base_semantic)))
    return base + blend * dense


def combined_entity_rerank_score(
    passage: dict[str, Any],
    query: str,
    entity_context: dict[str, Any] | None,
    base_semantic: float,
) -> float:
    """Backward-compatible alias; prefer :func:`final_entity_grounded_score`."""
    return final_entity_grounded_score(passage, query, entity_context, base_semantic)


def hard_filter_passages_by_entity_overlap(
    passages: list[dict[str, Any]],
    match_strings: list[str],
) -> list[dict[str, Any]]:
    """Discard passages with zero entity overlap (no exceptions when match_strings is non-empty)."""
    if not match_strings:
        return list(passages)
    return [p for p in passages if passage_entity_overlap_score(p, match_strings) >= 0.5]


def demote_passages_without_entity_overlap(
    passages: list[dict[str, Any]],
    match_strings: list[str],
) -> list[dict[str, Any]]:
    """If hard filtering is not applied, demote zero-overlap passages to the end."""
    if not match_strings:
        return passages
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for p in passages:
        if passage_entity_overlap_score(p, match_strings) >= 0.5:
            good.append(p)
        else:
            bad.append(p)
    if not good:
        return passages
    return good + bad


def edge_entity_score(edge: dict[str, Any], entity_context: dict[str, Any] | None) -> float:
    if not entity_context or not entity_context.get("match_strings"):
        return 0.0
    match_strings = list(entity_context.get("match_strings") or [])
    parts = " ".join(
        [
            str(edge.get("node_from", "")),
            str(edge.get("node_to", "")),
            str(edge.get("relation", "")),
            str(edge.get("text", "")),
        ]
    )
    folded = re.sub(r"[^a-z0-9]", "", parts.lower())
    for m in match_strings:
        if _contains_folded(folded, m):
            return 1.0
    return 0.0
