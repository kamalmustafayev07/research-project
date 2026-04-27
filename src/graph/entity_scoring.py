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
        "alma mater",
        "read ",
        "ba ",
        "ma ",
        "phd",
        "degree",
        "graduate of",
    ),
    "born_in": (
        "born",
        "birth",
        "birthplace",
        "b.",
        "native of",
        "raised",
        "grew up",
        "hometown",
        "originally from",
        "born and raised",
    ),
    "directed_by": ("directed", "director", "film", "movie"),
    "located_in": ("located", "based in", "headquartered", "city", "country", "situat"),
    "founded_by": ("founded", "co-founded", "founder", "established", "co-founder"),
    "invented_by": ("invented", "inventor", "patented", "creator", "developed"),
    "composed_by": ("composed", "composer", "wrote the score", "music"),
    "produced_by": ("produced", "producer"),
    "designed_by": ("designed", "designer"),
    "created_by": ("created", "creator"),
    "capital_of": ("capital", "capital city", "seat of government"),
}


# Answer-shape patterns: when a passage contains an *explicit answer slot*
# for a relation (e.g. "born in <ProperNoun>"), it almost certainly answers
# the active hop and should rank above passages that merely mention the
# entity. The patterns are anchored on Title-Cased spans so they only fire
# on passages with an obvious answer shape — they do not trigger on generic
# prose without a proper-noun candidate. All patterns are case-sensitive on
# the answer span so "born in 1955" does not falsely match.
_RELATION_ANSWER_PATTERNS: dict[str, tuple[str, ...]] = {
    "born_in": (
        r"\bborn(?:\s+and\s+raised)?\s+in\s+([A-Z][a-zA-Z\.\-']+(?:[,\s]+[A-Z][a-zA-Z\.\-']+){0,3})",
        r"\braised\s+in\s+([A-Z][a-zA-Z\.\-']+(?:[,\s]+[A-Z][a-zA-Z\.\-']+){0,3})",
        r"\b(?:hometown|birthplace)\s+(?:of|is|:)\s+([A-Z][a-zA-Z\.\-']+(?:[,\s]+[A-Z][a-zA-Z\.\-']+){0,3})",
        r"\bnative\s+of\s+([A-Z][a-zA-Z\.\-']+(?:[,\s]+[A-Z][a-zA-Z\.\-']+){0,3})",
        r"\boriginally\s+from\s+([A-Z][a-zA-Z\.\-']+(?:[,\s]+[A-Z][a-zA-Z\.\-']+){0,3})",
    ),
    "studied_at": (
        r"\b(?:studied|attended|graduated\s+from|enrolled\s+at|read\s+[A-Z][a-z]+\s+at)\s+(?:the\s+)?([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,4})",
        r"\balma\s+mater\s+(?:was|is|:)\s+(?:the\s+)?([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,4})",
    ),
    "founded_by": (
        r"\b(?:founded|co-?founded|established)\s+by\s+([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,4})",
        r"\b([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,4})\s+(?:founded|co-?founded|established)\s+",
    ),
    "directed_by": (
        r"\bdirected\s+by\s+([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,4})",
    ),
    "author_of": (
        r"\b(?:written|authored)\s+by\s+([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,4})",
    ),
    "capital_of": (
        r"\bcapital\s+(?:city\s+)?(?:of\s+[A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,3}\s+is\s+)?([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,3})",
    ),
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


def _relation_answer_pattern_score(passage: dict[str, Any], relation_hints: list[str]) -> float:
    """Return a 0..1 score for whether the passage contains an explicit
    answer-shape pattern for any of the active-hop relations.

    Higher score = the passage exposes a Title-Cased proper-noun candidate
    in the exact syntactic slot expected by the relation (e.g. "born in
    Seattle"). This is a soft signal — a value of 0 never zeroes the
    overall passage score, it just removes a small bonus.
    """
    if not relation_hints:
        return 0.0
    # Use raw (un-lowercased) text so the Title-Cased pattern fires.
    text = f"{passage.get('title', '')}\n{passage.get('text', '')}"
    if not text.strip():
        return 0.0
    hits = 0
    total_patterns = 0
    for rel in relation_hints:
        patterns = _RELATION_ANSWER_PATTERNS.get(rel, ())
        if not patterns:
            continue
        total_patterns += 1
        for pat in patterns:
            if re.search(pat, text):
                hits += 1
                break  # one match per relation is enough
    if total_patterns == 0:
        return 0.0
    return hits / float(total_patterns)


def _prototype_relation_score(
    passage: dict[str, Any],
    prototype: dict[str, Any] | None,
) -> float | None:
    """Cue-based relation consistency using an LLM-generated prototype.

    The prototype is produced once per active hop by
    :func:`src.agents.relation_scorer.generate_relation_prototype` and embedded
    in ``entity_context["relation_prototype"]`` as a plain dict. It contains
    short lowercase cue phrases that an answer-bearing passage is expected
    to contain (and "negative cues" indicating an off-topic same-subject
    passage).

    Returning ``None`` signals "no prototype signal" so the caller can fall
    back to the keyword table; otherwise we return a number in [0, 1].
    """
    if not prototype:
        return None
    cues = prototype.get("cue_phrases") or []
    neg_cues = prototype.get("negative_cue_phrases") or []
    if not cues:
        return None
    text_low = _blob(passage).lower()
    if not text_low.strip():
        return 0.0
    pos_hits = sum(1 for c in cues if c and c in text_low)
    neg_hits = sum(1 for c in neg_cues if c and c in text_low)
    reward = pos_hits / max(1, len(cues))
    penalty = 0.5 * (neg_hits / max(1, len(neg_cues))) if neg_cues else 0.0
    raw = reward - penalty
    return max(0.0, min(1.0, raw))


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


def relation_consistency_score(
    passage: dict[str, Any],
    relation_hints: list[str],
    prototype: dict[str, Any] | None = None,
) -> float:
    """Relevance signal that an LLM-generated prototype's cues match the passage.

    Preference order:
    1. LLM-generated prototype cues (hop-aware, conditioned on prior hops).
    2. Static ``_RELATION_KEYWORDS`` keyword table (legacy fallback).
    3. Neutral 0.5 if neither signal is available.

    The prototype-based signal is purely substring-driven and inherits all the
    benefits of the older keyword approach (cheap, deterministic) without its
    main weakness: keyword coverage that is fixed at code-write time and not
    conditioned on the current hop chain.
    """
    proto_score = _prototype_relation_score(passage, prototype)
    if proto_score is not None:
        return proto_score
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

    The "answer-shape" boost rewards passages that contain a Title-Cased
    proper-noun span in the exact syntactic slot that the active relation
    expects (e.g. "born in <City>" for ``born_in``). It is a small bonus —
    weighted to break ties between passages that all match the entity, in
    favour of the one that *literally answers* the active sub-question.
    """
    d = normalize_dense_score(base_semantic)
    if not entity_context or not entity_context.get("match_strings"):
        return float(base_semantic)

    ms = list(entity_context.get("match_strings") or [])
    rel_h = list(entity_context.get("relation_hints") or [])
    prototype = entity_context.get("relation_prototype")
    ent = passage_entity_overlap_score(passage, ms)
    if rel_h or prototype:
        rel = relation_consistency_score(passage, rel_h, prototype=prototype)
    else:
        rel = 0.45
    if (rel_h or prototype) and rel < 0.12:
        rel = 0.18
    title = title_overlap_score(passage, query)
    answer_shape = _relation_answer_pattern_score(passage, rel_h) if rel_h else 0.0

    w_d = float(entity_context.get("w_dense", 0.36))
    w_e = float(entity_context.get("w_entity", 0.28))
    w_r = float(entity_context.get("w_relation", 0.14))
    w_t = float(entity_context.get("w_title", 0.10))
    w_ans = float(entity_context.get("w_answer_shape", 0.12))
    return w_d * d + w_e * ent + w_r * rel + w_t * title + w_ans * answer_shape


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
    prototype = entity_context.get("relation_prototype")

    exact = exact_entity_match_score(passage, match_strings)
    rel = relation_consistency_score(passage, rel_hints, prototype=prototype)
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
