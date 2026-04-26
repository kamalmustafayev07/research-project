"""Build diverse retrieval queries (OR-style recall) from question + entities + relations.

Key design principles:
- OR-merge: each query variant is embedded independently; passages are ranked by
  max dense score across all variants (high recall).
- Referential anchors (e.g. {"role": "writer", "work": "The Hobbit"}) trigger dedicated
  work→creator queries so the retriever can find the actual person even when only the
  pivot work is mentioned in the question.
- Relation phrases expand canonical entities into targeted retrieval strings (e.g.
  "Oxford university education").
"""

from __future__ import annotations

import re
from typing import Any


_RELATION_PHRASES: dict[str, tuple[str, ...]] = {
    "studied_at": (
        "education",
        "university",
        "college",
        "studied",
        "attended",
        "graduated",
        "alma mater",
        "school",
        "academic",
        "degree",
    ),
    "author_of": (
        "author",
        "wrote",
        "written by",
        "written",
        "novel",
        "book",
        "penned",
        "published",
        "novelist",
        "creator",
        "write",
        "works",
        "biography",
        "bibliography",
    ),
    "born_in": (
        "born",
        "birthplace",
        "birth",
        "native",
        "hometown",
        "grew up",
        "nationality",
        "origin",
    ),
    "directed_by": (
        "director",
        "directed",
        "directed by",
        "film",
        "movie",
        "filmmaker",
        "cinematographer",
    ),
    "located_in": (
        "located",
        "based in",
        "situated",
        "capital",
        "city",
        "country",
        "region",
    ),
    "capital_of": (
        "capital",
        "capital city",
        "seat of government",
        "administrative center",
    ),
    "composed_by": (
        "composer",
        "composed",
        "composed by",
        "music",
        "musician",
        "wrote the music",
        "symphony",
        "opera",
    ),
    "invented_by": (
        "inventor",
        "invented",
        "invention",
        "discovered",
        "patent",
        "discovery",
    ),
    "founded_by": (
        "founder",
        "founded",
        "established",
        "created",
        "started",
        "co-founder",
        "founding",
    ),
    "created_by": (
        "creator",
        "created",
        "made by",
        "artist",
        "designed",
        "designed by",
    ),
    "produced_by": (
        "producer",
        "produced",
        "production",
        "produced by",
    ),
    "designed_by": (
        "designer",
        "designed",
        "architect",
        "designed by",
    ),
}

# Maps referential role noun → past-tense verb for "who [verb] [WORK]" queries.
_ROLE_TO_VERB: dict[str, str] = {
    "writer": "wrote",
    "writers": "wrote",
    "author": "wrote",
    "authors": "wrote",
    "novelist": "wrote",
    "novelists": "wrote",
    "playwright": "wrote",
    "screenwriter": "wrote",
    "poet": "wrote",
    "lyricist": "wrote",
    "librettist": "wrote",
    "biographer": "wrote",
    "journalist": "wrote",
    "creator": "created",
    "creators": "created",
    "illustrator": "illustrated",
    "cartoonist": "drew",
    "director": "directed",
    "directors": "directed",
    "producer": "produced",
    "producers": "produced",
    "composer": "composed",
    "composers": "composed",
    "inventor": "invented",
    "inventors": "invented",
    "founder": "founded",
    "founders": "founded",
    "designer": "designed",
    "designers": "designed",
    "painter": "painted",
    "sculptor": "sculpted",
    "photographer": "photographed",
    "choreographer": "choreographed",
}


def _dedupe_preserve_order(items: list[str], max_items: int = 28) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        t = (x or "").strip()
        if len(t) < 2:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_items:
            break
    return out


def strip_hop_placeholders(text: str) -> str:
    """Remove <answer of hop N> style placeholders as a last-resort fallback."""
    s = (text or "").strip()
    s = re.sub(r"<\s*answer\s+of\s+hop\s+\d+\s*>", "", s, flags=re.I)
    s = re.sub(r"<\s*hop\s*\d+\s*>", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def resolve_placeholders(
    text: str,
    referential_anchors: list[dict[str, str]],
) -> str:
    """Replace <answer of hop N> with a descriptive phrase from referential anchors.

    Transforms "Which university did <answer of hop 1> attend?" into
    "Which university did the writer of The Hobbit attend?" — a meaningful,
    grammatically correct retrieval query that has high semantic similarity to
    passages about the author's education.

    Falls back to stripping if no anchor covers the hop index.
    """

    def _replace(m: re.Match) -> str:
        hop_num = int(m.group(1))
        idx = hop_num - 1
        if 0 <= idx < len(referential_anchors):
            anchor = referential_anchors[idx]
            role = anchor.get("role", "author")
            work = anchor.get("work", "")
            if work:
                return f"the {role} of {work}"
        return ""

    s = re.sub(r"<\s*answer\s+of\s+hop\s+(\d+)\s*>", _replace, (text or "").strip(), flags=re.I)
    s = re.sub(r"<\s*hop\s*\d+\s*>", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def build_retrieval_query_pack(
    question: str,
    decomposition: dict[str, Any] | None,
    entity_context: dict[str, Any] | None,
) -> tuple[list[str], str]:
    """Return (query_variants, debug_summary).

    OR-logic: each variant is embedded separately; results are merged with max dense score.

    Referential anchors (e.g. the writer of [WORK]) receive dedicated work→creator
    queries so recall on the first hop is high even when the actual person's name is
    unknown at query-build time.
    """
    ec = entity_context or {}
    decomp = decomposition or {}
    canonical = list(ec.get("canonical_entities") or [])
    match_strings = list(ec.get("match_strings") or [])
    rel_hints = list(ec.get("relation_hints") or [])
    mention_map = dict(ec.get("mention_to_canonical") or {})
    referential_anchors = list(ec.get("referential_anchors") or [])

    # Resolve <answer of hop N> → "the [role] of [work]" so sub-questions become
    # grammatically correct, semantically rich retrieval queries.
    sub_questions = [
        resolve_placeholders(s, referential_anchors)
        for s in (decomp.get("sub_questions") or [])
        if str(s).strip()
    ]

    queries: list[str] = []

    # ------------------------------------------------------------------
    # 1. Referential anchor queries — work→creator first-hop resolution
    #
    # For "the writer of [WORK]" we generate targeted queries that will
    # find the actual person in the corpus without knowing their name yet.
    # ------------------------------------------------------------------
    for anchor in referential_anchors:
        work = (anchor.get("work") or "").strip()
        role = (anchor.get("role") or "author").strip()
        if not work:
            continue

        # High-recall work title queries
        queries.append(work)
        queries.append(f"{work} {role}")
        queries.append(f"{work} author")
        queries.append(f"{work} written by")
        queries.append(f"{work} creator")
        queries.append(f"{work} biography")

        # Question-form query (very high semantic similarity to relevant passages)
        action_verb = _ROLE_TO_VERB.get(role, f"is the {role} of")
        queries.append(f"who {action_verb} {work}")

        # Cross-queries: combine work with the downstream relation
        # e.g. "The Hobbit author university" to pull author+education passages
        for rh in rel_hints:
            if rh in _RELATION_PHRASES:
                phrases = _RELATION_PHRASES[rh]
                if phrases:
                    queries.append(f"{work} {role} {phrases[0]}")
                    queries.append(f"{work} {phrases[0]}")

    # ------------------------------------------------------------------
    # 2. Canonical entity queries
    # ------------------------------------------------------------------
    for c in canonical[:6]:
        queries.append(c)
    for m in match_strings[:10]:
        if m and m not in queries:
            queries.append(m)

    # ------------------------------------------------------------------
    # 3. Relation-expanded queries (canonical × relation phrase)
    # ------------------------------------------------------------------
    rel_phrases: list[str] = []
    for r in rel_hints:
        rel_phrases.extend(_RELATION_PHRASES.get(r, ()))
    rel_phrases = list(dict.fromkeys(rel_phrases))[:10]

    for c in canonical[:3]:
        for kw in rel_phrases[:4]:
            queries.append(f"{c} {kw}")
    if canonical and rel_phrases:
        queries.append(f"{canonical[0]} {' '.join(rel_phrases[:3])}")

    # ------------------------------------------------------------------
    # 4. Sub-question queries (from decomposer — include hop 1 as-is)
    # ------------------------------------------------------------------
    for sq in sub_questions[:5]:
        if len(sq) > 5:
            queries.append(sq[:320])

    # ------------------------------------------------------------------
    # 5. Full and shortened question
    # ------------------------------------------------------------------
    q_short = re.sub(
        r"^(what|which|who|where|when|how)\s+",
        "",
        (question or "").strip(),
        flags=re.I,
    )
    if len(q_short) > 15:
        queries.append(q_short[:400])
    queries.append((question or "")[:400])

    # ------------------------------------------------------------------
    # 6. Mention → canonical alias cross-queries
    # ------------------------------------------------------------------
    for surf, canon in list(mention_map.items())[:6]:
        if canon and surf.lower() != canon.lower():
            queries.append(f"{canon} {surf}"[:200])

    queries = _dedupe_preserve_order(queries, max_items=28)
    debug = (
        f"n_variants={len(queries)}; rel={rel_hints[:4]}; "
        f"canon={canonical[:3]}; ref_anchors={len(referential_anchors)}"
    )
    return queries, debug
