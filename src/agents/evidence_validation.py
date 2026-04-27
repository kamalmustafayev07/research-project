"""Programmatic evidence grounding checks (non-LLM).

Surface substring checks are complemented by a token-overlap alignment score
so answers grounded in *paraphrased* evidence (different wording, same
content words) are not incorrectly abstained. No entity lists or dataset
patterns — only generic tokenisation and set overlap.
"""

from __future__ import annotations

import re
from typing import Any


def _fold(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# General English function words — not domain keywords.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "or",
        "of",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "as",
        "is",
        "was",
        "were",
        "been",
        "be",
        "are",
        "did",
        "does",
        "do",
        "that",
        "this",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "how",
        "why",
        "not",
        "no",
        "yes",
        "also",
        "into",
        "over",
        "under",
        "than",
        "then",
        "there",
        "their",
        "they",
        "his",
        "her",
        "its",
        "he",
        "she",
        "it",
        "we",
        "you",
    }
)


def _significant_tokens(text: str, min_len: int = 3) -> list[str]:
    """Lowercase alphanumeric tokens, excluding very short and stop tokens."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']*", (text or "").lower())
    out: list[str] = []
    for w in words:
        if len(w) < min_len or w in _STOPWORDS:
            continue
        out.append(w)
    return out


def answer_evidence_alignment_score(answer: str, blob_lower: str) -> float:
    """Return a soft alignment score in ``[0, 1]`` between answer and evidence.

    Combines:
    - **Folded phrase**: the full answer (punctuation-stripped) appears as a
      substring of the folded evidence blob — strong signal for multi-token
      spans (e.g. city + state).
    - **Token recall**: fraction of significant answer tokens that appear as
      substrings in the lowercased blob — supports paraphrases that keep
      content words (e.g. ``discovered`` / ``discovery`` share no token but
      ``penicillin`` still matches).
    - **Jaccard** on significant-token sets between answer and a bag of words
      from the blob — smooth signal when wording diverges but entities overlap.

    Abstention phrases return ``1.0`` so downstream logic can treat them
    separately (they are not "supported" answers in the factual sense).
    """
    low = (answer or "").strip()
    if not low or low.lower().startswith("not enough"):
        return 1.0
    if "insufficient" in low.lower() or "cannot determine" in low.lower():
        return 1.0

    blob = (blob_lower or "").lower()
    folded_blob = _fold(blob)
    folded_ans = _fold(low)

    if len(folded_ans) >= 4 and folded_ans in folded_blob:
        return 1.0

    ans_toks = _significant_tokens(low)
    if not ans_toks:
        return 0.45

    hits = sum(1 for t in ans_toks if t in blob)
    recall = hits / len(ans_toks)

    blob_words = set(_significant_tokens(blob))
    ans_set = set(ans_toks)
    inter = ans_set & blob_words
    union = ans_set | blob_words
    jaccard = len(inter) / len(union) if union else 0.0

    # Weight recall higher (answers are short); Jaccard stabilises long blobs.
    score = 0.55 * recall + 0.45 * jaccard
    return max(0.0, min(1.0, float(score)))


def answer_grounded_in_evidence(
    answer: str,
    blob_lower: str,
    *,
    min_alignment: float = 0.18,
) -> bool:
    """True if the answer is supported by substring rules OR alignment score.

    ``min_alignment`` is intentionally low — the alignment score is one signal
    among several; combined with :func:`answer_strings_supported_by_evidence`
    we avoid false abstentions on correct paraphrases.
    """
    if answer_strings_supported_by_evidence(answer, blob_lower):
        return True
    return answer_evidence_alignment_score(answer, blob_lower) >= min_alignment


def evidence_blob(evidence_chain: list[dict[str, Any]], passages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for hop in evidence_chain:
        parts.append(
            f"{hop.get('node_from', '')} {hop.get('relation', '')} {hop.get('node_to', '')} {hop.get('text', '')}"
        )
    for p in passages:
        parts.append(f"{p.get('title', '')} {p.get('text', '')}")
    return " ".join(parts).lower()


def resolved_entities_from_context(entity_context: dict[str, Any] | None) -> list[str]:
    """Extract the CURRENT-LOOP pivot entities from entity context.

    Only uses ``match_strings`` and ``canonical_entities``, which are updated
    per retrieval loop to reflect the current pivot entity.

    Deliberately excludes ``sub_question_entities``: those carry entities from
    ALL decomposed hops (e.g. hop-1's "Microsoft" is still present when we
    are in hop-2 reasoning about Bill Gates' birthplace).  Including them causes
    ``chain_covers_resolved_entities`` to demand that EVERY hop's entities appear
    in the CURRENT loop's evidence blob, which is impossible in multi-hop
    reasoning and produces spurious "Not enough information" early returns.
    """
    if not entity_context:
        return []
    ms = list(entity_context.get("match_strings") or [])
    for c in entity_context.get("canonical_entities") or []:
        if str(c).strip():
            ms.append(str(c).strip())
    seen: set[str] = set()
    out: list[str] = []
    for m in ms:
        t = str(m).strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def chain_covers_resolved_entities(
    evidence_chain: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    required: list[str],
) -> bool:
    """Return True if at least ONE resolved entity is mentioned in the evidence.

    Relaxed from the previous ``min(2, len(required))`` threshold.  In multi-hop
    reasoning each retrieval loop brings evidence for only the CURRENT hop, not
    all prior hops, so requiring two entities causes false negatives in the
    intermediate loops (e.g. the final loop has "Bill Gates birthplace" passages
    but the old "Microsoft" entity from hop-1 is also in ``required`` → fail).

    Requiring just one entity is a meaningful sanity check (at least something
    relevant was retrieved) without over-constraining multi-hop scenarios.
    """
    if not required:
        return True
    blob_f = _fold(evidence_blob(evidence_chain, passages))
    long_req = [e for e in required if len(_fold(e)) >= 3]
    if not long_req:
        return True
    for ent in long_req:
        if _fold(ent) in blob_f:
            return True  # At least one entity found — sufficient
    return False


def answer_strings_supported_by_evidence(answer: str, blob_lower: str) -> bool:
    """Return True if the answer is grounded in the evidence blob.

    Checks that at least one significant content word from the answer (length
    >= 4, not a stop word) appears in the blob.  This is intentionally lenient:

    - It allows for morphological variation (the blob has "Seattle" regardless
      of whether the passage says "born in Seattle" or "Seattle-born").
    - It avoids rejecting single-entity answers that are confirmed by just one
      matching token (e.g. "London" matches if the blob contains "london").
    - "Not enough information" and similar abstain phrases always pass so they
      propagate to the critic for proper handling.

    The previous implementation required ALL significant words to be present,
    which rejected correct answers whenever the evidence blob did not contain
    every word of the answer (e.g. the capital "London" when the evidence blob
    covered only "United Kingdom" passages without mentioning the capital).
    """
    low = (answer or "").strip()
    if not low or low.lower().startswith("not enough"):
        return True
    if "insufficient" in low.lower():
        return True
    stop = {"the", "and", "or", "of", "a", "an", "in", "on", "at", "to", "for", "did"}
    folded_blob = _fold(blob_lower)
    significant = [
        w for w in re.findall(r"\b\w+\b", low) if len(w) >= 4 and w.lower() not in stop
    ]
    if not significant:
        return True  # Nothing meaningful to verify; allow
    for w in significant:
        if w.lower() in blob_lower or _fold(w) in folded_blob:
            return True  # At least one anchor word is grounded
    return False
