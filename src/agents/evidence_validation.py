"""Programmatic evidence grounding checks (non-LLM)."""

from __future__ import annotations

import re
from typing import Any


def _fold(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


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
    if not entity_context:
        return []
    ms = list(entity_context.get("match_strings") or [])
    for c in entity_context.get("canonical_entities") or []:
        if str(c).strip():
            ms.append(str(c).strip())
    for row in entity_context.get("sub_question_entities") or []:
        for e in row.get("resolved_entities") or []:
            ms.append(str(e))
        em = row.get("entity_map") or {}
        for k, v in em.items():
            ms.extend([str(k), str(v)])
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
    if not required:
        return True
    blob_f = _fold(evidence_blob(evidence_chain, passages))
    long_req = [e for e in required if len(_fold(e)) >= 3]
    if not long_req:
        return True
    hits = 0
    for ent in long_req:
        ef = _fold(ent)
        if ef in blob_f:
            hits += 1
    need = 1 if len(long_req) == 1 else min(2, len(long_req))
    return hits >= need


def answer_strings_supported_by_evidence(answer: str, blob_lower: str) -> bool:
    low = (answer or "").strip()
    if not low or low.lower().startswith("not enough"):
        return True
    if "insufficient" in low.lower():
        return True
    stop = {"the", "and", "or", "of", "a", "an", "in", "on", "at", "to", "for", "did"}
    folded_blob = _fold(blob_lower)
    for w in re.findall(r"\b\w+\b", low):
        wl = w.lower()
        if len(w) < 4 or wl in stop:
            continue
        if wl not in blob_lower and _fold(w) not in folded_blob:
            return False
    return True
