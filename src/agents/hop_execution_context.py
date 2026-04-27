"""Shared hop execution contract for multi-hop QA (module coordination).

All agents that touch a hop (retriever, reasoner, critic, evidence checks) read
the same structured snapshot from ``entity_context["hop_execution_context"]``.
This reduces *state drift*: no module silently assumes a different active
subject, relation, or plan than the HopMemory state machine.

The contract is built only from ``HopMemory`` + the active ``HopRecord`` — no
dataset names, no benchmark entities, no answer-specific rules.
"""

from __future__ import annotations

from typing import Any

from src.agents.hop_safety import (
    HopMemory,
    HopRecord,
    expected_answer_granularity,
    expected_answer_type,
)


def build_hop_execution_context(
    memory: HopMemory,
    hop: HopRecord,
    resolved_subquestion: str,
) -> dict[str, Any]:
    """Return a JSON-serialisable dict describing the active hop for all modules."""
    total = len(memory.hops)
    idx = hop.hop_index
    prior_chain: list[dict[str, Any]] = []
    for p in memory.previous_hops(idx):
        prior_chain.append(
            {
                "hop_index": p.hop_index,
                "subject": p.main_entity.text,
                "subject_type": p.main_entity.entity_type,
                "relation": p.relation,
                "answer": p.answer,
                "confidence": p.confidence,
            }
        )

    rsq = (resolved_subquestion or hop.subquestion_text or "").strip()
    return {
        "schema_version": 1,
        "active_hop_index": idx,
        "total_hops": total,
        "is_final_hop": idx >= total,
        "remaining_hops_after": max(0, total - idx),
        "original_question": memory.original_question,
        "resolved_objective": rsq,
        "subject": {
            "text": hop.main_entity.text,
            "entity_type": hop.main_entity.entity_type,
            "role": hop.main_entity.role,
        },
        "relation": hop.relation,
        "expected_answer_type": expected_answer_type(hop.relation, hop.subquestion_text),
        "expected_answer_granularity": expected_answer_granularity(
            hop.relation, hop.subquestion_text
        ),
        "prior_resolved_chain": prior_chain,
    }


def format_hop_execution_context_for_prompt(ctx: dict[str, Any]) -> str:
    """Short natural-language block for LLM prompts (reasoner / optional critic)."""
    if not ctx or not ctx.get("schema_version"):
        return ""
    subj = ctx.get("subject") or {}
    st = subj.get("text", "")
    stype = subj.get("entity_type", "")
    role = subj.get("role", "")
    lines = [
        "HOP EXECUTION CONTRACT (authoritative — all modules use this state):",
        f"  Hop {ctx.get('active_hop_index')}/{ctx.get('total_hops')} "
        f"({'final' if ctx.get('is_final_hop') else 'intermediate'}).",
        f"  Objective: {ctx.get('resolved_objective', '')}",
        f"  Locked subject: {st} ({stype}, role={role})",
        f"  Relation: {ctx.get('relation', '')}",
        f"  Expected answer type: {ctx.get('expected_answer_type', '')}",
    ]
    gran = ctx.get("expected_answer_granularity") or ""
    if gran:
        lines.append(f"  Expected answer granularity: {gran}")
    prior = ctx.get("prior_resolved_chain") or []
    if prior:
        lines.append("  Prior hops (already resolved; do not repeat as this hop's answer):")
        for row in prior:
            ans = row.get("answer") or "[pending]"
            lines.append(
                f"    — hop {row.get('hop_index')}: "
                f"{row.get('subject')!r} --{row.get('relation')}--> {ans!r}"
            )
    return "\n".join(lines) + "\n"


def attach_hop_execution_context(
    entity_context: dict[str, Any],
    memory: HopMemory,
    hop: HopRecord,
    resolved_subquestion: str,
) -> dict[str, Any]:
    """Return a shallow copy of ``entity_context`` with the contract attached."""
    out = dict(entity_context)
    out["hop_execution_context"] = build_hop_execution_context(
        memory, hop, resolved_subquestion
    )
    return out
