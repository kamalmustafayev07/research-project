"""Reasoner / critic confidence calibration from evidence alignment.

Confidence should track how well the answer is supported by retrieved text,
not only the LLM's self-reported number. This module applies a *generic*
monotonic mapping from an evidence-alignment score to a ceiling on confidence.
No dataset- or entity-specific tuning.
"""

from __future__ import annotations


def calibrate_reasoner_confidence(
    raw_confidence: float,
    alignment_score: float,
    *,
    is_final_hop: bool | None = None,
) -> float:
    """Cap self-reported confidence by how strongly the answer aligns with evidence.

    Parameters
    ----------
    raw_confidence:
        Value from the reasoner JSON (typically in [0, 1]).
    alignment_score:
        Score in [0, 1] from :func:`src.agents.evidence_validation.answer_evidence_alignment_score`.
    is_final_hop:
        When True, slightly tighten the ceiling for weak alignment so the
        critic is less likely to approve under-supported final answers.
    """
    a = max(0.0, min(1.0, float(alignment_score)))
    r = max(0.0, min(1.0, float(raw_confidence)))

    # Monotonic ceiling: strong alignment allows almost full self-confidence;
    # weak alignment pulls the ceiling down smoothly (no step functions on
    # entity classes).
    ceiling = 0.28 + 0.72 * a
    if is_final_hop and a < 0.28:
        ceiling = min(ceiling, 0.48)
    out = min(r, ceiling)
    return max(0.0, min(1.0, out))
