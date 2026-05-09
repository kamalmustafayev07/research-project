"""Cap reasoner/critic confidence by evidence-alignment score (monotonic, no dataset tuning)."""

from __future__ import annotations


def calibrate_reasoner_confidence(
    raw_confidence: float,
    alignment_score: float,
    *,
    is_final_hop: bool | None = None,
) -> float:
    """``min(raw, ceiling(alignment))``; final hop tightens ceiling when alignment is weak."""
    a = max(0.0, min(1.0, float(alignment_score)))
    r = max(0.0, min(1.0, float(raw_confidence)))

    ceiling = 0.28 + 0.72 * a
    if is_final_hop and a < 0.28:
        ceiling = min(ceiling, 0.48)
    out = min(r, ceiling)
    return max(0.0, min(1.0, out))
