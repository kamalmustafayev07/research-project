"""HotpotQA metrics: EM, token-level F1, and explainability scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.utils.helpers import normalize_text


@dataclass(slots=True)
class EvalMetrics:
    """Container for aggregated evaluation metrics."""

    exact_match: float
    f1: float
    explainability: float


def exact_match_score(prediction: str, gold: str) -> float:
    """Compute exact match between normalized strings."""
    return float(normalize_text(prediction) == normalize_text(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def heuristic_explainability_score(evidence_chain: list[dict[str, Any]], answer: str) -> float:
    """Estimate explainability by checking evidence depth and answer grounding."""
    if not evidence_chain:
        return 0.0

    coverage = min(1.0, len(evidence_chain) / 3.0)
    grounded = 0.0
    answer_tokens = set(normalize_text(answer).split())
    if answer_tokens:
        evidence_tokens = set()
        for hop in evidence_chain:
            evidence_tokens.update(normalize_text(hop.get("text", "")).split())
        intersection = answer_tokens & evidence_tokens
        grounded = len(intersection) / max(1, len(answer_tokens))

    return 0.6 * coverage + 0.4 * grounded


def aggregate_metrics(rows: list[dict[str, Any]]) -> EvalMetrics:
    """Aggregate per-example rows into final metrics."""
    if not rows:
        return EvalMetrics(exact_match=0.0, f1=0.0, explainability=0.0)

    em = sum(row["em"] for row in rows) / len(rows)
    f1 = sum(row["f1"] for row in rows) / len(rows)
    explainability = sum(row["explainability"] for row in rows) / len(rows)
    return EvalMetrics(exact_match=em, f1=f1, explainability=explainability)
