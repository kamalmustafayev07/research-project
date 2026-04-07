"""End-to-end evaluation helpers for baselines and full pipeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tqdm import tqdm

from src.evaluation.hotpotqa_eval import aggregate_metrics, exact_match_score, f1_score, heuristic_explainability_score


def score_predictions(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate metrics for a list of prediction rows."""
    rows: list[dict[str, float]] = []
    for row in predictions:
        em = exact_match_score(row["prediction"], row["gold"])
        f1 = f1_score(row["prediction"], row["gold"])
        explainability = heuristic_explainability_score(row.get("evidence_chain", []), row["prediction"])
        rows.append({"em": em, "f1": f1, "explainability": explainability})

    metrics = aggregate_metrics(rows)
    return asdict(metrics)


def run_method_on_dataset(
    method_name: str,
    data: list[dict[str, Any]],
    infer_fn: Any,
) -> dict[str, Any]:
    """Run a given inference function over dataset and return metrics and outputs."""
    predictions: list[dict[str, Any]] = []
    for item in tqdm(data, desc=f"Evaluating {method_name}"):
        output = infer_fn(item)
        predictions.append(
            {
                "id": item["qid"],
                "question": item["question"],
                "gold": item["answer"],
                "prediction": output.get("answer", ""),
                "evidence_chain": output.get("evidence_chain", []),
                "confidence": output.get("confidence", 0.0),
            }
        )

    return {
        "method": method_name,
        "metrics": score_predictions(predictions),
        "predictions": predictions,
    }
