"""End-to-end evaluation helpers for baselines and full pipeline."""

from __future__ import annotations

from dataclasses import asdict
from collections import Counter
import re
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


def analyze_evidence_chains(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute basic structural and quality diagnostics for evidence chains."""
    total = len(predictions)
    if total == 0:
        return {
            "total_predictions": 0,
            "empty_chain_rate": 0.0,
            "avg_chain_length": 0.0,
            "avg_hop_score": 0.0,
            "avg_source_diversity": 0.0,
            "malformed_hop_rate": 0.0,
            "numeric_node_rate": 0.0,
            "top_relations": [],
        }

    chain_lengths: list[int] = []
    avg_scores: list[float] = []
    source_diversity: list[float] = []
    empty_chains = 0
    malformed_hops = 0
    total_hops = 0
    numeric_nodes = 0
    relation_counter: Counter[str] = Counter()

    for row in predictions:
        chain = row.get("evidence_chain", []) or []
        if not chain:
            empty_chains += 1
            chain_lengths.append(0)
            source_diversity.append(0.0)
            continue

        chain_lengths.append(len(chain))
        hop_scores = [float(hop.get("score", 0.0)) for hop in chain]
        avg_scores.append(sum(hop_scores) / max(1, len(hop_scores)))

        sources = {str(hop.get("source", "")).strip() for hop in chain if str(hop.get("source", "")).strip()}
        source_diversity.append(len(sources) / len(chain))

        for hop in chain:
            total_hops += 1
            text = str(hop.get("text", "")).strip()
            source = str(hop.get("source", "")).strip()
            if not text or not source:
                malformed_hops += 1

            relation = str(hop.get("relation", "")).strip()
            if relation:
                relation_counter[relation] += 1

            for node_key in ["node_from", "node_to"]:
                node_value = str(hop.get(node_key, "")).strip()
                if node_value and re.fullmatch(r"\d+", node_value):
                    numeric_nodes += 1

    return {
        "total_predictions": total,
        "empty_chain_rate": empty_chains / total,
        "avg_chain_length": sum(chain_lengths) / total,
        "avg_hop_score": (sum(avg_scores) / len(avg_scores)) if avg_scores else 0.0,
        "avg_source_diversity": sum(source_diversity) / total,
        "malformed_hop_rate": malformed_hops / max(1, total_hops),
        "numeric_node_rate": numeric_nodes / max(1, total_hops * 2),
        "top_relations": relation_counter.most_common(8),
    }
