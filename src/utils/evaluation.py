"""End-to-end evaluation helpers for baselines and full pipeline."""

from __future__ import annotations

from dataclasses import asdict
from collections import Counter
from pathlib import Path
from time import perf_counter
import re
from typing import Any

import numpy as np
import pandas as pd
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


def latency_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate latency metrics from per-example predictions."""
    if not predictions:
        return {
            "latency_mean": 0.0,
            "latency_p95": 0.0,
            "latency_min": 0.0,
            "latency_max": 0.0,
            "agent_latency_mean": {},
        }

    totals = [float(row.get("latency_total", 0.0)) for row in predictions]
    agent_totals: dict[str, list[float]] = {}

    for row in predictions:
        breakdown = row.get("latency_breakdown", {})
        if not isinstance(breakdown, dict):
            continue
        for key, value in breakdown.items():
            agent_totals.setdefault(str(key), []).append(float(value))

    return {
        "latency_mean": float(np.mean(totals)),
        "latency_p95": float(np.percentile(totals, 95)),
        "latency_min": float(np.min(totals)),
        "latency_max": float(np.max(totals)),
        "agent_latency_mean": {
            key: float(np.mean(values))
            for key, values in sorted(agent_totals.items())
            if values
        },
    }


def save_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    """Save predictions as CSV with flattened latency columns."""
    rows: list[dict[str, Any]] = []
    for row in predictions:
        breakdown = row.get("latency_breakdown", {}) if isinstance(row.get("latency_breakdown", {}), dict) else {}
        rows.append(
            {
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "method": row.get("method"),
                "question": row.get("question"),
                "gold": row.get("gold"),
                "prediction": row.get("prediction"),
                "confidence": row.get("confidence"),
                "latency_total": row.get("latency_total", 0.0),
                "latency_decomposer": breakdown.get("decomposer", 0.0),
                "latency_retriever": breakdown.get("retriever", 0.0),
                "latency_reasoner": breakdown.get("reasoner", 0.0),
                "latency_critic": breakdown.get("critic", 0.0),
                "evidence_hops": len(row.get("evidence_chain", []) or []),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_latency_records_csv(path: Path, latency_records: list[dict[str, Any]]) -> None:
    """Save structured per-question latency records to CSV."""
    rows: list[dict[str, Any]] = []
    for row in latency_records:
        breakdown = row.get("latency_breakdown", {}) if isinstance(row.get("latency_breakdown", {}), dict) else {}
        rows.append(
            {
                "question_id": row.get("question_id"),
                "dataset": row.get("dataset"),
                "latency_total": row.get("latency_total", 0.0),
                "latency_decomposer": breakdown.get("decomposer", 0.0),
                "latency_retriever": breakdown.get("retriever", 0.0),
                "latency_reasoner": breakdown.get("reasoner", 0.0),
                "latency_critic": breakdown.get("critic", 0.0),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_method_on_dataset(
    method_name: str,
    data: list[dict[str, Any]],
    infer_fn: Any,
    dataset_name: str = "unknown",
) -> dict[str, Any]:
    """Run a given inference function over dataset and return metrics and outputs."""
    predictions: list[dict[str, Any]] = []
    for item in tqdm(data, desc=f"Evaluating {method_name}"):
        started = perf_counter()
        output = infer_fn(item)
        elapsed = perf_counter() - started
        latency_total = float(output.get("latency_total", elapsed))
        latency_breakdown = output.get("latency_breakdown", {})
        if not isinstance(latency_breakdown, dict):
            latency_breakdown = {}

        predictions.append(
            {
                "id": item.get("qid") or item.get("id"),
                "dataset": item.get("dataset", dataset_name),
                "method": method_name,
                "question": item["question"],
                "gold": item["answer"],
                "prediction": output.get("answer", ""),
                "evidence_chain": output.get("evidence_chain", []),
                "confidence": output.get("confidence", 0.0),
                "latency_total": latency_total,
                "latency_breakdown": latency_breakdown,
            }
        )

    latency_records = [
        {
            "question_id": row.get("id"),
            "dataset": row.get("dataset", dataset_name),
            "latency_total": row.get("latency_total", 0.0),
            "latency_breakdown": row.get("latency_breakdown", {}),
        }
        for row in predictions
    ]

    metrics = score_predictions(predictions)
    metrics.update(latency_summary(predictions))

    return {
        "method": method_name,
        "dataset": dataset_name,
        "metrics": metrics,
        "latency_records": latency_records,
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
