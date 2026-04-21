"""Compare benchmark performance and latency trade-offs across datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import SETTINGS
from src.utils.helpers import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dataset-level benchmark outputs.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(SETTINGS.paths.output_experiments),
        help=(
            "Base experiments directory (recommended), run directory, run/summaries directory, "
            "or legacy outputs/results directory."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run ID to compare from outputs/experiments/runs/<run_id>/summaries.",
    )
    return parser.parse_args()


def _latest_run_id(runs_root: Path) -> str:
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under {runs_root}")
    latest = max(run_dirs, key=lambda item: item.stat().st_mtime)
    return latest.name


def _resolve_summary_dir(input_dir: Path, run_id: str) -> tuple[Path, str | None, Path | None]:
    # Legacy flat folder support (outputs/results style).
    if (input_dir / "benchmark_summary_all_datasets.json").exists() or any(input_dir.glob("benchmark_summary_*.json")):
        return input_dir, None, None

    # Handle passing a run root directly.
    if (input_dir / "summaries").exists() and (input_dir / "metadata").exists():
        return input_dir / "summaries", input_dir.name, input_dir.parents[1] if len(input_dir.parents) > 1 else None

    # Handle base experiments directory.
    runs_root = input_dir / "runs"
    datasets_root = input_dir / "datasets"
    if runs_root.exists() and runs_root.is_dir():
        selected = run_id.strip() or _latest_run_id(runs_root)
        summary_dir = runs_root / selected / "summaries"
        if not summary_dir.exists():
            raise FileNotFoundError(f"Could not find summary directory for run '{selected}' at {summary_dir}")
        return summary_dir, selected, input_dir

    # Handle passing summaries directory directly.
    if input_dir.name == "summaries" and input_dir.exists():
        run_root = input_dir.parent
        selected = run_root.name if run_root.name else None
        experiments_root = run_root.parents[1] if len(run_root.parents) > 1 else None
        return input_dir, selected, experiments_root

    # Unknown layout: try as-is and surface a clear error later.
    return input_dir, None, (input_dir if datasets_root.exists() else None)


def _load_dataset_summaries(results_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}

    combined_path = results_dir / "benchmark_summary_all_datasets.json"
    if combined_path.exists():
        data = load_json(combined_path)
        if isinstance(data, dict) and data:
            summaries.update(data)

    for path in sorted(results_dir.glob("benchmark_summary_*.json")):
        if path.name == "benchmark_summary_all_datasets.json":
            continue
        dataset = path.stem.replace("benchmark_summary_", "")
        payload = load_json(path)
        if isinstance(payload, dict) and payload:
            summaries[dataset] = payload
    return summaries


def _load_dataset_summaries_from_tree(experiments_root: Path, run_id: str) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    datasets_root = experiments_root / "datasets"
    if not datasets_root.exists():
        return summaries

    for dataset_dir in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        summary_path = dataset_dir / run_id / "results" / "benchmark_summary.json"
        if not summary_path.exists():
            continue
        payload = load_json(summary_path)
        if isinstance(payload, dict) and payload:
            summaries[dataset_dir.name] = payload
    return summaries


def _flatten_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, methods in summaries.items():
        for method, metrics in methods.items():
            latency_mean = float(metrics.get("latency_mean", 0.0) or 0.0)
            f1 = float(metrics.get("f1", 0.0) or 0.0)
            em = float(metrics.get("exact_match", 0.0) or 0.0)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "exact_match": em,
                    "f1": f1,
                    "explainability": float(metrics.get("explainability", 0.0) or 0.0),
                    "latency_mean": latency_mean,
                    "latency_p95": float(metrics.get("latency_p95", 0.0) or 0.0),
                    "accuracy_latency_tradeoff": f1 / max(latency_mean, 1e-6),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary_dir, selected_run_id, experiments_root = _resolve_summary_dir(results_dir, args.run_id)

    summaries = _load_dataset_summaries(summary_dir)
    if not summaries and selected_run_id and experiments_root is not None:
        summaries = _load_dataset_summaries_from_tree(experiments_root, selected_run_id)

    if not summaries:
        raise FileNotFoundError(
            f"No benchmark summaries found in {summary_dir}."
        )

    rows = _flatten_rows(summaries)
    df = pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)

    per_dataset_best = (
        df.sort_values(["dataset", "f1", "latency_mean"], ascending=[True, False, True])
        .groupby("dataset", as_index=False)
        .first()
        .to_dict(orient="records")
    )

    comparison_payload = {
        "rows": rows,
        "best_f1_per_dataset": per_dataset_best,
        "run_id": selected_run_id,
    }

    save_json(summary_dir / "experiment_comparison.json", comparison_payload)
    df.to_csv(summary_dir / "experiment_comparison.csv", index=False)

    tradeoff_df = df[["dataset", "method", "f1", "latency_mean", "accuracy_latency_tradeoff"]]
    tradeoff_df.to_csv(summary_dir / "latency_accuracy_tradeoff.csv", index=False)

    print("Saved:")
    print(f"- {summary_dir / 'experiment_comparison.json'}")
    print(f"- {summary_dir / 'experiment_comparison.csv'}")
    print(f"- {summary_dir / 'latency_accuracy_tradeoff.csv'}")


if __name__ == "__main__":
    main()
