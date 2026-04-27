"""End-to-end execution script for multi-dataset benchmark runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Load environment variables before importing modules that initialize SETTINGS.
load_dotenv()

from src.config import SETTINGS
from src.pipeline import AgentEnhancedGraphRAG
from src.utils.data_loader import (
    load_prepared_disjoint_split,
    load_prepared_subset,
    prepare_dataset_disjoint_splits,
    prepare_dataset_subset,
)
from src.utils.evaluation import (
    analyze_evidence_chains,
    run_method_on_dataset,
    save_latency_records_csv,
    save_predictions_csv,
)
from src.utils.experiment_paths import DatasetRunPaths, ExperimentPaths, create_experiment_paths
from src.utils.helpers import save_json, setup_logger
from src.utils.plotting import generate_benchmark_plots, generate_reranker_plots
from app import run_pipeline as streamlit_run_pipeline


def _resolve_datasets(value: str) -> list[str]:
    aliases = {
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "musique": "musique",
        "2wiki": "2wikimultihopqa",
        "2wikimultihop": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
    }
    requested = value.strip().lower()
    if requested == "all":
        return ["hotpotqa", "musique", "2wikimultihopqa"]

    datasets: list[str] = []
    for part in requested.split(","):
        key = aliases.get(part.strip().lower())
        if key is None:
            raise ValueError(f"Unsupported dataset '{part.strip()}'.")
        if key not in datasets:
            datasets.append(key)
    if not datasets:
        raise ValueError("No valid datasets provided.")
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent-Enhanced GraphRAG benchmark across datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=SETTINGS.data.dataset,
        help="Dataset selection: hotpotqa, musique, 2wikimultihopqa, comma-separated list, or 'all'.",
    )
    parser.add_argument("--subset-size", type=int, default=SETTINGS.data.dataset_subset_size)
    parser.add_argument("--split", type=str, default=SETTINGS.data.dataset_split)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for quick run.")
    parser.add_argument("--prepare-data", action="store_true", help="Download and preprocess dataset first.")
    parser.add_argument(
        "--use-disjoint-splits",
        action="store_true",
        help="Create/load train-validation-test splits and evaluate on held-out test split.",
    )
    parser.add_argument("--test-size", type=int, default=200, help="Held-out test split size.")
    parser.add_argument("--val-size", type=int, default=1000, help="Validation split size.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=0,
        help="Optional train split size; 0 uses all remaining examples.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="",
        help="Source split for deterministic disjoint partitioning (defaults to --split).",
    )
    parser.add_argument(
        "--train-reranker",
        action="store_true",
        help="Force retraining of passage reranker when disjoint split mode is enabled.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional label for this run (e.g. baseline_1 or ablation_retriever_weight_0_3).",
    )
    return parser.parse_args()


def _summary_rows(dataset: str, b1: dict[str, Any], b2: dict[str, Any], ours: dict[str, Any]) -> list[dict[str, Any]]:
    methods = {"B1": b1, "B2": b2, "OURS": ours}
    rows: list[dict[str, Any]] = []
    for method_key, result in methods.items():
        metrics = result["metrics"]
        rows.append(
            {
                "dataset": dataset,
                "method": method_key,
                "exact_match": metrics.get("exact_match", 0.0),
                "f1": metrics.get("f1", 0.0),
                "explainability": metrics.get("explainability", 0.0),
                "latency_mean": metrics.get("latency_mean", 0.0),
                "latency_p95": metrics.get("latency_p95", 0.0),
            }
        )
    return rows


def _save_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_run_manifest(args: argparse.Namespace, datasets: list[str], experiment_paths: ExperimentPaths) -> dict[str, Any]:
    dataset_layout = {
        dataset: {
            "root": str(experiment_paths.for_dataset(dataset).root),
            "metadata": str(experiment_paths.for_dataset(dataset).metadata_dir),
            "results": str(experiment_paths.for_dataset(dataset).results_dir),
            "predictions": str(experiment_paths.for_dataset(dataset).predictions_dir),
            "latency": str(experiment_paths.for_dataset(dataset).latency_dir),
            "evidence": str(experiment_paths.for_dataset(dataset).evidence_dir),
            "plots": str(experiment_paths.for_dataset(dataset).plots_dir),
            "logs": str(experiment_paths.for_dataset(dataset).logs_dir),
        }
        for dataset in datasets
    }

    return {
        "run_id": experiment_paths.run_id,
        "created_at_utc": experiment_paths.created_at_utc,
        "datasets": datasets,
        "args": vars(args),
        "settings": _json_safe(SETTINGS),
        "output_layout": {
            "run_root": str(experiment_paths.run_root),
            "run_metadata": str(experiment_paths.metadata_dir),
            "run_logs": str(experiment_paths.logs_dir),
            "run_summaries": str(experiment_paths.summaries_dir),
            "datasets": dataset_layout,
        },
    }


def _evaluate_dataset(
    dataset: str,
    args: argparse.Namespace,
    pipeline: AgentEnhancedGraphRAG,
    logger: Any,
    dataset_paths: DatasetRunPaths,
) -> dict[str, Any]:
    split = args.split
    source_split = args.source_split or split

    if args.prepare_data:
        if args.use_disjoint_splits:
            logger.info(
                "Preparing disjoint splits | dataset=%s source_split=%s test=%s val=%s train=%s",
                dataset,
                source_split,
                args.test_size,
                args.val_size,
                "all_remaining" if args.train_size <= 0 else args.train_size,
            )
            prepare_dataset_disjoint_splits(
                dataset=dataset,
                test_size=args.test_size,
                val_size=args.val_size,
                train_size=(None if args.train_size <= 0 else args.train_size),
                source_split=source_split,
                save=True,
            )
        else:
            logger.info(
                "Preparing dataset subset | dataset=%s split=%s subset=%s",
                dataset,
                split,
                args.subset_size,
            )
            prepare_dataset_subset(
                dataset=dataset,
                subset_size=args.subset_size,
                save=True,
                dataset_split=split,
            )

    if args.use_disjoint_splits:
        examples = load_prepared_disjoint_split(
            "test",
            test_size=args.test_size,
            val_size=args.val_size,
            train_size=(None if args.train_size <= 0 else args.train_size),
            source_split=source_split,
            dataset=dataset,
        )
    else:
        examples = load_prepared_subset(
            subset_size=args.subset_size,
            dataset_split=split,
            dataset=dataset,
        )

    data = [asdict(example) for example in examples]
    if args.limit > 0:
        data = data[: args.limit]

    logger.info("Loaded %s examples | dataset=%s", len(data), dataset)
    save_json(
        dataset_paths.metadata_dir / "dataset_selection.json",
        {
            "dataset": dataset,
            "num_examples": len(data),
            "split": split,
            "source_split": source_split,
            "subset_size": args.subset_size,
            "limit": args.limit,
            "use_disjoint_splits": args.use_disjoint_splits,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "train_size": args.train_size,
        },
    )

    if args.use_disjoint_splits and SETTINGS.retrieval.use_reranker:
        should_train_reranker = args.train_reranker or not pipeline.has_trained_reranker()
        if should_train_reranker:
            logger.info("Training passage reranker | dataset=%s", dataset)
            train_examples = load_prepared_disjoint_split(
                "train",
                test_size=args.test_size,
                val_size=args.val_size,
                train_size=(None if args.train_size <= 0 else args.train_size),
                source_split=source_split,
                dataset=dataset,
            )
            validation_examples = load_prepared_disjoint_split(
                "validation",
                test_size=args.test_size,
                val_size=args.val_size,
                train_size=(None if args.train_size <= 0 else args.train_size),
                source_split=source_split,
                dataset=dataset,
            )
            reranker_metrics = pipeline.train_retriever_reranker(
                train_examples=[asdict(example) for example in train_examples],
                validation_examples=[asdict(example) for example in validation_examples],
                test_examples=[asdict(example) for example in examples],
            )
            plot_paths: dict[str, str] = {}
            try:
                plot_paths = generate_reranker_plots(reranker_metrics, dataset_paths.plots_dir / "reranker")
            except Exception as exc:
                logger.warning("Could not generate reranker plots: %s", exc)
            if plot_paths:
                reranker_metrics["plots"] = plot_paths
            save_json(dataset_paths.results_dir / "reranker_metrics.json", reranker_metrics)

    def _ours_infer(item: dict[str, Any]) -> dict[str, Any]:
        """Run the same inference logic as Streamlit and normalize keys.

        Streamlit's ``app.run_pipeline`` returns UI-centric keys:
        - ``chain`` / ``selected`` / ``latency`` / ``total``
        Evaluation expects:
        - ``evidence_chain`` / ``latency_breakdown`` / ``latency_total``
        """
        output = streamlit_run_pipeline(
            pipeline=pipeline,
            question=item["question"],
            contexts=item["contexts"],
        )

        normalized = dict(output)
        if "evidence_chain" not in normalized:
            normalized["evidence_chain"] = list(normalized.get("chain") or [])

        if "latency_total" not in normalized:
            normalized["latency_total"] = float(normalized.get("total") or 0.0)

        if "latency_breakdown" not in normalized:
            lat = normalized.get("latency") if isinstance(normalized.get("latency"), dict) else {}
            normalized["latency_breakdown"] = {
                "decomposer": float(lat.get("decomposer", 0.0)),
                # Streamlit names this bucket retriever+rereanker.
                "retriever": float(lat.get("retriever+rereanker", lat.get("retriever", 0.0))),
                "reasoner": float(lat.get("reasoner", 0.0)),
                "critic": float(lat.get("critic", 0.0)),
            }

        return normalized

    b1 = run_method_on_dataset(
        "B1_Standard_Dense_RAG",
        data,
        lambda item: pipeline.dense_rag_baseline(item["question"], item["contexts"]),
        dataset_name=dataset,
    )
    b2 = run_method_on_dataset(
        "B2_Basic_GraphRAG",
        data,
        lambda item: pipeline.basic_graphrag_baseline(item["question"], item["contexts"]),
        dataset_name=dataset,
    )
    ours = run_method_on_dataset(
        "Ours_Agent_Enhanced_GraphRAG",
        data,
        _ours_infer,
        dataset_name=dataset,
    )

    summary = {"B1": b1["metrics"], "B2": b2["metrics"], "OURS": ours["metrics"]}
    evidence_analysis = analyze_evidence_chains(ours["predictions"])

    benchmark_plot_paths: dict[str, str] = {}
    try:
        benchmark_plot_paths = generate_benchmark_plots(
            dataset=dataset,
            summary=summary,
            methods={"B1": b1, "B2": b2, "OURS": ours},
            output_dir=dataset_paths.plots_dir,
        )
    except Exception as exc:
        logger.warning("Could not generate benchmark plots | dataset=%s err=%s", dataset, exc)

    save_json(dataset_paths.results_dir / "benchmark_summary.json", summary)
    save_json(dataset_paths.predictions_dir / "benchmark_b1_predictions.json", b1["predictions"])
    save_json(dataset_paths.predictions_dir / "benchmark_b2_predictions.json", b2["predictions"])
    save_json(dataset_paths.predictions_dir / "benchmark_ours_predictions.json", ours["predictions"])
    save_json(dataset_paths.results_dir / "evidence_chain_analysis.json", evidence_analysis)
    save_json(dataset_paths.evidence_dir / "benchmark_ours_evidence_chains.json", ours["predictions"])

    save_predictions_csv(
        dataset_paths.predictions_dir / "benchmark_predictions_b1.csv",
        b1["predictions"],
    )
    save_predictions_csv(
        dataset_paths.predictions_dir / "benchmark_predictions_b2.csv",
        b2["predictions"],
    )
    save_predictions_csv(
        dataset_paths.predictions_dir / "benchmark_predictions_ours.csv",
        ours["predictions"],
    )

    save_json(dataset_paths.latency_dir / "latency_records_b1.json", b1["latency_records"])
    save_json(dataset_paths.latency_dir / "latency_records_b2.json", b2["latency_records"])
    save_json(dataset_paths.latency_dir / "latency_records_ours.json", ours["latency_records"])
    save_latency_records_csv(
        dataset_paths.latency_dir / "latency_records_b1.csv",
        b1["latency_records"],
    )
    save_latency_records_csv(
        dataset_paths.latency_dir / "latency_records_b2.csv",
        b2["latency_records"],
    )
    save_latency_records_csv(
        dataset_paths.latency_dir / "latency_records_ours.csv",
        ours["latency_records"],
    )

    summary_rows = _summary_rows(dataset=dataset, b1=b1, b2=b2, ours=ours)
    _save_summary_csv(dataset_paths.results_dir / "benchmark_summary.csv", summary_rows)

    save_json(
        dataset_paths.metadata_dir / "dataset_run_artifacts.json",
        {
            "dataset": dataset,
            "run_id": dataset_paths.run_id,
            "plots": benchmark_plot_paths,
            "result_files": {
                "summary_json": str(dataset_paths.results_dir / "benchmark_summary.json"),
                "summary_csv": str(dataset_paths.results_dir / "benchmark_summary.csv"),
                "predictions": {
                    "b1_json": str(dataset_paths.predictions_dir / "benchmark_b1_predictions.json"),
                    "b2_json": str(dataset_paths.predictions_dir / "benchmark_b2_predictions.json"),
                    "ours_json": str(dataset_paths.predictions_dir / "benchmark_ours_predictions.json"),
                    "b1_csv": str(dataset_paths.predictions_dir / "benchmark_predictions_b1.csv"),
                    "b2_csv": str(dataset_paths.predictions_dir / "benchmark_predictions_b2.csv"),
                    "ours_csv": str(dataset_paths.predictions_dir / "benchmark_predictions_ours.csv"),
                },
                "latency": {
                    "b1_json": str(dataset_paths.latency_dir / "latency_records_b1.json"),
                    "b2_json": str(dataset_paths.latency_dir / "latency_records_b2.json"),
                    "ours_json": str(dataset_paths.latency_dir / "latency_records_ours.json"),
                    "b1_csv": str(dataset_paths.latency_dir / "latency_records_b1.csv"),
                    "b2_csv": str(dataset_paths.latency_dir / "latency_records_b2.csv"),
                    "ours_csv": str(dataset_paths.latency_dir / "latency_records_ours.csv"),
                },
                "evidence_analysis_json": str(dataset_paths.results_dir / "evidence_chain_analysis.json"),
                "evidence_chains_json": str(dataset_paths.evidence_dir / "benchmark_ours_evidence_chains.json"),
            },
        },
    )

    logger.info("Benchmark complete | dataset=%s summary=%s", dataset, summary)
    logger.info("Evidence chain diagnostics | dataset=%s diagnostics=%s", dataset, evidence_analysis)

    return {
        "dataset": dataset,
        "summary": summary,
        "summary_rows": summary_rows,
        "methods": {"B1": b1, "B2": b2, "OURS": ours},
        "evidence_analysis": evidence_analysis,
    }


def main() -> None:
    args = parse_args()
    datasets = _resolve_datasets(args.datasets)
    run_name = args.run_name.strip() or None
    experiment_paths = create_experiment_paths(
        base_output_dir=SETTINGS.paths.output_experiments,
        datasets=datasets,
        run_name=run_name,
    )

    logger = setup_logger("run_pipeline", experiment_paths.logs_dir / "pipeline.log")
    logger.info("Datasets selected: %s", datasets)
    logger.info("Run initialized | run_id=%s run_root=%s", experiment_paths.run_id, experiment_paths.run_root)

    run_manifest = _build_run_manifest(args=args, datasets=datasets, experiment_paths=experiment_paths)
    save_json(experiment_paths.run_manifest_path, run_manifest)

    pipeline = AgentEnhancedGraphRAG()
    all_results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        result = _evaluate_dataset(
            dataset=dataset,
            args=args,
            pipeline=pipeline,
            logger=logger,
            dataset_paths=experiment_paths.for_dataset(dataset),
        )
        all_results.append(result)
        all_rows.extend(result["summary_rows"])

    cross_dataset_summary = {item["dataset"]: item["summary"] for item in all_results}
    save_json(experiment_paths.summaries_dir / "benchmark_summary_all_datasets.json", cross_dataset_summary)
    _save_summary_csv(experiment_paths.summaries_dir / "benchmark_summary_all_datasets.csv", all_rows)

    latency_comparison = {
        dataset: {
            "B1": summary.get("B1", {}).get("latency_mean", 0.0),
            "B2": summary.get("B2", {}).get("latency_mean", 0.0),
            "OURS": summary.get("OURS", {}).get("latency_mean", 0.0),
            "B1_p95": summary.get("B1", {}).get("latency_p95", 0.0),
            "B2_p95": summary.get("B2", {}).get("latency_p95", 0.0),
            "OURS_p95": summary.get("OURS", {}).get("latency_p95", 0.0),
        }
        for dataset, summary in cross_dataset_summary.items()
    }
    save_json(experiment_paths.summaries_dir / "latency_comparison_all_datasets.json", latency_comparison)

    save_json(
        experiment_paths.metadata_dir / "run_completion.json",
        {
            "run_id": experiment_paths.run_id,
            "datasets": datasets,
            "summary_files": {
                "benchmark_summary_all_datasets_json": str(
                    experiment_paths.summaries_dir / "benchmark_summary_all_datasets.json"
                ),
                "benchmark_summary_all_datasets_csv": str(
                    experiment_paths.summaries_dir / "benchmark_summary_all_datasets.csv"
                ),
                "latency_comparison_all_datasets_json": str(
                    experiment_paths.summaries_dir / "latency_comparison_all_datasets.json"
                ),
            },
            "dataset_roots": {
                dataset: str(experiment_paths.for_dataset(dataset).root)
                for dataset in datasets
            },
        },
    )

    logger.info("All dataset runs complete. Outputs saved under: %s", experiment_paths.run_root)


if __name__ == "__main__":
    main()
