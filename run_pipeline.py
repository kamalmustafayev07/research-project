"""End-to-end execution script for full benchmark run."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from dotenv import load_dotenv

# Load environment variables before importing modules that initialize SETTINGS.
load_dotenv()

from src.config import SETTINGS
from src.pipeline import AgentEnhancedGraphRAG
from src.utils.data_loader import (
    load_prepared_disjoint_split,
    load_prepared_subset,
    prepare_hotpotqa_disjoint_splits,
    prepare_hotpotqa_subset,
)
from src.utils.evaluation import analyze_evidence_chains, run_method_on_dataset
from src.utils.helpers import save_json, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Agent-Enhanced GraphRAG benchmark.")
    parser.add_argument("--subset-size", type=int, default=200)
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
        default="validation",
        help="Hotpot source split used for deterministic disjoint partitioning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("run_pipeline", SETTINGS.paths.output_logs / "pipeline.log")

    if args.prepare_data:
        if args.use_disjoint_splits:
            logger.info(
                "Preparing disjoint HotpotQA splits from %s (test=%s, val=%s, train=%s)",
                args.source_split,
                args.test_size,
                args.val_size,
                "all_remaining" if args.train_size <= 0 else args.train_size,
            )
            prepare_hotpotqa_disjoint_splits(
                test_size=args.test_size,
                val_size=args.val_size,
                train_size=(None if args.train_size <= 0 else args.train_size),
                source_split=args.source_split,
                save=True,
            )
        else:
            logger.info("Preparing HotpotQA subset: %s", args.subset_size)
            prepare_hotpotqa_subset(subset_size=args.subset_size, save=True)

    if args.use_disjoint_splits:
        examples = load_prepared_disjoint_split(
            "test",
            test_size=args.test_size,
            val_size=args.val_size,
            train_size=(None if args.train_size <= 0 else args.train_size),
            source_split=args.source_split,
        )
    else:
        examples = load_prepared_subset(subset_size=args.subset_size)

    data = [asdict(example) for example in examples]
    if args.limit > 0:
        data = data[: args.limit]

    logger.info("Loaded %s examples", len(data))
    pipeline = AgentEnhancedGraphRAG()

    b1 = run_method_on_dataset(
        "B1_Standard_Dense_RAG",
        data,
        lambda item: pipeline.dense_rag_baseline(item["question"], item["context"]),
    )
    b2 = run_method_on_dataset(
        "B2_Basic_GraphRAG",
        data,
        lambda item: pipeline.basic_graphrag_baseline(item["question"], item["context"]),
    )
    ours = run_method_on_dataset(
        "Ours_Agent_Enhanced_GraphRAG",
        data,
        lambda item: pipeline.invoke(item["question"], item["context"]),
    )

    summary = {"B1": b1["metrics"], "B2": b2["metrics"], "OURS": ours["metrics"]}
    evidence_analysis = analyze_evidence_chains(ours["predictions"])
    save_json(SETTINGS.paths.output_results / "benchmark_summary.json", summary)
    save_json(SETTINGS.paths.output_results / "benchmark_b1_predictions.json", b1["predictions"])
    save_json(SETTINGS.paths.output_results / "benchmark_b2_predictions.json", b2["predictions"])
    save_json(SETTINGS.paths.output_results / "benchmark_ours_predictions.json", ours["predictions"])
    save_json(SETTINGS.paths.output_results / "evidence_chain_analysis.json", evidence_analysis)
    save_json(SETTINGS.paths.output_evidence / "benchmark_ours_evidence_chains.json", ours["predictions"])

    logger.info("Benchmark complete. Summary: %s", summary)
    logger.info("Evidence chain diagnostics: %s", evidence_analysis)


if __name__ == "__main__":
    main()
