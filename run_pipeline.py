"""End-to-end execution script for full benchmark run."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from dotenv import load_dotenv

from src.config import SETTINGS
from src.pipeline import AgentEnhancedGraphRAG
from src.utils.data_loader import load_prepared_subset, prepare_hotpotqa_subset
from src.utils.evaluation import run_method_on_dataset
from src.utils.helpers import save_json, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Agent-Enhanced GraphRAG benchmark.")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for quick run.")
    parser.add_argument("--prepare-data", action="store_true", help="Download and preprocess dataset first.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    logger = setup_logger("run_pipeline", SETTINGS.paths.output_logs / "pipeline.log")

    if args.prepare_data:
        logger.info("Preparing HotpotQA subset: %s", args.subset_size)
        prepare_hotpotqa_subset(subset_size=args.subset_size, save=True)

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
    save_json(SETTINGS.paths.output_results / "benchmark_summary.json", summary)
    save_json(SETTINGS.paths.output_results / "benchmark_b1_predictions.json", b1["predictions"])
    save_json(SETTINGS.paths.output_results / "benchmark_b2_predictions.json", b2["predictions"])
    save_json(SETTINGS.paths.output_results / "benchmark_ours_predictions.json", ours["predictions"])
    save_json(SETTINGS.paths.output_evidence / "benchmark_ours_evidence_chains.json", ours["predictions"])

    logger.info("Benchmark complete. Summary: %s", summary)


if __name__ == "__main__":
    main()
