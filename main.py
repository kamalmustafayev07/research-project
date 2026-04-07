"""CLI entry point for Agent-Enhanced GraphRAG."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.config import SETTINGS
from src.pipeline import AgentEnhancedGraphRAG
from src.utils.data_loader import HotpotExample, load_prepared_subset, prepare_hotpotqa_subset
from src.utils.evaluation import run_method_on_dataset
from src.utils.helpers import save_json

app = typer.Typer(help="Agent-Enhanced GraphRAG CLI")
console = Console()


def _example_to_dict(example: HotpotExample) -> dict:
    return asdict(example)


@app.command()
def prepare_data(subset_size: int = typer.Option(200, help="Number of HotpotQA samples to prepare.")) -> None:
    """Download and preprocess HotpotQA subset."""
    load_dotenv()
    examples = prepare_hotpotqa_subset(subset_size=subset_size)
    console.print(f"Prepared {len(examples)} samples in data/processed.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question."),
    sample_index: int = typer.Option(0, help="Dataset sample index used for context passages."),
) -> None:
    """Run a single query through full agent pipeline."""
    load_dotenv()
    examples = load_prepared_subset()
    sample = examples[sample_index]

    pipeline = AgentEnhancedGraphRAG()
    output = pipeline.invoke(question=question, context_passages=sample.context)

    table = Table(title="Single Query Output")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Question", question)
    table.add_row("Answer", output["answer"])
    table.add_row("Confidence", f"{output['confidence']:.3f}")
    table.add_row("Retrieval Loops", str(output.get("retrieval_loops", 0)))
    table.add_row("Evidence Hops", str(len(output.get("evidence_chain", []))))
    console.print(table)


@app.command()
def evaluate(
    subset_size: int = typer.Option(200, help="Evaluation subset size."),
    limit: int = typer.Option(0, help="Optional limit for fast testing; 0 = full subset."),
) -> None:
    """Run B1, B2, and Agent-Enhanced GraphRAG on HotpotQA subset."""
    load_dotenv()
    examples = load_prepared_subset(subset_size=subset_size)
    data = [_example_to_dict(item) for item in examples]
    if limit > 0:
        data = data[:limit]

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

    results = {"B1": b1, "B2": b2, "OURS": ours}
    save_json(SETTINGS.paths.output_results / "evaluation_results.json", results)
    save_json(SETTINGS.paths.output_evidence / "ours_evidence_chains.json", ours["predictions"])

    table = Table(title="Evaluation Summary")
    table.add_column("Method")
    table.add_column("EM")
    table.add_column("F1")
    table.add_column("Explainability")
    for key, value in results.items():
        m = value["metrics"]
        table.add_row(key, f"{m['exact_match']:.3f}", f"{m['f1']:.3f}", f"{m['explainability']:.3f}")
    console.print(table)
    console.print(f"Saved results to {SETTINGS.paths.output_results}")


@app.command()
def smoke_test() -> None:
    """Run a short deterministic test on 10 samples (set LLM_BACKEND=mock for speed)."""
    load_dotenv()
    examples = load_prepared_subset(subset_size=10)
    data = [_example_to_dict(item) for item in examples]
    pipeline = AgentEnhancedGraphRAG()

    ours = run_method_on_dataset(
        "Ours_Agent_Enhanced_GraphRAG",
        data,
        lambda item: pipeline.invoke(item["question"], item["context"]),
    )
    save_json(SETTINGS.paths.output_results / "smoke_test_results.json", ours)
    console.print("Smoke test completed and saved to outputs/results/smoke_test_results.json")


if __name__ == "__main__":
    app()
