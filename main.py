"""CLI entry point for Agent-Enhanced GraphRAG."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load environment variables before importing modules that initialize SETTINGS.
load_dotenv()

from src.config import SETTINGS
from src.pipeline import AgentEnhancedGraphRAG
from src.utils.data_loader import MultiHopQAExample, load_prepared_subset, prepare_dataset_subset
from src.utils.evaluation import run_method_on_dataset, save_predictions_csv
from src.utils.experiment_paths import create_experiment_paths
from src.utils.helpers import save_json
from src.utils.plotting import generate_benchmark_plots

app = typer.Typer(help="Agent-Enhanced GraphRAG CLI")
console = Console()


def _resolve_datasets(value: str) -> list[str]:
    aliases = {
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "musique": "musique",
        "2wiki": "2wikimultihopqa",
        "2wikimultihop": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
    }
    selected = value.strip().lower()
    if selected == "all":
        return ["hotpotqa", "musique", "2wikimultihopqa"]

    datasets: list[str] = []
    for part in selected.split(","):
        key = aliases.get(part.strip().lower())
        if key is None:
            raise typer.BadParameter(f"Unsupported dataset: {part.strip()}")
        if key not in datasets:
            datasets.append(key)
    return datasets


def _example_to_dict(example: MultiHopQAExample) -> dict[str, Any]:
    return asdict(example)


@app.command("prepare-data")
def prepare_data(
    dataset: str = typer.Option(SETTINGS.data.dataset, help="Dataset: hotpotqa | musique | 2wikimultihopqa"),
    subset_size: int = typer.Option(SETTINGS.data.dataset_subset_size, help="Number of samples to prepare."),
    split: str = typer.Option(SETTINGS.data.dataset_split, help="Dataset split."),
) -> None:
    """Download and preprocess a dataset subset."""
    examples = prepare_dataset_subset(dataset=dataset, subset_size=subset_size, dataset_split=split)
    console.print(f"Prepared {len(examples)} samples for '{dataset}' in data/processed.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question."),
    dataset: str = typer.Option(SETTINGS.data.dataset, help="Dataset used to source context passages."),
    split: str = typer.Option(SETTINGS.data.dataset_split, help="Dataset split."),
    sample_index: int = typer.Option(0, help="Dataset sample index used for context passages."),
) -> None:
    """Run a single query through full agent pipeline."""
    examples = load_prepared_subset(dataset=dataset, dataset_split=split)
    sample = examples[sample_index]

    pipeline = AgentEnhancedGraphRAG()
    output = pipeline.invoke(question=question, context_passages=sample.contexts)

    table = Table(title="Single Query Output")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Dataset", dataset)
    table.add_row("Question", question)
    table.add_row("Answer", output["answer"])
    table.add_row("Confidence", f"{output['confidence']:.3f}")
    table.add_row("Retrieval Loops", str(output.get("retrieval_loops", 0)))
    table.add_row("Evidence Hops", str(len(output.get("evidence_chain", []))))
    table.add_row("Latency Total (s)", f"{output.get('latency_total', 0.0):.4f}")
    breakdown = output.get("latency_breakdown", {})
    table.add_row(
        "Latency Breakdown (s)",
        " | ".join(
            [
                f"decomposer={float(breakdown.get('decomposer', 0.0)):.4f}",
                f"retriever={float(breakdown.get('retriever', 0.0)):.4f}",
                f"reasoner={float(breakdown.get('reasoner', 0.0)):.4f}",
                f"critic={float(breakdown.get('critic', 0.0)):.4f}",
            ]
        ),
    )
    console.print(table)


@app.command()
def evaluate(
    datasets: str = typer.Option(SETTINGS.data.dataset, help="Dataset key, comma-separated list, or 'all'."),
    subset_size: int = typer.Option(SETTINGS.data.dataset_subset_size, help="Evaluation subset size."),
    split: str = typer.Option(SETTINGS.data.dataset_split, help="Dataset split."),
    limit: int = typer.Option(0, help="Optional limit for fast testing; 0 = full subset."),
    run_name: str = typer.Option("", help="Optional run label used in output folder naming."),
) -> None:
    """Run B1, B2, and Agent-Enhanced GraphRAG across one or multiple datasets."""
    dataset_list = _resolve_datasets(datasets)
    experiment_paths = create_experiment_paths(
        base_output_dir=SETTINGS.paths.output_experiments,
        datasets=dataset_list,
        run_name=(run_name.strip() or "cli_evaluate"),
    )
    pipeline = AgentEnhancedGraphRAG()

    all_results: dict[str, Any] = {}
    summary_table = Table(title="Evaluation Summary")
    summary_table.add_column("Dataset")
    summary_table.add_column("Method")
    summary_table.add_column("EM")
    summary_table.add_column("F1")
    summary_table.add_column("Explainability")
    summary_table.add_column("Latency Mean (s)")
    summary_table.add_column("Latency P95 (s)")

    for dataset in dataset_list:
        dataset_paths = experiment_paths.for_dataset(dataset)
        examples = load_prepared_subset(subset_size=subset_size, dataset_split=split, dataset=dataset)
        data = [_example_to_dict(item) for item in examples]
        if limit > 0:
            data = data[:limit]

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
            lambda item: pipeline.invoke(item["question"], item["contexts"]),
            dataset_name=dataset,
        )

        all_results[dataset] = {"B1": b1, "B2": b2, "OURS": ours}

        save_json(dataset_paths.results_dir / "evaluation_results.json", all_results[dataset])
        save_json(dataset_paths.evidence_dir / "ours_evidence_chains.json", ours["predictions"])
        save_predictions_csv(dataset_paths.predictions_dir / "evaluation_b1.csv", b1["predictions"])
        save_predictions_csv(dataset_paths.predictions_dir / "evaluation_b2.csv", b2["predictions"])
        save_predictions_csv(dataset_paths.predictions_dir / "evaluation_ours.csv", ours["predictions"])

        summary = {"B1": b1["metrics"], "B2": b2["metrics"], "OURS": ours["metrics"]}
        save_json(dataset_paths.results_dir / "evaluation_summary.json", summary)
        plot_paths = generate_benchmark_plots(
            dataset=dataset,
            summary=summary,
            methods={"B1": b1, "B2": b2, "OURS": ours},
            output_dir=dataset_paths.plots_dir,
        )
        save_json(dataset_paths.metadata_dir / "evaluation_artifacts.json", {"plots": plot_paths})

        for method_name, result in [("B1", b1), ("B2", b2), ("OURS", ours)]:
            metrics = result["metrics"]
            summary_table.add_row(
                dataset,
                method_name,
                f"{metrics['exact_match']:.3f}",
                f"{metrics['f1']:.3f}",
                f"{metrics['explainability']:.3f}",
                f"{metrics.get('latency_mean', 0.0):.3f}",
                f"{metrics.get('latency_p95', 0.0):.3f}",
            )

    save_json(experiment_paths.summaries_dir / "evaluation_results_all_datasets.json", all_results)
    console.print(summary_table)
    console.print(f"Saved results to {experiment_paths.run_root}")


@app.command("smoke-test")
def smoke_test(
    dataset: str = typer.Option(SETTINGS.data.dataset, help="Dataset for smoke run."),
    run_name: str = typer.Option("", help="Optional run label used in output folder naming."),
) -> None:
    """Run a short deterministic test on 10 samples (set LLM_BACKEND=mock for speed)."""
    examples = load_prepared_subset(subset_size=10, dataset=dataset)
    data = [_example_to_dict(item) for item in examples]
    pipeline = AgentEnhancedGraphRAG()

    ours = run_method_on_dataset(
        "Ours_Agent_Enhanced_GraphRAG",
        data,
        lambda item: pipeline.invoke(item["question"], item["contexts"]),
        dataset_name=dataset,
    )
    experiment_paths = create_experiment_paths(
        base_output_dir=SETTINGS.paths.output_experiments,
        datasets=[dataset],
        run_name=(run_name.strip() or "smoke_test"),
    )
    dataset_paths = experiment_paths.for_dataset(dataset)
    save_json(dataset_paths.results_dir / "smoke_test_results.json", ours)
    console.print(f"Smoke test completed and saved to {dataset_paths.results_dir / 'smoke_test_results.json'}")


if __name__ == "__main__":
    app()
