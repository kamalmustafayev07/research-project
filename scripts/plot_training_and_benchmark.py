"""Plot training and benchmark comparisons for Agent-Enhanced GraphRAG.

This script reads existing artifacts and creates figures for:
1) Training process overview (reranker training metadata + runtime config).
2) Output comparison across B1, B2, and OURS benchmarks.

Expected input files in outputs/results:
- benchmark_summary.json
- reranker_metrics.json
- benchmark_b1_predictions.json
- benchmark_b2_predictions.json
- benchmark_ours_predictions.json

Optional:
- .env (for runtime/model config annotation)
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def _f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    gold_counts: dict[str, int] = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1

    overlap = 0
    for tok, cnt in pred_counts.items():
        overlap += min(cnt, gold_counts.get(tok, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, gold: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(gold))


def _method_stats(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    em_values = []
    f1_values = []
    conf_values = []

    for row in predictions:
        pred = str(row.get("prediction", ""))
        gold = str(row.get("gold", ""))
        em_values.append(_exact_match(pred, gold))
        f1_values.append(_f1_score(pred, gold))
        conf_values.append(float(row.get("confidence", 0.0)))

    total = len(predictions)
    em_count = int(sum(em_values))
    mean_f1 = sum(f1_values) / total if total else 0.0
    mean_conf = sum(conf_values) / total if total else 0.0

    return {
        "total": total,
        "em_count": em_count,
        "mean_f1_from_preds": mean_f1,
        "mean_confidence": mean_conf,
        "confidence_values": conf_values,
    }


def _plot_training_overview(
    reranker_metrics: dict[str, Any],
    env: dict[str, str],
    out_path: Path,
) -> None:
    train_pairs = int(reranker_metrics.get("train_pairs", 0))
    positive_rate = float(reranker_metrics.get("positive_rate", 0.0))
    validation_accuracy = float(reranker_metrics.get("validation_accuracy", 0.0))

    pos_count = int(round(train_pairs * positive_rate))
    neg_count = max(0, train_pairs - pos_count)
    training_history = reranker_metrics.get("training_history", [])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Training history curve (fallback to message if history is absent)
    if training_history:
        iters = [int(item.get("iteration", 0)) for item in training_history]
        train_losses = [float(item.get("train_log_loss", 0.0)) for item in training_history]
        val_accs = [float(item.get("validation_accuracy", 0.0)) for item in training_history]

        ax_loss = axes[0]
        ax_acc = ax_loss.twinx()

        ax_loss.plot(iters, train_losses, marker="o", color="#1d3557", label="Train log loss")
        ax_acc.plot(iters, val_accs, marker="s", color="#e76f51", label="Validation accuracy")

        ax_loss.set_title("Reranker Training History")
        ax_loss.set_xlabel("Max Iteration Cap")
        ax_loss.set_ylabel("Train Log Loss", color="#1d3557")
        ax_acc.set_ylabel("Validation Accuracy", color="#e76f51")
        ax_acc.set_ylim(0, 1.0)

        line_labels = [
            (ax_loss.lines[0], "Train log loss"),
            (ax_acc.lines[0], "Validation accuracy"),
        ]
        axes[0].legend(
            [item[0] for item in line_labels],
            [item[1] for item in line_labels],
            loc="center right",
        )
    else:
        axes[0].axis("off")
        axes[0].text(
            0.02,
            0.98,
            "Training history not found in reranker_metrics.json\n"
            "Re-run training with updated code to collect\n"
            "per-iteration loss/validation points.",
            va="top",
            fontsize=10,
            family="monospace",
            transform=axes[0].transAxes,
        )

    # (2) Class balance in training pairs
    axes[1].bar(["Positive", "Negative"], [pos_count, neg_count], color=["#2a9d8f", "#e76f51"])
    axes[1].set_title("Reranker Train Pair Balance")
    axes[1].set_ylabel("Count")
    axes[1].text(0.5, max(pos_count, neg_count) * 0.9 if train_pairs else 0.0, f"Total: {train_pairs}", ha="center")

    # (3) Runtime/training config annotation
    axes[2].axis("off")
    config_lines = [
        "Runtime & Training Configuration",
        "",
        f"LLM_BACKEND={env.get('LLM_BACKEND', 'N/A')}",
        f"OLLAMA_MODEL={env.get('OLLAMA_MODEL', 'N/A')}",
        f"HF_MODEL_NAME={env.get('HF_MODEL_NAME', 'N/A')}",
        f"EMBEDDING_MODEL={env.get('EMBEDDING_MODEL', 'N/A')}",
        f"MAX_RETRIEVAL_LOOPS={env.get('MAX_RETRIEVAL_LOOPS', 'N/A')}",
        f"TEMPERATURE={env.get('TEMPERATURE', 'N/A')}",
        f"MAX_NEW_TOKENS={env.get('MAX_NEW_TOKENS', 'N/A')}",
        "",
        f"Reranker trained: {reranker_metrics.get('trained', False)}",
        f"Positive rate: {positive_rate:.3f}",
        f"Validation accuracy: {validation_accuracy:.3f}",
        f"History points: {len(training_history)}",
    ]
    axes[2].text(
        0.02,
        0.98,
        "\n".join(config_lines),
        va="top",
        fontsize=10,
        family="monospace",
        transform=axes[2].transAxes,
    )

    fig.suptitle("Training Process Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_output_comparison(
    benchmark_summary: dict[str, Any],
    b1_preds: list[dict[str, Any]],
    b2_preds: list[dict[str, Any]],
    ours_preds: list[dict[str, Any]],
    out_path: Path,
) -> None:
    methods = ["B1", "B2", "OURS"]
    metric_names = ["exact_match", "f1", "explainability"]

    # Scores from benchmark_summary.json
    metric_matrix: dict[str, list[float]] = {m: [] for m in metric_names}
    for method in methods:
        block = benchmark_summary.get(method, {})
        for metric in metric_names:
            metric_matrix[metric].append(float(block.get(metric, 0.0)))

    # Per-prediction derived stats
    stats = {
        "B1": _method_stats(b1_preds),
        "B2": _method_stats(b2_preds),
        "OURS": _method_stats(ours_preds),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) grouped bars for EM/F1/Explainability
    x = list(range(len(methods)))
    width = 0.24
    colors = {"exact_match": "#457b9d", "f1": "#1d3557", "explainability": "#e76f51"}
    for idx, metric in enumerate(metric_names):
        xs = [val + (idx - 1) * width for val in x]
        axes[0, 0].bar(xs, metric_matrix[metric], width=width, label=metric, color=colors[metric])

    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title("Benchmark Metrics")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend()

    # (2) EM correct count by method
    em_counts = [stats[m]["em_count"] for m in methods]
    totals = [stats[m]["total"] for m in methods]
    axes[0, 1].bar(methods, em_counts, color=["#8ecae6", "#ffb703", "#2a9d8f"])
    axes[0, 1].set_title("Exact Match Correct Predictions")
    axes[0, 1].set_ylabel("Count")
    for idx, (count, total) in enumerate(zip(em_counts, totals)):
        axes[0, 1].text(idx, count + max(1, int(0.01 * max(totals))), f"{count}/{total}", ha="center")

    # (3) confidence distribution
    conf_data = [stats[m]["confidence_values"] for m in methods]
    axes[1, 0].boxplot(conf_data, tick_labels=methods, showmeans=True)
    axes[1, 0].set_title("Confidence Distribution by Method")
    axes[1, 0].set_ylabel("Confidence")
    axes[1, 0].set_ylim(0, 1.0)

    # (4) Delta vs B1 for fast interpretation
    b1_em = metric_matrix["exact_match"][0]
    b1_f1 = metric_matrix["f1"][0]
    b1_exp = metric_matrix["explainability"][0]

    delta_labels = ["B2 vs B1", "OURS vs B1"]
    delta_em = [metric_matrix["exact_match"][1] - b1_em, metric_matrix["exact_match"][2] - b1_em]
    delta_f1 = [metric_matrix["f1"][1] - b1_f1, metric_matrix["f1"][2] - b1_f1]
    delta_exp = [metric_matrix["explainability"][1] - b1_exp, metric_matrix["explainability"][2] - b1_exp]

    x2 = [0, 1]
    w2 = 0.24
    axes[1, 1].bar([v - w2 for v in x2], delta_em, width=w2, label="EM", color="#457b9d")
    axes[1, 1].bar(x2, delta_f1, width=w2, label="F1", color="#1d3557")
    axes[1, 1].bar([v + w2 for v in x2], delta_exp, width=w2, label="Explainability", color="#e76f51")
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels(delta_labels)
    axes[1, 1].set_title("Metric Delta Relative to B1")
    axes[1, 1].set_ylabel("Absolute Change")
    axes[1, 1].legend()

    fig.suptitle("Benchmark Output Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training process and benchmark output comparison.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs") / "results",
        help="Directory containing benchmark/training JSON outputs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env used for the experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "figures",
        help="Directory to save generated plots.",
    )
    args = parser.parse_args()

    summary_path = args.results_dir / "benchmark_summary.json"
    reranker_path = args.results_dir / "reranker_metrics.json"
    b1_path = args.results_dir / "benchmark_b1_predictions.json"
    b2_path = args.results_dir / "benchmark_b2_predictions.json"
    ours_path = args.results_dir / "benchmark_ours_predictions.json"

    missing = [
        str(p)
        for p in [summary_path, reranker_path, b1_path, b2_path, ours_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n- " + "\n- ".join(missing)
        )

    benchmark_summary = _load_json(summary_path)
    reranker_metrics = _load_json(reranker_path)
    b1_preds = _load_json(b1_path)
    b2_preds = _load_json(b2_path)
    ours_preds = _load_json(ours_path)
    env = _load_env(args.env_file)

    training_plot_path = args.output_dir / "training_process_overview.png"
    comparison_plot_path = args.output_dir / "benchmark_output_comparison.png"

    _plot_training_overview(reranker_metrics, env, training_plot_path)
    _plot_output_comparison(benchmark_summary, b1_preds, b2_preds, ours_preds, comparison_plot_path)

    print("Saved plots:")
    print(f"- {training_plot_path}")
    print(f"- {comparison_plot_path}")


if __name__ == "__main__":
    main()
