"""Plotting utilities for model training diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.hotpotqa_eval import exact_match_score


def _get_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_reranker_training_history(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    """Plot train/validation loss and accuracy across epochs."""
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        return None

    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    train_acc: list[float] = []
    val_acc: list[float] = []

    for item in history:
        if not isinstance(item, dict):
            continue
        epoch = item.get("epoch")
        tr_loss = item.get("train_loss")
        tr_acc = item.get("train_accuracy")
        va_loss = item.get("validation_loss")
        va_acc = item.get("validation_accuracy")

        if epoch is None or tr_loss is None or tr_acc is None:
            continue

        epochs.append(int(epoch))
        train_loss.append(float(tr_loss))
        train_acc.append(float(tr_acc))
        val_loss.append(float(va_loss) if va_loss is not None else float("nan"))
        val_acc.append(float(va_acc) if va_acc is not None else float("nan"))

    if not epochs:
        return None

    plt = _get_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, train_loss, marker="o", linewidth=1.8, label="Train")
    axes[0].plot(epochs, val_loss, marker="o", linewidth=1.8, label="Validation")
    axes[0].set_title("Reranker Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(epochs, train_acc, marker="o", linewidth=1.8, label="Train")
    axes[1].plot(epochs, val_acc, marker="o", linewidth=1.8, label="Validation")
    axes[1].set_title("Reranker Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    plot_path = output_dir / "reranker_training_history.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def plot_reranker_test_confusion_matrix(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    """Plot confusion matrix for held-out test split."""
    matrix = metrics.get("test_confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 2:
        return None
    if not all(isinstance(row, list) and len(row) == 2 for row in matrix):
        return None

    cm = [[int(matrix[0][0]), int(matrix[0][1])], [int(matrix[1][0]), int(matrix[1][1])]]
    labels = metrics.get("confusion_matrix_labels", ["negative", "positive"])
    if not isinstance(labels, list) or len(labels) != 2:
        labels = ["negative", "positive"]

    plt = _get_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([str(labels[0]), str(labels[1])])
    ax.set_yticklabels([str(labels[0]), str(labels[1])])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Reranker Test Confusion Matrix")

    max_val = max(max(row) for row in cm) if cm else 0
    threshold = max_val / 2.0
    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=11)

    fig.tight_layout()
    plot_path = output_dir / "reranker_test_confusion_matrix.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def generate_reranker_plots(metrics: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Generate all reranker plots and return saved file paths."""
    output: dict[str, str] = {}

    history_path = plot_reranker_training_history(metrics, output_dir)
    if history_path is not None:
        output["training_history_plot"] = str(history_path)

    cm_path = plot_reranker_test_confusion_matrix(metrics, output_dir)
    if cm_path is not None:
        output["test_confusion_matrix_plot"] = str(cm_path)

    return output


def _method_labels() -> list[str]:
    return ["B1", "B2", "OURS"]


def plot_benchmark_metrics(summary: dict[str, Any], output_dir: Path, dataset: str) -> Path | None:
    """Plot EM/F1/Explainability bars per method for a dataset."""
    methods = [name for name in _method_labels() if isinstance(summary.get(name), dict)]
    if not methods:
        return None

    metric_names = ["exact_match", "f1", "explainability"]
    metric_labels = ["Exact Match", "F1", "Explainability"]

    values = {
        method: [float(summary[method].get(metric, 0.0) or 0.0) for metric in metric_names]
        for method in methods
    }

    plt = _get_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    x_pos = list(range(len(metric_names)))
    width = 0.22 if len(methods) >= 3 else 0.3
    offsets = [
        (idx - (len(methods) - 1) / 2.0) * width
        for idx in range(len(methods))
    ]

    for idx, method in enumerate(methods):
        xs = [x + offsets[idx] for x in x_pos]
        ax.bar(xs, values[method], width=width, label=method, alpha=0.88)

    ax.set_title(f"Benchmark Metrics by Method ({dataset})")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    plot_path = output_dir / f"benchmark_metrics_{dataset}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def plot_benchmark_latency(summary: dict[str, Any], output_dir: Path, dataset: str) -> Path | None:
    """Plot mean and P95 latency bars per method for a dataset."""
    methods = [name for name in _method_labels() if isinstance(summary.get(name), dict)]
    if not methods:
        return None

    means = [float(summary[method].get("latency_mean", 0.0) or 0.0) for method in methods]
    p95 = [float(summary[method].get("latency_p95", 0.0) or 0.0) for method in methods]

    plt = _get_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    x_pos = list(range(len(methods)))
    width = 0.36
    xs_mean = [x - width / 2.0 for x in x_pos]
    xs_p95 = [x + width / 2.0 for x in x_pos]

    ax.bar(xs_mean, means, width=width, label="Mean", alpha=0.88)
    ax.bar(xs_p95, p95, width=width, label="P95", alpha=0.88)

    ax.set_title(f"Latency by Method ({dataset})")
    ax.set_xlabel("Method")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    plot_path = output_dir / f"benchmark_latency_{dataset}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def plot_confidence_confusion_matrix(
    predictions: list[dict[str, Any]],
    output_dir: Path,
    dataset: str,
    method: str,
    confidence_threshold: float = 0.5,
) -> Path | None:
    """Plot confidence-vs-correctness confusion matrix for one method."""
    if not predictions:
        return None

    tp = fp = fn = tn = 0
    for row in predictions:
        prediction = str(row.get("prediction", "") or "")
        gold = str(row.get("gold", "") or "")
        confidence = float(row.get("confidence", 0.0) or 0.0)

        actual_correct = exact_match_score(prediction, gold) > 0.0
        predicted_correct = confidence >= confidence_threshold

        if actual_correct and predicted_correct:
            tp += 1
        elif (not actual_correct) and predicted_correct:
            fp += 1
        elif actual_correct and (not predicted_correct):
            fn += 1
        else:
            tn += 1

    cm = [[tn, fp], [fn, tp]]
    labels = ["Pred Incorrect", "Pred Correct"]
    y_labels = ["Actual Incorrect", "Actual Correct"]

    plt = _get_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(f"Predicted (confidence >= {confidence_threshold:.2f})")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confidence Confusion ({dataset}, {method})")

    max_val = max(max(row) for row in cm) if cm else 0
    threshold = max_val / 2.0
    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=11)

    fig.tight_layout()
    method_slug = method.strip().lower().replace(" ", "_")
    plot_path = output_dir / f"confidence_confusion_{dataset}_{method_slug}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def generate_benchmark_plots(
    dataset: str,
    summary: dict[str, Any],
    methods: dict[str, dict[str, Any]],
    output_dir: Path,
    confidence_threshold: float = 0.5,
) -> dict[str, str]:
    """Generate benchmark plots for one dataset and return file paths."""
    output: dict[str, str] = {}

    metrics_path = plot_benchmark_metrics(summary=summary, output_dir=output_dir, dataset=dataset)
    if metrics_path is not None:
        output["benchmark_metrics_plot"] = str(metrics_path)

    latency_path = plot_benchmark_latency(summary=summary, output_dir=output_dir, dataset=dataset)
    if latency_path is not None:
        output["benchmark_latency_plot"] = str(latency_path)

    for method_name in _method_labels():
        payload = methods.get(method_name, {})
        predictions = payload.get("predictions", []) if isinstance(payload, dict) else []
        if not isinstance(predictions, list):
            continue
        cm_path = plot_confidence_confusion_matrix(
            predictions=predictions,
            output_dir=output_dir,
            dataset=dataset,
            method=method_name,
            confidence_threshold=confidence_threshold,
        )
        if cm_path is not None:
            output[f"confidence_confusion_{method_name.lower()}"] = str(cm_path)

    return output
