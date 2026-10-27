#!/usr/bin/env python3
"""Visualize Sky Force congestion-classification model metrics.

Reads the existing ml/reports/congestion_metrics.json produced by Sky Force's
training pipeline. It does not train models or modify backend/frontend artifacts.

Run from the repository root:
    python scripts/visualize_congestion_model_metrics.py

Or specify paths explicitly:
    python scripts/visualize_congestion_model_metrics.py \
        --input ml/reports/congestion_metrics.json \
        --output ml/reports/figures/congestion_model_comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("ml/reports/congestion_metrics.json")
DEFAULT_OUTPUT = Path("ml/reports/figures/congestion_model_comparison")
MODEL_LABELS = {
    "dummy_most_frequent": "Dummy",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
}
CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]
OVERALL_METRICS = [
    ("accuracy", "Accuracy", True),
    ("balanced_accuracy", "Balanced accuracy", True),
    ("macro_f1", "Macro-F1", True),
    ("weighted_f1", "Weighted-F1", True),
    ("roc_auc_ovr_macro", "Macro ROC-AUC", True),
    ("brier_multiclass", "Multiclass Brier score", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to congestion_metrics.json (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for generated figures and CSV summaries (default: %(default)s)")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png"],
                        help="Figure formats to save (default: png)")
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find metrics file: {path}\n"
            "Run this script from your Sky Force repository root, or pass --input with the correct path."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Metrics file is empty or has an unexpected structure: {path}")
    return data


def model_names(metrics: dict[str, Any]) -> list[str]:
    preferred = list(MODEL_LABELS)
    present = [name for name in preferred if name in metrics]
    present.extend(name for name in metrics if name not in present and isinstance(metrics[name], dict))
    return present


def metric_value(split_data: dict[str, Any], key: str) -> float:
    value = split_data.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_overall_summary(data: dict[str, Any], names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in names:
        payload = data.get(name, {})
        for split in ("validation", "test"):
            split_data = payload.get(split, {})
            if not isinstance(split_data, dict):
                continue
            row: dict[str, Any] = {
                "model_name": name,
                "model_label": MODEL_LABELS.get(name, name),
                "split": split,
                "n_samples": split_data.get("n"),
                "fit_seconds": payload.get("fit_seconds"),
            }
            for key, _, _ in OVERALL_METRICS:
                row[key] = metric_value(split_data, key)
            rows.append(row)
    return pd.DataFrame(rows)


def build_class_summary(data: dict[str, Any], names: list[str], split: str = "test") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in names:
        split_data = data.get(name, {}).get(split, {})
        per_class = split_data.get("per_class", {}) if isinstance(split_data, dict) else {}
        for cls in CLASS_ORDER:
            values = per_class.get(cls, {})
            if not values:
                continue
            rows.append({
                "model_name": name,
                "model_label": MODEL_LABELS.get(name, name),
                "split": split,
                "class": cls,
                "precision": metric_value(values, "precision"),
                "recall": metric_value(values, "recall"),
                "f1": metric_value(values, "f1"),
                "support": values.get("support"),
            })
    return pd.DataFrame(rows)


def plot_overall_metrics(summary: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    metrics_to_plot = [(k, label, higher) for k, label, higher in OVERALL_METRICS
                       if k in summary.columns and summary[k].notna().any()]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes_flat = axes.flatten()
    models = list(dict.fromkeys(summary["model_name"].tolist()))
    display_labels = [MODEL_LABELS.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.36

    for ax, (key, label, higher) in zip(axes_flat, metrics_to_plot):
        val_by_model = summary[summary["split"] == "validation"].set_index("model_name")[key]
        test_by_model = summary[summary["split"] == "test"].set_index("model_name")[key]
        val = [val_by_model.get(m, np.nan) for m in models]
        test = [test_by_model.get(m, np.nan) for m in models]
        b1 = ax.bar(x - width / 2, val, width, label="Validation")
        b2 = ax.bar(x + width / 2, test, width, label="Test")
        ax.set_title(label + (" (higher is better)" if higher else " (lower is better)"), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=18, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        finite = [v for v in val + test if np.isfinite(v)]
        if finite and min(finite) >= 0 and max(finite) <= 1.02:
            ax.set_ylim(0, 1.05)
        for bars in (b1, b2):
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)
    for ax in axes_flat[len(metrics_to_plot):]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Sky Force Congestion Models — Validation vs Test", fontsize=15, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "01_overall_metrics_validation_vs_test", formats)


def plot_test_class_metrics(class_summary: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    if class_summary.empty:
        print("No per-class metrics found; skipping per-class plot.")
        return
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    models = list(dict.fromkeys(class_summary["model_name"].tolist()))
    labels = [MODEL_LABELS.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.23
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for i, cls in enumerate(CLASS_ORDER):
            cls_data = class_summary[class_summary["class"] == cls].set_index("model_name")
            values = [cls_data.loc[m, metric] if m in cls_data.index else np.nan for m in models]
            bars = ax.bar(x + (i - 1) * width, values, width, label=cls)
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, rotation=90)
        ax.set_title(f"Test {title} by class")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[-1].legend(title="Congestion class", frameon=False)
    fig.suptitle("Per-Class Test Performance — LOW, MEDIUM and HIGH", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "02_test_per_class_precision_recall_f1", formats)


def plot_minority_class_metrics(class_summary: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    subset = class_summary[class_summary["class"].isin(["MEDIUM", "HIGH"])]
    if subset.empty:
        return
    models = list(dict.fromkeys(subset["model_name"].tolist()))
    labels = [MODEL_LABELS.get(m, m) for m in models]
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    x = np.arange(len(models))
    width = 0.36
    for ax, (metric, title) in zip(axes, metrics):
        for i, cls in enumerate(["MEDIUM", "HIGH"]):
            rows = subset[subset["class"] == cls].set_index("model_name")
            values = [rows.loc[m, metric] if m in rows.index else np.nan for m in models]
            bars = ax.bar(x + (i - 0.5) * width, values, width, label=cls)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[-1].legend(title="Minority class", frameon=False)
    fig.suptitle("Minority-Class Test Metrics — MEDIUM and HIGH", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "03_minority_class_test_metrics", formats)


def plot_confusion_matrices(data: dict[str, Any], names: list[str], output_dir: Path, formats: list[str]) -> None:
    included = []
    for name in names:
        cm = data.get(name, {}).get("test", {}).get("confusion_matrix", {})
        matrix = cm.get("matrix") if isinstance(cm, dict) else None
        labels = cm.get("labels", CLASS_ORDER) if isinstance(cm, dict) else CLASS_ORDER
        if matrix is not None and len(matrix):
            included.append((name, np.asarray(matrix, dtype=float), list(labels)))
    if not included:
        print("No confusion matrices found; skipping confusion-matrix plot.")
        return
    ncols = 2
    nrows = int(np.ceil(len(included) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.8 * nrows), squeeze=False)
    for ax, (name, matrix, labels) in zip(axes.flatten(), included):
        image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
        ax.set_title(f"{MODEL_LABELS.get(name, name)} — test confusion matrix")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        threshold = matrix.max() * 0.55 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{int(matrix[i, j]):,}", ha="center", va="center",
                        fontsize=9, color="white" if matrix[i, j] > threshold else "black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.flatten()[len(included):]:
        ax.set_visible(False)
    fig.suptitle("Test Confusion Matrices (rows = actual, columns = predicted)", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_test_confusion_matrices", formats)


def plot_normalized_confusion_matrices(data: dict[str, Any], names: list[str], output_dir: Path,
                                       formats: list[str]) -> None:
    included = []
    for name in names:
        cm = data.get(name, {}).get("test", {}).get("confusion_matrix", {})
        matrix = cm.get("matrix") if isinstance(cm, dict) else None
        labels = cm.get("labels", CLASS_ORDER) if isinstance(cm, dict) else CLASS_ORDER
        if matrix is not None and len(matrix):
            matrix = np.asarray(matrix, dtype=float)
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
            included.append((name, normalized, list(labels)))
    if not included:
        return
    ncols = 2
    nrows = int(np.ceil(len(included) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.8 * nrows), squeeze=False)
    for ax, (name, matrix, labels) in zip(axes.flatten(), included):
        image = ax.imshow(matrix, interpolation="nearest", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{MODEL_LABELS.get(name, name)} — row-normalized test matrix")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > 0.55 else "black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.flatten()[len(included):]:
        ax.set_visible(False)
    fig.suptitle("Normalized Test Confusion Matrices (per actual class)", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "05_test_confusion_matrices_normalized", formats)


def plot_accuracy_vs_balanced(summary: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    test = summary[summary["split"] == "test"].copy()
    test = test.dropna(subset=["accuracy", "balanced_accuracy", "macro_f1"])
    if test.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(test["accuracy"], test["balanced_accuracy"], s=100 + 500 * test["macro_f1"], alpha=0.8)
    for _, row in test.iterrows():
        ax.annotate(MODEL_LABELS.get(row["model_name"], row["model_name"]),
                    (row["accuracy"], row["balanced_accuracy"]), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Test accuracy")
    ax.set_ylabel("Test balanced accuracy")
    ax.set_xlim(max(0, test["accuracy"].min() - 0.04), min(1, test["accuracy"].max() + 0.04))
    ax.set_ylim(max(0, test["balanced_accuracy"].min() - 0.06), min(1, test["balanced_accuracy"].max() + 0.06))
    ax.grid(alpha=0.25)
    ax.set_title("Accuracy vs Balanced Accuracy (bubble size reflects Macro-F1)")
    save_figure(fig, output_dir, "06_accuracy_vs_balanced_accuracy", formats)


def plot_fit_times(summary: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    test = summary[summary["split"] == "test"].drop_duplicates("model_name")
    if test.empty or test["fit_seconds"].isna().all():
        return
    labels = [MODEL_LABELS.get(m, m) for m in test["model_name"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, test["fit_seconds"])
    ax.bar_label(bars, fmt="%.2f s", padding=3)
    ax.set_ylabel("Fit time (seconds)")
    ax.set_title("Reported Model Fit Time")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "07_model_fit_time", formats)


def main() -> int:
    args = parse_args()
    try:
        data = load_metrics(args.input)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    names = model_names(data)
    if not names:
        print(f"No model entries found in {args.input}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    overall = build_overall_summary(data, names)
    class_summary = build_class_summary(data, names, split="test")
    overall_path = args.output / "overall_metrics_summary.csv"
    class_path = args.output / "test_per_class_metrics.csv"
    overall.to_csv(overall_path, index=False)
    class_summary.to_csv(class_path, index=False)

    plot_overall_metrics(overall, args.output, args.formats)
    plot_test_class_metrics(class_summary, args.output, args.formats)
    plot_minority_class_metrics(class_summary, args.output, args.formats)
    plot_confusion_matrices(data, names, args.output, args.formats)
    plot_normalized_confusion_matrices(data, names, args.output, args.formats)
    plot_accuracy_vs_balanced(overall, args.output, args.formats)
    plot_fit_times(overall, args.output, args.formats)

    print("\nVisualization complete.")
    print(f"Input metrics: {args.input}")
    print(f"Output directory: {args.output}")
    print("Generated files:")
    for path in sorted(args.output.iterdir()):
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
