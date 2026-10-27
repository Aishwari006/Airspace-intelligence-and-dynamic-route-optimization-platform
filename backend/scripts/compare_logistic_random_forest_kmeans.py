#!/usr/bin/env python3
"""Create a vivid, leakage-aware visual comparison of Logistic Regression,
Random Forest classification, and K-Means clustering for Sky Force.

This is a READ-ONLY visualization/reporting script. It does not train models,
change model artifacts, update the database, or touch the frontend.

Default inputs (project-root relative):
  ml/reports/congestion_metrics.json
  ml/reports/logistic_regression_tuning/logistic_regression_metrics.json (preferred if present)
  ml/reports/kmeans_standalone/kmeans_metrics.json
  ml/reports/kmeans_standalone/kmeans_k_selection.csv
  ml/reports/kmeans_standalone/kmeans_cluster_vs_congestion_counts.csv

Run from repository root:
  python backend/scripts/compare_logistic_random_forest_kmeans.py

Optional:
  python backend/scripts/compare_logistic_random_forest_kmeans.py --logistic-source baseline
  python backend/scripts/compare_logistic_random_forest_kmeans.py --output-dir ml/reports/figures/ml_model_comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Visual palette is intentionally vivid and consistent across plots.
COLORS = {
    "Logistic Regression": "#7C3AED",
    "Random Forest": "#0891B2",
    "K-Means": "#EA580C",
    "train": "#94A3B8",
    "validation": "#7C3AED",
    "test": "#0F766E",
    "LOW": "#2563EB",
    "MEDIUM": "#F59E0B",
    "HIGH": "#DC2626",
    "background": "#F8FAFC",
    "grid": "#CBD5E1",
    "text": "#0F172A",
}

CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]
CLASSIFIER_METRICS = [
    ("accuracy", "Accuracy", True),
    ("balanced_accuracy", "Balanced accuracy", True),
    ("macro_f1", "Macro-F1", True),
    ("weighted_f1", "Weighted-F1", True),
    ("roc_auc_ovr_macro", "Macro ROC-AUC", True),
    ("brier_multiclass", "Multiclass Brier score", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Sky Force repository root (default: current directory).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; default: ml/reports/figures/lr_rf_kmeans_comparison")
    parser.add_argument("--logistic-source", choices=["auto", "tuned", "baseline"], default="auto",
                        help="auto prefers standalone tuned Logistic Regression results if available.")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png"],
                        help="Output image formats (default: png).")
    return parser.parse_args()


def resolve_project_root(root: Path) -> Path:
    root = root.resolve()
    # If launched while the working directory is backend/, use its parent.
    if not (root / "ml" / "reports").exists() and (root.parent / "ml" / "reports").exists():
        root = root.parent
    # If launched from backend/scripts or another nested folder, walk upward.
    if not (root / "ml" / "reports").exists():
        for parent in root.parents:
            if (parent / "ml" / "reports").exists():
                return parent
    return root


def load_json(path: Path, label: str, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing {label}: {path}\nRun the relevant training experiment first, "
                "or pass the correct --project-root."
            )
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label} at {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected JSON structure for {label}: {path}")
    return obj


def get_num(obj: dict[str, Any], key: str) -> float:
    try:
        val = float(obj.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def load_logistic_metrics(reports: Path, choice: str) -> tuple[dict[str, Any], str]:
    tuned_path = reports / "logistic_regression_tuning" / "logistic_regression_metrics.json"
    baseline_path = reports / "congestion_metrics.json"
    if choice == "tuned":
        data = load_json(tuned_path, "tuned Logistic Regression metrics", required=True)
        assert data is not None
        if "metrics" not in data or not isinstance(data["metrics"], dict):
            raise ValueError(f"Tuned Logistic Regression JSON has no 'metrics' split dictionary: {tuned_path}")
        return data["metrics"], f"Tuned Logistic Regression ({tuned_path.relative_to(reports.parent.parent) if tuned_path.is_relative_to(reports.parent.parent) else tuned_path})"
    if choice == "baseline":
        data = load_json(baseline_path, "congestion candidate metrics", required=True)
        assert data is not None
        if "logistic_regression" not in data:
            raise ValueError(f"No 'logistic_regression' candidate in {baseline_path}")
        lr = data["logistic_regression"]
        return {split: lr[split] for split in ("train", "validation", "test") if split in lr}, "Existing candidate Logistic Regression"

    # auto: prefer tuned results if present, otherwise use baseline candidate.
    if tuned_path.exists():
        return load_logistic_metrics(reports, "tuned")
    data = load_json(baseline_path, "congestion candidate metrics", required=True)
    assert data is not None
    if "logistic_regression" not in data:
        raise ValueError(f"No 'logistic_regression' candidate in {baseline_path}")
    lr = data["logistic_regression"]
    return {split: lr[split] for split in ("train", "validation", "test") if split in lr}, "Existing candidate Logistic Regression (baseline)"


def load_classifier_data(reports: Path, logistic_choice: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    candidates = load_json(reports / "congestion_metrics.json", "classification candidate metrics", required=True)
    assert candidates is not None
    if "random_forest" not in candidates:
        raise ValueError("The congestion_metrics.json file does not contain random_forest metrics.")
    logistic_splits, logistic_label = load_logistic_metrics(reports, logistic_choice)
    rf_splits = {split: candidates["random_forest"][split]
                 for split in ("train", "validation", "test")
                 if split in candidates["random_forest"]}
    return logistic_splits, rf_splits, logistic_label


def load_kmeans_data(reports: Path) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    km_dir = reports / "kmeans_standalone"
    metrics = load_json(km_dir / "kmeans_metrics.json", "K-Means metrics", required=True)
    assert metrics is not None
    k_path = km_dir / "kmeans_k_selection.csv"
    table_path = km_dir / "kmeans_cluster_vs_congestion_counts.csv"
    k_df = pd.read_csv(k_path) if k_path.exists() else None
    table_df = pd.read_csv(table_path, index_col=0) if table_path.exists() else None
    return metrics, k_df, table_df


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    fig.patch.set_facecolor("white")
    for ext in formats:
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def nice_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(colors=COLORS["text"], labelsize=9)


def collect_overall_rows(lr: dict[str, Any], rf: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, metrics in [("Logistic Regression", lr), ("Random Forest", rf)]:
        for split in ("train", "validation", "test"):
            part = metrics.get(split)
            if not isinstance(part, dict):
                continue
            for key, display, _ in CLASSIFIER_METRICS:
                rows.append({"model": label, "split": split, "metric": key,
                             "metric_label": display, "value": get_num(part, key),
                             "n_samples": part.get("n")})
    return pd.DataFrame(rows)


def collect_class_rows(lr: dict[str, Any], rf: dict[str, Any], split: str = "test") -> pd.DataFrame:
    rows = []
    for model, split_dict in [("Logistic Regression", lr), ("Random Forest", rf)]:
        part = split_dict.get(split, {})
        per_class = part.get("per_class", {}) if isinstance(part, dict) else {}
        for cls in CLASS_ORDER:
            vals = per_class.get(cls, {}) if isinstance(per_class, dict) else {}
            if not vals:
                continue
            rows.append({"model": model, "split": split, "class": cls,
                         "precision": get_num(vals, "precision"),
                         "recall": get_num(vals, "recall"),
                         "f1": get_num(vals, "f1"), "support": vals.get("support")})
    return pd.DataFrame(rows)


def plot_supervised_overall(overall: pd.DataFrame, out: Path, formats: list[str]) -> None:
    metrics = [(key, label, higher) for key, label, higher in CLASSIFIER_METRICS
               if key in overall.columns and overall.loc[overall.metric == key, "value"].notna().any()]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), facecolor=COLORS["background"])
    bar_cols = [COLORS["Logistic Regression"], COLORS["Random Forest"]]
    for ax, (key, label, higher) in zip(axes.flat, metrics):
        nice_axes(ax)
        sub = overall[overall.metric == key]
        splits = [s for s in ("validation", "test") if s in sub.split.values]
        models = [m for m in ("Logistic Regression", "Random Forest") if m in sub.model.values]
        x = np.arange(len(splits))
        width = 0.34
        for i, model in enumerate(models):
            vals = []
            for split in splits:
                found = sub[(sub.model == model) & (sub.split == split)]
                vals.append(float(found.value.iloc[0]) if not found.empty else np.nan)
            bars = ax.bar(x + (i - (len(models)-1)/2) * width, vals, width,
                          label=model, color=bar_cols[i], edgecolor="white", linewidth=0.7)
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.annotate(f"{val:.3f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, rotation=0)
        ax.set_title(f"{label}\n{'Higher is better' if higher else 'Lower is better'}", fontsize=11, fontweight="bold", color=COLORS["text"])
        ax.set_xticks(x, [s.title() for s in splits])
        finite = sub.value[np.isfinite(sub.value)].to_numpy()
        if key != "brier_multiclass" and len(finite) and finite.min() >= 0 and finite.max() <= 1.02:
            ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.suptitle("Sky Force | Supervised Congestion Models", fontsize=17, fontweight="bold", y=1.035, color=COLORS["text"])
    fig.text(0.5, 0.005, "K-Means is excluded from these panels because it does not have conventional supervised-classification metrics.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    save_fig(fig, out, "01_supervised_overall_metrics", formats)


def plot_class_metrics(classes: pd.DataFrame, out: Path, formats: list[str]) -> None:
    if classes.empty:
        return
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True, facecolor=COLORS["background"])
    models = [m for m in ("Logistic Regression", "Random Forest") if m in classes.model.values]
    x = np.arange(len(models))
    width = 0.34
    for ax, (metric, title) in zip(axes, metrics):
        nice_axes(ax)
        for ci, cls in enumerate(CLASS_ORDER):
            vals = []
            for model in models:
                hit = classes[(classes.model == model) & (classes["class"] == cls)]
                vals.append(float(hit[metric].iloc[0]) if not hit.empty else np.nan)
            offsets = x + (ci - 1) * width
            bars = ax.bar(offsets, vals, width, label=cls, color=COLORS[cls], edgecolor="white")
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.annotate(f"{val:.2f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, rotation=90)
        ax.set_title(f"Test {title} by class", fontweight="bold", color=COLORS["text"])
        ax.set_xticks(x, models, rotation=10, ha="right")
        ax.set_ylim(0, 1.14)
    axes[0].set_ylabel("Score")
    axes[-1].legend(title="Congestion class", frameon=True, facecolor="white")
    fig.suptitle("Per-Class Performance | Test Set", fontsize=16, fontweight="bold", y=1.02, color=COLORS["text"])
    fig.tight_layout()
    save_fig(fig, out, "02_test_per_class_metrics", formats)


def get_cm(split_data: dict[str, Any]) -> tuple[np.ndarray, list[str]] | None:
    cm = split_data.get("confusion_matrix")
    if not isinstance(cm, dict) or not cm.get("matrix"):
        return None
    return np.asarray(cm["matrix"], dtype=float), list(cm.get("labels", CLASS_ORDER))


def plot_confusion_matrices(lr: dict[str, Any], rf: dict[str, Any], out: Path, formats: list[str]) -> None:
    found = []
    for model, data in [("Logistic Regression", lr), ("Random Forest", rf)]:
        cm = get_cm(data.get("test", {}))
        if cm:
            found.append((model, cm[0], cm[1]))
    if not found:
        return
    fig, axes = plt.subplots(1, len(found), figsize=(6.8 * len(found), 5.8), squeeze=False, facecolor=COLORS["background"])
    for ax, (model, matrix, labels) in zip(axes.flat, found):
        im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
        ax.set_title(f"{model}\nTest confusion matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        threshold = matrix.max() * 0.55 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{int(matrix[i,j]):,}", ha="center", va="center", fontsize=10,
                        color="white" if matrix[i,j] > threshold else COLORS["text"], fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.suptitle("Where the Supervised Classifiers Make Errors", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out, "03_test_confusion_matrices", formats)


def plot_normalized_confusions(lr: dict[str, Any], rf: dict[str, Any], out: Path, formats: list[str]) -> None:
    found = []
    for model, data in [("Logistic Regression", lr), ("Random Forest", rf)]:
        cm = get_cm(data.get("test", {}))
        if cm:
            matrix, labels = cm
            row_totals = matrix.sum(axis=1, keepdims=True)
            norm = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
            found.append((model, norm, labels))
    if not found:
        return
    fig, axes = plt.subplots(1, len(found), figsize=(6.8 * len(found), 5.8), squeeze=False, facecolor=COLORS["background"])
    for ax, (model, matrix, labels) in zip(axes.flat, found):
        im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{model}\nRow-normalized (%)", fontweight="bold")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i,j]*100:.1f}%", ha="center", va="center", fontsize=10,
                        color="white" if matrix[i,j] > 0.5 else "black", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.suptitle("Class-wise Recall and Misclassification Pattern", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out, "04_test_normalized_confusion_matrices", formats)


def metric_by_split(km: dict[str, Any], group: str, key: str) -> list[float]:
    out = []
    metrics_by_split = km.get("metrics_by_split", {})
    for split in ("train", "validation", "test"):
        payload = metrics_by_split.get(split, {}).get(group, {})
        out.append(get_num(payload, key))
    return out


def plot_kmeans_quality(km: dict[str, Any], out: Path, formats: list[str]) -> None:
    split_labels = ["Train", "Validation", "Test"]
    x = np.arange(len(split_labels))
    cq_keys = [
        ("silhouette", "Silhouette score", True, "#0F766E"),
        ("davies_bouldin", "Davies–Bouldin index", False, "#EA580C"),
        ("calinski_harabasz", "Calinski–Harabasz score", True, "#7C3AED"),
    ]
    posthoc_keys = [
        ("adjusted_rand_index", "Adjusted Rand index", "#2563EB"),
        ("normalized_mutual_information", "Normalized mutual information", "#9333EA"),
        ("cluster_purity", "Cluster purity (post-hoc)", "#D97706"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=COLORS["background"])
    for ax, (key, title, higher, color) in zip(axes[0], cq_keys):
        nice_axes(ax)
        vals = metric_by_split(km, "cluster_quality", key)
        bars = ax.bar(x, vals, color=color, edgecolor="white", width=0.62)
        ax.bar_label(bars, fmt="%.3f" if key != "calinski_harabasz" else "%.0f", fontsize=9, padding=3)
        ax.set_xticks(x, split_labels)
        ax.set_title(title + (" ↑" if higher else " ↓"), fontweight="bold")
    for ax, (key, title, color) in zip(axes[1], posthoc_keys):
        nice_axes(ax)
        vals = metric_by_split(km, "label_agreement_posthoc", key)
        bars = ax.bar(x, vals, color=color, edgecolor="white", width=0.62)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        ax.set_xticks(x, split_labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontweight="bold")
    fig.suptitle("K-Means | Cluster Quality and Post-hoc Label Agreement", fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, 0.005, "ARI, NMI and purity compare clusters with congestion labels after fitting; they are not classification accuracy.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    save_fig(fig, out, "05_kmeans_cluster_quality", formats)


def plot_k_selection(k_results: pd.DataFrame | None, km: dict[str, Any], out: Path, formats: list[str]) -> None:
    if k_results is None or k_results.empty or "k" not in k_results.columns:
        print("NOTE: kmeans_k_selection.csv not found or has no data; skipping k-selection plot.")
        return
    selection = km.get("selection", {})
    elbow_k = selection.get("elbow_selected_k")
    silhouette_k = selection.get("highest_validation_silhouette_k")
    fig, ax1 = plt.subplots(figsize=(11, 6), facecolor=COLORS["background"])
    nice_axes(ax1)
    ks = pd.to_numeric(k_results["k"], errors="coerce")
    inertia = pd.to_numeric(k_results.get("training_inertia"), errors="coerce")
    l1 = ax1.plot(ks, inertia, color="#2563EB", marker="o", linewidth=2.6, label="Training inertia")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Training inertia", color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.set_xticks(ks.dropna().astype(int).tolist())
    if elbow_k is not None:
        ax1.axvline(int(elbow_k), color="#EA580C", linestyle="--", linewidth=2,
                    label=f"Elbow-selected k = {elbow_k}")
    ax2 = ax1.twinx()
    silhouette = pd.to_numeric(k_results.get("validation_silhouette"), errors="coerce")
    l2 = ax2.plot(ks, silhouette, color="#059669", marker="s", linewidth=2.4, label="Validation silhouette")
    ax2.set_ylabel("Validation silhouette", color="#059669")
    ax2.tick_params(axis="y", labelcolor="#059669")
    if silhouette_k is not None:
        ax2.scatter([int(silhouette_k)], [silhouette.loc[ks == int(silhouette_k)].iloc[0]],
                    s=140, color="#7C3AED", edgecolors="white", linewidths=1.2, zorder=5,
                    label=f"Best validation silhouette k = {silhouette_k}")
    handles = l1 + l2
    labels = [line.get_label() for line in handles]
    h2, l2labels = ax2.get_legend_handles_labels()
    ax1.legend(handles + h2, labels + l2labels, loc="best", frameon=True, facecolor="white")
    ax1.set_title("K-Means Model Selection: Elbow vs Validation Silhouette", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out, "06_kmeans_k_selection", formats)


def plot_cluster_congestion_heatmap(table: pd.DataFrame | None, out: Path, formats: list[str]) -> None:
    if table is None or table.empty:
        print("NOTE: kmeans_cluster_vs_congestion_counts.csv not found; skipping cluster/class heatmap.")
        return
    # Keep known class order first, then any other columns.
    cols = [c for c in CLASS_ORDER if c in table.columns] + [c for c in table.columns if c not in CLASS_ORDER]
    table = table[cols]
    arr = table.to_numpy(dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    perc = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0) * 100
    fig, ax = plt.subplots(figsize=(9, max(4.8, 0.7 * len(table.index))), facecolor=COLORS["background"])
    im = ax.imshow(perc, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(table.index)), labels=[str(v) for v in table.index])
    ax.set_xlabel("Congestion label (post-hoc only)")
    ax.set_ylabel("K-Means cluster")
    ax.set_title("Cluster Composition by Congestion Class (%)\nLabels are used only after clustering", fontweight="bold")
    for i in range(perc.shape[0]):
        for j in range(perc.shape[1]):
            ax.text(j, i, f"{perc[i,j]:.1f}%\n(n={int(arr[i,j]):,})", ha="center", va="center",
                    fontsize=9, color="white" if perc[i,j] > 52 else COLORS["text"], fontweight="bold")
    fig.colorbar(im, ax=ax, label="Within-cluster percentage", fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, out, "07_kmeans_cluster_vs_congestion", formats)


def build_export_tables(overall: pd.DataFrame, classes: pd.DataFrame, km: dict[str, Any], out: Path) -> None:
    overall.to_csv(out / "supervised_overall_metrics.csv", index=False)
    classes.to_csv(out / "supervised_test_per_class_metrics.csv", index=False)
    rows = []
    for split in ("train", "validation", "test"):
        for metric_group in ("cluster_quality", "label_agreement_posthoc"):
            payload = km.get("metrics_by_split", {}).get(split, {}).get(metric_group, {})
            for metric, value in payload.items():
                if isinstance(value, (int, float)) or value is None:
                    rows.append({"model": "K-Means", "split": split, "metric_group": metric_group,
                                 "metric": metric, "value": value})
    pd.DataFrame(rows).to_csv(out / "kmeans_metrics_long.csv", index=False)

    # A readable, compact CSV with key TEST metrics across both paradigm types.
    summary = []
    for model, splits in [("Logistic Regression", None), ("Random Forest", None)]:
        sub = overall[(overall.model == model) & (overall.split == "test")]
        row = {"model": model, "evaluation_type": "supervised classification"}
        for _, display, _ in CLASSIFIER_METRICS:
            hit = sub[sub.metric_label == display]
            row[display] = float(hit.value.iloc[0]) if not hit.empty else np.nan
        summary.append(row)
    for group in ("cluster_quality", "label_agreement_posthoc"):
        payload = km.get("metrics_by_split", {}).get("test", {}).get(group, {})
        for metric, value in payload.items():
            summary.append({"model": "K-Means", "evaluation_type": group, "metric": metric, "value": value})
    pd.DataFrame(summary).to_csv(out / "key_test_metrics_summary.csv", index=False)


def main() -> int:
    args = parse_args()
    root = resolve_project_root(args.project_root)
    reports = root / "ml" / "reports"
    output = Path(args.output_dir) if args.output_dir else reports / "figures" / "lr_rf_kmeans_comparison"
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    lr, rf, lr_source = load_classifier_data(reports, args.logistic_source)
    km, k_results, cluster_table = load_kmeans_data(reports)
    overall = collect_overall_rows(lr, rf)
    classes = collect_class_rows(lr, rf, "test")

    plot_supervised_overall(overall, output, args.formats)
    plot_class_metrics(classes, output, args.formats)
    plot_confusion_matrices(lr, rf, output, args.formats)
    plot_normalized_confusions(lr, rf, output, args.formats)
    plot_kmeans_quality(km, output, args.formats)
    plot_k_selection(k_results, km, output, args.formats)
    plot_cluster_congestion_heatmap(cluster_table, output, args.formats)
    build_export_tables(overall, classes, km, output)

    # Record exact sources so plots are auditable.
    provenance = {
        "project_root": str(root),
        "classification_metrics_source": str(reports / "congestion_metrics.json"),
        "logistic_regression_source_used": lr_source,
        "kmeans_metrics_source": str(reports / "kmeans_standalone" / "kmeans_metrics.json"),
        "kmeans_k_selection_source": str(reports / "kmeans_standalone" / "kmeans_k_selection.csv"),
        "output_directory": str(output),
        "interpretation_note": "Do not compare classification accuracy/F1 directly with K-Means silhouette or cluster-quality metrics. K-Means ARI/NMI/purity are post-hoc label agreement, not trained classification scores.",
    }
    (output / "visualization_sources.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("\nModel comparison visualizations completed.")
    print(f"Project root: {root}")
    print(f"Logistic Regression source: {lr_source}")
    print(f"Output directory: {output}")
    print("Generated figures (where inputs were available):")
    for path in sorted(output.iterdir()):
        if path.suffix.lower() in {".png", ".pdf", ".svg"}:
            print(f"  - {path.name}")
    print("\nReminder: supervised classification metrics and unsupervised cluster-quality metrics measure different things.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
