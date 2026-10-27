"""Standalone K-Means experiment for Sky Force congestion features.

This script reuses Sky Force's existing congestion feature-generation code, but it
runs an independent clustering experiment. It does NOT register models, update the
DB, overwrite existing artifacts, or interact with the frontend.

K-Means is unsupervised: accuracy / precision / recall / F1 are not its primary
metrics. The script selects k using the training-set elbow curve, cross-checks
with validation silhouette score, and reports silhouette / Davies-Bouldin /
Calinski-Harabasz / inertia plus post-hoc ARI, NMI and cluster purity versus the
existing LOW/MEDIUM/HIGH congestion labels. Labels are NEVER used to fit K-Means,
select features, or select k.

Run from the project root:
    python scripts/train_kmeans_clustering.py
    python scripts/train_kmeans_clustering.py --k-min 2 --k-max 12
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

# This file lives in backend/scripts. Add backend/ to sys.path so imports such as
# `from app.core.config import get_settings` work when the script is launched from
# the repository root with `python backend/scripts/train_kmeans_clustering.py`.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

from app.core.config import get_settings
from app.services.features.congestion import (
    CATEGORICAL_FEATURES,
    CLASSES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_COUNT,
    Thresholds,
    build_feature_frame,
    chronological_split,
    classify,
)
from app.services.features.splits import demo_boundaries
from app.services.geospatial.grid import grid_from_settings
from app.services.ingestion.bootstrap import airports_frame


SEED = 20260115
ID_LIKE_NAMES = {
    "cell_id", "aircraft_id", "flight_id", "track_id", "sample_id", "edge_id",
    "window_end_utc", "window_start_utc", "timestamp", "datetime", "time",
    "y", "target", "label", "next_window_class", TARGET_COUNT,
}
# Prioritize these features when removing nearly duplicate numeric columns.
# The rest of the existing curated numeric feature list remains eligible.
FEATURE_PRIORITY = [
    "aircraft_count", "traffic_change", "neighbor_count", "count_roll4_mean",
    "mean_speed", "speed_variability", "mean_altitude", "wind_speed",
    "precipitation", "cloud_cover",
]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to training-fit quantiles, avoiding test leakage."""
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):  # noqa: ANN001
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper, axis=0)
        self.lower_bounds_ = np.where(np.isfinite(self.lower_bounds_), self.lower_bounds_, -np.inf)
        self.upper_bounds_ = np.where(np.isfinite(self.upper_bounds_), self.upper_bounds_, np.inf)
        return self

    def transform(self, X):  # noqa: ANN001
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


def _load_training_frame(settings):  # noqa: ANN001
    data_dir = settings.data_dir
    processed = data_dir / "processed"
    synthetic = data_dir / "synthetic"
    required = [
        processed / "congestion_agg.parquet",
        processed / "congestion_thresholds.json",
        synthetic / "weather_synthetic.parquet",
        synthetic / "edges.csv",
        synthetic / "waypoints.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Sky Force inputs are missing:\n  " + "\n  ".join(missing)
            + "\nRun the project's demo bootstrap/data-preparation process first."
        )

    agg = pd.read_parquet(processed / "congestion_agg.parquet")
    threshold_dict = json.loads((processed / "congestion_thresholds.json").read_text(encoding="utf-8"))
    weather = pd.read_parquet(synthetic / "weather_synthetic.parquet")
    edges = pd.read_csv(synthetic / "edges.csv")
    waypoints = pd.read_csv(synthetic / "waypoints.csv")

    grid = grid_from_settings(settings)
    bounds = demo_boundaries(settings)
    airports, _ = airports_frame()
    nodes = pd.concat(
        [airports[["node_id", "latitude", "longitude"]], waypoints[["node_id", "latitude", "longitude"]]],
        ignore_index=True,
    ).drop_duplicates("node_id", keep="first")
    pos = {r.node_id: (r.latitude, r.longitude) for r in nodes.itertuples(index=False)}
    missing_nodes = sorted({n for e in edges.itertuples(index=False) for n in (e.source_node, e.target_node) if n not in pos})
    if missing_nodes:
        raise ValueError(f"Some edge endpoints have no coordinates: {missing_nodes[:10]}")
    corridor = sorted({
        cell
        for e in edges.itertuples(index=False)
        for cell in grid.cells_along(*pos[e.source_node], *pos[e.target_node])
    })
    thresholds = Thresholds(
        threshold_dict["low_max"], threshold_dict["medium_max"], threshold_dict["method"]
    )
    frame = build_feature_frame(
        agg, weather, grid, corridor, bounds.start, bounds.end, settings.window_minutes
    )
    if TARGET_COUNT not in frame.columns:
        raise KeyError(f"Expected target count column {TARGET_COUNT!r} was not generated.")
    frame = frame[frame[TARGET_COUNT].notna()].copy()
    frame["_congestion_label_for_evaluation_only"] = classify(frame[TARGET_COUNT], thresholds).astype(str)
    parts = chronological_split(frame, bounds.val_start, bounds.test_start, settings.window_minutes)
    return parts, bounds, thresholds, settings


def _select_features(X_train: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    """Select varied, sufficiently observed numeric features and low-cardinality categories.

    Selection uses only the training features, never the congestion labels.
    Very high-cardinality identifiers (especially cell_id) are not one-hot encoded,
    because that would make Euclidean distances mostly reflect identifier identity.
    """
    diagnostics: list[dict[str, Any]] = []
    numeric_candidates = [c for c in NUMERIC_FEATURES if c in X_train.columns and c not in ID_LIKE_NAMES]
    preferred_order = [c for c in FEATURE_PRIORITY if c in numeric_candidates]
    numeric_order = preferred_order + [c for c in numeric_candidates if c not in preferred_order]
    numeric_selected: list[str] = []
    numeric_train = pd.DataFrame(index=X_train.index)
    for col in numeric_order:
        values = pd.to_numeric(X_train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_rate = float(values.isna().mean())
        nunique = int(values.nunique(dropna=True))
        reason = "selected"
        if missing_rate >= 0.80:
            reason = "dropped: >=80% missing in training data"
        elif nunique <= 1:
            reason = "dropped: constant or all-missing in training data"
        else:
            numeric_train[col] = values
            numeric_selected.append(col)
        diagnostics.append({"feature": col, "role": "numeric", "training_missing_pct": round(missing_rate * 100, 4),
                            "training_unique_values": nunique, "decision": reason})

    # Drop only extremely redundant numeric features (|Pearson r| >= 0.98), on training only.
    # Median imputation and quantile clipping here are only for feature screening; final
    # preprocessing is still fitted through the transformer using the training partition.
    if len(numeric_selected) >= 2:
        screen = numeric_train[numeric_selected].copy()
        for col in numeric_selected:
            med = screen[col].median()
            screen[col] = screen[col].fillna(med)
            lo, hi = screen[col].quantile(0.01), screen[col].quantile(0.99)
            screen[col] = screen[col].clip(lo, hi)
        corr = screen.corr(numeric_only=True).abs()
        kept: list[str] = []
        dropped_corr: set[str] = set()
        for col in numeric_selected:
            if any(corr.loc[col, prev] >= 0.98 for prev in kept):
                dropped_corr.add(col)
                for row in diagnostics:
                    if row["feature"] == col:
                        row["decision"] = "dropped: |training correlation| >= 0.98 with an earlier selected feature"
                        break
            else:
                kept.append(col)
        numeric_selected = kept

    categorical_selected: list[str] = []
    cat_candidates = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    for col in cat_candidates:
        lower_name = col.lower()
        values = X_train[col].astype("string")
        nunique = int(values.nunique(dropna=True))
        missing_rate = float(values.isna().mean())
        if lower_name in ID_LIKE_NAMES or lower_name.endswith("_id") or lower_name in {"cell_id"}:
            reason = "dropped: identifier/high-cardinality key; not one-hot encoded for Euclidean clustering"
        elif nunique <= 1:
            reason = "dropped: constant or all-missing in training data"
        elif nunique > 12:
            reason = "dropped: >12 categories; high-cardinality encoding can dominate distance"
        else:
            reason = "selected"
            categorical_selected.append(col)
        diagnostics.append({"feature": col, "role": "categorical", "training_missing_pct": round(missing_rate * 100, 4),
                            "training_unique_values": nunique, "decision": reason})

    if not numeric_selected and not categorical_selected:
        raise ValueError("No usable clustering features remain. Inspect feature-generation outputs and diagnostics.")
    return numeric_selected, categorical_selected, pd.DataFrame(diagnostics)


def _make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clip_outliers", QuantileClipper(lower=0.01, upper=0.99)),
            ("scale", RobustScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _safe_cluster_metrics(X: np.ndarray, labels: np.ndarray, seed: int, max_silhouette_samples: int) -> dict[str, Any]:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    out: dict[str, Any] = {"n_samples": int(len(labels)), "n_clusters_assigned": int(len(unique))}
    if len(unique) < 2 or len(unique) >= len(labels):
        out.update({"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
                    "reason": "requires at least 2 non-degenerate clusters and fewer clusters than samples"})
        return out
    try:
        kwargs = {}
        if len(labels) > max_silhouette_samples:
            kwargs = {"sample_size": max_silhouette_samples, "random_state": seed}
        out["silhouette"] = float(silhouette_score(X, labels, **kwargs))
    except Exception as exc:  # noqa: BLE001
        out["silhouette"] = None
        out["silhouette_error"] = str(exc)
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception as exc:  # noqa: BLE001
        out["davies_bouldin"] = None
        out["davies_bouldin_error"] = str(exc)
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception as exc:  # noqa: BLE001
        out["calinski_harabasz"] = None
        out["calinski_harabasz_error"] = str(exc)
    return out


def _elbow_k(ks: list[int], inertias: list[float]) -> int:
    """Approximate elbow as the point farthest from the line joining curve endpoints."""
    if len(ks) < 3:
        return ks[0]
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x_norm = (x - x.min()) / max(float(x.max() - x.min()), 1e-12)
    y_span = float(y.max() - y.min())
    y_norm = (y - y.min()) / max(y_span, 1e-12)
    # Perpendicular distance from the straight line connecting the normalized endpoints.
    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    distances = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1) / max(denom, 1e-12)
    # Ignore endpoints, which cannot represent a useful elbow.
    return int(ks[1 + int(np.argmax(distances[1:-1]))])


def _cluster_purity(labels_true: np.ndarray, clusters: np.ndarray) -> float | None:
    table = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(labels_true, name="congestion_class"))
    if table.to_numpy().sum() == 0:
        return None
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def _posthoc_label_metrics(y_true: pd.Series, cluster_ids: np.ndarray) -> dict[str, Any]:
    """External agreement metrics only; never used in model or k selection."""
    y = np.asarray(y_true).astype(str)
    return {
        "note": "Post-hoc comparison only. Congestion labels are not used to fit K-Means or select k.",
        "adjusted_rand_index": float(adjusted_rand_score(y, cluster_ids)),
        "normalized_mutual_information": float(normalized_mutual_info_score(y, cluster_ids)),
        "cluster_purity": _cluster_purity(y, cluster_ids),
    }


def run(args: argparse.Namespace) -> Path:
    settings = get_settings()
    parts, bounds, thresholds, settings = _load_training_frame(settings)
    if "train" not in parts or "val" not in parts or "test" not in parts:
        raise ValueError("Chronological splitter must return train, val and test partitions.")

    label_col = "_congestion_label_for_evaluation_only"
    # Guard against accidental target leakage if the feature list changes later.
    safe_features = [c for c in FEATURES if c != TARGET_COUNT and c != label_col]
    x_train = parts["train"][safe_features].copy()
    x_val = parts["val"][safe_features].copy()
    x_test = parts["test"][safe_features].copy()
    y_train = parts["train"][label_col].astype(str)
    y_val = parts["val"][label_col].astype(str)
    y_test = parts["test"][label_col].astype(str)

    numeric, categorical, feature_diagnostics = _select_features(x_train)
    preprocessor = _make_preprocessor(numeric, categorical)
    X_train = np.asarray(preprocessor.fit_transform(x_train), dtype=float)
    X_val = np.asarray(preprocessor.transform(x_val), dtype=float)
    X_test = np.asarray(preprocessor.transform(x_test), dtype=float)
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all() or not np.isfinite(X_test).all():
        raise ValueError("Non-finite values remain after preprocessing. Inspect data quality and selected features.")

    n_train = len(X_train)
    k_min = max(2, args.k_min)
    k_max = min(args.k_max, n_train - 1)
    if k_max < k_min:
        raise ValueError(f"Invalid k range after checking training rows: {k_min}..{k_max}")
    ks = list(range(k_min, k_max + 1))
    k_rows: list[dict[str, Any]] = []
    model_by_k: dict[int, KMeans] = {}

    for k in ks:
        model = KMeans(n_clusters=k, n_init=args.n_init, max_iter=500, random_state=args.seed, algorithm="lloyd")
        model.fit(X_train)
        model_by_k[k] = model
        val_clusters = model.predict(X_val)
        val_metrics = _safe_cluster_metrics(X_val, val_clusters, args.seed, args.max_silhouette_samples)
        min_dist_val = model.transform(X_val).min(axis=1)
        val_inertia = float(np.square(min_dist_val).sum())
        k_rows.append({
            "k": k,
            "training_inertia": float(model.inertia_),
            "validation_inertia": val_inertia,
            "validation_silhouette": val_metrics.get("silhouette"),
            "validation_davies_bouldin": val_metrics.get("davies_bouldin"),
            "validation_calinski_harabasz": val_metrics.get("calinski_harabasz"),
            "validation_clusters_assigned": val_metrics.get("n_clusters_assigned"),
        })

    k_results = pd.DataFrame(k_rows)
    inertias = [float(v) for v in k_results["training_inertia"].tolist()]
    k_elbow = _elbow_k(ks, inertias)
    # Keep k selected by the elbow method; report validation silhouette's top k separately.
    # This avoids selecting k using congestion labels or overfitting a single metric.
    silhouette_series = k_results["validation_silhouette"].dropna()
    best_silhouette_k = int(k_results.loc[silhouette_series.idxmax(), "k"]) if not silhouette_series.empty else None
    selected_k = k_elbow
    selected_model = model_by_k[selected_k]

    train_clusters = selected_model.predict(X_train)
    val_clusters = selected_model.predict(X_val)
    test_clusters = selected_model.predict(X_test)
    split_metrics = {
        "train": {
            "cluster_quality": _safe_cluster_metrics(X_train, train_clusters, args.seed, args.max_silhouette_samples),
            "label_agreement_posthoc": _posthoc_label_metrics(y_train, train_clusters),
        },
        "validation": {
            "cluster_quality": _safe_cluster_metrics(X_val, val_clusters, args.seed, args.max_silhouette_samples),
            "label_agreement_posthoc": _posthoc_label_metrics(y_val, val_clusters),
        },
        "test": {
            "cluster_quality": _safe_cluster_metrics(X_test, test_clusters, args.seed, args.max_silhouette_samples),
            "label_agreement_posthoc": _posthoc_label_metrics(y_test, test_clusters),
        },
    }

    out_dir = Path(settings.reports_dir) / "kmeans_standalone"
    out_dir.mkdir(parents=True, exist_ok=True)
    k_results.to_csv(out_dir / "kmeans_k_selection.csv", index=False)
    feature_diagnostics.to_csv(out_dir / "kmeans_feature_diagnostics.csv", index=False)

    # Persist row-level cluster assignments with labels explicitly identified as post-hoc.
    assignment_frames = []
    for split_name, partition, assignments in [
        ("train", parts["train"], train_clusters),
        ("validation", parts["val"], val_clusters),
        ("test", parts["test"], test_clusters),
    ]:
        ids = pd.DataFrame(index=partition.index)
        for candidate in ("cell_id", "window_end_utc", TARGET_COUNT):
            if candidate in partition.columns:
                ids[candidate] = partition[candidate]
        ids["split"] = split_name
        ids["cluster_id"] = assignments.astype(int)
        ids["congestion_label_posthoc_only"] = partition[label_col].astype(str).values
        assignment_frames.append(ids.reset_index(drop=True))
    assignments_df = pd.concat(assignment_frames, ignore_index=True)
    assignments_df.to_csv(out_dir / "kmeans_cluster_assignments.csv", index=False)

    posthoc_table = pd.crosstab(assignments_df["cluster_id"], assignments_df["congestion_label_posthoc_only"])
    posthoc_table.to_csv(out_dir / "kmeans_cluster_vs_congestion_counts.csv")
    split_table_rows = []
    for split_name in ("train", "validation", "test"):
        sub = assignments_df[assignments_df["split"] == split_name]
        counts = sub["cluster_id"].value_counts().sort_index()
        for cluster, count in counts.items():
            split_table_rows.append({"split": split_name, "cluster_id": int(cluster), "count": int(count),
                                     "percentage": float(count / max(len(sub), 1) * 100)})
    pd.DataFrame(split_table_rows).to_csv(out_dir / "kmeans_cluster_sizes_by_split.csv", index=False)

    # Elbow + validation silhouette chart.
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(k_results["k"], k_results["training_inertia"], marker="o", label="Training inertia")
    ax1.axvline(selected_k, linestyle="--", label=f"Elbow-selected k = {selected_k}")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Training inertia (lower is better; use elbow for diminishing returns)")
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(k_results["k"], k_results["validation_silhouette"], marker="s", linestyle="-.", label="Validation silhouette")
    ax2.set_ylabel("Validation silhouette (higher is generally better)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax1.set_title("K-Means model selection: elbow curve and validation silhouette")
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_elbow_silhouette.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "experiment": "standalone K-Means clustering on engineered congestion features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": "synthetic demonstration data as configured by the existing Sky Force pipeline",
        "warning": "Unsupervised clusters do not have conventional classification accuracy. ARI/NMI/purity are post-hoc label-agreement measures only; the labels are not used for fitting, feature selection, or k selection.",
        "selection": {
            "method": "elbow method on training inertia; validation silhouette is reported as an independent cross-check",
            "k_min": k_min,
            "k_max": k_max,
            "elbow_selected_k": selected_k,
            "highest_validation_silhouette_k": best_silhouette_k,
            "validation_silhouette_at_elbow": next((r["validation_silhouette"] for r in k_rows if r["k"] == selected_k), None),
        },
        "configuration": {"random_state": args.seed, "n_init": args.n_init, "max_iter": 500,
                          "quantile_clip": [0.01, 0.99], "scaler": "RobustScaler", "max_silhouette_samples": args.max_silhouette_samples},
        "features": {"numeric": numeric, "categorical": categorical,
                     "excluded_categorical_policy": "cell_id and high-cardinality identifiers are not one-hot encoded because they can dominate Euclidean distance"},
        "split_boundaries": bounds.to_dict() if hasattr(bounds, "to_dict") else str(bounds),
        "thresholds_used_for_posthoc_label_comparison_only": thresholds.to_dict() if hasattr(thresholds, "to_dict") else str(thresholds),
        "rows": {name: int(len(parts[name])) for name in ("train", "val", "test")},
        "transformed_feature_count": int(X_train.shape[1]),
        "metrics_by_split": split_metrics,
        "feature_selection_diagnostics_file": "kmeans_feature_diagnostics.csv",
        "k_search_file": "kmeans_k_selection.csv",
        "cluster_assignments_file": "kmeans_cluster_assignments.csv",
    }
    (out_dir / "kmeans_metrics.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")

    joblib.dump({
        "preprocessor": preprocessor,
        "kmeans": selected_model,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "selected_k": selected_k,
        "random_state": args.seed,
        "note": "Standalone experimental artifact; not registered in Sky Force model registry.",
    }, out_dir / "kmeans_standalone_model.joblib")

    print("\nK-Means experiment completed.")
    print(f"Output directory: {out_dir}")
    print(f"Selected k by elbow: {selected_k}")
    print(f"k with highest validation silhouette: {best_silhouette_k}")
    print("\nValidation metrics around selected k:")
    print(k_results[["k", "training_inertia", "validation_silhouette", "validation_davies_bouldin", "validation_calinski_harabasz"]].to_string(index=False))
    print("\nSelected model cluster-quality metrics:")
    print(json.dumps(_json_safe(split_metrics), indent=2))
    print("\nNote: do not report conventional accuracy/precision/recall/F1 for K-Means. Use cluster-quality metrics and post-hoc ARI/NMI/purity instead.")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-min", type=int, default=2, help="Smallest k to evaluate (default: 2)")
    parser.add_argument("--k-max", type=int, default=10, help="Largest k to evaluate (default: 10)")
    parser.add_argument("--n-init", type=int, default=20, help="K-Means initializations per k (default: 20)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED})")
    parser.add_argument("--max-silhouette-samples", type=int, default=3000,
                        help="Maximum samples for silhouette calculation to keep runtime manageable")
    args = parser.parse_args()
    if args.k_min < 2:
        parser.error("--k-min must be at least 2")
    if args.k_max < args.k_min:
        parser.error("--k-max must be >= --k-min")
    if args.n_init < 1:
        parser.error("--n-init must be at least 1")
    if args.max_silhouette_samples < 100:
        parser.error("--max-silhouette-samples should be at least 100")
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"K-Means experiment stopped: {exc}", RuntimeWarning)
        raise


if __name__ == "__main__":
    main()
