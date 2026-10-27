"""Standalone, validation-tuned Logistic Regression for Sky Force congestion labels.

This script reuses Sky Force's existing feature-generation functions and chronological
train/validation/test boundaries, but does NOT call the main training orchestrator,
register an artifact in the database, or replace the active Random Forest model.

Run from the repository's Python environment, from the project root:
    python scripts/train_logistic_regression_tuned.py

Outputs are written to ml/reports/logistic_regression_tuning/ (or settings.reports_dir
when the configured reports directory is used). Hyperparameters and class weights are
selected using the training period and the existing validation period only. The test
period is reported after selection and is not used to choose parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Allow execution from either a root-level scripts/ folder or backend/scripts/.
_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, _SCRIPT_DIR.parent, Path.cwd(), Path.cwd() / "backend"):
    if (_candidate / "app").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer

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


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def load_congestion_splits(settings: Any) -> tuple[dict[str, pd.DataFrame], Any, int]:
    """Recreate the existing congestion features and chronological splits."""
    processed = settings.data_dir / "processed"
    synthetic = settings.data_dir / "synthetic"
    required = [
        processed / "congestion_agg.parquet",
        processed / "congestion_thresholds.json",
        synthetic / "weather_synthetic.parquet",
        synthetic / "edges.csv",
        synthetic / "waypoints.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required project data files were not found:\n  - " + "\n  - ".join(missing)
        )

    agg = pd.read_parquet(processed / "congestion_agg.parquet")
    threshold_data = json.loads((processed / "congestion_thresholds.json").read_text(encoding="utf-8"))
    weather = pd.read_parquet(synthetic / "weather_synthetic.parquet")
    edges = pd.read_csv(synthetic / "edges.csv")
    waypoints = pd.read_csv(synthetic / "waypoints.csv")

    grid = grid_from_settings(settings)
    bounds = demo_boundaries(settings)
    airports, _ = airports_frame()
    nodes = pd.concat(
        [
            airports[["node_id", "latitude", "longitude"]],
            waypoints[["node_id", "latitude", "longitude"]],
        ],
        ignore_index=True,
    )
    positions = {row.node_id: (row.latitude, row.longitude) for row in nodes.itertuples()}
    missing_nodes = sorted(
        {node for node in pd.concat([edges["source_node"], edges["target_node"]]).dropna().unique()
         if node not in positions}
    )
    if missing_nodes:
        raise ValueError(f"Some airway edge endpoints have no coordinates: {missing_nodes[:10]}")

    corridor = sorted(
        {
            cell
            for edge in edges.itertuples()
            for cell in grid.cells_along(
                *positions[edge.source_node], *positions[edge.target_node]
            )
        }
    )
    thresholds = Thresholds(
        threshold_data["low_max"], threshold_data["medium_max"], threshold_data["method"]
    )
    frame = build_feature_frame(
        agg, weather, grid, corridor, bounds.start, bounds.end, settings.window_minutes
    )
    excluded_missing_target = int(frame[TARGET_COUNT].isna().sum())
    frame = frame[frame[TARGET_COUNT].notna()].copy()
    frame["y"] = classify(frame[TARGET_COUNT], thresholds).astype(str)
    splits = chronological_split(frame, bounds.val_start, bounds.test_start, settings.window_minutes)

    for split_name in ("train", "val", "test"):
        part = splits[split_name]
        if part.empty:
            raise ValueError(f"Chronological '{split_name}' split is empty; inspect dates and data.")
        absent_features = [column for column in FEATURES if column not in part.columns]
        if absent_features:
            raise ValueError(f"Missing expected features in {split_name}: {absent_features}")
        missing_classes = [label for label in CLASSES if label not in set(part["y"].astype(str))]
        if missing_classes:
            print(f"WARNING: {split_name} split has no examples of classes: {missing_classes}")

    if "window_end_utc" in splits["train"].columns and "window_end_utc" in splits["test"].columns:
        train_max = pd.to_datetime(splits["train"]["window_end_utc"], utc=True).max()
        test_min = pd.to_datetime(splits["test"]["window_end_utc"], utc=True).min()
        if not train_max < test_min:
            raise ValueError(f"Chronological split check failed: train max {train_max}, test min {test_min}")

    print(f"Rows excluded because target count was missing: {excluded_missing_target}")
    print("Chronological split sizes:", {name: len(part) for name, part in splits.items()})
    print("Training class distribution:", splits["train"]["y"].value_counts().to_dict())
    return splits, bounds, int(settings.demo_seed)


def make_preprocessor() -> ColumnTransformer:
    """Match the existing LR preprocessing: impute, scale numeric, one-hot categorical."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if NUMERIC_FEATURES:
        transformers.append(("numeric", numeric_pipeline, list(NUMERIC_FEATURES)))
    if CATEGORICAL_FEATURES:
        transformers.append(("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)))
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def metric_bundle(y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray,
                  labels: list[str]) -> dict[str, Any]:
    y = np.asarray(y_true).astype(str)
    pred = np.asarray(y_pred).astype(str)
    p, r, f, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y, pred, labels=labels)
    result: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
    onehot = np.stack([(y == label).astype(float) for label in labels], axis=1)
    if probabilities.shape == onehot.shape:
        result["brier_multiclass"] = float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))
        try:
            result["roc_auc_ovr_macro"] = float(roc_auc_score(onehot, probabilities, average="macro"))
        except ValueError:
            result["roc_auc_ovr_macro"] = None
    else:
        result["brier_multiclass"] = None
        result["roc_auc_ovr_macro"] = None
    return result


def ordered_probabilities(model: Pipeline, X: pd.DataFrame, labels: list[str]) -> np.ndarray:
    raw = model.predict_proba(X)
    model_classes = list(model.named_steps["clf"].classes_)
    aligned = np.zeros((len(X), len(labels)), dtype=float)
    for idx, label in enumerate(labels):
        if label in model_classes:
            aligned[:, idx] = raw[:, model_classes.index(label)]
    return aligned


def class_weight_candidates(labels: list[str]) -> list[Any]:
    """Compare no weights, sklearn balanced weights, and moderate/high minority emphasis."""
    low, medium, high = labels[0], labels[1], labels[2]
    return [
        None,
        "balanced",
        {low: 1.0, medium: 3.0, high: 6.0},
        {low: 1.0, medium: 6.0, high: 12.0},
        {low: 1.0, medium: 10.0, high: 25.0},
        {low: 1.0, medium: 15.0, high: 35.0},
    ]


def flatten_metric_rows(split_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    simple_metrics = [
        "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "brier_multiclass", "roc_auc_ovr_macro",
    ]
    for split_name, metrics in split_metrics.items():
        for metric in simple_metrics:
            rows.append({"split": split_name, "class": "ALL", "metric": metric, "value": metrics.get(metric)})
        for class_name, class_metrics in metrics["per_class"].items():
            for metric in ("precision", "recall", "f1", "support"):
                rows.append({
                    "split": split_name,
                    "class": class_name,
                    "metric": metric,
                    "value": class_metrics[metric],
                })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    reports_dir = Path(settings.reports_dir)
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir / "logistic_regression_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits, bounds, seed = load_congestion_splits(settings)
    labels = list(CLASSES)
    x_train = splits["train"][FEATURES].copy()
    y_train = splits["train"]["y"].astype(str).copy()
    x_val = splits["val"][FEATURES].copy()
    y_val = splits["val"]["y"].astype(str).copy()
    x_test = splits["test"][FEATURES].copy()
    y_test = splits["test"]["y"].astype(str).copy()

    # GridSearchCV uses a single, explicit chronological fold: fit on the project's
    # training period and score on its validation period. This intentionally avoids
    # shuffled folds that could mix future and past observations.
    x_train_val = pd.concat([x_train, x_val], axis=0, ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    train_indices = np.arange(len(x_train))
    val_indices = np.arange(len(x_train), len(x_train) + len(x_val))
    chronological_cv = [(train_indices, val_indices)]
    macro_f1_scorer = make_scorer(
        f1_score, labels=labels, average="macro", zero_division=0
    )

    weight_grid = class_weight_candidates(labels)
    param_grid = {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        "clf__class_weight": weight_grid,
    }
    search = GridSearchCV(
        estimator=build_pipeline(seed),
        param_grid=param_grid,
        scoring=macro_f1_scorer,
        cv=chronological_cv,
        refit=False,
        n_jobs=max(1, min(2, args.jobs)),
        verbose=args.verbose,
        error_score="raise",
        return_train_score=True,
        pre_dispatch="2*n_jobs",
    )

    print(f"Running {len(param_grid['clf__C']) * len(weight_grid)} candidate configurations.")
    print("Selection metric: validation Macro-F1; test data is held out until selection is complete.")
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        search.fit(x_train_val, y_train_val)
    search_seconds = time.perf_counter() - start

    search_frame = pd.DataFrame(search.cv_results_)
    keep_cols = [col for col in [
        "rank_test_score", "mean_test_score", "std_test_score", "mean_train_score",
        "mean_fit_time", "params", "param_clf__C", "param_clf__class_weight",
    ] if col in search_frame.columns]
    search_frame = search_frame[keep_cols].sort_values("rank_test_score", kind="stable")
    search_frame["params"] = search_frame["params"].map(lambda v: json.dumps(v, default=_json_default))
    search_frame["param_clf__class_weight"] = search_frame["param_clf__class_weight"].map(
        lambda v: json.dumps(v, default=_json_default)
    )
    search_frame.to_csv(output_dir / "grid_search_results.csv", index=False)

    best_params = dict(search.best_params_)
    best_score = float(search.best_score_)
    # Refit only on the training partition, so final metrics are directly comparable
    # with the existing Sky Force candidate models, which also fit on train only.
    final_model = build_pipeline(seed)
    final_model.set_params(**best_params)
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as final_warnings:
        warnings.simplefilter("always")
        final_model.fit(x_train, y_train)
    final_fit_seconds = time.perf_counter() - fit_start

    all_metrics: dict[str, dict[str, Any]] = {}
    prediction_frames: list[pd.DataFrame] = []
    split_data = {
        "train": (splits["train"], x_train, y_train),
        "validation": (splits["val"], x_val, y_val),
        "test": (splits["test"], x_test, y_test),
    }
    for split_name, (part, X, y) in split_data.items():
        y_pred = final_model.predict(X)
        proba = ordered_probabilities(final_model, X, labels)
        all_metrics[split_name] = metric_bundle(y, y_pred, proba, labels)
        if split_name == "test":
            columns_to_keep = [col for col in ["cell_id", "window_end_utc", TARGET_COUNT] if col in part.columns]
            preds = part[columns_to_keep].copy().reset_index(drop=True)
            preds["y_true"] = y.to_numpy()
            preds["y_pred"] = np.asarray(y_pred)
            for idx, label in enumerate(labels):
                preds[f"p_{label}"] = proba[:, idx]
            prediction_frames.append(preds)

        report = classification_report(
            y, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        pd.DataFrame(report).transpose().to_csv(
            output_dir / f"classification_report_{split_name}.csv", index=True, index_label="class"
        )
        cm = all_metrics[split_name]["confusion_matrix"]
        cm_df = pd.DataFrame(cm["matrix"], index=labels, columns=labels)
        cm_df.index.name = "actual_class"
        cm_df.to_csv(output_dir / f"confusion_matrix_{split_name}.csv")

    # Explicitly save selected candidate details and all split metrics.
    metadata = {
        "model_name": "logistic_regression_tuned_standalone",
        "task": "three-class research congestion proxy for the next observation window",
        "data_notice": "Sky Force synthetic demonstration data; metrics are not evidence of real-world aviation accuracy.",
        "selection": {
            "method": "GridSearchCV with one chronological train-to-validation split",
            "metric": "macro_f1",
            "best_validation_macro_f1": best_score,
            "candidate_count": int(len(search.cv_results_["params"])),
            "search_seconds": round(search_seconds, 3),
            "best_params": best_params,
        },
        "training": {
            "random_seed": seed,
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
            "features": list(FEATURES),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "fit_seconds": round(final_fit_seconds, 3),
            "split_boundaries": bounds.to_dict() if hasattr(bounds, "to_dict") else str(bounds),
        },
        "metrics": all_metrics,
        "warnings": [str(w.message) for w in caught_warnings + final_warnings],
        "isolation_note": "Writes only to this standalone output directory; does not register or replace active models.",
    }
    write_json(output_dir / "logistic_regression_metrics.json", metadata)
    flatten_metric_rows(all_metrics).to_csv(output_dir / "logistic_regression_metrics.csv", index=False)
    prediction_frames[0].to_csv(output_dir / "test_predictions.csv", index=False)
    write_json(output_dir / "best_params.json", best_params)
    joblib.dump(final_model, output_dir / "logistic_regression_pipeline.joblib")

    # A compact model-comparison export with the current test scores for quick review.
    compact_rows = []
    for split_name in ("train", "validation", "test"):
        m = all_metrics[split_name]
        row = {"model": "Tuned Logistic Regression", "split": split_name}
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "brier_multiclass", "roc_auc_ovr_macro"):
            row[key] = m.get(key)
        for cls in labels:
            row[f"{cls}_precision"] = m["per_class"][cls]["precision"]
            row[f"{cls}_recall"] = m["per_class"][cls]["recall"]
            row[f"{cls}_f1"] = m["per_class"][cls]["f1"]
        compact_rows.append(row)
    pd.DataFrame(compact_rows).to_csv(output_dir / "metrics_comparison_ready.csv", index=False)

    # Read-only comparison against the existing candidate report, if it exists.
    # This is an aid to compare the tuned run with the prior LR and RF results;
    # the source report is never overwritten.
    existing_report = reports_dir / "congestion_metrics.json"
    comparison_rows: list[dict[str, Any]] = []
    if existing_report.exists():
        try:
            old_data = json.loads(existing_report.read_text(encoding="utf-8"))
            model_aliases = {
                "Existing Logistic Regression": "logistic_regression",
                "Existing Random Forest": "random_forest",
                "Existing Histogram Gradient Boosting": "hist_gradient_boosting",
                "Existing Dummy Classifier": "dummy_most_frequent",
            }
            for split_label, old_split_key in (("validation", "validation"), ("test", "test")):
                for model_label, model_key in model_aliases.items():
                    old_metrics = old_data.get(model_key, {}).get(old_split_key, {})
                    for metric_name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                                        "roc_auc_ovr_macro", "brier_multiclass"):
                        if old_metrics.get(metric_name) is not None:
                            comparison_rows.append({
                                "model": model_label, "split": split_label, "metric": metric_name,
                                "value": old_metrics[metric_name], "source": str(existing_report),
                            })
                tuned_metrics = all_metrics[split_label]
                for metric_name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                                    "roc_auc_ovr_macro", "brier_multiclass"):
                    if tuned_metrics.get(metric_name) is not None:
                        comparison_rows.append({
                            "model": "Tuned Logistic Regression", "split": split_label,
                            "metric": metric_name, "value": tuned_metrics[metric_name],
                            "source": "this standalone tuned run",
                        })
            pd.DataFrame(comparison_rows).to_csv(
                output_dir / "baseline_vs_tuned_model_metrics.csv", index=False
            )
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            print(f"NOTE: Could not read existing model metrics report for comparison: {exc}")

    print("\nBEST HYPERPARAMETERS")
    print(json.dumps(best_params, indent=2, default=_json_default))
    print(f"Validation Macro-F1 used for selection: {best_score:.4f}")
    print("\nFINAL MODEL TEST METRICS")
    print(json.dumps(all_metrics["test"], indent=2, default=_json_default))
    print(f"\nAll outputs saved to: {output_dir}")
    print("The active Random Forest model, registry, database, and frontend were not changed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: settings.reports_dir/logistic_regression_tuning",
    )
    parser.add_argument(
        "--jobs", type=int, default=2,
        help="Parallel GridSearchCV jobs (default: 2 to limit memory use).",
    )
    parser.add_argument("--verbose", type=int, default=1, help="GridSearchCV verbosity (0-3).")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


if __name__ == "__main__":
    run(parse_args())
