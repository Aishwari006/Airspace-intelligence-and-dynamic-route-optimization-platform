"""Training & evaluation for (A) congestion classification and (B) segment-ETA regression.

Both use chronological train / validation / test partitions. Model selection
uses the VALIDATION split only; the TEST split is touched once, for reporting.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.features.congestion import (
    CATEGORICAL_FEATURES,
    CLASSES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_COUNT,
    Thresholds,
    chronological_split,
    classify,
)
from app.services.features.eta import ETA_FEATURES, ETA_NUMERIC, ETA_TARGET
from app.services.features.splits import SplitBoundaries


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def _ranges(df: pd.DataFrame, cols: list[str]) -> dict[str, list[float]]:
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s):
            out[c] = [float(s.quantile(0.01)), float(s.quantile(0.99))]
    return out


def classification_metrics(y_true, y_pred, proba: np.ndarray | None, classes: list[str]) -> dict[str, Any]:  # noqa: ANN001
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist()},
    }
    if proba is not None:
        onehot = np.stack([(np.asarray(y_true) == c).astype(float) for c in classes], axis=1)
        out["brier_multiclass"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        try:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(onehot, proba, average="macro"))
        except ValueError:
            out["roc_auc_ovr_macro"] = None
    return out


def _proba_ordered(model, X: pd.DataFrame, classes: list[str]) -> np.ndarray:  # noqa: ANN001
    raw = model.predict_proba(X)
    cls = list(model.classes_)
    out = np.zeros((len(X), len(classes)))
    for i, c in enumerate(classes):
        if c in cls:
            out[:, i] = raw[:, cls.index(c)]
    return out


def _reliability(y_true, proba_high: np.ndarray, bins: int = 10) -> list[dict]:  # noqa: ANN001
    y = (np.asarray(y_true) == "HIGH").astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (proba_high >= lo) & (proba_high < hi if hi < 1 else proba_high <= hi)
        if m.sum():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "mean_predicted": float(proba_high[m].mean()),
                        "observed_frequency": float(y[m].mean()), "n": int(m.sum())})
    return out


# ---------------------------------------------------------------------------
# (A) congestion
# ---------------------------------------------------------------------------

def congestion_candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline([("prep", _preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, False)),
                                         ("clf", DummyClassifier(strategy="most_frequent"))]),
        "logistic_regression": Pipeline([("prep", _preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, True)),
                                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))]),
        "random_forest": Pipeline([("prep", _preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, False)),
                                   ("clf", RandomForestClassifier(n_estimators=400, max_depth=18, min_samples_leaf=3,
                                                                  max_features=0.45,
                                                                  class_weight={"LOW": 1.0, "MEDIUM": 2.5, "HIGH": 5.0},
                                                                  n_jobs=-1,
                                                                  random_state=seed))]),
        "random_forest_balanced": Pipeline([("prep", _preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, False)),
                                            ("clf", RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=3,
                                                                           class_weight="balanced_subsample", n_jobs=-1,
                                                                           random_state=seed))]),
        "hist_gradient_boosting": Pipeline([("prep", _preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, False)),
                                            ("clf", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                                                   class_weight="balanced",
                                                                                   early_stopping=False, random_state=seed))]),
    }


def train_congestion(frame: pd.DataFrame, bounds: SplitBoundaries, th: Thresholds, window_min: int,
                     seed: int) -> dict[str, Any]:
    df = frame[frame[TARGET_COUNT].notna()].copy()
    excluded_missing_target = int(frame[TARGET_COUNT].isna().sum())
    df["y"] = classify(df[TARGET_COUNT], th)
    parts = chronological_split(df, bounds.val_start, bounds.test_start, window_min)
    Xtr, ytr = parts["train"][FEATURES], parts["train"]["y"].astype(str)
    Xva, yva = parts["val"][FEATURES], parts["val"]["y"].astype(str)
    Xte, yte = parts["test"][FEATURES], parts["test"]["y"].astype(str)
    assert parts["train"].window_end_utc.max() < parts["test"].window_end_utc.min(), "chronology violated"

    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in congestion_candidates(seed).items():
        t = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t
        fitted[name] = pipe
        results[name] = {
            "fit_seconds": round(fit_s, 2),
            "validation": classification_metrics(yva, pipe.predict(Xva), _proba_ordered(pipe, Xva, CLASSES), CLASSES),
            "test": classification_metrics(yte, pipe.predict(Xte), _proba_ordered(pipe, Xte, CLASSES), CLASSES),
            "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in pipe.named_steps["clf"].get_params().items()},
        }
    selectable = [n for n in results if not n.startswith("dummy")]
    best = max(selectable, key=lambda n: results[n]["validation"]["macro_f1"])
    model = fitted[best]

    # explainability on the selected model
    perm = permutation_importance(model, Xva, yva, scoring="f1_macro", n_repeats=5, random_state=seed, n_jobs=1)
    perm_imp = sorted(({"feature": f, "importance_mean": float(m), "importance_std": float(s)}
                       for f, m, s in zip(FEATURES, perm.importances_mean, perm.importances_std)),
                      key=lambda d: -d["importance_mean"])
    builtin = _builtin_importance(model, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    proba_te = _proba_ordered(model, Xte, CLASSES)
    pred_te = model.predict(Xte)
    test_preds = parts["test"][["cell_id", "window_end_utc", TARGET_COUNT]].copy()
    test_preds["y_true"] = yte.values
    test_preds["y_pred"] = pred_te
    for i, c in enumerate(CLASSES):
        test_preds[f"p_{c}"] = proba_te[:, i]

    reference = {c: float(pd.to_numeric(Xtr[c], errors="coerce").median()) for c in NUMERIC_FEATURES
                 if pd.to_numeric(Xtr[c], errors="coerce").notna().any()}
    class_dist = {k: {c: int((parts[k]["y"] == c).sum()) for c in CLASSES} for k in ("train", "val", "test")}
    metadata = {
        "model_name": best,
        "task": "3-class research congestion proxy for the NEXT window",
        "target_definition": {
            "target": "next_window_class",
            "source_column": TARGET_COUNT,
            "description": "unique-aircraft count in the same research cell during the next window, bucketed by thresholds",
            "thresholds": th.to_dict(),
            "window_minutes": window_min,
            "note": "Research traffic-density proxy - NOT an official ATC capacity/congestion classification.",
        },
        "feature_schema": {"features": FEATURES, "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
                           "categories": {"cell_id": sorted(Xtr.cell_id.astype(str).unique().tolist())}},
        "splits": bounds.to_dict(),
        "rows": {"train": int(len(Xtr)), "validation": int(len(Xva)), "test": int(len(Xte)),
                 "excluded_missing_target": excluded_missing_target},
        "class_distribution": class_dist,
        "selection": {"criterion": "validation macro-F1", "candidates": list(results)},
        "metrics": results,
        "selected_test_metrics": results[best]["test"],
        "reliability_high_class_test": _reliability(yte, proba_te[:, CLASSES.index("HIGH")]),
        "permutation_importance_validation": perm_imp,
        "builtin_importance": builtin,
        "training_ranges": _ranges(Xtr, NUMERIC_FEATURES),
        "reference_values": reference,
        "random_seed": seed,
    }
    return {"pipeline": model, "metadata": metadata, "test_predictions": test_preds}


def _builtin_importance(pipe: Pipeline, numeric: list[str], categorical: list[str]) -> list[dict] | None:
    est = pipe.steps[-1][1]
    if not hasattr(est, "feature_importances_"):
        return None
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = est.feature_importances_
    agg: dict[str, float] = {}
    for n, v in zip(names, imp):
        raw = n.split("__", 1)[1]
        key = next((c for c in categorical if raw.startswith(c + "_")), raw)
        agg[key] = agg.get(key, 0.0) + float(v)
    return sorted(({"feature": k, "importance": v} for k, v in agg.items()), key=lambda d: -d["importance"])


# ---------------------------------------------------------------------------
# (B) segment ETA
# ---------------------------------------------------------------------------

def regression_metrics(y, p) -> dict[str, Any]:  # noqa: ANN001
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    hist, edges = np.histogram(np.abs(err), bins=[0, 0.5, 1, 2, 3, 5, 8, 12, 1e9])
    abs_err = np.abs(err)
    return {
        "n": int(len(y)), "mae_min": float(mean_absolute_error(y, p)),
        "rmse_min": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p)),
        "median_abs_error_min": float(np.median(abs_err)),
        "bias_min": float(err.mean()),
        "abs_error_histogram": [{"bin_min": f"{edges[i]:g}-{'inf' if edges[i + 1] > 1e8 else f'{edges[i + 1]:g}'}",
                                 "count": int(h)} for i, h in enumerate(hist)],
    }


def heuristic_eta_min(distance_km, speed_kmh: float):  # noqa: ANN001, ANN201
    return np.asarray(distance_km, dtype=float) / speed_kmh * 60.0


def train_eta(df: pd.DataFrame, bounds: SplitBoundaries, seed: int, heuristic_speed_kmh: float) -> dict[str, Any]:
    df = df[df[ETA_TARGET].notna() & (df[ETA_TARGET] > 0)].copy()
    # plausibility filter on target: implied speed between 150 and 1300 km/h
    implied = df.segment_distance_km / (df[ETA_TARGET] / 60)
    implausible = ~implied.between(150, 1300)
    excluded = int(implausible.sum())
    df = df[~implausible]
    tr = df[df.exit_time_utc < bounds.val_start]
    va = df[(df.entry_time_utc >= bounds.val_start) & (df.exit_time_utc < bounds.test_start)]
    te = df[df.entry_time_utc >= bounds.test_start]
    Xtr, ytr = tr[ETA_FEATURES], tr[ETA_TARGET]
    Xva, yva = va[ETA_FEATURES], va[ETA_TARGET]
    Xte, yte = te[ETA_FEATURES], te[ETA_TARGET]

    cands = {
        "median_baseline": Pipeline([("prep", _preprocessor(ETA_NUMERIC, [], False)), ("reg", DummyRegressor(strategy="median"))]),
        "ridge": Pipeline([("prep", _preprocessor(ETA_NUMERIC, [], True)), ("reg", Ridge(alpha=1.0))]),
        # Random Forest variants
        "random_forest": Pipeline([("prep", _preprocessor(ETA_NUMERIC, [], False)),
                                   ("reg", RandomForestRegressor(n_estimators=300, max_depth=22, min_samples_leaf=2,
                                                                 max_features=0.85, n_jobs=-1, random_state=seed))]),
        "random_forest_400_24_0.90": Pipeline([("prep", _preprocessor(ETA_NUMERIC, [], False)),
                                               ("reg", RandomForestRegressor(n_estimators=400, max_depth=24,
                                                                             min_samples_leaf=2,
                                                                             max_features=0.90, n_jobs=-1,
                                                                             random_state=seed))]),
        "random_forest_500_26_0.95": Pipeline([("prep", _preprocessor(ETA_NUMERIC, [], False)),
                                               ("reg", RandomForestRegressor(n_estimators=500, max_depth=26,
                                                                             min_samples_leaf=2,
                                                                             max_features=0.95, n_jobs=-1,
                                                                             random_state=seed))]),
        # Hist Gradient Boosting variants
        "hist_gradient_boosting": Pipeline([("prep", _preprocessor(ETA_NUMERIC, [], False)),
                                            ("reg", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                                                                  early_stopping=False, random_state=seed))]),
        "hist_gradient_boosting_600_0.05": Pipeline([("prep", _preprocessor(ETA_NUMERIC, [], False)),
                                                    ("reg", HistGradientBoostingRegressor(max_iter=600,
                                                                                          learning_rate=0.05,
                                                                                          early_stopping=False,
                                                                                          random_state=seed))]),
    }
    results: dict[str, Any] = {
        "heuristic_distance_over_speed": {
            "assumption": f"distance / {heuristic_speed_kmh:g} km/h (not fitted)",
            "validation": regression_metrics(yva, heuristic_eta_min(va.segment_distance_km, heuristic_speed_kmh)),
            "test": regression_metrics(yte, heuristic_eta_min(te.segment_distance_km, heuristic_speed_kmh)),
        }
    }
    fitted = {}
    for name, pipe in cands.items():
        t = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fitted[name] = pipe
        results[name] = {"fit_seconds": round(time.perf_counter() - t, 2),
                         "validation": regression_metrics(yva, pipe.predict(Xva)),
                         "test": regression_metrics(yte, pipe.predict(Xte))}
    selectable = [n for n in cands if not n.startswith("median")]
    best = min(selectable, key=lambda n: results[n]["validation"]["mae_min"])
    model = fitted[best]
    pred_te = model.predict(Xte)
    perm = permutation_importance(model, Xva, yva, scoring="neg_mean_absolute_error", n_repeats=5, random_state=seed, n_jobs=1)
    perm_imp = sorted(({"feature": f, "importance_mean": float(m), "importance_std": float(s)}
                       for f, m, s in zip(ETA_FEATURES, perm.importances_mean, perm.importances_std)),
                      key=lambda d: -d["importance_mean"])
    test_preds = te[["edge_id", "entry_time_utc", "segment_distance_km"]].copy()
    test_preds["actual_min"] = yte.values
    test_preds["predicted_min"] = pred_te
    test_preds["heuristic_min"] = heuristic_eta_min(te.segment_distance_km, heuristic_speed_kmh)

    planning = {
        "entry_speed_mps_by_edge": tr.groupby("edge_id").entry_speed_mps.median().round(2).to_dict(),
        "entry_altitude_m_by_edge": tr.groupby("edge_id").entry_altitude_m.median().round(1).to_dict(),
        "entry_speed_mps_by_from_airport": tr.groupby("from_airport").entry_speed_mps.median().round(2).to_dict(),
        "entry_altitude_m_by_from_airport": tr.groupby("from_airport").entry_altitude_m.median().round(1).to_dict(),
        "note": "At planning time the actual entry speed/altitude is unknown; the training-split median for the edge "
                "(or for airport/non-airport origin) is used as a documented proxy.",
    }
    metadata = {
        "model_name": best,
        "task": "segment traversal time regression (minutes)",
        "target_definition": {
            "target": ETA_TARGET,
            "description": "time between consecutive observed node passages (25 km radius, along-track refined) on a graph edge",
            "exclusions": {"implausible_implied_speed": excluded},
        },
        "feature_schema": {"features": ETA_FEATURES, "numeric": ETA_NUMERIC, "categorical": []},
        "splits": bounds.to_dict(),
        "rows": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
        "selection": {"criterion": "validation MAE", "candidates": list(results)},
        "metrics": results,
        "selected_test_metrics": results[best]["test"],
        "permutation_importance_validation": perm_imp,
        "builtin_importance": _builtin_importance(model, ETA_NUMERIC, []),
        "training_ranges": _ranges(Xtr, ETA_NUMERIC),
        "planning_defaults": planning,
        "heuristic_speed_kmh": heuristic_speed_kmh,
        "random_seed": seed,
    }
    return {"pipeline": model, "metadata": metadata, "test_predictions": test_preds}
