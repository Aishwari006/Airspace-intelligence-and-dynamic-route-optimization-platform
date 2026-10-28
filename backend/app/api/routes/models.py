"""Machine Learning Model metadata, metrics, and feature introspection endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter

from app.api.state import get_state
from app.core.config import get_settings

router = APIRouter(prefix="/models", tags=["Machine Learning"])


@router.get("/status")
def get_models_status() -> dict[str, Any]:
    state = get_state()
    settings = get_settings()

    congestion_info = None
    if state.congestion_model:
        m = state.congestion_model.metadata
        feats = m.get("feature_schema", {}).get("features", [])
        sha = m.get("sha256", "")
        raw_metrics = dict(m.get("metrics", {}))
        model_name = m.get("model_name", "random_forest")
        test_m = m.get("selected_test_metrics") or raw_metrics.get(model_name, {}).get("test", {})
        flat_metrics = {
            "accuracy": test_m.get("accuracy"),
            "test_accuracy": test_m.get("accuracy"),
            "balanced_accuracy": test_m.get("balanced_accuracy"),
            "macro_f1": test_m.get("macro_f1"),
            "test_samples": test_m.get("n"),
            "n_test_samples": test_m.get("n"),
            "per_class": test_m.get("per_class", {}),
            "confusion_matrix": test_m.get("confusion_matrix", {}),
            "brier_multiclass": test_m.get("brier_multiclass"),
            "roc_auc_ovr_macro": test_m.get("roc_auc_ovr_macro"),
            "ood_guard": "active",
            **raw_metrics,
        }
        congestion_info = {
            "name": model_name,
            "kind": state.congestion_model.kind,
            "version": state.congestion_model.version,
            "task": m.get("task"),
            "target_definition": m.get("target_definition"),
            "metrics": flat_metrics,
            "feature_names": feats,
            "feature_count": len(feats),
            "sha256": sha[:16] + "..." if sha else "",
            "is_loaded": True,
            "ood_detection": "active",
        }

    eta_info = None
    if state.eta_model:
        m = state.eta_model.metadata
        feats = m.get("feature_schema", {}).get("features", [])
        sha = m.get("sha256", "")
        raw_metrics = dict(m.get("metrics", {}))
        model_name = m.get("model_name", "random_forest")
        test_m = m.get("selected_test_metrics") or raw_metrics.get(model_name, {}).get("test", {})
        flat_metrics = {
            "mae_min": test_m.get("mae_min"),
            "rmse_min": test_m.get("rmse_min"),
            "r2": test_m.get("r2"),
            "bias_min": test_m.get("bias_min"),
            "test_samples": test_m.get("n"),
            "n_test_samples": test_m.get("n"),
            "ood_guard": "active",
            **raw_metrics,
        }
        eta_info = {
            "name": model_name,
            "kind": state.eta_model.kind,
            "version": state.eta_model.version,
            "task": m.get("task"),
            "target_definition": m.get("target_definition"),
            "metrics": flat_metrics,
            "feature_names": feats,
            "feature_count": len(feats),
            "sha256": sha[:16] + "..." if sha else "",
            "is_loaded": True,
            "ood_detection": "active",
        }

    # Load summary if present
    summary_path = settings.reports_dir / "training_summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            pass

    return {
        "status": "operational",
        "congestion_model": congestion_info,
        "eta_model": eta_info,
        "training_summary": summary,
        "disclaimer": "Academic ML models for spatial airspace density & travel-time proxies. Not for tactical ATC separation.",
    }


@router.get("/metrics")
def get_model_evaluation_metrics() -> dict[str, Any]:
    settings = get_settings()
    rep_dir = settings.reports_dir

    congestion_metrics = {}
    c_path = rep_dir / "congestion_metrics.json"
    if c_path.exists():
        try:
            congestion_metrics = json.loads(c_path.read_text())
        except Exception:
            pass

    eta_metrics = {}
    e_path = rep_dir / "eta_metrics.json"
    if e_path.exists():
        try:
            eta_metrics = json.loads(e_path.read_text())
        except Exception:
            pass

    c_best = congestion_metrics.get("random_forest", {}).get("test", {})
    e_best = eta_metrics.get("random_forest", {}).get("test", {})

    return {
        "congestion": congestion_metrics,
        "eta": eta_metrics,
        "selected": {
            "congestion": c_best,
            "eta": e_best,
        },
    }


@router.get("/feature-importance")
def get_feature_importance() -> dict[str, Any]:
    state = get_state()
    out: dict[str, Any] = {"congestion": [], "eta": []}

    # Extract feature importances if tree models are loaded
    if state.congestion_model:
        try:
            pipe = state.congestion_model.pipeline
            clf = pipe.named_steps.get("clf")
            feat_names = state.congestion_model.metadata.get("feature_schema", {}).get("features", [])
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                out["congestion"] = sorted(
                    [
                        {"feature": str(f), "importance": round(float(imp), 4)}
                        for f, imp in zip(feat_names, importances)
                    ],
                    key=lambda x: x["importance"],
                    reverse=True,
                )
        except Exception:
            pass

    if state.eta_model:
        try:
            pipe = state.eta_model.pipeline
            reg = pipe.named_steps.get("reg")
            if hasattr(reg, "feature_importances_"):
                importances = reg.feature_importances_
                try:
                    pre = pipe.named_steps.get("pre")
                    feat_names = list(pre.get_feature_names_out())
                except Exception:
                    feat_names = [f"feat_{i}" for i in range(len(importances))]

                out["eta"] = sorted(
                    [
                        {"feature": str(f).replace("cat__", "").replace("num__", ""), "importance": round(float(imp), 4)}
                        for f, imp in zip(feat_names, importances)
                    ],
                    key=lambda x: x["importance"],
                    reverse=True,
                )[:25]
        except Exception:
            pass

    return out
