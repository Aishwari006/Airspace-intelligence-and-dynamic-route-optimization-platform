"""Integration tests for Dataset Studio, Dataset-Aware Routing, and Explanation Services."""
from __future__ import annotations

import io
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.routing.context import EdgeAttrs
from app.services.routing.cost_model import CostModel, CostWeights


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_list_datasets(client: TestClient):
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    # Check default synthetic demo dataset exists
    ids = [d["id"] for d in data]
    assert "synthetic_demo_v1" in ids


def test_dataset_upload_and_pipeline(client: TestClient):
    csv_content = (
        "icao24,time,lat,lon,alt,velocity,callsign\n"
        "SF0101,2026-01-22T09:00:00Z,28.556,77.100,10000,220,SFD101\n"
        "SF0101,2026-01-22T09:02:00Z,27.500,76.500,10500,225,SFD101\n"
        "SF0102,2026-01-22T09:00:00Z,19.088,72.868,9000,210,SFD102\n"
        "SF0102,2026-01-22T09:02:00Z,19.800,73.500,9500,215,SFD102\n"
    )

    # 1. Upload CSV
    file_tuple = ("test_observations.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    resp_upload = client.post("/api/datasets/upload", files={"file": file_tuple})
    assert resp_upload.status_code == 200
    upload_data = resp_upload.json()
    dataset_id = upload_data["dataset_id"]
    assert dataset_id.startswith("ds_")
    assert upload_data["row_count"] == 4
    assert upload_data["is_valid_format"] is True
    assert "latitude" in upload_data["suggested_mapping"]
    assert "longitude" in upload_data["suggested_mapping"]

    # 2. Preview raw data
    resp_prev = client.get(f"/api/datasets/{dataset_id}/preview")
    assert resp_prev.status_code == 200
    prev_data = resp_prev.json()
    assert prev_data["total_rows"] == 4
    assert len(prev_data["rows"]) == 4

    # 3. Run Pipeline
    pipeline_payload = {
        "stages": ["quality", "segmentation", "features", "inference"],
        "custom_mapping": upload_data["suggested_mapping"],
    }
    resp_pipe = client.post(f"/api/datasets/{dataset_id}/pipeline", json=pipeline_payload)
    assert resp_pipe.status_code == 200
    pipe_data = resp_pipe.json()
    assert pipe_data["status"] == "completed"
    assert len(pipe_data["stages"]) >= 3

    # 4. Verify Dataset Details
    resp_detail = client.get(f"/api/datasets/{dataset_id}")
    assert resp_detail.status_code == 200
    detail_data = resp_detail.json()
    assert detail_data["status"] == "processed"
    assert "cleaned" in detail_data["available_artifacts"]

    # 5. Dataset-Aware Route Optimization
    opt_payload = {
        "origin": "VIDP",
        "destination": "VABB",
        "objective": "balanced",
        "dataset_id": dataset_id,
    }
    resp_route = client.post("/api/routes/optimize", json=opt_payload)
    assert resp_route.status_code == 200
    route_data = resp_route.json()
    assert route_data["found"] is True
    assert route_data["dataset_id"] == dataset_id
    assert len(route_data["segments"]) > 0


def test_missing_weather_penalty_in_cost_model():
    cm = CostModel(ref_km=100.0, ref_min=10.0)
    w = CostWeights.preset("weather")

    # Edge with missing weather
    edge_missing = EdgeAttrs(
        edge_id="E_MISS",
        distance_km=100.0,
        time_min=10.0,
        time_source="test",
        weather_severity=None,  # MISSING
        weather_source="unavailable",
        uncertainty=0.4,
    )
    # Edge with observed clear weather
    edge_clear = EdgeAttrs(
        edge_id="E_CLEAR",
        distance_km=100.0,
        time_min=10.0,
        time_source="test",
        weather_severity=0.0,  # Clear observed
        weather_source="test",
        uncertainty=0.0,
    )

    cost_missing = cm.edge_cost(edge_missing, w)
    cost_clear = cm.edge_cost(edge_clear, w)

    # Missing weather MUST be penalized more than observed clear weather!
    assert cost_missing > cost_clear


def test_explanation_service(client: TestClient):
    # 1. Status endpoint
    resp_status = client.get("/api/explain/status")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    assert "ollama_available" in status_data
    assert "active_engine" in status_data

    # 2. Generate explanation endpoint (works with Phi-3 or deterministic fallback)
    explain_payload = {
        "task": "route",
        "context_data": {
            "path": ["VIDP", "SFW01", "VABB"],
            "objective": "weather",
            "totals": {
                "distance_km": 1180.0,
                "estimated_time_min": 85.0,
                "weather_exposure_km": 0.0,
                "congestion_exposure_km": 15.0,
                "composite_score": 1.25,
            },
            "baseline_totals": {
                "distance_km": 1148.0,
                "estimated_time_min": 82.0,
                "weather_exposure_km": 80.0,
                "congestion_exposure_km": 20.0,
            },
            "closed_avoided": [],
            "warnings": [],
        },
        "user_query": "Why did the system add 32 km to the route?",
    }
    resp_exp = client.post("/api/explain", json=explain_payload)
    assert resp_exp.status_code == 200
    exp_data = resp_exp.json()
    assert "summary" in exp_data
    assert len(exp_data["summary"]) > 10
    assert len(exp_data["supporting_facts"]) > 0
    assert exp_data["engine"] in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-3.8-flash", "phi3:latest", "deterministic_fallback"]

