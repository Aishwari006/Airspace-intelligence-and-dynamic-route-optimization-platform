import sys
sys.path.insert(0, 'backend')
from fastapi.testclient import TestClient
from app.main import app

def main():
    with TestClient(app) as client:
        print("--- 1. Testing GET /api/datasets ---")
        r = client.get("/api/datasets")
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        datasets = r.json()
        print(f"Datasets count: {len(datasets)}")
        dataset_ids = [d.get("id") or d.get("dataset_id") for d in datasets]
        print(f"Dataset IDs: {dataset_ids}")
        assert "ds_demo_aircraft_surveillance" in dataset_ids
        assert "ds_demo_weather_grid" in dataset_ids
        assert "ds_demo_congestion_cells" in dataset_ids

        print("\n--- 2. Testing Preview ---")
        r = client.get("/api/datasets/ds_demo_aircraft_surveillance/preview?page=1&page_size=10")
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        preview = r.json()
        print(f"Preview total_rows: {preview.get('total_rows')}, rows returned: {len(preview.get('rows', []))}")
        assert len(preview["rows"]) == 10

        print("\n--- 3. Testing Models Status ---")
        r = client.get("/api/models/status")
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        models = r.json()
        cong = models.get("congestion_model") or models.get("congestion", {})
        eta = models.get("eta_model") or models.get("eta", {})
        print(f"Congestion model loaded: {cong.get('is_loaded')}, ETA model loaded: {eta.get('is_loaded')}")
        print(f"Congestion metrics: accuracy={cong.get('metrics', {}).get('accuracy')}, macro_f1={cong.get('metrics', {}).get('macro_f1')}")
        assert cong.get("metrics", {}).get("accuracy", 0) > 0.8

        print("\n--- 4. Testing Congestion Cells with dataset_id ---")
        r = client.get("/api/congestion/cells?dataset_id=ds_demo_congestion_cells")
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        cells = r.json()
        print(f"Congestion cells returned: {len(cells)}")
        assert len(cells) > 0

        print("\n--- 5. Testing Route Optimization with dataset_id & weather_dataset_id ---")
        payload = {
            "origin": "VIDP",
            "destination": "VABB",
            "altitude_ft": 35000,
            "airspeed_kts": 450,
            "dataset_id": "ds_demo_aircraft_surveillance",
            "weather_dataset_id": "ds_demo_weather_grid",
            "objective_weights": {"distance": 0.3, "time": 0.3, "congestion": 0.2, "weather": 0.2}
        }
        r = client.post("/api/routes/optimize", json=payload)
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        route_res = r.json()
        path = route_res.get("path", [])
        dist_km = route_res.get("totals", {}).get("distance_km")
        print(f"Route calculated: path={' -> '.join(path)}, distance={dist_km} km, segments={len(route_res.get('segments', []))}")
        print(f"Route dataset_id: {route_res.get('dataset_id')}, weather_dataset_id: {route_res.get('weather_dataset_id')}")
        assert route_res["dataset_id"] == "ds_demo_aircraft_surveillance"
        assert route_res["weather_dataset_id"] == "ds_demo_weather_grid"
        assert len(path) >= 2

        print("\n--- 6. Testing Route Compare ---")
        r = client.post("/api/routes/compare", json=payload)
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        comp_res = r.json()
        routes = comp_res.get("routes", [])
        print(f"Alternative candidate routes count: {len(routes)}")
        assert len(routes) > 0

        print("\n--- 7. Testing Parquet Upload Rejection ---")
        files = {"file": ("malicious_test.parquet", b"PAR1fakecontent", "application/octet-stream")}
        data = {"dataset_type": "aircraft_observations"}
        r = client.post("/api/datasets/upload", files=files, data=data)
        print(f"Parquet upload response code: {r.status_code}")
        assert r.status_code in (400, 422)
        print(f"Error detail: {r.json()}")

        print("\n--- 8. Testing Weather endpoint with dataset_id ---")
        r = client.get("/api/weather?dataset_id=ds_demo_weather_grid")
        assert r.status_code == 200
        weather_data = r.json().get("weather", [])
        print(f"Weather records returned: {len(weather_data)}")
        assert len(weather_data) > 0

        print("\n==========================================")
        print(">>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print("==========================================")

if __name__ == "__main__":
    main()
