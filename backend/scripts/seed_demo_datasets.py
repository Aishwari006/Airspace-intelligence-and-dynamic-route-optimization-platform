"""Seed and Idempotently Maintain Valid Demonstration Datasets.

This script cleans stale/duplicate development runs and seeds 3 fully validated
demo datasets:
1. ds_demo_aircraft_surveillance (Aircraft Observations)
2. ds_demo_weather_grid (Weather Grid Observations)
3. ds_demo_congestion_cells (Pre-aggregated Corridor Congestion Cells)

Usage (from repo root or backend/):
    python backend/scripts/seed_demo_datasets.py
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd
from sqlalchemy import delete, func, select, update

from app.core.config import get_settings
from app.db import models as m
from app.db.models import Base
from app.db.session import get_engine, get_sessionmaker
from app.services.datasets.manager import DatasetManager
from app.services.datasets.schema_mapper import auto_detect_mapping


DEMO_AIRCRAFT_ID = "ds_demo_aircraft_surveillance"
DEMO_WEATHER_ID = "ds_demo_weather_grid"
DEMO_CONGESTION_ID = "ds_demo_congestion_cells"
CANONICAL_DEMO_IDS = {DEMO_AIRCRAFT_ID, DEMO_WEATHER_ID, DEMO_CONGESTION_ID}


def clean_stale_and_duplicate_datasets(session_factory, settings, manager: DatasetManager) -> int:
    """Removes orphan/stale test upload records, their database rows, and physical files."""
    cleaned_count = 0
    with session_factory() as session:
        records = session.scalars(select(m.UploadedDataset)).all()
        for rec in records:
            # Keep synthetic_demo_v1 baseline, clean everything else to reseed fresh
            if rec.id == settings.demo_dataset_id:
                continue

            # Delete related observations & runs
            session.execute(delete(m.AircraftObservation).where(m.AircraftObservation.dataset == rec.id))
            session.execute(delete(m.InferredTrack).where(m.InferredTrack.dataset == rec.id))
            session.execute(delete(m.WeatherObservation).where(m.WeatherObservation.dataset == rec.id))
            session.execute(delete(m.CongestionObservation).where(m.CongestionObservation.dataset == rec.id))
            session.execute(delete(m.DatasetRun).where(m.DatasetRun.dataset_id == rec.id))
            session.delete(rec)
            cleaned_count += 1

        session.commit()

    # Clean orphaned artifacts in data/processed and stale uploads
    processed_dir = settings.data_dir / "processed"
    for p in processed_dir.glob("ds_*"):
        try:
            p.unlink()
        except Exception:
            pass

    return cleaned_count


def generate_seed_data(data_dir: Path) -> dict[str, Path]:
    """Generates clean CSV seed source files for the 3 demo datasets."""
    ref_dir = data_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    # 1. Weather observations seed
    wx_file = ref_dir / "seed_demo_weather_grid.csv"
    wx_rows = []
    # Grid cells around Delhi (VIDP, R11C04), Mumbai (VABB, R06C02), and transit corridor
    sample_cells = [
        ("R11C04", 28.61, 77.21, 0.0, 4.2, 7.8, 15.0, 18.5, 1014.0),
        ("R10C04", 27.00, 76.50, 2.5, 8.5, 14.2, 45.0, 21.0, 1012.5),
        ("R09C03", 25.50, 75.80, 12.0, 16.4, 25.0, 85.0, 23.2, 1008.0),  # Heavy rain cell
        ("R08C03", 23.80, 74.90, 8.5, 12.0, 19.5, 70.0, 24.5, 1009.2),
        ("R07C03", 21.50, 73.80, 0.0, 5.1, 8.0, 20.0, 27.0, 1011.0),
        ("R06C02", 19.08, 72.88, 0.0, 6.2, 9.5, 30.0, 29.5, 1010.5),
        ("R05C02", 17.50, 73.50, 0.5, 7.0, 11.0, 35.0, 28.0, 1011.5),
        ("R04C03", 15.20, 74.80, 0.0, 4.0, 6.5, 10.0, 27.5, 1012.0),
        ("R03C04", 13.00, 77.60, 0.0, 5.5, 8.0, 15.0, 26.0, 1013.0),  # Bengaluru area
        ("R02C05", 10.50, 78.50, 0.0, 6.0, 9.0, 20.0, 28.5, 1012.5),
    ]
    for cid, lat, lon, rain, ws, gust, cld, temp, p in sample_cells:
        wx_rows.append({
            "cell_id": cid,
            "latitude": lat,
            "longitude": lon,
            "valid_time_utc": "2026-01-22T03:00:00Z",
            "precipitation_mm": rain,
            "wind_speed_10m_ms": ws,
            "wind_gusts_10m_ms": gust,
            "cloud_cover_pct": cld,
            "temperature_2m_c": temp,
            "surface_pressure_hpa": p,
        })
    pd.DataFrame(wx_rows).to_csv(wx_file, index=False)

    # 2. Congestion cells seed
    cong_file = ref_dir / "seed_demo_congestion_cells.csv"
    cong_rows = [
        {"cell_id": "R11C04", "aircraft_count": 19, "congestion_score": 0.88, "predicted_class": "HIGH", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R10C04", "aircraft_count": 13, "congestion_score": 0.62, "predicted_class": "MEDIUM", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R09C03", "aircraft_count": 5, "congestion_score": 0.25, "predicted_class": "LOW", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R08C03", "aircraft_count": 4, "congestion_score": 0.18, "predicted_class": "LOW", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R07C03", "aircraft_count": 11, "congestion_score": 0.55, "predicted_class": "MEDIUM", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R06C02", "aircraft_count": 21, "congestion_score": 0.92, "predicted_class": "HIGH", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R05C02", "aircraft_count": 8, "congestion_score": 0.38, "predicted_class": "LOW", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R04C03", "aircraft_count": 6, "congestion_score": 0.28, "predicted_class": "LOW", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R03C04", "aircraft_count": 15, "congestion_score": 0.72, "predicted_class": "MEDIUM", "timestamp_utc": "2026-01-22T03:00:00Z"},
        {"cell_id": "R02C05", "aircraft_count": 7, "congestion_score": 0.32, "predicted_class": "LOW", "timestamp_utc": "2026-01-22T03:00:00Z"},
    ]
    pd.DataFrame(cong_rows).to_csv(cong_file, index=False)

    # 3. Aircraft observations seed
    ac_file = ref_dir / "seed_demo_aircraft_surveillance.csv"
    existing_ac = data_dir / "uploads" / "ds_1b807f28f1_sky_force_demo_flight_observations.csv"
    if existing_ac.exists():
        shutil.copy(existing_ac, ac_file)
    else:
        # Fallback generator if existing file not present
        flights = []
        for i in range(1, 25):
            callsign = f"AIC{100 + i}"
            icao = f"800{i:03d}"
            # Linear trajectory between Delhi (28.6, 77.2) and Mumbai (19.1, 72.9)
            steps = 15
            for s in range(steps):
                frac = s / (steps - 1)
                lat = 28.61 - (28.61 - 19.08) * frac + (i % 3) * 0.1
                lon = 77.21 - (77.21 - 72.88) * frac - (i % 2) * 0.1
                flights.append({
                    "observation_id": f"OBS-{icao}-{s:03d}",
                    "source": "surveillance_demo",
                    "is_synthetic": False,
                    "source_aircraft_id": icao,
                    "callsign": callsign,
                    "timestamp_utc": f"2026-01-22T03:{s*4:02d}:00Z",
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "altitude_m": round(9000.0 + (s * 50), 1),
                    "ground_speed_mps": round(230.0 + (i % 5), 1),
                    "heading_deg": 210.0,
                    "vertical_rate_mps": 0.0,
                    "on_ground": False,
                    "last_contact_utc": f"2026-01-22T03:{s*4:02d}:00Z",
                })
        pd.DataFrame(flights).to_csv(ac_file, index=False)

    return {"weather": wx_file, "congestion": cong_file, "aircraft": ac_file}


def seed_dataset(
    manager: DatasetManager,
    session_factory,
    dataset_id: str,
    name: str,
    csv_path: Path,
    dataset_type: str,
    stages: list[str],
) -> dict:
    """Idempotently seeds a single dataset with explicit dataset_id and executes its pipeline."""
    content = csv_path.read_bytes()
    df = pd.read_csv(io.BytesIO(content))
    cols = list(df.columns)
    mapping = auto_detect_mapping(cols, dataset_type=dataset_type)

    now = datetime.now(timezone.utc)
    target_upload_path = manager.upload_dir / f"{dataset_id}_{name}"
    target_upload_path.write_bytes(content)

    with session_factory() as session:
        # Check if already present and clean it first
        existing = session.get(m.UploadedDataset, dataset_id)
        if existing:
            session.execute(delete(m.AircraftObservation).where(m.AircraftObservation.dataset == dataset_id))
            session.execute(delete(m.InferredTrack).where(m.InferredTrack.dataset == dataset_id))
            session.execute(delete(m.WeatherObservation).where(m.WeatherObservation.dataset == dataset_id))
            session.execute(delete(m.CongestionObservation).where(m.CongestionObservation.dataset == dataset_id))
            session.execute(delete(m.DatasetRun).where(m.DatasetRun.dataset_id == dataset_id))
            session.delete(existing)
            session.commit()

        record = m.UploadedDataset(
            id=dataset_id,
            filename=name,
            dataset_type=dataset_type,
            stored_path=str(target_upload_path),
            file_size_bytes=len(content),
            row_count=len(df),
            col_count=len(cols),
            uploaded_at_utc=now,
            status="validated",
            schema_mapping=mapping,
            validation_report={
                "is_valid": True,
                "dataset_type": dataset_type,
                "warnings": [],
                "detected_columns": cols,
            },
            metadata_json={"source": "seed_demo", "original_filename": name},
        )
        session.add(record)
        session.commit()

    # Execute complete pipeline
    pipeline_res = manager.execute_pipeline(dataset_id=dataset_id, stages=stages)
    return {
        "dataset_id": dataset_id,
        "type": dataset_type,
        "rows": len(df),
        "status": pipeline_res.status,
        "stages": [s.stage for s in pipeline_res.stages],
    }


def run_seed():
    """Main seeding routine."""
    print("=" * 70)
    print("SKY FORCE — DEMO DATASET SEED & DATABASE CLEANUP")
    print("=" * 70)

    settings = get_settings()
    engine = get_engine()
    session_factory = get_sessionmaker()

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    manager = DatasetManager(settings=settings)

    # 1. Clean stale/orphaned datasets
    print("1. Cleaning stale development uploads and duplicate runs...")
    cleaned = clean_stale_and_duplicate_datasets(session_factory, settings, manager)
    print(f"   Cleaned {cleaned} stale/duplicate dataset records.")

    # 2. Generate valid seed files
    print("\n2. Preparing validated seed CSV files with corridor coordinates...")
    seed_files = generate_seed_data(settings.data_dir)
    for k, v in seed_files.items():
        print(f"   - {k}: {v.name} ({v.stat().st_size} bytes)")

    # 3. Seed Weather Observations
    print("\n3. Seeding demo weather grid (ds_demo_weather_grid)...")
    wx_res = seed_dataset(
        manager=manager,
        session_factory=session_factory,
        dataset_id=DEMO_WEATHER_ID,
        name="sky_force_weather_observations.csv",
        csv_path=seed_files["weather"],
        dataset_type="weather_observations",
        stages=["validation", "grid_alignment"],
    )
    print(f"   Status: {wx_res['status']} | Stages: {', '.join(wx_res['stages'])}")

    # 4. Seed Pre-aggregated Congestion Cells
    print("\n4. Seeding demo congestion cells (ds_demo_congestion_cells)...")
    cong_res = seed_dataset(
        manager=manager,
        session_factory=session_factory,
        dataset_id=DEMO_CONGESTION_ID,
        name="delhi_mumbai_corridor_congestion.csv",
        csv_path=seed_files["congestion"],
        dataset_type="congestion_cells",
        stages=["validation"],
    )
    print(f"   Status: {cong_res['status']} | Stages: {', '.join(cong_res['stages'])}")

    # 5. Seed Aircraft Observations
    print("\n5. Seeding demo aircraft surveillance (ds_demo_aircraft_surveillance)...")
    ac_res = seed_dataset(
        manager=manager,
        session_factory=session_factory,
        dataset_id=DEMO_AIRCRAFT_ID,
        name="sky_force_demo_flight_observations.csv",
        csv_path=seed_files["aircraft"],
        dataset_type="aircraft_observations",
        stages=["validation", "quality", "segmentation", "features", "inference"],
    )
    print(f"   Status: {ac_res['status']} | Stages: {', '.join(ac_res['stages'])}")

    # 6. Link weather dataset to aircraft dataset for route optimization context
    with session_factory() as session:
        session.execute(
            update(m.UploadedDataset)
            .where(m.UploadedDataset.id == DEMO_AIRCRAFT_ID)
            .values(linked_weather_dataset_id=DEMO_WEATHER_ID)
        )
        session.commit()

    # 7. Verification Summary
    print("\n" + "=" * 70)
    print("DATABASE VERIFICATION SUMMARY")
    print("=" * 70)
    with session_factory() as session:
        ds_records = session.scalars(select(m.UploadedDataset)).all()
        print(f"Uploaded Datasets ({len(ds_records)} total):")
        for d in ds_records:
            print(f"  • {d.id:30s} | {d.dataset_type:22s} | status={d.status:10s} | rows={d.row_count}")

        runs_count = session.scalar(select(func.count(m.DatasetRun.id)))
        ac_count = session.scalar(select(func.count(m.AircraftObservation.observation_id)))
        wx_count = session.scalar(select(func.count(m.WeatherObservation.id)))
        cong_count = session.scalar(select(func.count(m.CongestionObservation.id)))
        tracks_count = session.scalar(select(func.count(m.InferredTrack.track_id)))
        airports_count = session.scalar(select(func.count(m.Airport.node_id)))

        print("\nSQL Table Row Counts:")
        print(f"  • DatasetRun:              {runs_count}")
        print(f"  • AircraftObservation:     {ac_count}")
        print(f"  • InferredTrack:           {tracks_count}")
        print(f"  • WeatherObservation:      {wx_count}")
        print(f"  • CongestionObservation:   {cong_count}")
        print(f"  • Airport (Reference):     {airports_count}")

    print("\nIdempotent seeding completed successfully!")


if __name__ == "__main__":
    run_seed()
