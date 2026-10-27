"""Aircraft observations and track telemetry endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.db.session import get_db

router = APIRouter(prefix="/aircraft", tags=["Aircraft & Surveillance"])


@router.get("/observations")
def get_aircraft_observations(
    timestamp_utc: datetime | None = Query(None, description="Time slice in UTC"),
    window_seconds: int = Query(90, description="Window tolerance around timestamp in seconds"),
    dataset_id: str | None = Query(None, description="Optional uploaded dataset context identifier"),
    limit: int = Query(500, le=2000),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    query = select(m.AircraftObservation)
    if dataset_id:
        query = query.where(m.AircraftObservation.dataset == dataset_id)
    if timestamp_utc:
        t0 = timestamp_utc.replace(tzinfo=None)
        dt = timedelta(seconds=window_seconds)
        query = query.where(
            m.AircraftObservation.timestamp_utc >= (t0 - dt),
            m.AircraftObservation.timestamp_utc <= (t0 + dt),
        )
    query = query.order_by(m.AircraftObservation.timestamp_utc.desc()).limit(limit)
    rows = db.scalars(query).all()

    return {
        "timestamp_requested": timestamp_utc.isoformat() if timestamp_utc else None,
        "count": len(rows),
        "observations": [
            {
                "observation_id": r.observation_id,
                "source_aircraft_id": r.source_aircraft_id,
                "callsign": r.callsign,
                "timestamp_utc": r.timestamp_utc.isoformat(),
                "latitude": r.latitude,
                "longitude": r.longitude,
                "altitude_m": r.altitude_m,
                "ground_speed_mps": r.ground_speed_mps,
                "heading_deg": r.heading_deg,
                "vertical_rate_mps": r.vertical_rate_mps,
                "on_ground": r.on_ground,
                "track_id": r.track_id,
                "data_quality_flags": r.data_quality_flags,
                "is_synthetic": r.is_synthetic,
            }
            for r in rows
        ],
    }


@router.get("/tracks")
def get_inferred_tracks(
    dataset_id: str | None = Query(None, description="Optional uploaded dataset context identifier"),
    limit: int = Query(50, le=500),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    query = select(m.InferredTrack)
    if dataset_id:
        query = query.where(m.InferredTrack.dataset == dataset_id)
    tracks = db.scalars(query.limit(limit)).all()
    return [
        {
            "track_id": t.track_id,
            "source_aircraft_id": t.source_aircraft_id,
            "callsign": t.callsign,
            "start_utc": t.start_utc.isoformat(),
            "end_utc": t.end_utc.isoformat(),
            "n_observations": t.n_observations,
            "method": t.method,
            "is_inferred": t.is_inferred,
        }
        for t in tracks
    ]
