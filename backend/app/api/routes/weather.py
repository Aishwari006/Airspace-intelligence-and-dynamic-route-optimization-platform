"""Weather context and observation endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.db.session import get_db

router = APIRouter(prefix="/weather", tags=["Weather"])


@router.get("")
def get_weather_observations(
    time_utc: datetime | None = Query(None, description="Hourly time in UTC"),
    dataset_id: str | None = Query(None, description="Optional uploaded dataset context identifier"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    query = select(m.WeatherObservation)
    if dataset_id:
        query = query.where(m.WeatherObservation.dataset == dataset_id)
    if time_utc:
        t0 = time_utc.replace(tzinfo=None)
        query = query.where(m.WeatherObservation.valid_time_utc == t0)
    query = query.order_by(m.WeatherObservation.valid_time_utc.desc()).limit(limit)
    rows = db.scalars(query).all()

    return {
        "count": len(rows),
        "weather": [
            {
                "location_id": r.location_id,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "valid_time_utc": r.valid_time_utc.isoformat(),
                "temperature_2m_c": r.temperature_2m_c,
                "relative_humidity_2m_pct": r.relative_humidity_2m_pct,
                "surface_pressure_hpa": r.surface_pressure_hpa,
                "precipitation_mm": r.precipitation_mm,
                "cloud_cover_pct": r.cloud_cover_pct,
                "wind_speed_10m_ms": r.wind_speed_10m_ms,
                "wind_direction_10m_deg": r.wind_direction_10m_deg,
                "wind_gusts_10m_ms": r.wind_gusts_10m_ms,
                "source": r.source,
                "product": r.product,
                "is_synthetic": r.is_synthetic,
            }
            for r in rows
        ],
    }
