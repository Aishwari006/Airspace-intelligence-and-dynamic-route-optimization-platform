"""Typed application settings loaded from environment variables / .env.

All external URLs, credentials and research parameters live here so that they
are configured in exactly one place. No secrets are hard-coded.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_ROOT / ".env"), str(BACKEND_DIR / ".env")),
        env_prefix="SKYFORCE_",
        extra="ignore",
    )

    app_name: str = "Sky Force"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Persistence -------------------------------------------------------
    # SQLite is the verified local fallback; PostgreSQL is the documented
    # production-style configuration (see docker-compose.yml).
    database_url: str = f"sqlite:///{(REPO_ROOT / 'data' / 'skyforce.db').as_posix()}"

    data_dir: Path = REPO_ROOT / "data"
    ml_dir: Path = REPO_ROOT / "ml"

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"]
    )

    # --- Demo dataset ------------------------------------------------------
    demo_dataset_id: str = "synthetic_demo_v1"
    demo_seed: int = 20260115
    demo_start_utc: datetime = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
    demo_days: int = 8
    demo_flights_per_day: int = 450
    demo_sample_seconds: int = 90

    # Research aggregation parameters (NOT official ATC sectors).
    grid_lat_min: float = 6.0
    grid_lat_max: float = 34.0
    grid_lon_min: float = 66.0
    grid_lon_max: float = 94.0
    grid_deg: float = 2.0
    window_minutes: int = 15

    # Default scenario time: inside the chronological TEST partition of the
    # demo dataset so routing uses out-of-sample model predictions.
    default_scenario_time_utc: datetime = datetime(2026, 1, 22, 9, 0, tzinfo=timezone.utc)

    # --- Routing assumptions ---------------------------------------------
    heuristic_cruise_speed_kmh: float = 780.0  # documented assumption for heuristic ETA

    # --- External sources (all optional) ----------------------------------
    http_timeout_s: float = 15.0
    http_max_retries: int = 3

    opensky_base_url: str = "https://opensky-network.org/api"
    opensky_token_url: str = (
        "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
    )
    opensky_client_id: str | None = None
    opensky_client_secret: str | None = None
    opensky_min_interval_s: float = 30.0
    # Default bounding box: Indian subcontinent demo region (lamin, lomin, lamax, lomax)
    opensky_bbox: tuple[float, float, float, float] = (6.0, 66.0, 34.0, 94.0)

    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    aviation_weather_base_url: str = "https://aviationweather.gov/api/data"
    ourairports_csv_url: str = "https://davidmegginson.github.io/ourairports-data/airports.csv"

    # Upload limit for observation file import (configurable via SKYFORCE_MAX_UPLOAD_BYTES)
    max_upload_bytes: int = 25 * 1024 * 1024

    # LLM Explanation Service via Google Gemini API (replacing Ollama)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_s: float = 6.0

    # Retained for backwards compatibility
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemini-1.5-flash"
    ollama_timeout_s: float = 6.0

    @property
    def artifacts_dir(self) -> Path:
        return self.ml_dir / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.ml_dir / "reports"

    @property
    def experiments_dir(self) -> Path:
        return self.ml_dir / "experiments"


@lru_cache
def get_settings() -> Settings:
    return Settings()


DISCLAIMER = (
    "Research prototype for simulation and decision support. Not approved for operational "
    "flight planning, navigation, or air traffic control."
)
