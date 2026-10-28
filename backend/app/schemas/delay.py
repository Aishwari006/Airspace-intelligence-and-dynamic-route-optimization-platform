"""Typed Pydantic schemas for Flight Delay Prediction & Passenger-Centred Intelligence."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class DelayContributingFactor(BaseModel):
    """An explainable contributing factor with strict evidence source attribution."""
    name: str
    category: str  # INBOUND | TURNAROUND | WEATHER | QUEUE | AIRSPACE | UNKNOWN
    impact_minutes: float
    evidence_status: str  # CONFIRMED | MODEL_INFERRED | UNAVAILABLE | NOT_APPLICABLE
    description: str
    source: str


class OperationalStage(BaseModel):
    """One of the 8 chronological operational departure stages."""
    stage_number: int
    stage_id: str
    stage_name: str
    status: str  # COMPLETED | ACTIVE | PROJECTED | DELAYED | UNAVAILABLE
    timestamp_utc: datetime | None = None
    time_display: str
    evidence_status: str  # CONFIRMED | MODEL_INFERRED | UNAVAILABLE
    source: str
    explanation: str
    is_confirmed: bool = False


class FlightSummary(BaseModel):
    """Concise flight record suitable for search results, lists, and status badges."""
    flight_id: str
    flight_number: str
    callsign: str
    airline: str
    aircraft_tail: str
    origin_id: str
    origin_iata: str
    origin_name: str
    destination_id: str
    destination_iata: str
    destination_name: str
    scheduled_departure_utc: datetime
    estimated_departure_utc: datetime
    scheduled_departure_display: str
    estimated_departure_display: str
    scheduled_arrival_utc: datetime | None = None
    estimated_arrival_utc: datetime | None = None
    scheduled_arrival_display: str | None = None
    estimated_arrival_display: str | None = None
    arrival_delay_minutes: float | None = None
    predicted_delay_minutes: float
    delay_interval_min: float
    delay_interval_max: float
    status: str  # ON_SCHEDULE | DELAY_LIKELY | SIGNIFICANT_DELAY_LIKELY | DEPARTURE_UNCERTAIN | CANCELLED
    severity: str  # NOMINAL | MODERATE | HIGH
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_label: str  # HIGH | MEDIUM | LOW
    data_source: str
    is_demonstration: bool = True
    last_updated_utc: datetime


class FlightDetail(FlightSummary):
    """Full flight record including stage timeline, factor decomposition, and en-route context."""
    estimated_gate_departure_utc: datetime
    estimated_takeoff_utc: datetime
    taxi_time_minutes: float
    enroute_time_minutes: float
    inbound_aircraft_id: str | None = None
    inbound_flight_callsign: str | None = None
    inbound_arrival_utc: datetime | None = None
    inbound_delay_minutes: float = 0.0
    turnaround_buffer_minutes: float = 0.0
    factors: list[DelayContributingFactor] = Field(default_factory=list)
    timeline: list[OperationalStage] = Field(default_factory=list)
    departure_weather_summary: dict[str, Any] = Field(default_factory=dict)
    departure_queue_size: int = 0
    enroute_distance_km: float = 0.0
    enroute_corridor_congestion: str = "LOW"
    explanation_plain_english: str = ""
    alternative_flights: list[FlightSummary] = Field(default_factory=list)


class FlightSearchResponse(BaseModel):
    """Response containing matched flights and pagination metadata."""
    total: int
    page: int
    page_size: int
    flights: list[FlightSummary]


class AlternativeFlightsResponse(BaseModel):
    """Alternative flights on the same city pair and date for passenger re-accommodation."""
    flight_id: str
    origin_iata: str
    destination_iata: str
    alternatives: list[FlightSummary]


class DelayPredictionRequest(BaseModel):
    """What-if scenario parameter overrides for dynamic flight delay estimation."""
    flight_id: str
    inbound_delay_minutes: float | None = None
    departure_queue_additional: int = 0
    weather_severity_override: float | None = None  # 0.0 to 1.0
    slot_delay_minutes: float = 0.0


class DelayDashboardResponse(BaseModel):
    """Operations delay dashboard summary across monitored flights."""
    total_flights_monitored: int
    on_time_count: int
    moderate_delay_count: int
    significant_delay_count: int
    average_predicted_delay_min: float
    flights_requiring_attention: list[FlightSummary]
    factor_breakdown: dict[str, float]
    delay_distribution: list[dict[str, Any]]
    model_readiness: dict[str, Any]
    generated_at_utc: datetime


class DataReadinessReport(BaseModel):
    """Transparent gap analysis documenting current demonstration state vs real-world requirements."""
    status: str
    model_type: str
    current_capabilities: list[str]
    missing_operational_labels: list[dict[str, Any]]
    requirements_for_production: list[str]
    disclaimer: str

