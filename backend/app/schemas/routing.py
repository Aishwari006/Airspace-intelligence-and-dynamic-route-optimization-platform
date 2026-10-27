"""Routing and optimization API schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class RouteWeightsInput(BaseModel):
    distance: float = 0.25
    time: float = 0.25
    weather: float = 0.20
    congestion: float = 0.20
    uncertainty: float = 0.10


class OptimizeRouteRequest(BaseModel):
    origin: str
    destination: str
    objective: str = "balanced"  # 'distance', 'fastest', 'weather', 'congestion', 'balanced', 'custom'
    algorithm: str = "dijkstra"  # 'dijkstra', 'astar'
    weights: RouteWeightsInput | None = None
    scenario_id: str | None = None
    departure_time_utc: datetime | None = None
    dataset_id: str | None = None
    weather_dataset_id: str | None = None
    run_id: str | None = None


class SegmentDetail(BaseModel):
    edge_id: str
    from_node: str
    to_node: str
    distance_km: float
    time_min: float
    time_source: str
    weather_severity: float | None = None
    weather_source: str
    congestion_score: float | None = None
    congestion_class: str | None = None
    congestion_source: str
    uncertainty: float
    closed: bool
    closure_reasons: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    cost_components: dict[str, float]
    weighted_cost: dict[str, float]
    cost_total: float


class RouteTotals(BaseModel):
    distance_km: float
    estimated_time_min: float
    weather_exposure_km: float
    congestion_exposure_km: float
    mean_uncertainty: float
    composite_score: float
    hops: int
    closed_edges_in_route: int


class RouteDetailResponse(BaseModel):
    found: bool
    label: str
    algorithm: str
    route_id: str | None = None
    dataset_id: str | None = "synthetic_demo_v1"
    weather_dataset_id: str | None = None
    run_id: str | None = None
    path: list[str] = Field(default_factory=list)
    geometry: list[list[float]] = Field(default_factory=list)
    totals: RouteTotals | None = None
    cost_by_factor: dict[str, float] = Field(default_factory=dict)
    segments: list[SegmentDetail] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    search: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    explanation: list[str] = Field(default_factory=list)
    disclaimer: str = "Research prototype estimate only. Not approved for operational flight planning."


class CompareRoutesResponse(BaseModel):
    origin: str
    destination: str
    scenario_id: str | None = None
    dataset_id: str | None = "synthetic_demo_v1"
    weather_dataset_id: str | None = None
    run_id: str | None = None
    selected_objective: str
    algorithm: str
    routes: list[RouteDetailResponse]
    closed_edges_avoided: list[dict[str, Any]] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)


class ExplainRouteRequest(BaseModel):
    origin: str
    destination: str
    objective: str = "balanced"
    algorithm: str = "dijkstra"
    scenario_id: str | None = None
    dataset_id: str | None = None
    weather_dataset_id: str | None = None
    run_id: str | None = None
    weights: RouteWeightsInput | None = None

