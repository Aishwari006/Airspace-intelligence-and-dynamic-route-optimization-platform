"""Route optimization, alternatives, Pareto candidates, and explanations endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.state import get_state
from app.db import models as m
from app.db.session import get_db
from app.schemas.routing import (
    CompareRoutesResponse,
    ExplainRouteRequest,
    OptimizeRouteRequest,
    RouteDetailResponse,
)
from app.services.routing.cost_model import CostWeights
from app.services.datasets.manager import get_dataset_context_builder
from app.services.routing.explain import explain_route
from app.services.routing.planner import OBJECTIVE_NAMES, closed_edges_on_unconstrained
from app.services.scenario_engine.events import apply_events

router = APIRouter(prefix="/routes", tags=["Route Optimization"])


def _resolve_context(
    state,
    departure_time: datetime | None,
    scenario_id: str | None,
    db: Session,
    dataset_id: str | None = None,
    weather_dataset_id: str | None = None,
):  # noqa: ANN001
    t = departure_time or state.settings.default_scenario_time_utc
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    # Thread-safe dataset-specific context resolution:
    # Does NOT mutate shared global state.
    builder = get_dataset_context_builder(dataset_id, state.context_builder, weather_dataset_id=weather_dataset_id)
    base_ctx = builder.base_context(t)

    if not scenario_id:
        return base_ctx

    sc = db.scalar(select(m.Scenario).where(m.Scenario.id == scenario_id))
    if not sc or not sc.events:
        return base_ctx

    event_dicts = [
        {
            "id": e.id,
            "name": e.name,
            "event_type": e.event_type,
            "start_time_utc": e.start_time_utc,
            "end_time_utc": e.end_time_utc,
            "enabled": e.enabled,
            "scope": e.scope,
            "severity": e.severity,
            "source_status": e.source_status,
        }
        for e in sc.events
    ]
    return apply_events(base_ctx, event_dicts, state.graph, state.grid)


@router.post("/optimize", response_model=RouteDetailResponse)
def optimize_route(req: OptimizeRouteRequest, db: Session = Depends(get_db)) -> RouteDetailResponse:
    state = get_state()
    ctx = _resolve_context(
        state,
        req.departure_time_utc,
        req.scenario_id,
        db,
        dataset_id=req.dataset_id,
        weather_dataset_id=req.weather_dataset_id,
    )

    # Cost weights
    if req.weights:
        w = CostWeights.from_dict(req.weights.model_dump())
    else:
        w = CostWeights.preset(req.objective if req.objective in ["distance", "fastest", "weather", "congestion", "balanced"] else "balanced")

    res = state.planner.plan(
        origin=req.origin,
        destination=req.destination,
        w=w,
        ctx=ctx,
        algorithm=req.algorithm,
        label=req.objective,
    )

    if not res.get("found"):
        return RouteDetailResponse(
            found=False,
            label=req.objective,
            algorithm=req.algorithm,
            reason=res.get("reason"),
            explanation=[res.get("reason", "No route found."), "Research prototype estimate only."],
        )

    # Distance-only baseline comparison
    w_base = CostWeights.preset("distance")
    res_base = state.planner.plan(req.origin, req.destination, w_base, ctx, req.algorithm, label="distance")
    closed_on_direct = closed_edges_on_unconstrained(state.planner, req.origin, req.destination, ctx, w, req.algorithm)

    expl = explain_route(
        route=res,
        baseline=res_base if res_base.get("found") else None,
        ctx=ctx,
        objective_name=OBJECTIVE_NAMES.get(req.objective, req.objective),
        closed_on_direct=closed_on_direct,
    )

    return RouteDetailResponse(
        found=True,
        label=req.objective,
        algorithm=req.algorithm,
        route_id=res["route_id"],
        dataset_id=req.dataset_id or "synthetic_demo_v1",
        weather_dataset_id=req.weather_dataset_id,
        run_id=req.run_id,
        path=res["path"],
        geometry=res["geometry"],
        totals=res["totals"],
        cost_by_factor=res["cost_by_factor"],
        segments=res["segments"],
        provenance=res["provenance"],
        search=res["search"],
        explanation=expl,
    )


@router.post("/compare", response_model=CompareRoutesResponse)
def compare_routes(req: OptimizeRouteRequest, db: Session = Depends(get_db)) -> CompareRoutesResponse:
    state = get_state()
    ctx = _resolve_context(
        state,
        req.departure_time_utc,
        req.scenario_id,
        db,
        dataset_id=req.dataset_id,
        weather_dataset_id=req.weather_dataset_id,
    )

    if req.weights:
        selected_w = CostWeights.from_dict(req.weights.model_dump())
    else:
        selected_w = CostWeights.preset(req.objective if req.objective in ["distance", "fastest", "weather", "congestion", "balanced"] else "balanced")

    # Generate candidate alternatives
    alts = state.planner.alternatives(
        origin=req.origin,
        destination=req.destination,
        selected_w=selected_w,
        objective=req.objective,
        ctx=ctx,
        algorithm=req.algorithm,
    )

    closed_avoided = closed_edges_on_unconstrained(state.planner, req.origin, req.destination, ctx, selected_w, req.algorithm)

    # Baseline for explanation
    baseline_alt = next((a for a in alts if "distance" in a.get("labels", [])), alts[0] if alts else None)
    best_alt = alts[0] if alts else None

    expl = []
    if best_alt:
        expl = explain_route(
            route=best_alt,
            baseline=baseline_alt,
            ctx=ctx,
            objective_name=OBJECTIVE_NAMES.get(req.objective, req.objective),
            closed_on_direct=closed_avoided,
        )

    route_details = [
        RouteDetailResponse(
            found=True,
            label=", ".join(a.get("labels", [req.objective])),
            algorithm=req.algorithm,
            route_id=a["route_id"],
            dataset_id=req.dataset_id or "synthetic_demo_v1",
            weather_dataset_id=req.weather_dataset_id,
            run_id=req.run_id,
            path=a["path"],
            geometry=a["geometry"],
            totals=a["totals"],
            cost_by_factor=a["cost_by_factor"],
            segments=a["segments"],
            provenance=a["provenance"],
            search=a["search"],
            explanation=explain_route(
                route=a,
                baseline=baseline_alt,
                ctx=ctx,
                objective_name=a.get("label_names", [a.get("label", "")])[0],
                closed_on_direct=closed_avoided,
            ),
        )
        for a in alts
    ]

    return CompareRoutesResponse(
        origin=req.origin,
        destination=req.destination,
        scenario_id=req.scenario_id,
        dataset_id=req.dataset_id or "synthetic_demo_v1",
        weather_dataset_id=req.weather_dataset_id,
        run_id=req.run_id,
        selected_objective=req.objective,
        algorithm=req.algorithm,
        routes=route_details,
        closed_edges_avoided=closed_avoided,
        explanation=expl,
    )


@router.get("/pareto")
def get_pareto_routes(
    origin: str = Query(..., description="Origin node e.g. VIDP"),
    destination: str = Query(..., description="Destination node e.g. VABB"),
    scenario_id: str | None = Query(None),
    dataset_id: str | None = Query(None),
    algorithm: str = Query("dijkstra"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    state = get_state()
    ctx = _resolve_context(state, None, scenario_id, db, dataset_id=dataset_id)
    return state.planner.pareto(origin, destination, ctx, algorithm=algorithm)


@router.post("/explain")
def explain_candidate_route(req: ExplainRouteRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    state = get_state()
    ctx = _resolve_context(
        state,
        None,
        req.scenario_id,
        db,
        dataset_id=req.dataset_id,
        weather_dataset_id=req.weather_dataset_id,
    )
    w = CostWeights.from_dict(req.weights.model_dump()) if req.weights else CostWeights.preset(req.objective if req.objective in ["distance", "fastest", "weather", "congestion", "balanced"] else "balanced")

    res = state.planner.plan(req.origin, req.destination, w, ctx, req.algorithm, label=req.objective)
    w_base = CostWeights.preset("distance")
    res_base = state.planner.plan(req.origin, req.destination, w_base, ctx, req.algorithm, label="distance")
    closed = closed_edges_on_unconstrained(state.planner, req.origin, req.destination, ctx, w, req.algorithm)

    expl = explain_route(res, res_base if res_base.get("found") else None, ctx, OBJECTIVE_NAMES.get(req.objective, req.objective), closed)
    return {
        "route_id": res.get("route_id"),
        "explanation": expl,
        "dataset_id": req.dataset_id or "synthetic_demo_v1",
        "weather_dataset_id": req.weather_dataset_id,
    }

