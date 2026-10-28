"""Coordinator service for Flight Delay Intelligence & Passenger-Centred Features."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from app.schemas.delay import (
    AlternativeFlightsResponse,
    DataReadinessReport,
    DelayDashboardResponse,
    DelayPredictionRequest,
    FlightDetail,
    FlightSearchResponse,
    FlightSummary,
    OperationalStage,
)
from app.services.delay.estimator import DemonstrationDelayEstimator
from app.services.delay.readiness import DelayDataReadinessAnalyzer
from app.services.delay.repository import FlightDelayRepository
from app.services.delay.timeline import OperationalTimelineBuilder


class FlightDelayService:
    """Service integrating flight repository, delay estimator, operational timeline, and dashboard."""

    def __init__(self) -> None:
        self.repo = FlightDelayRepository.get_instance()
        self.estimator = DemonstrationDelayEstimator()

    def _to_flight_summary(self, flight: dict[str, Any], estimate: dict[str, Any]) -> FlightSummary:
        sched_dep = flight["scheduled_departure_utc"]
        gate_delay = estimate["predicted_delay_minutes"]
        est_dep = sched_dep + timedelta(minutes=gate_delay)
        sched_arr = estimate.get("scheduled_arrival_utc")
        est_arr = estimate.get("estimated_arrival_utc")
        arr_delay = None
        if sched_arr and est_arr:
            arr_delay = round((est_arr - sched_arr).total_seconds() / 60.0, 1)

        return FlightSummary(
            flight_id=flight["flight_id"],
            flight_number=flight["flight_number"],
            callsign=flight["callsign"],
            airline=flight["airline"],
            aircraft_tail=flight["aircraft_tail"],
            origin_id=flight["origin_id"],
            origin_iata=flight["origin_iata"],
            origin_name=flight["origin_name"],
            destination_id=flight["destination_id"],
            destination_iata=flight["destination_iata"],
            destination_name=flight["destination_name"],
            scheduled_departure_utc=sched_dep,
            estimated_departure_utc=est_dep,
            scheduled_departure_display=sched_dep.strftime("%I:%M %p UTC"),
            estimated_departure_display=est_dep.strftime("%I:%M %p UTC"),
            scheduled_arrival_utc=sched_arr,
            estimated_arrival_utc=est_arr,
            scheduled_arrival_display=sched_arr.strftime("%I:%M %p UTC") if sched_arr else None,
            estimated_arrival_display=est_arr.strftime("%I:%M %p UTC") if est_arr else None,
            arrival_delay_minutes=arr_delay,
            predicted_delay_minutes=gate_delay,
            delay_interval_min=estimate["delay_interval_min"],
            delay_interval_max=estimate["delay_interval_max"],
            status=estimate["status"],
            severity=estimate["severity"],
            confidence_score=estimate["confidence_score"],
            confidence_label=estimate["confidence_label"],
            data_source="Demonstration Flight Rotation & Surveillance Ingestion",
            is_demonstration=True,
            last_updated_utc=datetime.now(timezone.utc),
        )

    def search_flights(
        self,
        query: str | None = None,
        date_str: str | None = None,
        origin: str | None = None,
        destination: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> FlightSearchResponse:
        total, raw_flights = self.repo.search_flights(query, date_str, origin, destination, limit, offset)
        summaries: list[FlightSummary] = []

        for f in raw_flights:
            weather = self.repo.get_weather_at_airport(f["origin_lat"], f["origin_lon"], f["scheduled_departure_utc"])
            queue_size = self.repo.get_departure_queue_size(f["origin_id"], f["scheduled_departure_utc"])
            cong = self.repo.get_corridor_congestion(f["origin_lat"], f["origin_lon"], f["scheduled_departure_utc"])
            est = self.estimator.estimate(f, weather, queue_size, cong)
            summaries.append(self._to_flight_summary(f, est))

        return FlightSearchResponse(
            total=total,
            page=(offset // limit) + 1,
            page_size=limit,
            flights=summaries,
        )

    def get_flight_detail(
        self,
        flight_id: str,
        inbound_delay_override: float | None = None,
        queue_additional: int = 0,
        weather_override: float | None = None,
        slot_delay_override: float = 0.0,
    ) -> FlightDetail | None:
        f = self.repo.get_flight(flight_id)
        if not f:
            return None

        weather = self.repo.get_weather_at_airport(f["origin_lat"], f["origin_lon"], f["scheduled_departure_utc"])
        queue_size = self.repo.get_departure_queue_size(f["origin_id"], f["scheduled_departure_utc"])
        cong = self.repo.get_corridor_congestion(f["origin_lat"], f["origin_lon"], f["scheduled_departure_utc"])

        est = self.estimator.estimate(
            flight=f,
            weather=weather,
            queue_size=queue_size,
            corridor_congestion=cong,
            inbound_delay_override=inbound_delay_override,
            queue_additional=queue_additional,
            weather_override=weather_override,
            slot_delay_override=slot_delay_override,
        )

        timeline = OperationalTimelineBuilder.build_timeline(f, est, weather)
        summary = self._to_flight_summary(f, est)

        # Get alternatives on same route
        raw_alts = self.repo.get_alternatives(f, limit=4)
        alts: list[FlightSummary] = []
        for alt_f in raw_alts:
            alt_wx = self.repo.get_weather_at_airport(alt_f["origin_lat"], alt_f["origin_lon"], alt_f["scheduled_departure_utc"])
            alt_q = self.repo.get_departure_queue_size(alt_f["origin_id"], alt_f["scheduled_departure_utc"])
            alt_cong = self.repo.get_corridor_congestion(alt_f["origin_lat"], alt_f["origin_lon"], alt_f["scheduled_departure_utc"])
            alt_est = self.estimator.estimate(alt_f, alt_wx, alt_q, alt_cong)
            alts.append(self._to_flight_summary(alt_f, alt_est))

        return FlightDetail(
            **summary.model_dump(),
            estimated_gate_departure_utc=est["estimated_gate_departure_utc"],
            estimated_takeoff_utc=est["estimated_takeoff_utc"],
            taxi_time_minutes=est["taxi_time_minutes"],
            enroute_time_minutes=f.get("enroute_eta_min", 75.0),
            inbound_aircraft_id=f.get("inbound_flight_id"),
            inbound_flight_callsign=f.get("inbound_callsign"),
            inbound_arrival_utc=f.get("inbound_arrival_utc"),
            inbound_delay_minutes=est["inbound_delay_minutes"],
            turnaround_buffer_minutes=est["turnaround_buffer_minutes"],
            factors=est["factors"],
            timeline=timeline,
            departure_weather_summary=weather,
            departure_queue_size=est["departure_queue_size"],
            enroute_distance_km=f["distance_km"],
            enroute_corridor_congestion=cong,
            explanation_plain_english=est["explanation_plain_english"],
            alternative_flights=alts,
        )

    def get_timeline(self, flight_id: str) -> list[OperationalStage] | None:
        detail = self.get_flight_detail(flight_id)
        return detail.timeline if detail else None

    def get_alternatives(self, flight_id: str) -> AlternativeFlightsResponse | None:
        f = self.repo.get_flight(flight_id)
        if not f:
            return None

        raw_alts = self.repo.get_alternatives(f, limit=5)
        alts: list[FlightSummary] = []
        for alt_f in raw_alts:
            alt_wx = self.repo.get_weather_at_airport(alt_f["origin_lat"], alt_f["origin_lon"], alt_f["scheduled_departure_utc"])
            alt_q = self.repo.get_departure_queue_size(alt_f["origin_id"], alt_f["scheduled_departure_utc"])
            alt_cong = self.repo.get_corridor_congestion(alt_f["origin_lat"], alt_f["origin_lon"], alt_f["scheduled_departure_utc"])
            alt_est = self.estimator.estimate(alt_f, alt_wx, alt_q, alt_cong)
            alts.append(self._to_flight_summary(alt_f, alt_est))

        return AlternativeFlightsResponse(
            flight_id=flight_id,
            origin_iata=f["origin_iata"],
            destination_iata=f["destination_iata"],
            alternatives=alts,
        )

    def predict_scenario(self, request: DelayPredictionRequest) -> FlightDetail | None:
        return self.get_flight_detail(
            flight_id=request.flight_id,
            inbound_delay_override=request.inbound_delay_minutes,
            queue_additional=request.departure_queue_additional,
            weather_override=request.weather_severity_override,
            slot_delay_override=request.slot_delay_minutes,
        )

    def get_dashboard(self, sample_size: int = 150) -> DelayDashboardResponse:
        """Aggregate high-level delay risks across a representative sample of flights."""
        self.repo.load()
        sample = self.repo.flights[:sample_size]

        on_time = 0
        moderate = 0
        significant = 0
        delays = []
        attention: list[FlightSummary] = []

        factor_sums = {
            "Inbound Delay": 0.0,
            "Turnaround Buffer": 0.0,
            "Weather Exposure": 0.0,
            "Runway Queue": 0.0,
            "Corridor Congestion": 0.0,
        }

        for f in sample:
            wx = self.repo.get_weather_at_airport(f["origin_lat"], f["origin_lon"], f["scheduled_departure_utc"])
            q = self.repo.get_departure_queue_size(f["origin_id"], f["scheduled_departure_utc"])
            cong = self.repo.get_corridor_congestion(f["origin_lat"], f["origin_lon"], f["scheduled_departure_utc"])
            est = self.estimator.estimate(f, wx, q, cong)
            d = est["predicted_delay_minutes"]
            delays.append(d)

            if d < 15.0:
                on_time += 1
            elif d < 45.0:
                moderate += 1
            else:
                significant += 1

            for fact in est["factors"]:
                if "Inbound" in fact.name:
                    factor_sums["Inbound Delay"] += fact.impact_minutes
                elif "Turnaround" in fact.name:
                    factor_sums["Turnaround Buffer"] += fact.impact_minutes
                elif "Weather" in fact.name:
                    factor_sums["Weather Exposure"] += fact.impact_minutes
                elif "Queue" in fact.name:
                    factor_sums["Runway Queue"] += fact.impact_minutes
                elif "Corridor" in fact.name or "Airspace" in fact.name:
                    factor_sums["Corridor Congestion"] += fact.impact_minutes

            if est["severity"] in ("MODERATE", "HIGH") and len(attention) < 15:
                attention.append(self._to_flight_summary(f, est))

        # Sort attention by delay descending
        attention.sort(key=lambda x: x.predicted_delay_minutes, reverse=True)

        avg_delay = round(sum(delays) / max(1, len(delays)), 1)

        # Build distribution bins
        dist_bins = [
            {"bracket": "< 15 min (On-Time)", "count": on_time, "percentage": round((on_time / len(sample)) * 100, 1)},
            {"bracket": "15 – 30 min (Moderate)", "count": sum(1 for d in delays if 15 <= d < 30), "percentage": round((sum(1 for d in delays if 15 <= d < 30) / len(sample)) * 100, 1)},
            {"bracket": "30 – 60 min (Extended)", "count": sum(1 for d in delays if 30 <= d < 60), "percentage": round((sum(1 for d in delays if 30 <= d < 60) / len(sample)) * 100, 1)},
            {"bracket": "≥ 60 min (Significant)", "count": sum(1 for d in delays if d >= 60), "percentage": round((sum(1 for d in delays if d >= 60) / len(sample)) * 100, 1)},
        ]

        readiness = {
            "mode": "DEMONSTRATION_ESTIMATOR",
            "fleet_size": len(self.repo.rotations_by_tail),
            "total_indexed_flights": len(self.repo.flights),
            "supervised_ml_delay_model_trained": False,
            "data_readiness_status": "DEMONSTRATION_READY",
        }

        return DelayDashboardResponse(
            total_flights_monitored=len(sample),
            on_time_count=on_time,
            moderate_delay_count=moderate,
            significant_delay_count=significant,
            average_predicted_delay_min=avg_delay,
            flights_requiring_attention=attention,
            factor_breakdown=factor_sums,
            delay_distribution=dist_bins,
            model_readiness=readiness,
            generated_at_utc=datetime.now(timezone.utc),
        )

    def get_readiness_report(self) -> DataReadinessReport:
        return DelayDataReadinessAnalyzer.get_report()

