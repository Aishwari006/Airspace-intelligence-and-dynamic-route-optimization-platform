"""Explanation Service integrating Google Gemini API with deterministic fallback."""
from __future__ import annotations

import os
import time
from typing import Any
import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.explanation import ExplanationResponse
from app.services.explanation.prompts import SYSTEM_PROMPT, build_quality_prompt, build_route_prompt

logger = get_logger(__name__)


class ExplanationService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def get_api_key(self) -> str | None:
        """Resolves Gemini API key from settings or environment variables."""
        return (
            self.settings.gemini_api_key
            or os.environ.get("SKYFORCE_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )

    def check_gemini_availability(self) -> bool:
        """Returns True if a valid-length Gemini API key is configured."""
        key = self.get_api_key()
        return bool(key and len(key.strip()) > 8)

    def check_ollama_availability(self) -> bool:
        """Backward-compatible alias pointing to active AI availability."""
        return self.check_gemini_availability()

    def explain(self, task: str, context: dict[str, Any], query: str | None = None) -> ExplanationResponse:
        """Generates evidence-grounded natural language explanation with automatic fallback."""
        t0 = time.time()
        api_key = self.get_api_key()

        if api_key and len(api_key.strip()) > 8:
            try:
                if task == "route":
                    user_prompt = build_route_prompt(context, query)
                elif task in ["quality", "dataset"]:
                    user_prompt = build_quality_prompt(context)
                else:
                    user_prompt = f"Explain the following aviation intelligence data for task '{task}':\n{context}"

                model_name = self.settings.gemini_model or "gemini-1.5-flash"
                base_url = self.settings.gemini_base_url.rstrip("/")
                endpoint = f"{base_url}/models/{model_name}:generateContent?key={api_key.strip()}"

                payload = {
                    "system_instruction": {
                        "parts": [{"text": SYSTEM_PROMPT}]
                    },
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": user_prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024,
                    }
                }

                with httpx.Client(timeout=self.settings.gemini_timeout_s) as client:
                    resp = client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts).strip()
                            if text:
                                elapsed_ms = round((time.time() - t0) * 1000, 2)
                                lines = [
                                    line.strip("- *").strip()
                                    for line in text.split("\n")
                                    if line.strip().startswith(("-", "*", "1.", "2.", "3.", "•"))
                                ]
                                if not lines:
                                    lines = [s.strip() for s in text.split(". ") if len(s.strip()) > 15][:4]

                                return ExplanationResponse(
                                    task=task,
                                    summary=text.split("\n\n")[0],
                                    supporting_facts=lines[:5],
                                    limitations=[
                                        "Inference generated via Google Gemini API based on verified candidate airspace network.",
                                        "Research decision-support proxy; not an air traffic control clearance.",
                                    ],
                                    engine=model_name,
                                    model_name=f"Google {model_name}",
                                    ollama_available=True,
                                    gemini_available=True,
                                    latency_ms=elapsed_ms,
                                )
                    else:
                        logger.warning(
                            "Gemini API returned status %s (%s); engaging deterministic fallback.",
                            resp.status_code,
                            resp.text[:120],
                        )
            except Exception as exc:
                logger.warning("Gemini inference failed or timed out (%s); engaging deterministic fallback.", exc)

        # ------------------------------------------------ Deterministic Fallback Engine
        return self._deterministic_fallback(task, context, query, t0)

    def _deterministic_fallback(
        self, task: str, context: dict[str, Any], query: str | None, t0: float
    ) -> ExplanationResponse:
        """Rule-based factual synthesizer when LLM is offline, timed out, or using placeholder key."""
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        if task == "route":
            path = context.get("path", [])
            obj = context.get("objective", "balanced")
            totals = context.get("totals", {})
            base_tot = context.get("baseline_totals", {})
            dist = totals.get("distance_km", 0)
            base_dist = base_tot.get("distance_km", dist)
            eta = totals.get("estimated_time_min", 0)
            base_eta = base_tot.get("estimated_time_min", eta)
            wx_exp = totals.get("weather_exposure_km", 0)
            cg_exp = totals.get("congestion_exposure_km", 0)
            closed = context.get("closed_avoided", [])
            warnings = context.get("warnings", [])

            delta_d = round(dist - base_dist, 1)
            delta_t = round(eta - base_eta, 1)

            summary = (
                f"Candidate route {' -> '.join(path)} optimizes the '{obj}' objective "
                f"covering {dist:.0f} km in ~{eta:.0f} minutes across {totals.get('hops', len(path)-1)} segments."
            )

            facts = []
            if delta_d == 0:
                facts.append("Route matches the shortest geodesic candidate under active graph constraints.")
            elif delta_d > 0:
                facts.append(f"Trades off +{delta_d} km ({delta_d/base_dist*100:+.1f}%) in flight length to satisfy '{obj}' objective constraints.")
            else:
                facts.append(f"Saves {abs(delta_d)} km compared with unconstrained trajectory.")

            if delta_t != 0:
                facts.append(f"Estimated flight duration changes by {delta_t:+.1f} min relative to distance baseline.")

            if wx_exp > 0:
                facts.append(f"Traverses segments with {wx_exp:.1f} severity·km cumulative weather exposure.")
            else:
                facts.append("Route has zero observed convective weather hazard exposure.")

            if cg_exp > 0:
                facts.append(f"Traverses airspace with {cg_exp:.1f} score·km cumulative congestion exposure.")

            if closed:
                reasons = sorted({r for c in closed for r in c.get("reasons", [])})
                facts.append(f"Bypasses {len(closed)} closed segment(s) on direct path due to: {'; '.join(reasons)}.")

            # Missing data caveat
            if any("weather_missing" in str(w) for w in warnings):
                facts.append("Caution: Some segments lack weather observations. Missing data was penalized, but avoidance cannot be certified.")

            return ExplanationResponse(
                task="route",
                summary=summary,
                supporting_facts=facts,
                limitations=[
                    "Deterministic fallback engine active (Gemini API offline, rate-limited, or placeholder key).",
                    "Computed over research candidate network - not an operational clearance.",
                ],
                engine="deterministic_fallback",
                model_name="deterministic_rules_v1",
                ollama_available=False,
                gemini_available=False,
                latency_ms=elapsed_ms,
            )

        elif task in ["quality", "dataset"]:
            inp = context.get("input_rows", 0)
            ret = context.get("retained_rows", 0)
            drop = context.get("dropped", {})
            flag = context.get("flagged", {})

            summary = f"Data quality pipeline retained {ret:,} of {inp:,} surveillance records ({ret/max(inp,1)*100:.1f}% retention)."
            facts = [
                f"{drop.get('invalid_coordinate', 0)} coordinate errors and {drop.get('invalid_timestamp', 0)} timestamp errors excluded.",
                f"{drop.get('duplicate', 0)} duplicate observations removed.",
                f"{flag.get('implausible_speed', 0)} implausible ground speed records flagged and clamped to NaN.",
                f"{flag.get('missing_altitude', 0)} records flagged for missing altitude (retained without zero-imputation).",
            ]
            return ExplanationResponse(
                task=task,
                summary=summary,
                supporting_facts=facts,
                limitations=["Quality filtering based on civil aviation physical bounds."],
                engine="deterministic_fallback",
                model_name="deterministic_rules_v1",
                ollama_available=False,
                gemini_available=False,
                latency_ms=elapsed_ms,
            )

        return ExplanationResponse(
            task=task,
            summary=f"Analysis of {task} completed across verified data inputs.",
            supporting_facts=[f"Context keys analyzed: {', '.join(context.keys())}"],
            limitations=["Deterministic fallback analysis."],
            engine="deterministic_fallback",
            model_name="deterministic_rules_v1",
            ollama_available=False,
            gemini_available=False,
            latency_ms=elapsed_ms,
        )
