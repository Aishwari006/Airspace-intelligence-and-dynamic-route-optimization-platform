"""Schemas for LLM Explanation Service."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ExplanationRequest(BaseModel):
    task: str = Field(
        default="route",
        description="Explanation task: 'route', 'weather', 'congestion', 'dataset', 'quality'",
    )
    context_data: dict[str, Any] = Field(
        ...,
        description="Verified computed evidence (route totals, cost factors, segments, QA report, etc.)",
    )
    user_query: str | None = Field(
        default=None,
        description="Optional specific question, e.g., 'Why was this route chosen over the shortest path?'",
    )


class ExplanationResponse(BaseModel):
    task: str
    summary: str
    supporting_facts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    engine: str = Field(..., description="'gemini-1.5-flash', 'phi3:latest', or 'deterministic_fallback'")
    model_name: str
    ollama_available: bool = Field(default=False, description="Kept for backward compatibility")
    gemini_available: bool = Field(default=False, description="True when Gemini API is active")
    latency_ms: float
    disclaimer: str = (
        "AI explanation of verified computations. Research prototype only - not an operational clearance."
    )

