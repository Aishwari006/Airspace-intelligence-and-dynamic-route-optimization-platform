"""LLM Explanation API routes."""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.explanation import ExplanationRequest, ExplanationResponse
from app.services.explanation.service import ExplanationService

router = APIRouter(prefix="/explain", tags=["Explainability"])
service = ExplanationService()


@router.post("", response_model=ExplanationResponse)
def generate_explanation(req: ExplanationRequest) -> ExplanationResponse:
    """Generates an evidence-grounded natural language explanation for routing, weather, or QA."""
    return service.explain(
        task=req.task,
        context=req.context_data,
        query=req.user_query,
    )


@router.get("/status")
def get_explanation_engine_status() -> dict[str, Any]:
    """Returns whether Google Gemini API is configured or if deterministic fallback is running."""
    settings = get_settings()
    gemini_up = service.check_gemini_availability()
    return {
        "gemini_available": gemini_up,
        "ollama_available": gemini_up,  # Kept for backward compatibility
        "active_engine": settings.gemini_model if gemini_up else "deterministic_fallback",
        "model_name": settings.gemini_model,
        "endpoint": settings.gemini_base_url,
        "ollama_url": settings.gemini_base_url,
    }

