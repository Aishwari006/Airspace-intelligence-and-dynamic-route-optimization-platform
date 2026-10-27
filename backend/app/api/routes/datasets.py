"""Dataset Studio API Routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.errors import NotFoundError, ValidationFailed
from app.schemas.datasets import (
    DatasetDetailResponse,
    DatasetRawPreviewResponse,
    DatasetSummaryItem,
    DatasetUploadResponse,
    LinkWeatherRequest,
    LinkWeatherResponse,
    RunPipelineRequest,
    RunPipelineResponse,
)
from app.services.datasets.manager import DatasetManager

router = APIRouter(prefix="/datasets", tags=["Dataset Studio"])
manager = DatasetManager()


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = Form("aircraft_observations"),
) -> DatasetUploadResponse:
    """Handles drag-and-drop or file-picker CSV upload with explicit dataset type."""
    if not file.filename:
        raise ValidationFailed("Uploaded file must have a filename.")
    fname_lower = file.filename.lower()
    if fname_lower.endswith((".parquet", ".pq")):
        raise ValidationFailed(
            "Direct upload of Parquet (.parquet, .pq) files is not supported. Only CSV files can be uploaded. "
            "Parquet is reserved for internal pipeline processing. Please export your dataset as CSV and upload again.",
            details={"filename": file.filename},
        )
    if not fname_lower.endswith(".csv"):
        raise ValidationFailed(
            "Only CSV files are supported for dataset upload. Please choose a .csv file.",
            details={"filename": file.filename},
        )

    content = await file.read()
    return manager.handle_upload(filename=file.filename, content=content, dataset_type=dataset_type)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    """Deletes an uploaded dataset, associated runs, and artifacts."""
    manager.delete_dataset(dataset_id)
    return {"status": "deleted", "dataset_id": dataset_id}


@router.get("", response_model=list[DatasetSummaryItem])
def list_datasets() -> list[DatasetSummaryItem]:
    """Lists all available datasets with their processing stage statuses."""
    return manager.list_datasets()


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset_details(dataset_id: str) -> DatasetDetailResponse:
    """Fetches comprehensive dataset profile, schema mapping, and run history."""
    return manager.get_dataset(dataset_id)


@router.get("/{dataset_id}/preview", response_model=DatasetRawPreviewResponse)
def preview_dataset(
    dataset_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=200),
) -> DatasetRawPreviewResponse:
    """Returns paginated rows from the raw uploaded CSV."""
    return manager.get_raw_preview(dataset_id, page=page, page_size=page_size)


@router.post("/{dataset_id}/pipeline", response_model=RunPipelineResponse)
def run_dataset_pipeline(dataset_id: str, req: RunPipelineRequest) -> RunPipelineResponse:
    """Executes the multi-stage quality, track segmentation, features, and inference pipeline."""
    return manager.execute_pipeline(
        dataset_id=dataset_id,
        stages=req.stages,
        custom_mapping=req.custom_mapping,
    )


@router.post("/{dataset_id}/link-weather", response_model=LinkWeatherResponse)
def link_weather_data(dataset_id: str, req: LinkWeatherRequest) -> LinkWeatherResponse:
    """Spatially and temporally links an aircraft observation dataset with a measured weather dataset."""
    return manager.link_traffic_and_weather(
        aircraft_dataset_id=dataset_id,
        weather_dataset_id=req.weather_dataset_id,
        spatial_tolerance_deg=req.spatial_tolerance_deg,
        temporal_tolerance_seconds=req.temporal_tolerance_seconds,
    )


@router.get("/{dataset_id}/export/{artifact_name}")
def download_dataset_artifact(dataset_id: str, artifact_name: str):
    """Downloads a raw CSV or processed Parquet artifact."""
    details = manager.get_dataset(dataset_id)
    path_str = details.available_artifacts.get(artifact_name)
    if not path_str:
        raise NotFoundError(f"Artifact '{artifact_name}' not found for dataset '{dataset_id}'")

    p = Path(path_str)
    if not p.exists():
        raise NotFoundError(f"File missing on server: {path_str}")

    return FileResponse(
        path=str(p),
        filename=p.name,
        media_type="application/octet-stream",
    )
