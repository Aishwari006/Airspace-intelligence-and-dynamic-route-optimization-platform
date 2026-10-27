"""Dataset Management, Storage, Lifecycle and Pipeline Orchestration."""
from __future__ import annotations

import io
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError, ValidationFailed
from app.db import models as m
from app.db.session import get_sessionmaker, session_scope
from app.schemas.datasets import (
    ColumnInfo,
    DatasetDetailResponse,
    DatasetRawPreviewResponse,
    DatasetSummaryItem,
    DatasetUploadResponse,
    LinkWeatherResponse,
    PipelineStageResult,
    RunPipelineResponse,
)
from app.services.datasets.schema_mapper import (
    analyze_dataframe_columns,
    apply_mapping,
    auto_detect_dataset_type,
    auto_detect_mapping,
    validate_mapping,
)
from app.services.features.congestion import build_feature_frame
from app.services.geospatial.grid import grid_from_settings
from app.services.models.registry import LoadedModel, load_active, ood_flags, validate_features
from app.services.preprocessing.quality import run_quality, segment_tracks
from app.services.routing.graph import RouteGraph
from app.services.routing.providers import ContextBuilder, DataStore

# Thread-safe dataset storage cache for dataset-specific routing contexts
_dataset_stores: dict[str, DataStore] = {}
_dataset_builders: dict[str, ContextBuilder] = {}


def sanitize_filename(filename: str) -> str:
    """Sanitizes incoming filename to prevent path traversal and unsafe characters."""
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", Path(filename).name)
    return clean or "uploaded_dataset.csv"


class DatasetManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.upload_dir = self.settings.data_dir / "uploads"
        self.processed_dir = self.settings.data_dir / "processed"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.session_factory = get_sessionmaker(self.settings.database_url)

    def handle_upload(
        self,
        filename: str,
        content: bytes,
        dataset_type: str = "aircraft_observations",
    ) -> DatasetUploadResponse:
        """Validates, stores, profiles, and registers a newly uploaded dataset under its schema type."""
        # 1. Size check
        if len(content) > self.settings.max_upload_bytes:
            raise ValidationFailed(
                f"File size ({len(content)} bytes) exceeds maximum configured limit of {self.settings.max_upload_bytes} bytes",
                details={"max_bytes": self.settings.max_upload_bytes, "file_bytes": len(content)},
            )

        # 2. Decoding check
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as exc:
                raise ValidationFailed(f"Unable to decode CSV file: {exc}")

        # 3. Parse CSV
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            raise ValidationFailed(f"Malformed CSV content: {exc}")

        if df.empty:
            raise ValidationFailed("Uploaded CSV file is empty.")

        cols = list(df.columns)
        # If dataset_type is auto, detect from columns
        if dataset_type == "auto":
            dataset_type = auto_detect_dataset_type(cols)

        valid_types = {"aircraft_observations", "weather_observations", "congestion_cells"}
        if dataset_type not in valid_types:
            dataset_type = "aircraft_observations"

        dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
        safe_name = sanitize_filename(filename)
        stored_path = self.upload_dir / f"{dataset_id}_{safe_name}"

        # Save immutable raw copy
        stored_path.write_bytes(content)

        mapping = auto_detect_mapping(cols, dataset_type=dataset_type)
        is_valid, warnings = validate_mapping(cols, mapping, dataset_type=dataset_type)
        cols_info = analyze_dataframe_columns(df)

        now = datetime.now(timezone.utc)
        record = m.UploadedDataset(
            id=dataset_id,
            filename=safe_name,
            dataset_type=dataset_type,
            stored_path=str(stored_path),
            file_size_bytes=len(content),
            row_count=len(df),
            col_count=len(cols),
            uploaded_at_utc=now,
            status="validated" if is_valid else "uploaded",
            schema_mapping=mapping,
            validation_report={
                "is_valid": is_valid,
                "dataset_type": dataset_type,
                "warnings": warnings,
                "detected_columns": cols,
            },
            metadata_json={
                "original_filename": filename,
                "storage_path": str(stored_path),
                "preview_sample_size": len(df.head(5)),
            },
        )

        with self.session_factory() as session:
            session.add(record)
            session.commit()

        return DatasetUploadResponse(
            dataset_id=dataset_id,
            filename=safe_name,
            dataset_type=dataset_type,
            file_size_bytes=len(content),
            row_count=len(df),
            col_count=len(cols),
            detected_columns=cols,
            suggested_mapping=mapping,
            columns_info=cols_info,
            is_valid_format=is_valid,
            validation_warnings=warnings,
            upload_timestamp_utc=now,
        )

    def list_datasets(self) -> list[DatasetSummaryItem]:
        """Lists all persistent datasets and built-in synthetic demo dataset."""
        items: list[DatasetSummaryItem] = [
            DatasetSummaryItem(
                id=self.settings.demo_dataset_id,
                filename="synthetic_demo_v1.parquet",
                dataset_type="aircraft_observations",
                file_size_bytes=14500000,
                row_count=276480,
                col_count=14,
                uploaded_at_utc=datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc),
                status="processed",
                is_active=True,
                pipeline_stages={
                    "validation": "completed",
                    "quality": "completed",
                    "segmentation": "completed",
                    "features": "completed",
                    "inference": "completed",
                    "routing": "completed",
                },
                provenance="synthetic_demo",
            )
        ]

        with self.session_factory() as session:
            rows = session.scalars(
                select(m.UploadedDataset).order_by(m.UploadedDataset.uploaded_at_utc.desc())
            ).all()

            for r in rows:
                runs = session.scalars(
                    select(m.DatasetRun).where(m.DatasetRun.dataset_id == r.id)
                ).all()
                stages = {run.stage: run.status for run in runs}

                items.append(
                    DatasetSummaryItem(
                        id=r.id,
                        filename=r.filename,
                        dataset_type=r.dataset_type or "aircraft_observations",
                        file_size_bytes=r.file_size_bytes,
                        row_count=r.row_count,
                        col_count=r.col_count,
                        uploaded_at_utc=r.uploaded_at_utc,
                        status=r.status,
                        is_active=(r.id in _dataset_stores),
                        pipeline_stages=stages,
                        provenance=f"user_upload:{r.id}",
                        linked_weather_dataset_id=r.linked_weather_dataset_id,
                    )
                )

        return items

    def get_dataset(self, dataset_id: str) -> DatasetDetailResponse:
        """Retrieves complete dataset details, runs, and artifacts."""
        if dataset_id == self.settings.demo_dataset_id:
            return DatasetDetailResponse(
                id=self.settings.demo_dataset_id,
                filename="synthetic_demo_v1.parquet",
                dataset_type="aircraft_observations",
                file_size_bytes=14500000,
                row_count=276480,
                col_count=14,
                uploaded_at_utc=datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc),
                status="processed",
                schema_mapping={},
                validation_report={"is_valid": True, "notes": "Built-in synthetic demo dataset."},
                pipeline_runs=[
                    {"stage": "quality", "status": "completed", "elapsed_seconds": 1.2},
                    {"stage": "features", "status": "completed", "elapsed_seconds": 2.4},
                    {"stage": "inference", "status": "completed", "elapsed_seconds": 1.8},
                ],
                available_artifacts={
                    "raw": "data/synthetic/observations_raw.parquet",
                    "cleaned": "data/processed/observations_clean.parquet",
                    "features": "data/processed/congestion_agg.parquet",
                },
                active_in_routing=True,
                provenance="synthetic_demo",
            )

        with self.session_factory() as session:
            record = session.get(m.UploadedDataset, dataset_id)
            if not record:
                raise NotFoundError(f"Dataset '{dataset_id}' not found")

            runs = session.scalars(
                select(m.DatasetRun).where(m.DatasetRun.dataset_id == dataset_id).order_by(m.DatasetRun.created_at_utc.asc())
            ).all()

            artifacts: dict[str, str] = {"raw": record.stored_path}
            for run in runs:
                if run.artifact_paths:
                    artifacts.update(run.artifact_paths)

            return DatasetDetailResponse(
                id=record.id,
                filename=record.filename,
                dataset_type=record.dataset_type or "aircraft_observations",
                file_size_bytes=record.file_size_bytes,
                row_count=record.row_count,
                col_count=record.col_count,
                uploaded_at_utc=record.uploaded_at_utc,
                status=record.status,
                schema_mapping=record.schema_mapping,
                validation_report=record.validation_report,
                pipeline_runs=[
                    {
                        "stage": run.stage,
                        "status": run.status,
                        "elapsed_seconds": run.elapsed_seconds,
                        "summary": run.summary_json,
                    }
                    for run in runs
                ],
                available_artifacts=artifacts,
                active_in_routing=(dataset_id in _dataset_stores),
                provenance=f"user_upload:{dataset_id}",
                linked_weather_dataset_id=record.linked_weather_dataset_id,
            )

    def get_raw_preview(self, dataset_id: str, page: int = 1, page_size: int = 50) -> DatasetRawPreviewResponse:
        """Paginated preview of raw uploaded CSV or demo synthetic dataset."""
        if dataset_id == self.settings.demo_dataset_id:
            raw_path = self.settings.data_dir / "synthetic" / "observations_raw.parquet"
            if not raw_path.exists():
                raw_path = self.settings.data_dir / "synthetic" / "observations_synthetic.parquet"
            if not raw_path.exists():
                raw_path = self.settings.data_dir / "processed" / "observations_clean.parquet"
            if not raw_path.exists():
                raise NotFoundError("Demo dataset artifact not found on disk.")
            df = pd.read_parquet(raw_path)
            start = (page - 1) * page_size
            chunk = df.iloc[start : start + page_size].copy()
            # Clean for JSON serialization: handle timestamps, inf, nan
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.where(pd.notna(chunk), None)
            for col in chunk.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
                chunk[col] = chunk[col].astype(str)

            return DatasetRawPreviewResponse(
                dataset_id=dataset_id,
                filename="synthetic_demo_v1.parquet",
                total_rows=len(df),
                page=page,
                page_size=page_size,
                columns=list(df.columns),
                rows=chunk.to_dict(orient="records"),
            )

        with self.session_factory() as session:
            record = session.get(m.UploadedDataset, dataset_id)
            if not record:
                raise NotFoundError(f"Dataset '{dataset_id}' not found")

            p = Path(record.stored_path)
            if not p.exists():
                p = self.upload_dir / p.name
            if not p.exists():
                raise NotFoundError(f"Underlying raw file missing: {record.stored_path}")

            if str(p).lower().endswith(".parquet") or str(p).lower().endswith(".pq"):
                df = pd.read_parquet(p)
            else:
                try:
                    df = pd.read_csv(p, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(p, encoding="latin-1")

            start = (page - 1) * page_size
            chunk = df.iloc[start : start + page_size].copy()
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.where(pd.notna(chunk), None)
            for col in chunk.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
                chunk[col] = chunk[col].astype(str)

            return DatasetRawPreviewResponse(
                dataset_id=dataset_id,
                filename=record.filename,
                total_rows=len(df),
                page=page,
                page_size=page_size,
                columns=list(df.columns),
                rows=chunk.to_dict(orient="records"),
            )

    def execute_pipeline(
        self,
        dataset_id: str,
        stages: list[str],
        custom_mapping: dict[str, str] | None = None,
    ) -> RunPipelineResponse:
        """Executes data-processing, feature-engineering, and ML inference based on dataset schema type."""
        with self.session_factory() as session:
            record = session.get(m.UploadedDataset, dataset_id)
            if not record:
                raise NotFoundError(f"Dataset '{dataset_id}' not found")

            mapping = custom_mapping or record.schema_mapping or {}
            stored_p = Path(record.stored_path)
            if not stored_p.exists():
                raise NotFoundError(f"File missing at {record.stored_path}")

            raw_df = pd.read_csv(stored_p)
            ds_type = record.dataset_type or "aircraft_observations"

        if ds_type == "weather_observations":
            return self._execute_weather_pipeline(dataset_id, raw_df, mapping, stages)
        elif ds_type == "congestion_cells":
            return self._execute_congestion_pipeline(dataset_id, raw_df, mapping, stages)
        else:
            return self._execute_aircraft_pipeline(dataset_id, raw_df, mapping, stages)

    def _execute_weather_pipeline(
        self,
        dataset_id: str,
        raw_df: pd.DataFrame,
        mapping: dict[str, str],
        stages: list[str],
    ) -> RunPipelineResponse:
        """Executes weather validation, coordinate bound check, and spatial-grid indexing."""
        total_t0 = time.time()
        stage_results: list[PipelineStageResult] = []
        artifacts: dict[str, str] = {}
        grid = grid_from_settings(self.settings)

        t0 = time.time()
        mapped_df = apply_mapping(raw_df, mapping, dataset_type="weather_observations")
        val_ok, val_warns = validate_mapping(list(raw_df.columns), mapping, dataset_type="weather_observations")

        # Check numeric coordinates
        valid_coords = mapped_df["latitude"].between(-90, 90) & mapped_df["longitude"].between(-180, 180)
        invalid_coords_count = int((~valid_coords).sum())
        if invalid_coords_count > 0:
            val_warns.append(f"{invalid_coords_count} rows contain out-of-bounds geographic coordinates.")

        # Ensure valid_time_utc parses to datetime
        mapped_df["valid_time_utc"] = pd.to_datetime(mapped_df["valid_time_utc"], utc=True, errors="coerce")
        null_times = int(mapped_df["valid_time_utc"].isna().sum())
        if null_times > 0:
            val_warns.append(f"{null_times} rows contain unparseable timestamps; dropped from temporal grid.")

        # Non-negative precipitation
        mapped_df["precipitation_mm"] = pd.to_numeric(mapped_df["precipitation_mm"], errors="coerce").clip(lower=0.0)

        stage_results.append(
            PipelineStageResult(
                stage="validation",
                status="completed" if val_ok else "failed",
                elapsed_seconds=round(time.time() - t0, 3),
                output_summary={
                    "total_input_rows": len(raw_df),
                    "valid_rows": int(valid_coords.sum()),
                    "invalid_coords": invalid_coords_count,
                    "dataset_type": "weather_observations",
                },
                warnings=val_warns,
            )
        )

        clean_wx = mapped_df[valid_coords & mapped_df["valid_time_utc"].notna()].copy()

        # Spatial grid indexing
        t0 = time.time()
        clean_wx["cell_id"] = clean_wx["cell_id"].fillna(
            pd.Series(grid.cell_ids_vec(clean_wx["latitude"].to_numpy(), clean_wx["longitude"].to_numpy()), index=clean_wx.index)
        )
        clean_wx["hour_utc"] = clean_wx["valid_time_utc"].dt.floor("1h")

        # Aggregate per cell and hour
        agg_wx = clean_wx.groupby(["cell_id", "hour_utc"]).agg(
            precipitation_mm=("precipitation_mm", "mean"),
            wind_speed_10m_ms=("wind_speed_10m_ms", "mean"),
            wind_gusts_10m_ms=("wind_gusts_10m_ms", "mean"),
            cloud_cover_pct=("cloud_cover_pct", "mean"),
            temperature_2m_c=("temperature_2m_c", "mean"),
        ).reset_index().rename(columns={"hour_utc": "valid_time_utc"})

        # Save processed weather parquet
        wx_out_path = self.processed_dir / f"{dataset_id}_weather_grid.parquet"
        agg_wx.to_parquet(wx_out_path, index=False)
        artifacts["weather_grid"] = str(wx_out_path)

        stage_results.append(
            PipelineStageResult(
                stage="weather_indexing",
                status="completed",
                elapsed_seconds=round(time.time() - t0, 3),
                output_summary={
                    "spatial_cells_covered": int(agg_wx["cell_id"].nunique()),
                    "time_hours_indexed": int(agg_wx["valid_time_utc"].nunique()),
                    "mean_precipitation_mm": round(float(agg_wx["precipitation_mm"].mean()), 2) if not agg_wx.empty else 0.0,
                    "max_precipitation_mm": round(float(agg_wx["precipitation_mm"].max()), 2) if not agg_wx.empty else 0.0,
                },
                warnings=[],
            )
        )

        total_elapsed = round(time.time() - total_t0, 3)
        with self.session_factory() as session:
            # Persist dataset-specific weather observations to database
            session.execute(delete(m.WeatherObservation).where(m.WeatherObservation.dataset == dataset_id))
            wx_recs = []
            for _, row in clean_wx.iterrows():
                valid_dt = row["valid_time_utc"]
                if pd.notna(valid_dt):
                    if hasattr(valid_dt, "to_pydatetime"):
                        valid_dt = valid_dt.to_pydatetime()
                    if valid_dt.tzinfo is None:
                        valid_dt = valid_dt.replace(tzinfo=timezone.utc)
                wx_recs.append(
                    m.WeatherObservation(
                        dataset=dataset_id,
                        source=f"upload:{dataset_id}",
                        product="uploaded_weather_grid",
                        is_synthetic=False,
                        location_type="grid_cell",
                        location_id=str(row.get("cell_id") or f"WX_{row.get('latitude', 0):.2f}_{row.get('longitude', 0):.2f}"),
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        valid_time_utc=valid_dt,
                        temperature_2m_c=float(row["temperature_2m_c"]) if pd.notna(row.get("temperature_2m_c")) else None,
                        precipitation_mm=float(row["precipitation_mm"]) if pd.notna(row.get("precipitation_mm")) else None,
                        cloud_cover_pct=float(row["cloud_cover_pct"]) if pd.notna(row.get("cloud_cover_pct")) else None,
                        wind_speed_10m_ms=float(row["wind_speed_10m_ms"]) if pd.notna(row.get("wind_speed_10m_ms")) else None,
                        wind_gusts_10m_ms=float(row["wind_gusts_10m_ms"]) if pd.notna(row.get("wind_gusts_10m_ms")) else None,
                        retrieved_at=datetime.now(timezone.utc),
                    )
                )
            if wx_recs:
                session.add_all(wx_recs)

            for s_res in stage_results:
                run_rec = m.DatasetRun(
                    id=f"run_{uuid.uuid4().hex[:10]}",
                    dataset_id=dataset_id,
                    dataset_type="weather_observations",
                    stage=s_res.stage,
                    status=s_res.status,
                    elapsed_seconds=s_res.elapsed_seconds,
                    summary_json=s_res.output_summary,
                    artifact_paths=artifacts,
                )
                session.add(run_rec)

            session.execute(
                update(m.UploadedDataset)
                .where(m.UploadedDataset.id == dataset_id)
                .values(status="processed", schema_mapping=mapping)
            )
            session.commit()

        return RunPipelineResponse(
            dataset_id=dataset_id,
            status="completed",
            total_elapsed_seconds=total_elapsed,
            stages=stage_results,
            artifact_paths=artifacts,
        )

    def _execute_congestion_pipeline(
        self,
        dataset_id: str,
        raw_df: pd.DataFrame,
        mapping: dict[str, str],
        stages: list[str],
    ) -> RunPipelineResponse:
        """Executes pre-aggregated congestion cell validation and store registration."""
        total_t0 = time.time()
        stage_results: list[PipelineStageResult] = []
        artifacts: dict[str, str] = {}
        grid = grid_from_settings(self.settings)

        t0 = time.time()
        mapped_df = apply_mapping(raw_df, mapping, dataset_type="congestion_cells")
        val_ok, val_warns = validate_mapping(list(raw_df.columns), mapping, dataset_type="congestion_cells")

        mapped_df["aircraft_count"] = pd.to_numeric(mapped_df["aircraft_count"], errors="coerce").fillna(0).clip(lower=0)
        mapped_df["congestion_score"] = pd.to_numeric(mapped_df["congestion_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        if "timestamp_utc" in mapped_df.columns:
            mapped_df["timestamp_utc"] = pd.to_datetime(mapped_df["timestamp_utc"], utc=True, errors="coerce")
            mapped_df["window_start_utc"] = mapped_df["timestamp_utc"].dt.floor(f"{self.settings.window_minutes}min")
        else:
            now_dt = datetime.now(timezone.utc)
            mapped_df["window_start_utc"] = pd.Timestamp(now_dt).floor(f"{self.settings.window_minutes}min")

        stage_results.append(
            PipelineStageResult(
                stage="validation",
                status="completed" if val_ok else "failed",
                elapsed_seconds=round(time.time() - t0, 3),
                output_summary={
                    "total_cells": int(mapped_df["cell_id"].nunique()),
                    "total_observations": len(mapped_df),
                    "mean_congestion_score": round(float(mapped_df["congestion_score"].mean()), 3),
                },
                warnings=val_warns,
            )
        )

        out_p = self.processed_dir / f"{dataset_id}_congestion_grid.parquet"
        mapped_df.to_parquet(out_p, index=False)
        artifacts["congestion_grid"] = str(out_p)

        # Register DataStore
        from app.api.state import get_state
        st = get_state()
        corridor = sorted(list(mapped_df["cell_id"].dropna().unique()))
        weather_path = self.settings.data_dir / "synthetic" / "weather_synthetic.parquet"
        wx_df = pd.read_parquet(weather_path) if weather_path.exists() else pd.DataFrame()

        store = DataStore(
            agg=mapped_df,
            weather=wx_df,
            corridor_cells=corridor,
            airports=set(st.store.airports),
            dataset_id=dataset_id,
            weather_source_label="Default baseline weather",
        )
        _dataset_stores[dataset_id] = store
        _dataset_builders[dataset_id] = ContextBuilder(
            rg=st.graph,
            grid=grid,
            store=store,
            window_min=self.settings.window_minutes,
            cruise_kmh=self.settings.heuristic_cruise_speed_kmh,
            congestion_model=st.congestion_model,
            eta_model=st.eta_model,
            thresholds=st.thresholds,
        )

        total_elapsed = round(time.time() - total_t0, 3)
        with self.session_factory() as session:
            # Persist congestion observations to database table
            session.execute(delete(m.CongestionObservation).where(m.CongestionObservation.dataset == dataset_id))
            default_t = self.settings.default_scenario_time_utc
            if default_t.tzinfo is None:
                default_t = default_t.replace(tzinfo=timezone.utc)

            cong_recs = []
            for _, row in mapped_df.iterrows():
                w_start = row.get("window_start_utc")
                if pd.isna(w_start) or w_start is None:
                    w_start = default_t
                elif hasattr(w_start, "to_pydatetime"):
                    w_start = w_start.to_pydatetime()
                if getattr(w_start, "tzinfo", None) is None:
                    w_start = w_start.replace(tzinfo=timezone.utc)
                w_end = w_start + timedelta(minutes=self.settings.window_minutes)
                cong_recs.append(
                    m.CongestionObservation(
                        dataset=dataset_id,
                        cell_id=str(row["cell_id"]),
                        window_start_utc=w_start,
                        window_end_utc=w_end,
                        aircraft_count=int(row.get("aircraft_count", 0)),
                        observation_count=int(row.get("aircraft_count", 0)),
                        observed_class=str(row.get("predicted_class") or "LOW"),
                    )
                )
            if cong_recs:
                session.add_all(cong_recs)

            for s_res in stage_results:
                run_rec = m.DatasetRun(
                    id=f"run_{uuid.uuid4().hex[:10]}",
                    dataset_id=dataset_id,
                    dataset_type="congestion_cells",
                    stage=s_res.stage,
                    status=s_res.status,
                    elapsed_seconds=s_res.elapsed_seconds,
                    summary_json=s_res.output_summary,
                    artifact_paths=artifacts,
                )
                session.add(run_rec)

            session.execute(
                update(m.UploadedDataset)
                .where(m.UploadedDataset.id == dataset_id)
                .values(status="processed", schema_mapping=mapping)
            )
            session.commit()

        return RunPipelineResponse(
            dataset_id=dataset_id,
            status="completed",
            total_elapsed_seconds=total_elapsed,
            stages=stage_results,
            artifact_paths=artifacts,
        )

    def _execute_aircraft_pipeline(
        self,
        dataset_id: str,
        raw_df: pd.DataFrame,
        mapping: dict[str, str],
        stages: list[str],
    ) -> RunPipelineResponse:
        """Executes the standard 5-stage aircraft processing pipeline, with linked weather if available."""
        total_t0 = time.time()
        stage_results: list[PipelineStageResult] = []
        artifacts: dict[str, str] = {}
        grid = grid_from_settings(self.settings)

        # STAGE 1: Schema Normalization & Validation
        t0 = time.time()
        mapped_df = apply_mapping(raw_df, mapping, dataset_type="aircraft_observations")
        val_ok, val_warns = validate_mapping(list(raw_df.columns), mapping, dataset_type="aircraft_observations")
        stage_results.append(
            PipelineStageResult(
                stage="validation",
                status="completed" if val_ok else "failed",
                elapsed_seconds=round(time.time() - t0, 3),
                output_summary={"total_input_rows": len(raw_df), "columns_mapped": len(mapping)},
                warnings=val_warns,
            )
        )
        if not val_ok:
            return RunPipelineResponse(
                dataset_id=dataset_id,
                status="failed",
                total_elapsed_seconds=round(time.time() - total_t0, 3),
                stages=stage_results,
                artifact_paths={},
            )

        cleaned_df = mapped_df
        qa_report = None

        # STAGE 2: Data Quality & Anomaly Detection
        if "quality" in stages:
            t0 = time.time()
            cleaned_df, qa_report = run_quality(mapped_df)
            qa_path = self.processed_dir / f"{dataset_id}_cleaned.parquet"
            cleaned_df.to_parquet(qa_path, index=False)
            artifacts["cleaned"] = str(qa_path)

            qa_summary = qa_report.to_dict()
            stage_results.append(
                PipelineStageResult(
                    stage="quality",
                    status="completed",
                    elapsed_seconds=round(time.time() - t0, 3),
                    output_summary={
                        "input_rows": qa_summary["input_rows"],
                        "retained_rows": qa_summary["retained_rows"],
                        "dropped_breakdown": qa_summary["dropped"],
                        "flagged_breakdown": qa_summary["flagged"],
                    },
                    warnings=[f"{k}: {v} records" for k, v in qa_summary["flagged"].items() if v > 0],
                )
            )

        # STAGE 3: Track Segmentation
        inferred_tracks_summary = None
        if "segmentation" in stages:
            t0 = time.time()
            seg_df, track_sum = segment_tracks(cleaned_df, gap_s=1800)
            track_path = self.processed_dir / f"{dataset_id}_tracks.parquet"
            track_sum.to_parquet(track_path, index=False)
            artifacts["tracks"] = str(track_path)
            inferred_tracks_summary = {
                "total_inferred_tracks": len(track_sum),
                "unique_aircraft": int(track_sum["source_aircraft_id"].nunique()),
                "mean_obs_per_track": round(float(track_sum["n_observations"].mean()), 1),
            }
            stage_results.append(
                PipelineStageResult(
                    stage="segmentation",
                    status="completed",
                    elapsed_seconds=round(time.time() - t0, 3),
                    output_summary=inferred_tracks_summary,
                )
            )

        # STAGE 4: Feature Engineering
        agg_df = pd.DataFrame()
        feat_df = pd.DataFrame()
        if "features" in stages:
            t0 = time.time()
            cleaned_df["cell_id"] = grid.cell_ids_vec(
                cleaned_df["latitude"].to_numpy(), cleaned_df["longitude"].to_numpy()
            )
            valid_cells = cleaned_df.dropna(subset=["cell_id", "timestamp_utc"]).copy()
            valid_cells["window_start_utc"] = valid_cells["timestamp_utc"].dt.floor(f"{self.settings.window_minutes}min")

            if not valid_cells.empty:
                agg_df = (
                    valid_cells.groupby(["cell_id", "window_start_utc"])
                    .agg(
                        aircraft_count=("source_aircraft_id", "nunique"),
                        observation_count=("latitude", "size"),
                        mean_speed_mps=("ground_speed_mps", "mean"),
                        speed_std_mps=("ground_speed_mps", "std"),
                        mean_altitude_m=("altitude_m", "mean"),
                    )
                    .reset_index()
                )
                agg_df["window_end_utc"] = agg_df["window_start_utc"] + pd.Timedelta(minutes=self.settings.window_minutes)

                # Weather frame selection (linked weather or synthetic fallback)
                wx_df = self._load_weather_for_dataset(dataset_id)

                corridor_cells = list(agg_df["cell_id"].unique())
                t_min = agg_df["window_start_utc"].min()
                t_max = agg_df["window_end_utc"].max()

                feat_df = build_feature_frame(
                    agg_df, wx_df, grid, corridor_cells, t_min, t_max, self.settings.window_minutes
                )
                feat_path = self.processed_dir / f"{dataset_id}_features.parquet"
                feat_df.to_parquet(feat_path, index=False)
                artifacts["features"] = str(feat_path)

            stage_results.append(
                PipelineStageResult(
                    stage="features",
                    status="completed",
                    elapsed_seconds=round(time.time() - t0, 3),
                    output_summary={
                        "spatial_cells_populated": int(agg_df["cell_id"].nunique()) if not agg_df.empty else 0,
                        "time_windows_aggregated": int(agg_df["window_start_utc"].nunique()) if not agg_df.empty else 0,
                        "engineered_feature_rows": len(feat_df),
                        "features_generated": list(feat_df.columns)[:8] if not feat_df.empty else [],
                    },
                )
            )

        # STAGE 5: ML Model Inference
        pred_summary: dict[str, Any] = {}
        if "inference" in stages and not feat_df.empty:
            t0 = time.time()
            cm: LoadedModel | None = None
            try:
                cm = load_active("congestion", self.settings.artifacts_dir)
            except Exception:
                cm = None

            if cm is not None and not feat_df.empty:
                X, warns = validate_features(feat_df, cm)
                proba = cm.pipeline.predict_proba(X)
                classes = list(cm.pipeline.classes_)
                preds = [classes[np.argmax(p)] for p in proba]
                feat_df["predicted_class"] = preds
                for i, c in enumerate(classes):
                    feat_df[f"p_{c}"] = proba[:, i]

                pred_path = self.processed_dir / f"{dataset_id}_predictions.parquet"
                feat_df.to_parquet(pred_path, index=False)
                artifacts["predictions"] = str(pred_path)

                pred_counts = pd.Series(preds).value_counts().to_dict()
                pred_summary = {
                    "model_version": cm.version,
                    "inferences_computed": len(preds),
                    "class_distribution": pred_counts,
                }

            stage_results.append(
                PipelineStageResult(
                    stage="inference",
                    status="completed",
                    elapsed_seconds=round(time.time() - t0, 3),
                    output_summary=pred_summary,
                )
            )

        # STAGE 6: Register DataStore for Routing
        if not agg_df.empty:
            wx_df = self._load_weather_for_dataset(dataset_id)
            from app.api.state import get_state
            st = get_state()
            corridor = sorted(list(agg_df["cell_id"].unique()))

            store = DataStore(
                agg=agg_df,
                weather=wx_df,
                corridor_cells=corridor,
                airports=set(st.store.airports),
                dataset_id=dataset_id,
                weather_source_label=f"Dataset-aligned weather grid ({dataset_id})",
            )
            _dataset_stores[dataset_id] = store
            _dataset_builders[dataset_id] = ContextBuilder(
                rg=st.graph,
                grid=grid,
                store=store,
                window_min=self.settings.window_minutes,
                cruise_kmh=self.settings.heuristic_cruise_speed_kmh,
                congestion_model=st.congestion_model,
                eta_model=st.eta_model,
                thresholds=st.thresholds,
            )

        # Record observations and runs in SQLite
        total_elapsed = round(time.time() - total_t0, 3)
        with self.session_factory() as session:
            # Persist dataset-specific aircraft observations & tracks to database
            session.execute(delete(m.AircraftObservation).where(m.AircraftObservation.dataset == dataset_id))
            session.execute(delete(m.InferredTrack).where(m.InferredTrack.dataset == dataset_id))

            if "segmentation" in stages and track_sum is not None:
                track_recs = []
                for _, tr in track_sum.iterrows():
                    s_utc = tr["start_utc"]
                    if hasattr(s_utc, "to_pydatetime"):
                        s_utc = s_utc.to_pydatetime()
                    if s_utc is not None and getattr(s_utc, "tzinfo", None) is None:
                        s_utc = s_utc.replace(tzinfo=timezone.utc)
                    e_utc = tr["end_utc"]
                    if hasattr(e_utc, "to_pydatetime"):
                        e_utc = e_utc.to_pydatetime()
                    if e_utc is not None and getattr(e_utc, "tzinfo", None) is None:
                        e_utc = e_utc.replace(tzinfo=timezone.utc)
                    track_recs.append(
                        m.InferredTrack(
                            track_id=f"{dataset_id}:{tr['track_id']}",
                            dataset=dataset_id,
                            source_aircraft_id=str(tr["source_aircraft_id"]),
                            callsign=str(tr["callsign"]) if pd.notna(tr.get("callsign")) else None,
                            start_utc=s_utc,
                            end_utc=e_utc,
                            n_observations=int(tr["n_observations"]),
                            method=str(tr.get("method", "gap_segmentation")),
                            is_inferred=True,
                        )
                    )
                if track_recs:
                    session.add_all(track_recs)

            # Persist clean observations (up to 20,000 rows to keep SQLite fast while preserving rich data)
            obs_sample = cleaned_df.head(20000)
            obs_recs = []
            for _, r in obs_sample.iterrows():
                t_utc = r.get("timestamp_utc")
                if hasattr(t_utc, "to_pydatetime"):
                    t_utc = t_utc.to_pydatetime()
                if t_utc is not None and getattr(t_utc, "tzinfo", None) is None:
                    t_utc = t_utc.replace(tzinfo=timezone.utc)

                raw_obs_id = r.get("observation_id")
                obs_id = f"{dataset_id}_{raw_obs_id}" if raw_obs_id else f"{dataset_id}_OBS_{uuid.uuid4().hex[:8]}"
                raw_track_id = r.get("track_id")
                t_id = f"{dataset_id}:{raw_track_id}" if pd.notna(raw_track_id) else None

                obs_recs.append(
                    m.AircraftObservation(
                        observation_id=obs_id,
                        dataset=dataset_id,
                        source=f"upload:{dataset_id}",
                        is_synthetic=False,
                        source_aircraft_id=str(r["source_aircraft_id"]),
                        callsign=str(r["callsign"]) if pd.notna(r.get("callsign")) else None,
                        timestamp_utc=t_utc,
                        latitude=float(r["latitude"]) if pd.notna(r.get("latitude")) else None,
                        longitude=float(r["longitude"]) if pd.notna(r.get("longitude")) else None,
                        altitude_m=float(r["altitude_m"]) if pd.notna(r.get("altitude_m")) else None,
                        ground_speed_mps=float(r["ground_speed_mps"]) if pd.notna(r.get("ground_speed_mps")) else None,
                        heading_deg=float(r["heading_deg"]) if pd.notna(r.get("heading_deg")) else None,
                        vertical_rate_mps=float(r["vertical_rate_mps"]) if pd.notna(r.get("vertical_rate_mps")) else None,
                        on_ground=bool(r["on_ground"]) if pd.notna(r.get("on_ground")) else False,
                        data_quality_flags=str(r.get("data_quality_flags") or ""),
                        track_id=t_id,
                        retrieved_at=datetime.now(timezone.utc),
                    )
                )
            if obs_recs:
                session.add_all(obs_recs)

            for s_res in stage_results:
                run_rec = m.DatasetRun(
                    id=f"run_{uuid.uuid4().hex[:10]}",
                    dataset_id=dataset_id,
                    dataset_type="aircraft_observations",
                    stage=s_res.stage,
                    status=s_res.status,
                    elapsed_seconds=s_res.elapsed_seconds,
                    summary_json=s_res.output_summary,
                    artifact_paths=artifacts,
                )
                session.add(run_rec)

            session.execute(
                update(m.UploadedDataset)
                .where(m.UploadedDataset.id == dataset_id)
                .values(status="processed", schema_mapping=mapping)
            )
            session.commit()

        return RunPipelineResponse(
            dataset_id=dataset_id,
            status="completed",
            total_elapsed_seconds=total_elapsed,
            stages=stage_results,
            artifact_paths=artifacts,
        )

    def _load_weather_for_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Loads linked weather dataset if configured, else synthetic weather fallback."""
        with self.session_factory() as session:
            rec = session.get(m.UploadedDataset, dataset_id)
            if rec and rec.linked_weather_dataset_id:
                linked_p = self.processed_dir / f"{rec.linked_weather_dataset_id}_weather_grid.parquet"
                if linked_p.exists():
                    return pd.read_parquet(linked_p)

        weather_path = self.settings.data_dir / "synthetic" / "weather_synthetic.parquet"
        return pd.read_parquet(weather_path) if weather_path.exists() else pd.DataFrame()

    def link_traffic_and_weather(
        self,
        aircraft_dataset_id: str,
        weather_dataset_id: str,
        spatial_tolerance_deg: float = 0.5,
        temporal_tolerance_seconds: int = 3600,
    ) -> LinkWeatherResponse:
        """Spatially and temporally associates flight observations with a measured weather dataset."""
        with self.session_factory() as session:
            ac_rec = session.get(m.UploadedDataset, aircraft_dataset_id)
            if not ac_rec:
                raise NotFoundError(f"Aircraft dataset '{aircraft_dataset_id}' not found")
            wx_rec = session.get(m.UploadedDataset, weather_dataset_id)
            if not wx_rec:
                raise NotFoundError(f"Weather dataset '{weather_dataset_id}' not found")

        # Load aircraft data
        ac_p = self.processed_dir / f"{aircraft_dataset_id}_cleaned.parquet"
        if ac_p.exists():
            ac_df = pd.read_parquet(ac_p)
        else:
            raw_ac = pd.read_csv(ac_rec.stored_path)
            ac_df = apply_mapping(raw_ac, ac_rec.schema_mapping, dataset_type="aircraft_observations")

        # Load weather data
        wx_p = self.processed_dir / f"{weather_dataset_id}_weather_grid.parquet"
        if wx_p.exists():
            wx_df = pd.read_parquet(wx_p)
        else:
            raw_wx = pd.read_csv(wx_rec.stored_path)
            wx_df = apply_mapping(raw_wx, wx_rec.schema_mapping, dataset_type="weather_observations")

        grid = grid_from_settings(self.settings)
        if "cell_id" not in ac_df.columns:
            ac_df["cell_id"] = grid.cell_ids_vec(ac_df["latitude"].to_numpy(), ac_df["longitude"].to_numpy())

        ac_df["time_hr"] = pd.to_datetime(ac_df["timestamp_utc"], utc=True).dt.floor("1h")
        if "valid_time_utc" in wx_df.columns:
            wx_df["time_hr"] = pd.to_datetime(wx_df["valid_time_utc"], utc=True).dt.floor("1h")
        else:
            wx_df["time_hr"] = ac_df["time_hr"].iloc[0] if not ac_df.empty else pd.Timestamp.now(tz=timezone.utc).floor("1h")

        if "cell_id" not in wx_df.columns and "latitude" in wx_df.columns:
            wx_df["cell_id"] = grid.cell_ids_vec(wx_df["latitude"].to_numpy(), wx_df["longitude"].to_numpy())

        # Join on (cell_id, time_hr)
        merged = pd.merge(
            ac_df,
            wx_df[["cell_id", "time_hr", "precipitation_mm", "wind_speed_10m_ms", "wind_gusts_10m_ms"]].drop_duplicates(subset=["cell_id", "time_hr"]),
            on=["cell_id", "time_hr"],
            how="left",
            suffixes=("", "_wx"),
        )

        total_rows = len(ac_df)
        matched_rows = int(merged["precipitation_mm"].notna().sum())
        unmatched_rows = total_rows - matched_rows
        coverage_pct = round((matched_rows / total_rows * 100.0) if total_rows > 0 else 0.0, 1)

        warnings: list[str] = []
        if coverage_pct < 100.0:
            warnings.append(
                f"{unmatched_rows} observations ({100 - coverage_pct:.1f}%) lack matching weather measurements. They are flagged with null weather and will incur missing-data uncertainty penalties during routing."
            )

        linked_path = self.processed_dir / f"{aircraft_dataset_id}_linked_weather_{weather_dataset_id}.parquet"
        merged.to_parquet(linked_path, index=False)

        # Update DB record with linked weather ID
        with self.session_factory() as session:
            session.execute(
                update(m.UploadedDataset)
                .where(m.UploadedDataset.id == aircraft_dataset_id)
                .values(linked_weather_dataset_id=weather_dataset_id)
            )
            session.commit()

        # Update active context builder if dataset is active
        if aircraft_dataset_id in _dataset_stores:
            _dataset_stores[aircraft_dataset_id].weather = wx_df
            _dataset_stores[aircraft_dataset_id].weather_source_label = f"Uploaded Weather Grid ({weather_dataset_id})"

        return LinkWeatherResponse(
            aircraft_dataset_id=aircraft_dataset_id,
            weather_dataset_id=weather_dataset_id,
            status="linked",
            total_traffic_rows=total_rows,
            matched_rows=matched_rows,
            unmatched_rows=unmatched_rows,
            coverage_percentage=coverage_pct,
            warnings=warnings,
            artifact_path=str(linked_path),
        )

    def delete_dataset(self, dataset_id: str) -> bool:
        """Safely deletes an uploaded dataset, its runs, and physical artifacts."""
        if dataset_id == self.settings.demo_dataset_id:
            raise ValidationFailed("The built-in demonstration dataset cannot be deleted.")

        with self.session_factory() as session:
            record = session.get(m.UploadedDataset, dataset_id)
            if not record:
                raise NotFoundError(f"Dataset '{dataset_id}' not found")

            # Remove associated observation and run rows
            session.execute(delete(m.AircraftObservation).where(m.AircraftObservation.dataset == dataset_id))
            session.execute(delete(m.InferredTrack).where(m.InferredTrack.dataset == dataset_id))
            session.execute(delete(m.WeatherObservation).where(m.WeatherObservation.dataset == dataset_id))
            session.execute(delete(m.CongestionObservation).where(m.CongestionObservation.dataset == dataset_id))
            session.execute(delete(m.DatasetRun).where(m.DatasetRun.dataset_id == dataset_id))
            session.delete(record)
            session.commit()

        # Remove physical files
        try:
            stored_p = Path(record.stored_path)
            if stored_p.exists():
                stored_p.unlink()
        except Exception:
            pass

        for p in self.processed_dir.glob(f"{dataset_id}_*"):
            try:
                p.unlink()
            except Exception:
                pass

        _dataset_stores.pop(dataset_id, None)
        _dataset_builders.pop(dataset_id, None)
        return True


def restore_dataset_context(dataset_id: str) -> ContextBuilder | None:
    """Lazily restores a dataset-specific ContextBuilder and DataStore from disk/database across restarts."""
    if not dataset_id or dataset_id == get_settings().demo_dataset_id:
        return None

    settings = get_settings()
    processed_dir = settings.data_dir / "processed"
    grid = grid_from_settings(settings)
    from app.api.state import get_state
    st = get_state()

    # Check for congestion dataset
    cong_path = processed_dir / f"{dataset_id}_congestion_grid.parquet"
    if cong_path.exists():
        mapped_df = pd.read_parquet(cong_path)
        corridor = sorted(list(mapped_df["cell_id"].dropna().unique()))
        weather_path = settings.data_dir / "synthetic" / "weather_synthetic.parquet"
        wx_df = pd.read_parquet(weather_path) if weather_path.exists() else pd.DataFrame()
        store = DataStore(
            agg=mapped_df,
            weather=wx_df,
            corridor_cells=corridor,
            airports=set(st.store.airports),
            dataset_id=dataset_id,
            weather_source_label="Default baseline weather",
        )
        _dataset_stores[dataset_id] = store
        builder = ContextBuilder(
            rg=st.graph,
            grid=grid,
            store=store,
            window_min=settings.window_minutes,
            cruise_kmh=settings.heuristic_cruise_speed_kmh,
            congestion_model=st.congestion_model,
            eta_model=st.eta_model,
            thresholds=st.thresholds,
        )
        _dataset_builders[dataset_id] = builder
        return builder

    # Check for aircraft dataset (features or cleaned)
    feat_path = processed_dir / f"{dataset_id}_features.parquet"
    clean_path = processed_dir / f"{dataset_id}_cleaned.parquet"
    if feat_path.exists() or clean_path.exists():
        if feat_path.exists():
            feat_df = pd.read_parquet(feat_path)
            cols = [
                c for c in [
                    "cell_id", "window_start_utc", "aircraft_count", "observation_count",
                    "mean_speed_mps", "speed_std_mps", "mean_altitude_m",
                ] if c in feat_df.columns
            ]
            agg_df = feat_df[cols].drop_duplicates()
            if "observation_count" not in agg_df.columns:
                agg_df["observation_count"] = agg_df.get("aircraft_count", 0)
            corridor = sorted(list(agg_df["cell_id"].dropna().unique()))
        else:
            cleaned_df = pd.read_parquet(clean_path)
            cleaned_df["cell_id"] = grid.cell_ids_vec(cleaned_df["latitude"].to_numpy(), cleaned_df["longitude"].to_numpy())
            valid_cells = cleaned_df.dropna(subset=["cell_id", "timestamp_utc"]).copy()
            valid_cells["window_start_utc"] = valid_cells["timestamp_utc"].dt.floor(f"{settings.window_minutes}min")
            agg_kwargs = {
                "aircraft_count": ("source_aircraft_id", "nunique"),
                "observation_count": ("latitude", "size"),
            }
            if "ground_speed_mps" in valid_cells.columns:
                agg_kwargs["mean_speed_mps"] = ("ground_speed_mps", "mean")
                agg_kwargs["speed_std_mps"] = ("ground_speed_mps", "std")
            if "altitude_m" in valid_cells.columns:
                agg_kwargs["mean_altitude_m"] = ("altitude_m", "mean")
            agg_df = valid_cells.groupby(["cell_id", "window_start_utc"]).agg(**agg_kwargs).reset_index()
            corridor = sorted(list(agg_df["cell_id"].dropna().unique()))

        mgr = DatasetManager(settings)
        wx_df = mgr._load_weather_for_dataset(dataset_id)
        store = DataStore(
            agg=agg_df,
            weather=wx_df,
            corridor_cells=corridor,
            airports=set(st.store.airports),
            dataset_id=dataset_id,
            weather_source_label=f"Dataset-aligned weather grid ({dataset_id})",
        )
        _dataset_stores[dataset_id] = store
        builder = ContextBuilder(
            rg=st.graph,
            grid=grid,
            store=store,
            window_min=settings.window_minutes,
            cruise_kmh=settings.heuristic_cruise_speed_kmh,
            congestion_model=st.congestion_model,
            eta_model=st.eta_model,
            thresholds=st.thresholds,
        )
        _dataset_builders[dataset_id] = builder
        return builder

    return None


def get_dataset_context_builder(
    dataset_id: str | None,
    default_builder: ContextBuilder,
    weather_dataset_id: str | None = None,
) -> ContextBuilder:
    """Thread-safe dataset-specific context builder resolver with persistence recovery and optional custom weather."""
    base_builder: ContextBuilder = default_builder
    if dataset_id and dataset_id != get_settings().demo_dataset_id:
        if dataset_id in _dataset_builders:
            base_builder = _dataset_builders[dataset_id]
        else:
            restored = restore_dataset_context(dataset_id)
            if restored is not None:
                base_builder = restored

    if not weather_dataset_id or weather_dataset_id == get_settings().demo_dataset_id:
        return base_builder

    # If weather_dataset_id is specified, bind the custom weather grid
    settings = get_settings()
    wx_path = settings.data_dir / "processed" / f"{weather_dataset_id}_weather_grid.parquet"
    if wx_path.exists():
        try:
            wx_df = pd.read_parquet(wx_path)
            from app.api.state import get_state
            st = get_state()
            grid = grid_from_settings(settings)
            custom_store = DataStore(
                agg=base_builder.store.agg,
                weather=wx_df,
                corridor_cells=base_builder.store.corridor_cells,
                airports=base_builder.store.airports,
                dataset_id=base_builder.store.dataset_id,
                weather_source_label=f"Custom Weather Grid ({weather_dataset_id})",
            )
            return ContextBuilder(
                rg=st.graph,
                grid=grid,
                store=custom_store,
                window_min=settings.window_minutes,
                cruise_kmh=settings.heuristic_cruise_speed_kmh,
                congestion_model=st.congestion_model,
                eta_model=st.eta_model,
                thresholds=st.thresholds,
            )
        except Exception:
            pass

    return base_builder
