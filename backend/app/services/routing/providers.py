"""Builds the per-edge RoutingContext for a scenario time from data + models.

Inputs at scenario time ``t`` (all strictly available before ``t``):
* weather: hourly cell record valid at floor(t, 1h)
* congestion: model prediction for the window containing ``t``, using features of the
  last COMPLETE window (ending at floor(t, window)). Fallback without a model: the
  OBSERVED class of that last complete window (clearly labelled, not a prediction).
* ETA: ETA model with planning-time features; fallback: distance / heuristic speed.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.services.features.congestion import CLASSES, Thresholds, build_feature_frame
from app.services.features.eta import ETA_FEATURES
from app.services.geospatial.grid import Grid
from app.services.ingestion.synthetic_demo import weather_severity
from app.services.models.registry import LoadedModel, ood_flags, validate_features
from app.services.routing.context import EdgeAttrs, RoutingContext
from app.services.routing.graph import RouteGraph

CLASS_SCORE = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}


@dataclass
class DataStore:
    agg: pd.DataFrame            # cell_id, window_start_utc, aircraft_count, ...
    weather: pd.DataFrame        # location_id, valid_time_utc, ...
    corridor_cells: list[str]
    airports: set[str]
    dataset_id: str
    weather_source_label: str

    @property
    def coverage(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if self.agg.empty:
            return None, None
        return self.agg.window_start_utc.min(), self.agg.window_start_utc.max()


class ContextBuilder:
    def __init__(self, rg: RouteGraph, grid: Grid, store: DataStore, window_min: int, cruise_kmh: float,
                 congestion_model: LoadedModel | None, eta_model: LoadedModel | None, thresholds: Thresholds | None):
        self.rg, self.grid, self.store = rg, grid, store
        self.window_min, self.cruise_kmh = window_min, cruise_kmh
        self.cm, self.em, self.th = congestion_model, eta_model, thresholds
        self.edge_cells = {(u, v): grid.cells_along(*rg.pos(u), *rg.pos(v), step_km=40) for u, v in rg.g.edges()}
        self._cache: OrderedDict[datetime, RoutingContext] = OrderedDict()
        if len(store.weather):
            wx_df = store.weather.copy()
            if "location_id" not in wx_df.columns and "cell_id" in wx_df.columns:
                wx_df["location_id"] = wx_df["cell_id"]
            if "location_id" in wx_df.columns and "valid_time_utc" in wx_df.columns:
                self._wx = wx_df.set_index(["location_id", "valid_time_utc"]).sort_index()
            else:
                self._wx = None
        else:
            self._wx = None
        self._counts = (store.agg.set_index(["cell_id", "window_start_utc"])["aircraft_count"]
                        if len(store.agg) else pd.Series(dtype=float))

    # ---------------------------------------------------------------- helpers
    def _cell_weather(self, t: pd.Timestamp) -> dict[str, dict]:
        hour = t.floor("1h")
        out = {}
        if self._wx is None:
            return out
        try:
            block = self._wx.xs(hour, level="valid_time_utc")
        except KeyError:
            return out
        precip = block["precipitation_mm"] if "precipitation_mm" in block else pd.Series(0.0, index=block.index)
        gusts = block["wind_gusts_10m_ms"] if "wind_gusts_10m_ms" in block else pd.Series(0.0, index=block.index)
        clouds = block["cloud_cover_pct"] if "cloud_cover_pct" in block else pd.Series(0.0, index=block.index)
        sev = weather_severity(precip, gusts, clouds)
        ws_col = (
            block["wind_speed_100m_ms"]
            if "wind_speed_100m_ms" in block
            else (block["wind_speed_10m_ms"] if "wind_speed_10m_ms" in block else pd.Series(0.0, index=block.index))
        )
        wd_col = (
            block["wind_direction_10m_deg"]
            if "wind_direction_10m_deg" in block
            else pd.Series(0.0, index=block.index)
        )
        for cid, s, ws, wd in zip(block.index, sev, ws_col, wd_col):
            out[cid] = {"severity": float(s), "ws": float(ws), "wd": float(wd)}
        return out

    def predict_cells(self, t: pd.Timestamp) -> tuple[dict[str, dict], str, list[str]]:
        """Per-cell congestion estimate for the window containing t."""
        wend = t.floor(f"{self.window_min}min")
        warnings: list[str] = []
        start = wend - pd.Timedelta(hours=2)
        lo, hi = self.store.coverage
        if lo is None or wend - pd.Timedelta(minutes=self.window_min) < lo or wend > hi + pd.Timedelta(minutes=self.window_min):
            return {}, "unavailable (scenario time outside data coverage)", ["Congestion inputs unavailable: scenario time outside data coverage."]
        agg = self.store.agg[(self.store.agg.window_start_utc >= start) & (self.store.agg.window_start_utc < wend)]
        frame = build_feature_frame(agg, self.store.weather, self.grid, self.store.corridor_cells, start, wend, self.window_min)
        rows = frame[frame.window_end_utc == wend].copy()
        if self.cm is not None:
            X, w = validate_features(rows, self.cm)
            warnings += w
            proba = self.cm.pipeline.predict_proba(X)
            cls = list(self.cm.pipeline.classes_)
            out = {}
            for i, cid in enumerate(rows.cell_id):
                p = {c: float(proba[i, cls.index(c)]) if c in cls else 0.0 for c in CLASSES}
                pred = max(p, key=p.get)
                ood = ood_flags(rows.iloc[i], self.cm)
                out[cid] = {"probs": p, "class": pred, "score": p["MEDIUM"] * 0.5 + p["HIGH"] * 1.0,
                            "confidence": p[pred], "ood": ood}
            src = (f"ML: {self.cm.metadata['model_name']} {self.cm.version} (synthetic-trained), "
                   f"next {self.window_min}-min window from {wend.strftime('%Y-%m-%d %H:%M')}Z")
            return out, src, warnings
        out = {}
        for cid, cnt in zip(rows.cell_id, rows.aircraft_count):
            c = ("LOW" if cnt <= self.th.low_max else "MEDIUM" if cnt <= self.th.medium_max else "HIGH") if self.th else None
            out[cid] = {"probs": None, "class": c, "score": CLASS_SCORE.get(c, 0.0), "confidence": None, "ood": []}
        warnings.append("Congestion model unavailable: using OBSERVED class of the last complete window (not a prediction).")
        return out, f"Observed proxy (no model): last complete window ending {wend.strftime('%H:%M')}Z", warnings

    def predict_edge_times(self, t: pd.Timestamp, cell_wx: dict[str, dict]) -> tuple[dict, str, list[str]]:
        edges = [(u, v, d) for u, v, d in self.rg.g.edges(data=True)]
        if self.em is None:
            src = f"Heuristic: distance / {self.cruise_kmh:g} km/h (not ML)"
            return {(u, v): d["distance_km"] / self.cruise_kmh * 60 for u, v, d in edges}, src, ["ETA model unavailable: heuristic times used."]
        pdft = self.em.metadata.get("planning_defaults", {})
        spd_e, alt_e = pdft.get("entry_speed_mps_by_edge", {}), pdft.get("entry_altitude_m_by_edge", {})
        spd_a, alt_a = pdft.get("entry_speed_mps_by_from_airport", {}), pdft.get("entry_altitude_m_by_from_airport", {})
        prev_ws = t.floor(f"{self.window_min}min") - pd.Timedelta(minutes=self.window_min)
        hour = t.hour + t.minute / 60
        rows = []
        for u, v, d in edges:
            fa = int(u in self.store.airports)
            cells = self.edge_cells[(u, v)]
            wx = cell_wx.get(cells[0]) if cells else None
            brg = d["bearing_deg"]
            rows.append({
                "segment_distance_km": d["distance_km"], "bearing_sin": np.sin(np.radians(brg)), "bearing_cos": np.cos(np.radians(brg)),
                "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                "from_airport": fa, "to_airport": int(v in self.store.airports),
                "entry_speed_mps": spd_e.get(d["edge_id"], spd_a.get(str(fa), spd_a.get(fa, np.nan))),
                "entry_altitude_m": alt_e.get(d["edge_id"], alt_a.get(str(fa), alt_a.get(fa, np.nan))),
                "tailwind_proxy_ms": (-wx["ws"] * np.cos(np.radians(brg - wx["wd"]))) if wx else np.nan,
                "weather_severity": wx["severity"] if wx else np.nan,
                "corridor_count_prev_window": float(np.mean([self._counts.get((c, prev_ws), 0) for c in cells])) if cells else np.nan,
            })
        X, warns = validate_features(pd.DataFrame(rows, columns=ETA_FEATURES), self.em)
        pred = self.em.pipeline.predict(X)
        out = {}
        for (u, v, d), p in zip(edges, pred):
            lo, hi = d["distance_km"] / 1300 * 60, d["distance_km"] / 150 * 60
            out[(u, v)] = float(np.clip(p, lo, hi))
        src = f"ML: {self.em.metadata['model_name']} ETA {self.em.version} (synthetic-trained)"
        return out, src, warns

    # ------------------------------------------------------------------ main
    def base_context(self, t: datetime) -> RoutingContext:
        t = t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)
        if t in self._cache:
            self._cache.move_to_end(t)
            return self._cache[t].copy()
        ts = pd.Timestamp(t)
        cell_wx = self._cell_weather(ts)
        cells_c, c_src, c_warn = self.predict_cells(ts)
        times, t_src, t_warn = self.predict_edge_times(ts, cell_wx)
        warnings = list(dict.fromkeys(c_warn + t_warn))
        wx_src = f"{self.store.weather_source_label} valid {ts.floor('1h').strftime('%Y-%m-%d %H:00')}Z" if cell_wx else "unavailable"
        if not cell_wx:
            warnings.append("Weather unavailable at scenario time: weather component is 0 and uncertainty is raised.")
        edges: dict[tuple[str, str], EdgeAttrs] = {}
        for u, v, d in self.rg.g.edges(data=True):
            cells = self.edge_cells[(u, v)]
            sev = [cell_wx[c]["severity"] for c in cells if c in cell_wx]
            cg = [cells_c[c] for c in cells if c in cells_c]
            flags, unc = [], 0.0
            if not sev:
                unc += 0.4
                flags.append("weather_missing")
            if not cg:
                unc += 0.4
                flags.append("congestion_missing")
            else:
                if cg[0]["probs"] is None:
                    unc += 0.2
                    flags.append("congestion_observed_proxy")
                elif min(c["confidence"] for c in cg) < 0.5:
                    unc += 0.2
                    flags.append("congestion_low_confidence")
                if any(c["ood"] for c in cg):
                    unc += 0.1
                    flags.append("congestion_input_out_of_training_range")
            if self.em is None:
                unc += 0.1
            worst = max(cg, key=lambda c: c["score"]) if cg else None
            probs = None
            if cg and cg[0]["probs"] is not None:
                probs = {k: round(float(np.mean([c["probs"][k] for c in cg])), 3) for k in CLASSES}
            edges[(u, v)] = EdgeAttrs(
                edge_id=d["edge_id"], distance_km=d["distance_km"], time_min=times[(u, v)], time_source=t_src,
                weather_severity=float(np.mean(sev)) if sev else None, weather_source=wx_src if sev else "unavailable",
                congestion_score=float(np.mean([c["score"] for c in cg])) if cg else None,
                congestion_class=worst["class"] if worst else None, congestion_probs=probs,
                congestion_source=c_src if cg else "unavailable", uncertainty=min(1.0, unc), flags=flags,
            )
        ctx = RoutingContext(t, None, edges, sources={"time": t_src, "weather": wx_src, "congestion": c_src,
                                                      "dataset": self.store.dataset_id}, warnings=warnings)
        self._cache[t] = ctx
        if len(self._cache) > 48:
            self._cache.popitem(last=False)
        return ctx.copy()

    def cell_predictions(self, t: datetime) -> tuple[dict[str, dict], str, list[str]]:
        t = t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)
        return self.predict_cells(pd.Timestamp(t))
