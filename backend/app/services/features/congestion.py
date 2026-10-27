"""Congestion feature engineering.

Aggregation unit: research grid cell x fixed time window (default 15 min).
* Windows are half-open ``[window_start, window_end)`` and labelled by ``window_end``.
* ``aircraft_count`` = number of UNIQUE aircraft identifiers observed in the cell
  during the window (not the number of position reports, which is stored
  separately as ``observation_count``).
* Weather matching rule: the hourly record whose ``valid_time`` equals
  ``floor(window_start, 1h)`` - i.e. a value that is already valid before the
  prediction cutoff (``window_end``). No hourly value is interpolated into
  sub-hourly "measurements".
* Every feature in row *t* uses only observations with timestamp < window_end(t).
  The target is the count class of window *t+1*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.geospatial.grid import Grid

CLASSES = ["LOW", "MEDIUM", "HIGH"]

NUMERIC_FEATURES = [
    "hour_sin", "hour_cos",
    "aircraft_count", "count_lag1", "count_lag2", "count_lag4", "count_roll4_mean", "traffic_change",
    "neighbor_count", "neighbor_count_lag1",
    "mean_speed_mps", "speed_std_mps", "mean_altitude_m",
    "wind_speed_10m_ms", "wind_dir_sin", "wind_dir_cos", "precipitation_mm", "cloud_cover_pct",
    "wind_gusts_10m_ms", "weather_missing",
]
CATEGORICAL_FEATURES = ["cell_id"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COUNT = "next_window_count"
TARGET = "next_window_class"


@dataclass(frozen=True)
class Thresholds:
    low_max: int     # count <= low_max          -> LOW
    medium_max: int  # low_max < count <= med_max -> MEDIUM ; above -> HIGH
    method: str = "quantiles of occupied (count>=1) training windows: q50 / q85"

    def to_dict(self) -> dict:
        return {"low_max": self.low_max, "medium_max": self.medium_max, "method": self.method}


def thresholds_from_training(train_counts: pd.Series) -> Thresholds:
    occ = train_counts[train_counts >= 1]
    if len(occ) < 10:
        raise ValueError("Not enough occupied training windows to derive thresholds")
    t1 = int(np.floor(occ.quantile(0.50)))
    t2 = int(np.floor(occ.quantile(0.85)))
    t2 = max(t2, t1 + 1)
    return Thresholds(t1, t2)


def classify(counts, th: Thresholds) -> np.ndarray:  # noqa: ANN001
    c = np.asarray(counts, dtype=float)
    out = np.where(c <= th.low_max, "LOW", np.where(c <= th.medium_max, "MEDIUM", "HIGH"))
    return np.where(np.isnan(c), None, out)


def aggregate_windows(obs: pd.DataFrame, grid: Grid, window_min: int) -> pd.DataFrame:
    df = obs[["source_aircraft_id", "timestamp_utc", "latitude", "longitude", "altitude_m", "ground_speed_mps"]].copy()
    df["cell_id"] = grid.cell_ids_vec(df.latitude.to_numpy(), df.longitude.to_numpy())
    df = df[df.cell_id.notna()]
    df["window_start_utc"] = df.timestamp_utc.dt.floor(f"{window_min}min")
    agg = (
        df.groupby(["cell_id", "window_start_utc"])
        .agg(aircraft_count=("source_aircraft_id", "nunique"), observation_count=("source_aircraft_id", "size"),
             mean_speed_mps=("ground_speed_mps", "mean"), speed_std_mps=("ground_speed_mps", "std"),
             mean_altitude_m=("altitude_m", "mean"))
        .reset_index()
    )
    agg["window_end_utc"] = agg.window_start_utc + pd.Timedelta(minutes=window_min)
    return agg


def _neighbors(grid: Grid, cells: list[str]) -> dict[str, list[str]]:
    cs = set(cells)
    out = {}
    for c in cells:
        r, k = int(c[1:3]), int(c[4:6])
        out[c] = [Grid.cell_id_from_rc(r + dr, k + dk) for dr in (-1, 0, 1) for dk in (-1, 0, 1)
                  if (dr or dk) and Grid.cell_id_from_rc(r + dr, k + dk) in cs]
    return out


def build_feature_frame(agg: pd.DataFrame, weather: pd.DataFrame | None, grid: Grid, cells: list[str],
                        t_start: pd.Timestamp, t_end: pd.Timestamp, window_min: int) -> pd.DataFrame:
    """Zero-filled cell x window frame with leakage-safe lag features and next-window target count."""
    windows = pd.date_range(t_start.floor(f"{window_min}min"), t_end, freq=f"{window_min}min", inclusive="left")
    full = pd.MultiIndex.from_product([cells, windows], names=["cell_id", "window_start_utc"]).to_frame(index=False)
    df = full.merge(agg.drop(columns=["window_end_utc"], errors="ignore"), on=["cell_id", "window_start_utc"], how="left")
    if "aircraft_count" in df.columns:
        df["aircraft_count"] = df["aircraft_count"].fillna(0).astype(int)
    else:
        df["aircraft_count"] = 0
    if "observation_count" in df.columns:
        df["observation_count"] = df["observation_count"].fillna(0).astype(int)
    else:
        df["observation_count"] = 0
    for col in ["mean_speed_mps", "speed_std_mps", "mean_altitude_m"]:
        if col not in df.columns:
            df[col] = np.nan
    df["window_end_utc"] = df.window_start_utc + pd.Timedelta(minutes=window_min)
    df = df.sort_values(["cell_id", "window_start_utc"]).reset_index(drop=True)

    g = df.groupby("cell_id")["aircraft_count"]
    df["count_lag1"] = g.shift(1)
    df["count_lag2"] = g.shift(2)
    df["count_lag4"] = g.shift(4)
    df["count_roll4_mean"] = g.transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["traffic_change"] = df.aircraft_count - df.count_lag1
    df[TARGET_COUNT] = g.shift(-1)

    # spatial context: sum of 8-neighbour counts in the same (current) window
    pivot = df.pivot(index="window_start_utc", columns="cell_id", values="aircraft_count")
    nb = _neighbors(grid, cells)
    nsum = pd.DataFrame({c: pivot[nb[c]].sum(axis=1) if nb[c] else 0 for c in cells}, index=pivot.index)
    nlong = nsum.stack().rename("neighbor_count").reset_index()
    nlong.columns = ["window_start_utc", "cell_id", "neighbor_count"]
    df = df.merge(nlong, on=["window_start_utc", "cell_id"], how="left")
    df = df.sort_values(["cell_id", "window_start_utc"]).reset_index(drop=True)
    df["neighbor_count_lag1"] = df.groupby("cell_id")["neighbor_count"].shift(1)

    hour = df.window_end_utc.dt.hour + df.window_end_utc.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    wx_cols = ["precipitation_mm", "cloud_cover_pct", "wind_speed_10m_ms", "wind_direction_10m_deg", "wind_gusts_10m_ms"]
    if weather is not None and len(weather):
        w = weather.copy()
        if "location_id" not in w.columns and "cell_id" in w.columns:
            w["location_id"] = w["cell_id"]
        for c in wx_cols:
            if c not in w.columns:
                w[c] = np.nan
        avail_cols = [c for c in ["location_id", "valid_time_utc", *wx_cols] if c in w.columns]
        w = w[avail_cols].rename(columns={"location_id": "cell_id"})
        if "valid_time_utc" in w.columns:
            w["valid_time_utc"] = pd.to_datetime(w["valid_time_utc"], utc=True)
            df["wx_hour"] = df.window_start_utc.dt.floor("1h")
            df = df.merge(w.rename(columns={"valid_time_utc": "wx_hour"}), on=["cell_id", "wx_hour"], how="left")
            df = df.drop(columns=["wx_hour"])
        else:
            for c in wx_cols:
                if c not in df.columns:
                    df[c] = np.nan
    else:
        for c in wx_cols:
            df[c] = np.nan
    df["weather_missing"] = df.get("precipitation_mm", pd.Series(np.nan, index=df.index)).isna().astype(int)
    rad = np.radians(df.wind_direction_10m_deg)
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)
    return df


def chronological_split(df: pd.DataFrame, val_start: pd.Timestamp, test_start: pd.Timestamp,
                        window_min: int) -> dict[str, pd.DataFrame]:
    """Split by window_end. Rows whose TARGET window lies in a later partition are purged."""
    target_end = df.window_end_utc + pd.Timedelta(minutes=window_min)
    train = df[(target_end <= val_start)]
    val = df[(df.window_end_utc > val_start) & (target_end <= test_start)]
    test = df[df.window_end_utc > test_start]
    return {"train": train, "val": val, "test": test}
