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
from functools import lru_cache
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from app.services.geospatial.geo import haversine_km_vec, interpolate_gc
from app.services.geospatial.grid import Grid

CLASSES = ["LOW", "MEDIUM", "HIGH"]

NUMERIC_FEATURES = [
    "hour_sin", "hour_cos",
    "aircraft_count", "observation_count", "obs_per_aircraft",
    "count_lag1", "count_lag2", "count_lag4", "count_roll4_mean", "count_roll8_mean", "traffic_change",
    "late_window_count", "early_window_count", "subwindow_trend",
    "new_dep_count", "new_dep_lag1", "new_dep_roll4_mean", "new_dep_roll8_mean", "landing_count",
    "kinematic_proj_count", "expected_proj_count", "max_branch_proj_count", "high_conf_proj_count",
    "staying_proj_count", "inflow_proj_count", "proj_obs_count",
    "cell_hub_weight", "is_airport_cell", "diurnal_dep_prior",
    "total_expected_load", "kinematic_plus_dep", "proj_minus_current", "branch_uncertainty",
    "load_dist_to_med", "load_dist_to_high", "proj_ge_5", "proj_eq_4", "proj_ge_3", "proj_eq_2",
    "neighbor_count", "neighbor_count_lag1", "neighbor_proj_count",
    "mean_speed_mps", "speed_std_mps", "mean_altitude_m", "mean_vertical_rate_mps",
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


@lru_cache(maxsize=2)
def _graph_kinematic_tables() -> dict[str, Any]:
    """Precompute airway graph geometry and Bayesian transition tables (purely topological)."""
    from app.core.config import get_settings
    from app.services.geospatial.grid import grid_from_settings
    from app.services.ingestion.bootstrap import airports_frame
    from app.services.ingestion.synthetic_demo import HUB_WEIGHTS

    settings = get_settings()
    grid = grid_from_settings(settings)
    airports, _ = airports_frame()
    ap_set = set(airports.node_id)
    syn_dir = settings.data_dir / "synthetic"
    edges_p = syn_dir / "edges.csv"
    wps_p = syn_dir / "waypoints.csv"
    if not edges_p.exists() or not wps_p.exists():
        return {}

    edges = pd.read_csv(edges_p)
    wps = pd.read_csv(wps_p)
    nodes = pd.concat([
        airports[["node_id", "latitude", "longitude"]],
        wps[["node_id", "latitude", "longitude"]],
    ], ignore_index=True)
    pos = {r.node_id: (float(r.latitude), float(r.longitude)) for r in nodes.itertuples()}

    g = nx.DiGraph()
    for e in edges.itertuples():
        g.add_edge(e.source_node, e.target_node, w=float(e.distance_km), brg=float(e.bearing_deg))

    ap_cells = {nid: grid.cell_id(*pos[nid]) for nid in ap_set}
    cell_hub_weight: dict[str, float] = {}
    for nid, cid in ap_cells.items():
        if cid:
            cell_hub_weight[cid] = cell_hub_weight.get(cid, 0.0) + float(HUB_WEIGHTS.get(nid, 1.0))

    trans_4: dict[tuple[str, str, str, str], list[tuple[str, str, float]]] = {}
    trans_3: dict[tuple[str, str, str], list[tuple[str, str, float]]] = {}
    trans_2: dict[tuple[str, str], list[tuple[str, str, float]]] = {}

    for o in sorted(ap_set):
        wo = float(HUB_WEIGHTS.get(o, 1.0))
        for de in sorted(ap_set):
            if o == de or not nx.has_path(g, o, de):
                continue
            wde = float(HUB_WEIGHTS.get(de, 1.0))
            od_w = wo * wde
            gen = nx.shortest_simple_paths(g, o, de, weight="w")
            p1 = next(gen)
            p2 = next(gen, None)
            for p, prob_p in [(p1, 0.75 if p2 else 1.0)] + ([(p2, 0.25)] if p2 else []):
                w_path = od_w * prob_p
                for idx in range(len(p) - 1):
                    u, v = p[idx], p[idx + 1]
                    u_prev = p[idx - 1] if idx > 0 else "START"
                    act = ("STOP", v, w_path) if idx == len(p) - 2 else ("NEXT", p[idx + 2], w_path)
                    trans_4.setdefault((o, u_prev, u, v), []).append(act)
                    trans_3.setdefault((o, u, v), []).append(act)
                    trans_2.setdefault((u, v), []).append(act)

    def _compress(act_list: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        agg_w: dict[tuple[str, str], float] = {}
        for kind, target, w_path in act_list:
            key = (kind, target)
            agg_w[key] = agg_w.get(key, 0.0) + w_path
        tot = sum(agg_w.values())
        return [(k[0], k[1], v / tot) for k, v in sorted(agg_w.items(), key=lambda x: -x[1])]

    e_u = edges["source_node"].to_numpy()
    e_v = edges["target_node"].to_numpy()
    e_dist = edges["distance_km"].to_numpy(float)
    e_brg = edges["bearing_deg"].to_numpy(float)

    n_pts = 100
    fs_grid = np.linspace(0.0, 1.0, n_pts + 1)
    edge_frac_cells: dict[tuple[str, str], list[str]] = {}
    edge_len_map: dict[tuple[str, str], float] = {}
    for u, v, d_km in zip(e_u, e_v, e_dist):
        edge_len_map[(u, v)] = float(d_km)
        la1, lo1 = pos[u]
        la2, lo2 = pos[v]
        pts = [interpolate_gc(la1, lo1, la2, lo2, float(f)) for f in fs_grid]
        las = np.array([p[0] for p in pts])
        lons = np.array([p[1] for p in pts])
        cids = grid.cell_ids_vec(las, lons)
        edge_frac_cells[(u, v)] = [str(c) if pd.notna(c) else "" for c in cids]

    return {
        "ap_set": ap_set,
        "ap_ids": airports["node_id"].to_numpy(),
        "ap_lats": airports["latitude"].to_numpy(float),
        "ap_lons": airports["longitude"].to_numpy(float),
        "ap_cells": ap_cells,
        "cell_hub_weight": cell_hub_weight,
        "pos": pos,
        "e_u": e_u,
        "e_v": e_v,
        "e_dist": e_dist,
        "e_brg": e_brg,
        "u_lat": np.array([pos[u][0] for u in e_u]),
        "u_lon": np.array([pos[u][1] for u in e_u]),
        "v_lat": np.array([pos[v][0] for v in e_v]),
        "v_lon": np.array([pos[v][1] for v in e_v]),
        "lookup_4": {k: _compress(v) for k, v in trans_4.items()},
        "lookup_3": {k: _compress(v) for k, v in trans_3.items()},
        "lookup_2": {k: _compress(v) for k, v in trans_2.items()},
        "edge_frac_cells": edge_frac_cells,
        "edge_len_map": edge_len_map,
        "n_pts": n_pts,
    }


def aggregate_windows(obs: pd.DataFrame, grid: Grid, window_min: int) -> pd.DataFrame:
    """Aggregate cell-window observations and compute causal graph-kinematic projections.

    Every feature in window *t* uses strictly observations with ``timestamp_utc < window_end_utc(t)``.
    """
    base_cols = ["source_aircraft_id", "timestamp_utc", "latitude", "longitude", "altitude_m", "ground_speed_mps"]
    opt_cols = [c for c in ["track_id", "heading_deg", "vertical_rate_mps"] if c in obs.columns]
    df = obs[base_cols + opt_cols].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["cell_id"] = grid.cell_ids_vec(df.latitude.to_numpy(), df.longitude.to_numpy())
    df = df[df.cell_id.notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "cell_id", "window_start_utc", "aircraft_count", "observation_count",
            "mean_speed_mps", "speed_std_mps", "mean_altitude_m", "mean_vertical_rate_mps",
            "late_window_count", "early_window_count", "new_dep_count", "landing_count",
            "kinematic_proj_count", "expected_proj_count", "max_branch_proj_count",
            "high_conf_proj_count", "staying_proj_count", "inflow_proj_count", "proj_obs_count",
            "window_end_utc",
        ])

    if "track_id" not in df.columns:
        df = df.sort_values(["source_aircraft_id", "timestamp_utc"]).reset_index(drop=True)
        gap = df.groupby("source_aircraft_id")["timestamp_utc"].diff().dt.total_seconds().gt(1800).fillna(False)
        t_num = gap.groupby(df["source_aircraft_id"]).cumsum()
        df["track_id"] = df["source_aircraft_id"].astype(str) + "-T" + t_num.astype(str)
    if "heading_deg" not in df.columns:
        df["heading_deg"] = 0.0
    if "vertical_rate_mps" not in df.columns:
        df["vertical_rate_mps"] = 0.0

    df = df.sort_values(["track_id", "timestamp_utc"]).reset_index(drop=True)
    df["window_start_utc"] = df.timestamp_utc.dt.floor(f"{window_min}min")
    df["window_end_utc"] = df.window_start_utc + pd.Timedelta(minutes=window_min)
    df["track_step"] = df.groupby("track_id").cumcount() + 1

    offset_min = (df["timestamp_utc"] - df["window_start_utc"]).dt.total_seconds() / 60.0
    df["is_late_5m"] = offset_min >= (window_min - 5.0)
    df["is_early_5m"] = offset_min < 5.0
    df["is_new_dep"] = (df["track_step"] == 1) & (df["altitude_m"].fillna(0) < 2500.0)
    df["is_landing_obs"] = (df["altitude_m"].fillna(10000) <= 50.0) & (df["track_step"] >= 3)

    agg = (
        df.groupby(["cell_id", "window_start_utc"])
        .agg(
            aircraft_count=("source_aircraft_id", "nunique"),
            observation_count=("source_aircraft_id", "size"),
            mean_speed_mps=("ground_speed_mps", "mean"),
            speed_std_mps=("ground_speed_mps", "std"),
            mean_altitude_m=("altitude_m", "mean"),
            mean_vertical_rate_mps=("vertical_rate_mps", "mean"),
        )
        .reset_index()
    )

    for col_name, flag_col in [
        ("late_window_count", "is_late_5m"),
        ("early_window_count", "is_early_5m"),
        ("new_dep_count", "is_new_dep"),
        ("landing_count", "is_landing_obs"),
    ]:
        s_cnt = df[df[flag_col]].groupby(["cell_id", "window_start_utc"])["source_aircraft_id"].nunique().rename(col_name)
        agg = agg.merge(s_cnt, on=["cell_id", "window_start_utc"], how="left")
        agg[col_name] = agg[col_name].fillna(0).astype(int)

    # Causal Graph-Kinematic Projection for next window [window_end_utc, window_end_utc + window_min)
    tables = _graph_kinematic_tables()
    if tables:
        first_obs = df.groupby("track_id", sort=False).first()[["latitude", "longitude"]]
        d_ap = haversine_km_vec(
            first_obs.latitude.to_numpy()[:, None],
            first_obs.longitude.to_numpy()[:, None],
            tables["ap_lats"][None, :],
            tables["ap_lons"][None, :],
        )
        nearest_ap_idx = np.argmin(d_ap, axis=1)
        track_origin_map = dict(zip(first_obs.index, tables["ap_ids"][nearest_ap_idx]))
        df["origin_ap"] = df["track_id"].map(track_origin_map)

        last_obs = (
            df.sort_values("timestamp_utc")
            .groupby(["window_start_utc", "source_aircraft_id"], sort=False)
            .last()
            .reset_index()
            .sort_values(["track_id", "window_start_utc"])
            .reset_index(drop=True)
        )

        all_lat = last_obs["latitude"].to_numpy(float)
        all_lon = last_obs["longitude"].to_numpy(float)
        all_hd = last_obs["heading_deg"].fillna(0.0).to_numpy(float)
        du_all = haversine_km_vec(all_lat[:, None], all_lon[:, None], tables["u_lat"][None, :], tables["u_lon"][None, :])
        dv_all = haversine_km_vec(all_lat[:, None], all_lon[:, None], tables["v_lat"][None, :], tables["v_lon"][None, :])
        detour_all = np.abs(du_all + dv_all - tables["e_dist"][None, :])
        ang_all = np.abs(((all_hd[:, None] - tables["e_brg"][None, :] + 180.0) % 360.0) - 180.0)
        best_e_all = np.argmin(detour_all + 0.5 * ang_all, axis=1)

        last_obs["matched_u"] = tables["e_u"][best_e_all]
        last_obs["matched_v"] = tables["e_v"][best_e_all]
        last_obs["rem_to_v_km"] = dv_all[np.arange(len(last_obs)), best_e_all]

        u_arr = last_obs["matched_u"].to_numpy()
        tid_arr = last_obs["track_id"].to_numpy()
        u_prev_arr = np.full(len(last_obs), "START", dtype=object)
        cur_tid = None
        last_u = "START"
        prev_distinct_u = "START"
        for idx_r in range(len(last_obs)):
            t_id = tid_arr[idx_r]
            u_curr = u_arr[idx_r]
            if t_id != cur_tid:
                cur_tid = t_id
                last_u = u_curr
                prev_distinct_u = "START"
            else:
                if u_curr != last_u:
                    prev_distinct_u = last_u
                    last_u = u_curr
            u_prev_arr[idx_r] = prev_distinct_u
        last_obs["u_prev"] = u_prev_arr

        alt0 = last_obs["altitude_m"].fillna(9450.0).to_numpy(float)
        vr0 = last_obs["vertical_rate_mps"].fillna(0.0).to_numpy(float)
        tstep = last_obs["track_step"].to_numpy(int)
        dt_to_wend = (last_obs["window_end_utc"] - last_obs["timestamp_utc"]).dt.total_seconds().to_numpy(float)

        already_landed = ((alt0 <= 24.0) & (tstep >= 3) & (vr0 <= 0.0)) | ((dt_to_wend > 90.0) & (alt0 < 2500.0) & (vr0 < -0.5))
        cont = last_obs[~already_landed].copy().reset_index(drop=True)

        matched_u = cont["matched_u"].to_numpy()
        matched_v = cont["matched_v"].to_numpy()
        u_prev_c = cont["u_prev"].to_numpy()
        rem_to_v_km = cont["rem_to_v_km"].to_numpy(float)
        cgs = cont["ground_speed_mps"].fillna(215.0).to_numpy(float)
        calt = cont["altitude_m"].fillna(9450.0).to_numpy(float)
        cvr = cont["vertical_rate_mps"].fillna(0.0).to_numpy(float)
        cdt = (cont["window_end_utc"] - cont["timestamp_utc"]).dt.total_seconds().to_numpy(float)
        corig = cont["origin_ap"].to_numpy()
        ccell = cont["cell_id"].to_numpy()
        cwstart = cont["window_start_utc"].to_numpy()
        cacid = cont["source_aircraft_id"].to_numpy()

        dist_step_km = np.zeros((len(cont), 11), dtype=float)
        cur_spd = cgs.copy()
        cur_alt = calt.copy()
        cur_d = np.zeros(len(cont), dtype=float)
        is_clmb = cvr > 2.0
        is_desc = cvr < -2.0
        for s_idx in range(11):
            cur_d = cur_d + (cur_spd * 90.0) / 1000.0
            dist_step_km[:, s_idx] = cur_d
            cur_alt = cur_alt + cvr * 90.0
            still_clmb = is_clmb & (cur_alt < 9800.0)
            cur_spd = np.where(still_clmb, np.minimum(222.0, cur_spd + 8.2), cur_spd)
            cur_spd = np.where(is_desc, np.maximum(116.7, cur_spd - 7.8), cur_spd)

        k_steps = np.arange(1, 12)[None, :] * 90.0
        in_next_win = (k_steps >= cdt[:, None] - 1.0) & (k_steps < cdt[:, None] + window_min * 60.0 - 1.0)
        ap_set = tables["ap_set"]
        v_is_ap = np.array([v in ap_set for v in matched_v])

        proj_unique_top1: dict[tuple[str, pd.Timestamp], set[str]] = {}
        proj_expected_prob: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
        proj_max_branch: dict[tuple[str, pd.Timestamp], set[str]] = {}
        proj_high_conf: dict[tuple[str, pd.Timestamp], set[str]] = {}
        proj_staying: dict[tuple[str, pd.Timestamp], set[str]] = {}
        proj_inflow: dict[tuple[str, pd.Timestamp], set[str]] = {}
        proj_obs_total: dict[tuple[str, pd.Timestamp], float] = {}

        lookup_4 = tables["lookup_4"]
        lookup_3 = tables["lookup_3"]
        lookup_2 = tables["lookup_2"]
        edge_frac_cells = tables["edge_frac_cells"]
        edge_len_map = tables["edge_len_map"]
        ap_cells = tables["ap_cells"]
        n_pts = tables["n_pts"]

        for i in range(len(cont)):
            u = matched_u[i]
            v = matched_v[i]
            up = u_prev_c[i]
            orig = corig[i]
            rem_v = rem_to_v_km[i]
            tot_u_v = max(edge_len_map.get((u, v), 1.0), 1.0)
            cur_f_on_uv = np.clip(1.0 - rem_v / tot_u_v, 0.0, 1.0)
            ws = pd.Timestamp(cwstart[i])
            acid = str(cacid[i])
            cur_cell = str(ccell[i])
            d_ticks = dist_step_km[i][in_next_win[i]]

            actions = (
                lookup_4.get((orig, up, u, v))
                or lookup_3.get((orig, u, v))
                or lookup_2.get((u, v))
                or [("STOP", v, 1.0)]
            )
            if v_is_ap[i]:
                if is_desc[i] or (calt[i] < 4500.0 and cvr[i] <= 0.0):
                    actions = [("STOP", v, 1.0)]
                elif calt[i] >= 9350.0 and cvr[i] >= -0.2 and rem_v < 248.0:
                    non_stop = [(k, tgt, p) for k, tgt, p in actions if k != "STOP"]
                    if non_stop:
                        s_p = sum(x[2] for x in non_stop)
                        actions = [(k, tgt, p / s_p) for k, tgt, p in non_stop]

            uv_cells = edge_frac_cells.get((u, v), [])
            before_mask = d_ticks <= rem_v
            if before_mask.any() and uv_cells:
                f_vals = np.clip(cur_f_on_uv + d_ticks[before_mask] / tot_u_v, 0.0, 1.0)
                idx_100 = np.clip(np.round(f_vals * n_pts).astype(int), 0, n_pts)
                for idx_c in idx_100:
                    cid = uv_cells[idx_c]
                    if not cid:
                        continue
                    key = (cid, ws)
                    proj_unique_top1.setdefault(key, set()).add(acid)
                    proj_max_branch.setdefault(key, set()).add(acid)
                    proj_high_conf.setdefault(key, set()).add(acid)
                    d_prob = proj_expected_prob.setdefault(key, {})
                    d_prob[acid] = max(d_prob.get(acid, 0.0), 1.0)
                    proj_obs_total[key] = proj_obs_total.get(key, 0.0) + 1.0
                    if cid == cur_cell:
                        proj_staying.setdefault(key, set()).add(acid)
                    else:
                        proj_inflow.setdefault(key, set()).add(acid)

            after_d = d_ticks[~before_mask] - rem_v
            if len(after_d) > 0:
                for b_idx, (kind, target, b_prob) in enumerate(actions):
                    if b_prob < 0.05:
                        continue
                    if kind == "STOP":
                        if rem_v > 1.5:
                            cid = ap_cells.get(v, "")
                            if cid:
                                key = (cid, ws)
                                if b_idx == 0:
                                    proj_unique_top1.setdefault(key, set()).add(acid)
                                    if cid == cur_cell:
                                        proj_staying.setdefault(key, set()).add(acid)
                                    else:
                                        proj_inflow.setdefault(key, set()).add(acid)
                                if b_prob >= 0.60:
                                    proj_high_conf.setdefault(key, set()).add(acid)
                                proj_max_branch.setdefault(key, set()).add(acid)
                                d_prob = proj_expected_prob.setdefault(key, {})
                                d_prob[acid] = min(1.0, d_prob.get(acid, 0.0) + b_prob)
                                proj_obs_total[key] = proj_obs_total.get(key, 0.0) + b_prob
                    else:
                        w = target
                        vw_cells = edge_frac_cells.get((v, w))
                        if not vw_cells:
                            continue
                        tot_v_w = max(edge_len_map.get((v, w), 200.0), 1.0)
                        valid_after = after_d[after_d <= tot_v_w]
                        if len(valid_after) == 0:
                            continue
                        idx_100 = np.clip(np.round((valid_after / tot_v_w) * n_pts).astype(int), 0, n_pts)
                        seen_in_branch = set()
                        for idx_c in idx_100:
                            cid = vw_cells[idx_c]
                            if not cid:
                                continue
                            key = (cid, ws)
                            proj_obs_total[key] = proj_obs_total.get(key, 0.0) + b_prob
                            if cid not in seen_in_branch:
                                seen_in_branch.add(cid)
                                if b_idx == 0:
                                    proj_unique_top1.setdefault(key, set()).add(acid)
                                    if cid == cur_cell:
                                        proj_staying.setdefault(key, set()).add(acid)
                                    else:
                                        proj_inflow.setdefault(key, set()).add(acid)
                                if b_prob >= 0.60:
                                    proj_high_conf.setdefault(key, set()).add(acid)
                                proj_max_branch.setdefault(key, set()).add(acid)
                                d_prob = proj_expected_prob.setdefault(key, {})
                                d_prob[acid] = min(1.0, d_prob.get(acid, 0.0) + b_prob)

        # Merge projected records into agg (including cells that had 0 current aircraft but >=1 inflow aircraft!)
        all_proj_keys = (
            set(proj_unique_top1.keys())
            | set(proj_expected_prob.keys())
            | set(proj_max_branch.keys())
        )
        existing_keys = set(zip(agg["cell_id"], agg["window_start_utc"]))
        new_keys = sorted(all_proj_keys - existing_keys)
        if new_keys:
            extra_df = pd.DataFrame(new_keys, columns=["cell_id", "window_start_utc"])
            for zc in ["aircraft_count", "observation_count", "late_window_count", "early_window_count", "new_dep_count", "landing_count"]:
                extra_df[zc] = 0
            agg = pd.concat([agg, extra_df], ignore_index=True)

        keys = list(zip(agg["cell_id"], agg["window_start_utc"]))
        agg["kinematic_proj_count"] = [len(proj_unique_top1.get(k, ())) for k in keys]
        agg["expected_proj_count"] = [float(sum(proj_expected_prob.get(k, {}).values())) for k in keys]
        agg["max_branch_proj_count"] = [len(proj_max_branch.get(k, ())) for k in keys]
        agg["high_conf_proj_count"] = [len(proj_high_conf.get(k, ())) for k in keys]
        agg["staying_proj_count"] = [len(proj_staying.get(k, ())) for k in keys]
        agg["inflow_proj_count"] = [len(proj_inflow.get(k, ())) for k in keys]
        agg["proj_obs_count"] = [float(proj_obs_total.get(k, 0.0)) for k in keys]
    else:
        agg["kinematic_proj_count"] = agg["aircraft_count"]
        agg["expected_proj_count"] = agg["aircraft_count"].astype(float)
        agg["max_branch_proj_count"] = agg["aircraft_count"]
        agg["high_conf_proj_count"] = agg["aircraft_count"]
        agg["staying_proj_count"] = agg["aircraft_count"]
        agg["inflow_proj_count"] = 0
        agg["proj_obs_count"] = agg["observation_count"].astype(float)

    agg["window_end_utc"] = agg.window_start_utc + pd.Timedelta(minutes=window_min)
    return agg.sort_values(["cell_id", "window_start_utc"]).reset_index(drop=True)


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
    """Zero-filled cell x window frame with leakage-safe lag, kinematic, and spatial features."""
    from app.services.ingestion.synthetic_demo import HUB_WEIGHTS, IST_PROFILE

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
        df["observation_count"] = df["aircraft_count"] * 4
    for col in ["mean_speed_mps", "speed_std_mps", "mean_altitude_m"]:
        if col not in df.columns:
            df[col] = np.nan

    df["window_end_utc"] = df.window_start_utc + pd.Timedelta(minutes=window_min)
    df = df.sort_values(["cell_id", "window_start_utc"]).reset_index(drop=True)

    # Ensure sub-window and kinematic columns exist with graceful causal fallbacks for pre-aggregated uploads
    for sub_col in ["late_window_count", "early_window_count", "new_dep_count", "landing_count"]:
        if sub_col not in df.columns:
            df[sub_col] = (df["aircraft_count"] * 0.5).round().astype(int) if "window" in sub_col else 0
        else:
            df[sub_col] = df[sub_col].fillna(0).astype(int)

    if "mean_vertical_rate_mps" not in df.columns:
        df["mean_vertical_rate_mps"] = 0.0

    has_kinematic = "expected_proj_count" in df.columns and df["expected_proj_count"].notna().any()
    if has_kinematic:
        for k_int in ["kinematic_proj_count", "max_branch_proj_count", "high_conf_proj_count", "staying_proj_count", "inflow_proj_count"]:
            df[k_int] = df[k_int].fillna(0).astype(int)
        for k_flt in ["expected_proj_count", "proj_obs_count"]:
            df[k_flt] = df[k_flt].fillna(0.0).astype(float)
    else:
        df["kinematic_proj_count"] = df["aircraft_count"]
        df["expected_proj_count"] = df["aircraft_count"].astype(float)
        df["max_branch_proj_count"] = df["aircraft_count"]
        df["high_conf_proj_count"] = df["aircraft_count"]
        df["staying_proj_count"] = (df["aircraft_count"] * 0.5).round().astype(int)
        df["inflow_proj_count"] = (df["aircraft_count"] * 0.5).round().astype(int)
        df["proj_obs_count"] = df["observation_count"].astype(float)

    g = df.groupby("cell_id")["aircraft_count"]
    df["count_lag1"] = g.shift(1)
    df["count_lag2"] = g.shift(2)
    df["count_lag4"] = g.shift(4)
    df["count_roll4_mean"] = g.transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["count_roll8_mean"] = g.transform(lambda s: s.rolling(8, min_periods=1).mean())
    df["traffic_change"] = df.aircraft_count - df.count_lag1
    df[TARGET_COUNT] = g.shift(-1)

    g_dep = df.groupby("cell_id")["new_dep_count"]
    df["new_dep_lag1"] = g_dep.shift(1).fillna(0)
    df["new_dep_roll4_mean"] = g_dep.transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["new_dep_roll8_mean"] = g_dep.transform(lambda s: s.rolling(8, min_periods=1).mean())
    df["subwindow_trend"] = df["late_window_count"] - df["early_window_count"]
    df["obs_per_aircraft"] = np.where(df["aircraft_count"] > 0, df["observation_count"] / df["aircraft_count"], 0.0)

    tables = _graph_kinematic_tables()
    cell_hub_w = tables.get("cell_hub_weight", {}) if tables else {}
    df["cell_hub_weight"] = df["cell_id"].map(lambda c: float(cell_hub_w.get(c, 0.0)))
    df["is_airport_cell"] = (df["cell_hub_weight"] > 0).astype(int)
    ist_hr = ((df["window_end_utc"].dt.hour + df["window_end_utc"].dt.minute / 60.0 + 5.5) % 24).astype(int)
    ist_mult = ist_hr.map(lambda h: float(IST_PROFILE[h]))
    total_hub_w = max(sum(HUB_WEIGHTS.values()), 1.0)
    total_ist_w = max(sum(IST_PROFILE), 1.0)
    df["diurnal_dep_prior"] = 45.0 * (df["cell_hub_weight"] / total_hub_w) * (ist_mult / (total_ist_w / 24.0)) / 24.0

    df["total_expected_load"] = df["expected_proj_count"] + np.where(
        df["is_airport_cell"] == 1,
        0.55 * df["diurnal_dep_prior"] + 0.45 * df["new_dep_roll4_mean"],
        0.0,
    )
    df["kinematic_plus_dep"] = df["kinematic_proj_count"] + df["diurnal_dep_prior"]
    df["proj_minus_current"] = df["expected_proj_count"] - df["aircraft_count"]
    df["branch_uncertainty"] = df["max_branch_proj_count"] - df["high_conf_proj_count"]
    df["load_dist_to_med"] = df["total_expected_load"] - 2.5
    df["load_dist_to_high"] = df["total_expected_load"] - 4.5
    df["proj_ge_5"] = (df["kinematic_proj_count"] >= 5).astype(int)
    df["proj_eq_4"] = (df["kinematic_proj_count"] == 4).astype(int)
    df["proj_ge_3"] = (df["kinematic_proj_count"] >= 3).astype(int)
    df["proj_eq_2"] = (df["kinematic_proj_count"] == 2).astype(int)

    # Spatial context: sum of 8-neighbour counts and projected counts in the current window
    pivot_cnt = df.pivot(index="window_start_utc", columns="cell_id", values="aircraft_count")
    pivot_proj = df.pivot(index="window_start_utc", columns="cell_id", values="expected_proj_count")
    nb = _neighbors(grid, cells)
    nsum_cnt = pd.DataFrame({c: pivot_cnt[nb[c]].sum(axis=1) if nb[c] else 0 for c in cells}, index=pivot_cnt.index)
    nsum_proj = pd.DataFrame({c: pivot_proj[nb[c]].sum(axis=1) if nb[c] else 0.0 for c in cells}, index=pivot_proj.index)

    nlong_cnt = nsum_cnt.stack().rename("neighbor_count").reset_index()
    nlong_cnt.columns = ["window_start_utc", "cell_id", "neighbor_count"]
    nlong_proj = nsum_proj.stack().rename("neighbor_proj_count").reset_index()
    nlong_proj.columns = ["window_start_utc", "cell_id", "neighbor_proj_count"]

    df = df.merge(nlong_cnt, on=["window_start_utc", "cell_id"], how="left").merge(nlong_proj, on=["window_start_utc", "cell_id"], how="left")
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
