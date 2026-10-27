# Sky Force — Data Quality, Missingness, and Outlier Analysis Report

## 1. Raw ADS-B / Telemetry Nine-Stage Data-Quality Pipeline
- **Total Raw Observations Inspected**: `274,775`
- **Clean Retained Observations**: `274,679`
- **Inferred Continuous Flight Tracks**: `3,678`
- **Removed Invalid Coordinates / Out-of-Range Lat-Lon**: `35`
- **Removed Impossible Speeds (>340 m/s or <0)**: `28`
- **Removed Exact Duplicate Telemetry Records**: `33`
- **Clipped Negative Altitudes (Sensor Noise near Ground)**: `0`

## 2. Missing-Value Audit & Leakage-Safe Imputation Strategy
### Model A — Congestion Classification (`68,263` cell-windows)
- **Zero-Occupied Cell Windows (`aircraft_count == 0`)**: When a grid cell has 0 aircraft during window $t$, kinematic summary columns (`mean_speed_mps`, `speed_std_mps`, `mean_altitude_m`, `mean_vertical_rate_mps`) are structurally `NaN` (`~78.9%` of all grid cell-windows).
- **Handling Strategy**:
  1. Structural absence of aircraft is directly represented by `aircraft_count == 0` and `obs_per_aircraft == 0`.
  2. Weather missingness is explicitly captured via the binary indicator `weather_missing`.
  3. All numerical imputation medians (`SimpleImputer(strategy="median")`) are fitted **exclusively on the training split** inside the scikit-learn `Pipeline`.

### Model B — Segment ETA Regression (`17,223` clean traversals)
- **Entry Speed (`entry_speed_mps`) Missingness**: `8` rows (`0.046%`) due to telemetry dropout at node passage.
- **Entry Altitude (`entry_altitude_m`) Missingness**: `340` rows (`1.974%`).
- **Handling Strategy**:
  1. Explicit binary missingness indicators (`entry_speed_missing`, `entry_altitude_missing`) are created prior to imputation so tree splits can distinguish imputed defaults from measured entry states.
  2. Edge-conditional and training-median imputation is fitted **strictly on the training split**.

## 3. Multi-Perspective Outlier Analysis (Physical, IQR, IsolationForest)
### Congestion Dataset (`42,631` training windows)
- **IQR Outliers on `aircraft_count`**: Because `78.9%` of cell-windows have `0` aircraft (`Q1 = 0, Q3 = 0, IQR = 0`), naive IQR filtering flags **every occupied window** as an outlier!
- **IsolationForest (`contamination=0.02`)**: Flagged `853` training rows, of which **`42.32%` (`361` rows)** are legitimate `MEDIUM` and `HIGH` congestion surge events.
- **Engineering Decision**: **RETAIN all high-traffic count observations.** Removing or winsorizing high traffic counts would erase the `HIGH` congestion class.

### Segment ETA Dataset (`17,223` raw traversals)
- **Physical Plausibility Filter**: Removed `0` traversals with implied speed outside `[150, 1300] km/h` (caused by track-ID stitching across turnarounds), retaining `17,223` valid traversals (`10,746` train, `2,105` validation, `4,356` test).
- **Sensor Noise Bounding**: Entry speeds are bounded to physical commercial jet operating envelopes (`[60, 310] m/s`) before computing inverse-speed kinematic ratios (`distance / speed`) to prevent division-by-near-zero spikes.
