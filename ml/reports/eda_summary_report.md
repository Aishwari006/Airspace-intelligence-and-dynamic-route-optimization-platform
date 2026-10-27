# Sky Force — Exploratory Data Analysis (EDA) & Spatiotemporal Profiling Report

## 1. Dataset Overview & Split Architecture
| Dataset | Total Clean Rows | Train Split | Validation Split | Test Split | Duplicate Rows |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Congestion Grid Windows** | `68,263` | `42,631` | `8,455` | `16,999` | `0` |
| **Segment ETA Traversals** | `17,223` | `10,746` | `2,105` | `4,356` | `0` |

## 2. Congestion Class Imbalance across Chronological Splits
| Split | LOW (count <= 2) | MEDIUM (3..4) | HIGH (>= 5) | Total |
| :--- | ---: | ---: | ---: | ---: |
| **Overall** | `60,387` (`88.46%`) | `5,655` (`8.28%`) | `2,221` (`3.25%`) | `68,263` |
| **Train** | `37,708` (`88.45%`) | `3,587` (`8.41%`) | `1,336` (`3.13%`) | `42,631` |
| **Validation** | `7,493` (`88.62%`) | `680` (`8.04%`) | `282` (`3.34%`) | `8,455` |
| **Test** | `15,009` (`88.29%`) | `1,387` (`8.16%`) | `603` (`3.55%`) | `16,999` |

## 3. Key Multivariate & Spatiotemporal Insights Guiding Feature Engineering
1. **Why Current-Window Cell Counts Alone Fail on Minority Congestion Classes**:
   - `aircraft_count` in window $t$ has correlation $r = 0.780$ with `next_window_count` ($t+1$) because cruising jets at $820\text{ km/h}$ travel $\sim 205\text{ km}$ per 15-minute window, crossing into downstream grid cells.
   - Projecting active aircraft trajectories forward along the directed airway graph from their last causal observation before `window_end_utc(t)` achieves **$r = 0.9675$** with `next_window_count`.
2. **Spatial Concentration at Major Terminal Hubs**:
   - Over 72% of `HIGH` congestion windows occur in or adjacent to the top 8 terminal hub cells (`VIDP`, `VABB`, `VOBL`, `VOMM`, `VOHS`, `VECC`, `VAAH`, `VOCI`) during diurnal departure waves.
3. **Non-Linear Aerodynamic Phase Regimes in Segment ETA**:
   - Segment traversal duration (`observed_segment_time_min`) is governed by inverse ground speed ($d / v_g$), where true airspeed ramps from $420\text{ km/h}$ to $820\text{ km/h}$ during climb (`from_airport == 1` with low entry altitude) and decelerates from $820\text{ km/h}$ to $420\text{ km/h}$ during terminal approach (`to_airport == 1` when the airport is the flight's final destination).
   - Crucial discovery: Several airport nodes (`VANP`, `VABB`, `VOBL`, `VAPO`, `VOHS`) act as **both** terminal destinations (where aircraft descend and slow down) and intermediate airway waypoints (where aircraft overfly at $820\text{ km/h}$ cruise). Incorporating causal track-prefix history (`origin_node`, `prev_node`, `alt_drop_from_peak_m`, `dist_to_cruise_fl_m`) resolves this bimodality.
