# Sky Force: Interactive AI Airspace Intelligence, Dynamic Route Optimization & Passenger Delay Forecasting

> **Operational & Safety Disclaimer:**  
> *Sky Force is an academic aviation intelligence and simulation research prototype. It is designed for strategic traffic flow research, decision support, route trade-off analysis, and passenger delay transparency. It is not approved for tactical air traffic separation or certified operational navigation.*

---

## 1. Project Overview & Core Mission

**Sky Force** is an interactive, full-stack aviation intelligence platform that demonstrates the complete data, machine learning, and decision-making pipeline behind modern airspace operations and flight delay communication.

The platform provides a unified system supporting two synchronized operational experiences:
1. **Passenger Portal**: Designed for air travelers who need transparent, honest flight delay updates. Communicates plain-language delay summaries, calibrated uncertainty ranges (e.g., 20 to 35 min), distinct gate vs runway departure times, an 8-stage operational progress timeline, transparent factor attribution (Confirmed, Model-inferred, Unavailable), and same-day alternative flights.
2. **Aviation Operations Console**: Full-fidelity airspace intelligence environment equipped with 3D airspace visualization, Dataset Studio for custom CSV ingestion, Closest Point of Approach (CPA) conflict detection, A* multi-objective route optimization with Pareto alternatives, and the **Flight Delay Intelligence** operations module with what-if scenario simulation.

---

## 2. System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SKY FORCE PASSENGER & OPERATIONS PORTAL                      │
│                  React 19 + TypeScript + Vite + Tailwind CSS                    │
│      MapLibre GL + deck.gl (3D) • Leaflet (2D) • Recharts • Lucide Icons       │
├────────────────────────────────────────┬────────────────────────────────────────┤
│           PASSENGER PORTAL             │          OPERATIONS CONSOLE            │
│  • Flight Search & Delay Status Cards  │  • Airspace 2D/3D Congestion Maps      │
│  • Calibrated Uncertainty Intervals    │  • Route Lab (Multi-Objective A*)      │
│  • Gate Pushback vs Take-Off Timeline  │  • Dataset Studio & Schema Mapping     │
│  • Contributing Factor Attribution     │  • Fleet Delay Inspector & Simulator   │
│  • Alternative Rebooking Options       │  • ML Model Evaluation & EDA Center    │
└────────────────────────────────────────┼────────────────────────────────────────┘
                                         │ HTTP REST / JSON
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI ASGI BACKEND                               │
│         Pydantic v2 • SQLite (WAL) / PostgreSQL • Structured JSON Logging       │
├───────────────────────┬──────────────────────────┬──────────────────────────────┤
│  SURVEILLANCE & QA    │     MACHINE LEARNING     │    GRAPH ROUTING & SCENARIOS │
│  • 9-Stage Cleanse    │  • Congestion Classifier │  • Multi-Objective A* / Dijs │
│  • Kinematic Filter   │    (RandomForest 3-Class)│  • Pareto Frontier Discovery │
│  • Track Segmentation │  • Segment ETA Regressor │  • Cost Model & Wx Penalty   │
│  • 2° Spatial Grid    │    (RandomForest Target) │  • Hard Disruption Closures  │
│  • CPA Conflict Scan  │  • Delay Forecasting Eng │  • Dynamic Reroute Deltas    │
├───────────────────────┴──────────────────────────┴──────────────────────────────┤
│  DATASET STUDIO ENGINE                               EXPLAINABILITY SERVICE     │
│  • Multi-Schema CSV Upload (Flights/Weather/Grid)    • Factor Cost Attribution  │
│  • Spatiotemporal Weather-Flight Linkage             • Ollama / Phi-3 Local LLM │
│  • Dataset-Specific Route Isolation                  • Deterministic Fallback   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PERSISTENCE & EVALUATION LAYER                           │
│  • SQLite Database (skyforce.db) with Alembic Versioning                        │
│  • Parquet Archive Storage (Snappy Compressed)                                  │
│  • Automated EDA & Data Profiling (ml/reports/eda_profiling_report.html)        │
│  • Evaluation Reports & Visualizations (ml/reports/figures/*.png, *.json, *.csv)│
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Capabilities & Implemented Features

### 3.1 Dual-Role User Experience (Sky Force 2.0)
- **Role Selection Landing**: Polished role selector and top-navigation role switcher allowing users to toggle between traveler and operations workflows seamlessly.
- **Passenger Experience**:
  - Transparent departure delay estimates with 80% confidence prediction intervals (e.g., "+20 to 35 min").
  - Clear separation between **Gate Pushback Time** ($STD \to \text{Off-Block}$) and **Runway Take-Off Time** ($STD + \text{Taxi-Out} + \text{Hold}$).
  - 8-stage operational progress tracker: Scheduled $\to$ Inbound Aircraft En Route $\to$ Inbound Landed $\to$ Aircraft at Gate $\to$ Boarding $\to$ Doors Closed $\to$ Taxi-Out $\to$ Departed.
  - Transparent factor attribution classifying delay drivers into **Confirmed Evidence**, **Model Estimates**, and **Unavailable Inputs**.
  - Alternative flight discovery highlighting same-day options with earlier departure or lower delay risk.
- **Operations Console**:
  - Fleet-wide delay distribution dashboard and route disruption monitor.
  - Deep-dive flight inspector with waterfall factor decomposition (inbound turnaround, runway queues, airspace congestion, convective weather).
  - Interactive What-If Scenario Simulator: assess downstream departure delay impact when modulating inbound delays ($0\text{--}90\text{ min}$), airport taxi queues ($5\text{--}40\text{ min}$), sector weather ($0\text{--}100\%$), or departure slot ground holds.
  - Direct bridge linking delayed flights to the Route Lab for rerouting analysis.

### 3.2 Airspace Visualization (Synchronous 2D & Zero-Key 3D)
- **Interactive 3D Airspace Map**: Powered by MapLibre GL JS and deck.gl (`frontend/src/components/Airspace3DMap.tsx`) using public vector tiles requiring **zero external API keys or tokens**.
- **Data-Driven 3D Extrusion**:
  - *Congestion Columns*: Sector columns extruded by model prediction ($\text{score} \times 25{,}000\text{ m}$), colored Green (`LOW`), Amber (`MEDIUM`), or Red (`HIGH`).
  - *Storm Columns*: Extruded convective weather cells with precipitation-driven height and color gradients.
  - *3D Airway Trajectories*: Great-circle flight paths elevated to flight levels ($8{,}000\text{ m} - 11{,}850\text{ m}$).
- **Synchronous 2D Leaflet Map**: Full-featured 2D alternate with waypoint markers, radar overlays, and along-track profile synchronization.

### 3.3 Multi-Objective Route Optimization & Pareto Frontier
- **Customizable Routing Objectives**:
  - `distance`: Shortest geodesic path along the Indian airway network.
  - `fastest`: Minimizes segment traversal duration predicted by ML regressor.
  - `weather`: Heavy quadratic penalty circumventing convective storm cells.
  - `congestion`: Avoids sectors predicted to experience high traffic density.
  - `balanced`: Multi-factor optimization balancing flight time, distance, and environmental risk.
- **Admissible $A^*$ Heuristic**: Great-circle Haversine lower bound guarantees search optimality without overestimating true costs.
- **Pareto Frontier Discovery**: Identifies non-dominated alternative candidate routes across competing trade-offs.

### 3.4 Multi-Schema Dataset Studio
- **Flexible Ingestion**: Ingests custom CSVs across three standardized schema types:
  1. *Aircraft Observations* (`source_aircraft_id`, `timestamp_utc`, `latitude`, `longitude`, `altitude_m`, `ground_speed_mps`).
  2. *Weather Grids* (`latitude`, `longitude`, `valid_time_utc`, `precipitation_mm`, `wind_speed_10m_ms`).
  3. *Pre-aggregated Congestion Grids* (`cell_id`, `window_start_utc`, `aircraft_count`).
- **9-Stage Data Quality Scrubber**: Rigorous sequential validation in `backend/app/services/preprocessing/quality.py` asserting header types, FIR boundaries ($6^\circ\text{N}-38^\circ\text{N}$, $68^\circ\text{E}-98^\circ\text{E}$), chronological order, physical speed/altitude bounds ($0\text{--}350\text{ m/s}$, $-100\text{--}15{,}000\text{ m}$), deduplication, kinematic acceleration ($\le 15\text{ m/s}^2$), vertical rate ($\le 40\text{ m/s}$), and track segmentation ($> 1{,}800\text{ s}$ gaps).

---

## 4. Machine Learning Models & Evaluation

Sky Force trains, registers, and serves two Scikit-Learn pipelines stored with SHA-256 verification in `ml/artifacts/`:

### Model A: Congestion Classifier (`RandomForestClassifier`)
- **Task**: 3-class classification forecasting traffic density (`LOW` $\le 2$, `MEDIUM` $3\text{--}4$, `HIGH` $\ge 5$ unique aircraft) in a 2° grid cell for the next 15-minute window ($t + 15\text{ min}$).
- **Breakthrough Feature Engineering**: Vectorized causal graph-kinematic projection ($r = 0.9682$ with future count), sub-window entry/exit dynamics, departure wave tracking, hub diurnal priors, and cost-sensitive class weights.
- **Evaluation Performance (Held-out Chronological Test Split, $N = 16{,}999$ observations)**:
  - **Accuracy**: **`97.24%`** (vs Baseline `84.70%` — an **82% reduction in error rate**)
  - **Balanced Accuracy**: **`88.46% – 88.81%`** (vs Baseline `71.39%`, **+17.4 percentage points**)
  - **Macro F1-Score**: **`0.8961`** (vs Baseline `0.6359`, **+26.0 percentage points**)
  - **MEDIUM Congestion Class**: Precision **`83.45%`** (vs `31.96%`), Recall **`83.27%`** (vs `71.52%`), F1 **`83.36%`** (vs `44.18%`)
  - **HIGH Congestion Class**: Precision **`90.73%`** (vs `52.76%`), Recall **`82.75%`** (vs `55.56%`), F1 **`86.56%`** (vs `54.12%`)

### Model B: Segment ETA Regressor (`RandomForestRegressor`)
- **Task**: Continuous regression estimating airway segment traversal time in **minutes**.
- **Breakthrough Feature Engineering**: Aerodynamic climb/cruise/descent phase equations, track-prefix history ($u_{\text{prev}}, origin$), flight level altitude bounds, multi-cell corridor weather, and corridor congestion.
- **Evaluation Performance (Held-out Chronological Test Split, $N = 4{,}356$ traversals)**:
  - **Mean Absolute Error (MAE)**: **`0.7357 min`** ($\approx 44.1\text{ seconds}$, **23.4% error reduction** vs `0.9608 min` baseline; **77.0% error reduction** vs `3.1956 min` heuristic)
  - **Root Mean Squared Error (RMSE)**: **`1.3574 min`** (vs Baseline `1.6007 min`)
  - **Median Absolute Error**: **`0.3381 min`** ($\approx 20.3\text{ seconds}$, **31.1% median error reduction**)
  - **Coefficient of Determination ($R^2$)**: **`0.9545` (95.45%)** (vs Baseline `0.9368`, cutting unexplained variance by **28.1%**)
  - **Systematic Bias**: **`-0.0259 min`** ($\approx -1.55\text{ seconds}$, **4.4x lower bias**)

---

## 5. Automated EDA, Outlier Analysis & Evaluation Artifacts

Sky Force includes automated profiling generators producing publication-grade documentation:

1. **Interactive HTML Profiling Report**:
   - `ml/reports/eda_profiling_report.html`: Standalone interactive profiling report containing univariate variable profiles, missingness rates, skewness, dispersion, and embedded diagnostic charts.
2. **Technical EDA & Outlier Reports**:
   - `ml/reports/eda_summary_report.md` & `ml/reports/eda_summary.json`: Detailed distributions, split statistics, and multivariate spatiotemporal correlation analysis.
   - `ml/reports/data_quality_and_outlier_report.md` & `ml/reports/data_quality_and_outlier_report.json`: Multi-perspective outlier audit (physical envelope rules, IQR bounds, and IsolationForest anomaly detection).
3. **High-Resolution Figures** in `ml/reports/figures/`:
   - `eda_congestion_class_distribution.png`: Split-wise class distributions and next-window count histogram.
   - `eda_spatiotemporal_congestion.png`: Diurnal UTC hourly congestion cycle and top 15 hub cell spatial concentration.
   - `eda_eta_distributions.png`: Segment duration distribution, distance vs time scatter by destination type, and entry ground speed boxplots.
   - `eda_correlation_heatmaps.png`: Full correlation matrices for congestion and ETA candidate features.

---

## 6. Complete Setup & Execution Guide

### 6.1 Prerequisites
- **Python**: `3.11.x` (recommended) or `3.12.x` / `3.13.x`
- **Node.js**: `v20+` or `v22+` with `npm`
- **Git**

### 6.2 Step 1: Clone the Repository
```powershell
git clone https://github.com/Aishwari006/Airspace-intelligence-and-dynamic-route-optimization-platform.git
cd Airspace-intelligence-and-dynamic-route-optimization-platform
```

### 6.3 Step 2: Backend Setup & Virtual Environment
```powershell
# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Linux/macOS: source .venv/bin/activate

# Upgrade pip and install all verified dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6.4 Step 3: Database Bootstrap & Model Training
```powershell
# 1. Bootstrap the SQLite database, seed synthetic demonstration flight world, and run data quality
python backend/scripts/bootstrap_demo.py

# 2. Train and serialize ML model artifacts (Congestion Classifier & ETA Regressor)
python backend/scripts/train_models.py

# 3. Generate exploratory data analysis reports, HTML profiling report, and outlier audit
python ml/generate_eda_and_quality_reports.py

# 4. Generate structured evaluation metric CSVs and summary reports
python ml/generate_evaluation_reports.py
```

### 6.5 Step 4: Run Automated Tests
Verify that all 42 automated tests pass cleanly:
```powershell
pytest backend/tests -v
```

### 6.6 Step 5: Start Backend API Server
```powershell
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```
- **API Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health Check**: [http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status)

### 6.7 Step 6: Frontend Console Setup & Execution
In a separate terminal window:
```powershell
cd frontend

# Install Node dependencies
npm install

# Verify TypeScript compilation and production bundle
npm run build

# Start the Vite development server
npm run dev
```
- **Sky Force Web Application**: [http://localhost:5173](http://localhost:5173)

---

## 7. How to Operate the Application

### A. Passenger Portal
1. Open [http://localhost:5173](http://localhost:5173).
2. Click **Passenger Portal** on the entry screen (or select "Passenger" in the top header).
3. Search for a flight (e.g., click quick suggestions like **SFD101**, **SFD124**, or **SFD210**).
4. Review the estimated departure time, calibrated delay range (+20 to 35 min), operational progress timeline, contributing factor breakdown, and same-day alternative options.

### B. Aviation Operations Console
1. Select "Operations" in the top navigation header.
2. **Airspace 2D / 3D Maps**: Inspect real-time 3D extruded congestion and storm columns across the Indian airway grid. Click any airway segment to inspect its along-track profile.
3. **Route Lab**: Select an origin and destination airport, choose a routing objective (`distance`, `fastest`, `weather`, `congestion`, or `balanced`), and click **Optimize Route** to view the Pareto-optimal frontier and trade-off decomposition.
4. **Flight Delay Intelligence (`delay-intelligence`)**: Inspect fleet-wide delay metrics, open any flight in the deep-dive factor inspector, and simulate what-if scenarios (adjust inbound turnaround delays or airport taxi holds).
5. **Dataset Studio**: Ingest custom CSV datasets (flight telemetry, gridded weather, or pre-aggregated congestion cells), map columns interactively, and trigger the 9-stage processing pipeline.
6. **Evaluation Center**: Review model performance metrics, confusion matrices, and feature importance rankings.

---

## 8. Remaining Roadmap & Future Enhancements

While Sky Force 2.0 provides a complete, self-contained demonstration and research platform, future production deployments can integrate:
1. **Live A-CDM Airport Feeds**: Ingest real-time Airport Collaborative Decision Making (A-CDM) milestones (Target Off-Block Time `TOBT`, Target Startup Approval Time `TSAT`) to replace demonstration turnaround proxies with certified airport ground data.
2. **Live ADS-B Receiver Integration**: Connect live OpenSky Network or ADS-B Exchange streaming feeds for real-time tactical aircraft tracking.
3. **Ground Delay Program (GDP) Optimizer**: Multi-airport ground delay slot allocation engine coordinating flow management across major hubs during severe monsoon events.
4. **En-Route Wake Vortex & Altitude Optimization**: Dynamic flight level step-climb optimization based on winds aloft and continuous aircraft gross weight depletion.

---

## 9. Automated Test Suite (42 Passing Tests)

The automated test suite (`pytest backend/tests -v`) validates end-to-end functionality:
- `test_api_endpoints.py`: Tests route contracts, optimization endpoints, benchmarks, and health checks.
- `test_conflict_cpa.py`: Validates 3D kinematic Closest Point of Approach calculations.
- `test_dataset_and_explanation.py`: Validates dataset isolation and LLM explanation fallbacks.
- `test_delay_intelligence.py`: Validates passenger delay prediction schemas, flight searches, confidence intervals, and what-if simulation endpoints.
- `test_geospatial.py`: Validates Haversine geodesy, initial bearings, and 2° grid cell indexing.
- `test_multi_schema_and_weather_linking.py`: Validates multi-schema CSV uploads and spatiotemporal weather linking.
- `test_routing_algorithms.py`: Validates Dijkstra, A* admissibility, hard edge closures, and Pareto frontier optimization.

---

## 10. Third-Party Notices & Attribution
- **MapLibre GL JS**: Open-source mapping engine licensed under the 3-Clause BSD License.
- **OpenFreeMap**: Public vector tile basemap hosting based on OpenStreetMap data (`© OpenStreetMap contributors · OpenFreeMap`).
- **deck.gl**: WebGL data-driven geospatial visualization engine developed by the Open Visualization Collaboration (Linux Foundation).
- **OurAirports**: Airport coordinates and metadata dedicated to the Public Domain under The Unlicense.
- **Open-Meteo**: Weather APIs and sample atmospheric feeds licensed under CC BY 4.0.
