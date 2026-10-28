export interface Airport {
  node_id: string;
  icao?: string;
  iata?: string;
  name: string;
  latitude: number;
  longitude: number;
  country?: string;
  elevation_m?: number;
  source?: string;
  provenance?: string;
}

export interface Waypoint {
  node_id: string;
  name: string;
  latitude: number;
  longitude: number;
  kind?: string;
  source?: string;
}

export interface RouteEdge {
  edge_id: string;
  source_node: string;
  target_node: string;
  distance_km: number;
  bearing_deg: number;
  is_synthetic?: boolean;
}

export interface AirspaceNetwork {
  airports: Airport[];
  waypoints: Waypoint[];
  edges: RouteEdge[];
  total_airports: number;
  total_waypoints: number;
  total_edges: number;
}

export interface CostWeights {
  distance: number;
  time: number;
  weather: number;
  congestion: number;
  uncertainty: number;
}

export interface RouteTotals {
  distance_km: number;
  estimated_time_min: number;
  weather_exposure_km: number;
  congestion_exposure_km: number;
  mean_uncertainty: number;
  composite_score: number;
  hops: number;
  closed_edges_in_route: number;
}

export interface RouteSegment {
  edge_id: string;
  from_node: string;
  to_node: string;
  distance_km: number;
  time_min: number;
  time_source: string;
  weather_severity: number | null;
  weather_source: string;
  congestion_score: number | null;
  congestion_class: string | null;
  congestion_probs: Record<string, number> | null;
  congestion_source: string;
  uncertainty: number;
  closed: boolean;
  closure_reasons: string[];
  event_ids: string[];
  flags: string[];
  cost_components: Record<string, number>;
  weighted_cost: Record<string, number>;
  cost_total: number;
}

export interface RouteDetail {
  found: boolean;
  label: string;
  algorithm: string;
  route_id?: string;
  dataset_id?: string;
  run_id?: string;
  path?: string[];
  geometry?: [number, number][];
  totals?: RouteTotals;
  cost_by_factor?: Record<string, number>;
  segments?: RouteSegment[];
  provenance?: Record<string, string[]>;
  search?: {
    nodes_expanded: number;
    latency_ms: number;
    search_cost?: number;
  };
  explanation?: string[];
  reason?: string;
}

export interface CompareRoutesResponse {
  origin: string;
  destination: string;
  scenario_id?: string;
  dataset_id?: string;
  run_id?: string;
  selected_objective: string;
  algorithm: string;
  routes: RouteDetail[];
  closed_edges_avoided: { edge_id: string; reasons: string[] }[];
  explanation: string[];
}


export interface ScenarioEvent {
  id: string;
  name: string;
  description: string;
  event_type: string;
  start_time_utc: string;
  end_time_utc: string;
  enabled: boolean;
  scope: Record<string, any>;
  severity: number;
  source_status: string;
}

export interface Scenario {
  id: string;
  name: string;
  description: string;
  is_prebuilt: boolean;
  is_active: boolean;
  seed: number;
  scenario_time_utc: string;
  created_at: string;
  updated_at: string;
  events: ScenarioEvent[];
}

export interface DynamicRerouteDelta {
  path_changed: boolean;
  distance_km_delta: number;
  estimated_time_min_delta: number;
  weather_penalty_delta: number;
  congestion_penalty_delta: number;
  composite_score_delta: number;
}

export interface DynamicRerouteResponse {
  scenario_id: string;
  scenario_name: string;
  origin: string;
  destination: string;
  baseline_route: RouteDetail;
  rerouted_route: RouteDetail;
  delta: DynamicRerouteDelta;
  explanation: string[];
  affected_events: any[];
}

export interface ConflictEvent {
  event_id: string;
  aircraft_1: string;
  callsign_1?: string;
  aircraft_2: string;
  callsign_2?: string;
  observation_time_utc: string;
  current_horizontal_dist_km: number;
  current_vertical_diff_m?: number | null;
  cpa_time_seconds: number;
  cpa_horizontal_dist_km: number;
  cpa_vertical_diff_m?: number | null;
  severity: 'LOW' | 'MEDIUM' | 'HIGH';
  is_projected: boolean;
  data_uncertainty_flags: string[];
  disclaimer: string;
}

export interface ConflictScanResponse {
  timestamp_utc: string;
  aircraft_scanned: number;
  pairs_evaluated: number;
  proximity_events_found: number;
  events: ConflictEvent[];
  parameters: Record<string, any>;
  disclaimer: string;
}

export interface CongestionCell {
  cell_id: string;
  aircraft_count: number;
  predicted_class: 'LOW' | 'MEDIUM' | 'HIGH';
  congestion_score: number;
  confidence?: number;
  center_lat: number;
  center_lon: number;
  bounds: [[number, number], [number, number]];
}

export interface ModelInfo {
  name: string;
  kind: string;
  version: string;
  task: string;
  metrics: Record<string, any>;
  feature_names: string[];
  feature_count: number;
  sha256: string;
  is_loaded: boolean;
}

export interface ModelsStatusResponse {
  status: string;
  congestion_model: ModelInfo | null;
  eta_model: ModelInfo | null;
  training_summary: Record<string, any>;
  disclaimer: string;
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  non_null_count: number;
  null_count: number;
  sample_values: any[];
}

export type DatasetType = 'aircraft_observations' | 'weather_observations' | 'congestion_cells';

export interface DatasetUploadResponse {
  dataset_id: string;
  filename: string;
  dataset_type: DatasetType;
  file_size_bytes: number;
  row_count: number;
  col_count: number;
  detected_columns: string[];
  suggested_mapping: Record<string, string>;
  columns_info: ColumnInfo[];
  is_valid_format: boolean;
  validation_warnings: string[];
  upload_timestamp_utc: string;
}

export interface DatasetSummaryItem {
  id: string;
  filename: string;
  dataset_type: DatasetType;
  file_size_bytes: number;
  row_count: number;
  col_count: number;
  uploaded_at_utc: string;
  status: string;
  is_active: boolean;
  pipeline_stages: Record<string, string>;
  provenance: string;
  linked_weather_dataset_id?: string | null;
}

export interface DatasetDetailResponse {
  id: string;
  filename: string;
  dataset_type: DatasetType;
  file_size_bytes: number;
  row_count: number;
  col_count: number;
  uploaded_at_utc: string;
  status: string;
  schema_mapping: Record<string, string>;
  validation_report: Record<string, any>;
  pipeline_runs: {
    stage: string;
    status: string;
    elapsed_seconds: number;
    summary?: Record<string, any>;
  }[];
  available_artifacts: Record<string, string>;
  active_in_routing: boolean;
  provenance: string;
  linked_weather_dataset_id?: string | null;
}

export interface LinkWeatherResponse {
  aircraft_dataset_id: string;
  weather_dataset_id: string;
  status: string;
  total_traffic_rows: number;
  matched_rows: number;
  unmatched_rows: number;
  coverage_percentage: number;
  warnings: string[];
  artifact_path: string;
}

export interface DatasetRawPreviewResponse {
  dataset_id: string;
  filename: string;
  total_rows: number;
  page: number;
  page_size: number;
  columns: string[];
  rows: Record<string, any>[];
}

export interface PipelineStageResult {
  stage: string;
  status: string;
  elapsed_seconds: number;
  output_summary: Record<string, any>;
  warnings: string[];
}

export interface RunPipelineResponse {
  dataset_id: string;
  status: string;
  total_elapsed_seconds: number;
  stages: PipelineStageResult[];
  artifact_paths: Record<string, string>;
}

export interface ExplanationResponse {
  task: string;
  summary: string;
  supporting_facts: string[];
  limitations: string[];
  engine: string;
  model_name: string;
  ollama_available?: boolean;
  gemini_available?: boolean;
  latency_ms: number;
  disclaimer: string;
}

export interface ExplanationEngineStatus {
  ollama_available?: boolean;
  gemini_available?: boolean;
  active_engine: string;
  model_name: string;
  ollama_url?: string;
  endpoint?: string;
}

export interface WeatherObservationItem {
  location_id: string;
  latitude: number;
  longitude: number;
  valid_time_utc: string;
  temperature_2m_c?: number;
  relative_humidity_2m_pct?: number;
  surface_pressure_hpa?: number;
  precipitation_mm?: number;
  cloud_cover_pct?: number;
  wind_speed_10m_ms?: number;
  wind_direction_10m_deg?: number;
  wind_gusts_10m_ms?: number;
  severity?: number;
  source?: string;
  product?: string;
  is_synthetic?: boolean;
}


// --- Sky Force 2.0: Flight Delay Intelligence & Passenger Types ---

export type EvidenceStatus = 'CONFIRMED' | 'MODEL_INFERRED' | 'UNAVAILABLE' | 'NOT_APPLICABLE';
export type FlightDelayStatus = 'ON_SCHEDULE' | 'DELAY_LIKELY' | 'SIGNIFICANT_DELAY_LIKELY' | 'DEPARTURE_UNCERTAIN' | 'CANCELLED';
export type UserRole = 'passenger' | 'operations';

export interface DelayContributingFactor {
  name: string;
  category: string;
  impact_minutes: number;
  evidence_status: EvidenceStatus;
  description: string;
  source: string;
}

export interface OperationalStage {
  stage_number: number;
  stage_id: string;
  stage_name: string;
  status: 'COMPLETED' | 'ACTIVE' | 'PROJECTED' | 'DELAYED' | 'UNAVAILABLE';
  timestamp_utc?: string | null;
  time_display: string;
  evidence_status: EvidenceStatus;
  source: string;
  explanation: string;
  is_confirmed: boolean;
}

export interface FlightSummary {
  flight_id: string;
  flight_number: string;
  callsign: string;
  airline: string;
  aircraft_tail: string;
  origin_id: string;
  origin_iata: string;
  origin_name: string;
  destination_id: string;
  destination_iata: string;
  destination_name: string;
  scheduled_departure_utc: string;
  estimated_departure_utc: string;
  scheduled_departure_display: string;
  estimated_departure_display: string;
  scheduled_arrival_utc?: string;
  estimated_arrival_utc?: string;
  scheduled_arrival_display?: string;
  estimated_arrival_display?: string;
  arrival_delay_minutes?: number;
  predicted_delay_minutes: number;
  delay_interval_min: number;
  delay_interval_max: number;
  status: FlightDelayStatus;
  severity: 'NOMINAL' | 'MODERATE' | 'HIGH';
  confidence_score: number;
  confidence_label: 'HIGH' | 'MEDIUM' | 'LOW';
  data_source: string;
  is_demonstration: boolean;
  last_updated_utc: string;
}

export interface FlightDetail extends FlightSummary {
  estimated_gate_departure_utc: string;
  estimated_takeoff_utc: string;
  scheduled_arrival_utc: string;
  estimated_arrival_utc: string;
  taxi_time_minutes: number;
  enroute_time_minutes: number;
  inbound_aircraft_id?: string | null;
  inbound_flight_callsign?: string | null;
  inbound_arrival_utc?: string | null;
  inbound_delay_minutes: number;
  turnaround_buffer_minutes: number;
  factors: DelayContributingFactor[];
  timeline: OperationalStage[];
  departure_weather_summary: Record<string, any>;
  departure_queue_size: number;
  enroute_distance_km: number;
  enroute_corridor_congestion: string;
  explanation_plain_english: string;
  alternative_flights: FlightSummary[];
}

export interface FlightSearchResponse {
  total: number;
  page: number;
  page_size: number;
  flights: FlightSummary[];
}

export interface AlternativeFlightsResponse {
  flight_id: string;
  origin_iata: string;
  destination_iata: string;
  alternatives: FlightSummary[];
}

export interface DelayPredictionRequest {
  flight_id: string;
  inbound_delay_minutes?: number | null;
  departure_queue_additional?: number;
  weather_severity_override?: number | null;
  slot_delay_minutes?: number;
}

export interface DelayDashboardResponse {
  total_flights_monitored: number;
  on_time_count: number;
  moderate_delay_count: number;
  significant_delay_count: number;
  average_predicted_delay_min: number;
  flights_requiring_attention: FlightSummary[];
  factor_breakdown: Record<string, number>;
  delay_distribution: { bracket: string; count: number; percentage: number }[];
  model_readiness: Record<string, any>;
  generated_at_utc: string;
}

export interface DataReadinessReport {
  status: string;
  model_type: string;
  current_capabilities: string[];
  missing_operational_labels: {
    field: string;
    category: string;
    importance: string;
    status: string;
    reason: string;
  }[];
  requirements_for_production: string[];
  disclaimer: string;
}
