import type {
  AirspaceNetwork,
  CompareRoutesResponse,
  ConflictScanResponse,
  DynamicRerouteResponse,
  ModelsStatusResponse,
  RouteDetail,
  Scenario,
} from '../types';

const API_BASE = '/api';

export async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const { headers: requestHeaders, ...requestOptions } = options || {};
  let response: Response;

  try {
    response = await fetch(`${API_BASE}${endpoint}`, {
      ...requestOptions,
      headers: {
        Accept: 'application/json',
        ...(requestOptions.body ? { 'Content-Type': 'application/json' } : {}),
        ...requestHeaders,
      },
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : 'Network request failed';
    throw new Error(`API connection failed for ${endpoint}: ${reason}. Is the FastAPI server running on port 8000?`);
  }

  const responseText = await response.text();
  let payload: unknown = null;
  if (responseText) {
    try {
      payload = JSON.parse(responseText);
    } catch {
      payload = responseText;
    }
  }

  if (!response.ok) {
    const body = payload && typeof payload === 'object' ? payload as Record<string, unknown> : null;
    const nestedError = body?.error && typeof body.error === 'object' ? body.error as Record<string, unknown> : null;
    const detail = body?.detail || nestedError?.message || body?.message || (typeof payload === 'string' ? payload : null);
    throw new Error(typeof detail === 'string' ? detail : `HTTP ${response.status}: ${response.statusText}`);
  }

  if (payload === null) return null as T;
  return payload as T;
}

export const api = {
  getHealth: () => fetchJson<{ status: string; app_name: string; database: string; disclaimer: string }>('/health'),
  getMeta: () => fetchJson<any>('/meta'),
  getDataSourcesStatus: () => fetchJson<any[]>('/data-sources/status'),
  getDataQualitySummary: () => fetchJson<any>('/data-quality/summary'),
  
  getAirports: () => fetchJson<any[]>('/airports'),
  getAirspaceNetwork: () => fetchJson<AirspaceNetwork>('/airports/network'),
  
  getAircraftObservations: (params?: number | { limit?: number; timestamp_utc?: string; window_seconds?: number; dataset_id?: string }) => {
    if (typeof params === 'number') {
      return fetchJson<any>(`/aircraft/observations?limit=${params}`);
    }
    const q = new URLSearchParams();
    if (params?.limit) q.set('limit', String(params.limit));
    if (params?.timestamp_utc) q.set('timestamp_utc', params.timestamp_utc);
    if (params?.window_seconds) q.set('window_seconds', String(params.window_seconds));
    if (params?.dataset_id) q.set('dataset_id', params.dataset_id);
    const queryStr = q.toString();
    return fetchJson<any>(`/aircraft/observations${queryStr ? `?${queryStr}` : ''}`);
  },
  getWeather: (params?: { time_utc?: string; dataset_id?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.time_utc) q.set('time_utc', params.time_utc);
    if (params?.dataset_id) q.set('dataset_id', params.dataset_id);
    if (params?.limit) q.set('limit', String(params.limit));
    const queryStr = q.toString();
    return fetchJson<any>(`/weather${queryStr ? `?${queryStr}` : ''}`);
  },
  
  getCongestionCells: (params?: { time_utc?: string; dataset_id?: string }) => {
    const q = new URLSearchParams();
    if (params?.time_utc) q.set('time_utc', params.time_utc);
    if (params?.dataset_id) q.set('dataset_id', params.dataset_id);
    const queryStr = q.toString();
    return fetchJson<any[]>(`/congestion/cells${queryStr ? `?${queryStr}` : ''}`);
  },
  getCongestionTimeSeries: (cellId: string, datasetId?: string) =>
    fetchJson<any>(`/congestion/timeseries?cell_id=${encodeURIComponent(cellId)}${datasetId ? `&dataset_id=${encodeURIComponent(datasetId)}` : ''}`),
  
  // Datasets Studio Endpoints
  getDatasets: () => fetchJson<any[]>('/datasets'),
  getDataset: (id: string) => fetchJson<any>(`/datasets/${encodeURIComponent(id)}`),
  getDatasetPreview: (id: string, page = 1, pageSize = 50) =>
    fetchJson<any>(`/datasets/${encodeURIComponent(id)}/preview?page=${page}&page_size=${pageSize}`),
  uploadDataset: async (file: File, datasetType = 'aircraft_observations') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataset_type', datasetType);
    const resp = await fetch('/api/datasets/upload', {
      method: 'POST',
      body: formData,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err?.error?.message || err?.detail || 'Upload failed');
    }
    return resp.json();
  },
  deleteDataset: (id: string) =>
    fetchJson<{ success: boolean; message: string }>(`/datasets/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  runDatasetPipeline: (id: string, payload: { stages: string[]; custom_mapping?: Record<string, string> }) =>
    fetchJson<any>(`/datasets/${encodeURIComponent(id)}/pipeline`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  linkWeather: (aircraftId: string, weatherId: string) =>
    fetchJson<any>(`/datasets/${encodeURIComponent(aircraftId)}/link-weather`, {
      method: 'POST',
      body: JSON.stringify({ weather_dataset_id: weatherId }),
    }),
  getDatasetArtifactUrl: (id: string, artifact: string) =>
    `/api/datasets/${encodeURIComponent(id)}/export/${encodeURIComponent(artifact)}`,

  // Explanation Endpoints
  getExplanationStatus: () => fetchJson<any>('/explain/status'),
  getExplanation: (payload: { task: string; context_data: any; user_query?: string }) =>
    fetchJson<any>('/explain', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  optimizeRoute: (payload: {
    origin: string;
    destination: string;
    algorithm?: string;
    objective?: string;
    weights?: Record<string, number>;
    scenario_id?: string;
    dataset_id?: string;
    weather_dataset_id?: string;
    run_id?: string;
  }) => fetchJson<RouteDetail>('/routes/optimize', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  
  compareRoutes: (payload: {
    origin: string;
    destination: string;
    algorithm?: string;
    objective?: string;
    scenario_id?: string;
    dataset_id?: string;
    weather_dataset_id?: string;
    run_id?: string;
  }) => fetchJson<CompareRoutesResponse>('/routes/compare', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  
  getParetoRoutes: (origin: string, destination: string, scenarioId?: string, datasetId?: string) =>
    fetchJson<any>(`/routes/pareto?origin=${encodeURIComponent(origin)}&destination=${encodeURIComponent(destination)}${scenarioId ? `&scenario_id=${encodeURIComponent(scenarioId)}` : ''}${datasetId ? `&dataset_id=${encodeURIComponent(datasetId)}` : ''}`),
  
  getScenarios: () => fetchJson<Scenario[]>('/scenarios'),
  activateScenario: (scenarioId: string) => fetchJson<Scenario>(`/scenarios/${encodeURIComponent(scenarioId)}/activate`, { method: 'POST' }),
  resetScenario: (scenarioId: string) => fetchJson<Scenario>(`/scenarios/${encodeURIComponent(scenarioId)}/reset`, { method: 'POST' }),
  dynamicReroute: (payload: {
    origin: string;
    destination: string;
    scenario_id: string;
    objective?: string;
    algorithm?: string;
  }) => fetchJson<DynamicRerouteResponse>('/scenarios/reroute', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  
  getLatestConflicts: (horizon = 300, hThresh = 9.26, vThresh = 300) =>
    fetchJson<ConflictScanResponse>(`/conflicts/latest?horizon_seconds=${encodeURIComponent(String(horizon))}&horiz_threshold_km=${encodeURIComponent(String(hThresh))}&vert_threshold_m=${encodeURIComponent(String(vThresh))}`),
  
  getModelsStatus: () => fetchJson<ModelsStatusResponse>('/models/status'),
  getModelsMetrics: () => fetchJson<any>('/models/metrics'),
  getFeatureImportance: () => fetchJson<any>('/models/feature-importance'),
  
  getExperiments: () => fetchJson<any[]>('/experiments'),
  runExperiment: (payload: { od_pairs: string[][]; scenario_ids: string[]; algorithm?: string }) =>
    fetchJson<any>('/experiments/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
};

