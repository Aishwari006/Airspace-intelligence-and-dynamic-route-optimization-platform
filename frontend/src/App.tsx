import React, { useEffect, useState } from 'react';
import { Activity, AlertCircle, ArrowUpRight, Database, ShieldAlert } from 'lucide-react';
import { Header, tabLabels } from './components/Header';
import { ErrorBoundary } from './components/ErrorBoundary';
import { OverviewView } from './views/OverviewView';
import { AirspaceExplorerView } from './views/AirspaceExplorerView';
import { RouteLabView } from './views/RouteLabView';
import { ScenarioStudioView } from './views/ScenarioStudioView';
import { CongestionIntelligenceView } from './views/CongestionIntelligenceView';
import { ModelEvaluationView } from './views/ModelEvaluationView';
import { DataQualityView } from './views/DataQualityView';
import { DocumentationView } from './views/DocumentationView';
import { DatasetStudioView } from './views/DatasetStudioView';
import { DatasetExplorerView } from './views/DatasetExplorerView';
import { api } from './services/api';
import type { Airport, AirspaceNetwork, Scenario, ModelsStatusResponse, CongestionCell } from './types';

interface SystemStatus {
  status: string;
  database: string;
  activeScenario?: string | null;
}

interface LoadResult<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

async function settle<T>(request: Promise<T>): Promise<LoadResult<T>> {
  try {
    return { ok: true, data: await request };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Request failed' };
  }
}

export function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [activeDatasetId, setActiveDatasetId] = useState<string>('synthetic_demo_v1');
  const [routingOrigin, setRoutingOrigin] = useState<string>('VIDP');
  const [routingDestination, setRoutingDestination] = useState<string>('VABB');
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({ status: 'CHECKING', database: 'Unknown' });
  const [airports, setAirports] = useState<Airport[]>([]);
  const [network, setNetwork] = useState<AirspaceNetwork | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const [congestionCells, setCongestionCells] = useState<CongestionCell[]>([]);
  const [aircraft, setAircraft] = useState<Record<string, unknown>[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [loadWarning, setLoadWarning] = useState<string | null>(null);

  const handleSelectAirportForRouting = (icao: string, role: 'origin' | 'destination') => {
    if (role === 'origin') {
      setRoutingOrigin(icao);
    } else {
      setRoutingDestination(icao);
    }
    setActiveTab('routing');
  };

  useEffect(() => {
    let cancelled = false;

    const initApp = async () => {
      const [health, airportResult, networkResult, scenarioResult, modelResult, cellResult, aircraftResult] = await Promise.all([
        settle(api.getHealth()),
        settle(api.getAirports()),
        settle(api.getAirspaceNetwork()),
        settle(api.getScenarios()),
        settle(api.getModelsStatus()),
        settle(api.getCongestionCells()),
        settle(api.getAircraftObservations(150)),
      ]);

      if (cancelled) return;

      const healthData = health.data;
      const airportData = Array.isArray(airportResult.data)
        ? airportResult.data.filter((airport) => airport && typeof airport.node_id === 'string' && Number.isFinite(Number(airport.latitude)) && Number.isFinite(Number(airport.longitude))) as Airport[]
        : [];
      const networkPayload = networkResult.data;
      const safeNetwork = networkPayload && Array.isArray(networkPayload.airports) && Array.isArray(networkPayload.waypoints) && Array.isArray(networkPayload.edges)
        ? networkPayload
        : null;
      const scenarioData = Array.isArray(scenarioResult.data)
        ? scenarioResult.data.filter((scenario) => scenario && typeof scenario.id === 'string' && typeof scenario.name === 'string')
        : [];
      const activeScenario = scenarioData.find((scenario) => scenario.is_active);
      const failedParts = [
        !health.ok && 'health check',
        !airportResult.ok && 'airports',
        !networkResult.ok && 'airspace network',
        !scenarioResult.ok && 'scenarios',
        !modelResult.ok && 'model status',
        !cellResult.ok && 'congestion data',
        !aircraftResult.ok && 'aircraft observations',
      ].filter(Boolean);

      setSystemStatus({
        status: health.ok ? (healthData?.status || 'OK') : 'DEGRADED',
        database: healthData?.database || 'Unknown',
        activeScenario: activeScenario?.name ?? null,
      });
      setAirports(airportData);
      setNetwork(safeNetwork);
      setScenarios(scenarioData);
      setModelsStatus(modelResult.data ?? null);
      setCongestionCells(Array.isArray(cellResult.data) ? cellResult.data.filter((cell) => cell && typeof cell.cell_id === 'string') : []);
      const observations = aircraftResult.data && typeof aircraftResult.data === 'object'
        ? (aircraftResult.data as { observations?: Record<string, unknown>[] }).observations
        : [];
      setAircraft(Array.isArray(observations) ? observations : []);
      setLoadWarning(failedParts.length ? `Some data could not be loaded: ${failedParts.join(', ')}. Check the backend/API status.` : null);
      setInitialLoading(false);
    };

    void initApp();
    return () => { cancelled = true; };
  }, []);

  const refreshScenarios = async () => {
    try {
      const result = await api.getScenarios();
      const safeScenarios = Array.isArray(result)
        ? result.filter((scenario) => scenario && typeof scenario.id === 'string' && typeof scenario.name === 'string')
        : [];
      setScenarios(safeScenarios);
      const activeScenario = safeScenarios.find((scenario) => scenario.is_active);
      setSystemStatus((previous) => ({ ...previous, activeScenario: activeScenario?.name ?? null }));
    } catch (error) {
      console.error('Could not refresh scenarios:', error);
      setLoadWarning(error instanceof Error ? `Scenario refresh failed: ${error.message}` : 'Scenario refresh failed.');
    }
  };

  const activeTitle = tabLabels[activeTab] || 'Mission Control';
  const isHealthy = ['ok', 'online', 'healthy', 'operational'].includes((systemStatus.status || '').toLowerCase());

  return (
    <div className="app-shell">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} systemStatus={systemStatus} />

      <div className="app-main-column">
        <header className="app-topbar">
          <div className="topbar-context">
            <p className="eyebrow">SKY FORCE <span className="breadcrumb-slash">/</span> RESEARCH WORKSPACE</p>
            <h1>{activeTitle}</h1>
          </div>
          <div className="topbar-meta">
            <div className={`connection-pill ${isHealthy ? 'is-good' : systemStatus.status === 'CHECKING' ? 'is-pending' : 'is-warning'}`}>
              <span className="status-indicator" />
              <span>{isHealthy ? 'API connected' : systemStatus.status === 'CHECKING' ? 'Checking API' : 'API degraded'}</span>
            </div>
            <div className="data-mode-pill font-mono text-xs" title="Currently selected dataset context for analysis and routing">
              <span className="text-cyan-400 font-bold">DATASET:</span> {activeDatasetId}
            </div>
            <div className="data-mode-pill"><Database size={14} /> {systemStatus.database || 'DB unknown'}</div>
          </div>
        </header>

        {loadWarning && (
          <div className="app-load-warning" role="status">
            <AlertCircle size={16} />
            <span>{loadWarning}</span>
            <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">Open API docs <ArrowUpRight size={13} /></a>
          </div>
        )}

        <main className="app-content">
          {initialLoading && (
            <div className="loading-strip"><span className="loading-spinner" /> Loading airspace data and model status…</div>
          )}

          <div className="view-transition" key={activeTab}>
            <ErrorBoundary key={activeTab}>
            {activeTab === 'overview' && (
              <OverviewView
                network={network}
                airports={airports}
                modelsStatus={modelsStatus}
                congestionCells={congestionCells}
                aircraft={aircraft}
                onNavigate={setActiveTab}
              />
            )}
            {activeTab === 'studio' && (
              <DatasetStudioView
                activeDatasetId={activeDatasetId}
                onSelectActiveDataset={setActiveDatasetId}
                onNavigate={setActiveTab}
              />
            )}
            {activeTab === 'dataset-explorer' && (
              <DatasetExplorerView
                activeDatasetId={activeDatasetId}
              />
            )}
            {activeTab === 'explorer' && (
              <AirspaceExplorerView
                network={network}
                airports={airports}
                congestionCells={congestionCells}
                aircraft={aircraft}
                onSelectAirportForRouting={handleSelectAirportForRouting}
              />
            )}
            {activeTab === 'routing' && (
              <RouteLabView
                network={network}
                airports={airports}
                activeDatasetId={activeDatasetId}
                onSelectDataset={setActiveDatasetId}
                initialOrigin={routingOrigin}
                initialDestination={routingDestination}
              />
            )}
            {activeTab === 'scenarios' && <ScenarioStudioView scenarios={scenarios} onScenarioChange={refreshScenarios} airports={airports} network={network} />}
            {activeTab === 'congestion' && <CongestionIntelligenceView modelsStatus={modelsStatus} />}
            {activeTab === 'evaluation' && <ModelEvaluationView modelsStatus={modelsStatus} />}
            {activeTab === 'quality' && <DataQualityView />}
            {activeTab === 'docs' && <DocumentationView />}
            </ErrorBoundary>
          </div>
        </main>

        <footer className="app-footer">
          <span><Activity size={13} /> Sky Force · AI-driven airspace analysis</span>
          <span>Academic prototype · Not for operational use</span>
        </footer>
      </div>
    </div>
  );
}

export default App;
