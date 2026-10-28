import React, { useEffect, useState } from 'react';
import { Activity, AlertCircle, CheckCircle2, FileJson, Play, RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import type { ModelsStatusResponse } from '../types';
import { asRecord, findCandidateMetricRows, findMetric, findNamedSection, formatNumber, formatPercent } from '../services/metricUtils';

interface ModelEvaluationViewProps {
  modelsStatus?: ModelsStatusResponse | null;
}

const MetricCard: React.FC<{ label: string; value: string; note: string }> = ({ label, value, note }) => (
  <div className="evaluation-metric-card">
    <span>{label}</span>
    <strong>{value}</strong>
    <small>{note}</small>
  </div>
);

export const ModelEvaluationView: React.FC<ModelEvaluationViewProps> = ({ modelsStatus = null }) => {
  const [metrics, setMetrics] = useState<unknown>(null);
  const [loadingMetrics, setLoadingMetrics] = useState(true);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [experimentResult, setExperimentResult] = useState<Record<string, unknown> | null>(null);
  const [runningExp, setRunningExp] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadMetrics = async () => {
    setLoadingMetrics(true);
    setMetricsError(null);
    try {
      const result = await api.getModelsMetrics();
      setMetrics(result);
    } catch (loadError) {
      setMetricsError(loadError instanceof Error ? loadError.message : 'Model metrics could not be loaded.');
    } finally {
      setLoadingMetrics(false);
    }
  };

  useEffect(() => { void loadMetrics(); }, []);

  const congestionModel = modelsStatus?.congestion_model;
  const etaModel = modelsStatus?.eta_model;

  const rawCongestion = asRecord(asRecord(metrics)?.congestion);
  const rawEta = asRecord(asRecord(metrics)?.eta);
  const rawSelected = asRecord(asRecord(metrics)?.selected);

  // Active loaded / selected model metrics
  const activeCongMetrics = asRecord(congestionModel?.metrics) ||
    asRecord(rawSelected?.congestion) ||
    asRecord(asRecord(rawCongestion?.random_forest)?.test) ||
    asRecord(rawCongestion?.random_forest) ||
    findNamedSection(metrics, ['congestion']) ||
    null;

  const activeEtaMetrics = asRecord(etaModel?.metrics) ||
    asRecord(rawSelected?.eta) ||
    asRecord(asRecord(rawEta?.random_forest)?.test) ||
    asRecord(rawEta?.random_forest) ||
    findNamedSection(metrics, ['eta', 'regression']) ||
    null;

  const classifierAccuracy = findMetric(activeCongMetrics, ['test_accuracy', 'accuracy', 'accuracy_score']);
  const classifierBalanced = findMetric(activeCongMetrics, ['balanced_accuracy', 'balanced_acc']);
  const classifierF1 = findMetric(activeCongMetrics, ['macro_f1', 'f1_macro', 'macro_f1_score']);
  const etaMae = findMetric(activeEtaMetrics, ['test_mae', 'mae_min', 'mae', 'mean_absolute_error']);
  const etaRmse = findMetric(activeEtaMetrics, ['test_rmse', 'rmse_min', 'rmse', 'root_mean_squared_error']);
  const etaR2 = findMetric(activeEtaMetrics, ['test_r2', 'r2', 'r2_score', 'r_squared']);
  const classificationCandidates = findCandidateMetricRows(metrics, 'classification');
  const regressionCandidates = findCandidateMetricRows(metrics, 'regression');

  const handleRunBenchmark = async () => {
    setRunningExp(true);
    setError(null);
    try {
      const result = await api.runExperiment({
        od_pairs: [['VIDP', 'VABB'], ['VABB', 'VIDP']],
        scenario_ids: ['clear', 'corridor-congestion', 'corridor-closure'],
        algorithm: 'dijkstra',
      });
      setExperimentResult(asRecord(result) || { result });
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : 'Failed to run benchmark experiment.');
    } finally {
      setRunningExp(false);
    }
  };

  const reportAvailable = metrics !== null && metrics !== undefined;

  return (
    <div className="space-y-6 page-model-evaluation">
      <section className="view-hero view-hero-cyan">
        <div className="view-hero-copy">
          <p className="eyebrow">SCIENTIFIC VALIDATION</p>
          <h2>Model evaluation & benchmarking</h2>
          <p>Inspect saved evaluation results, verify which artifacts are loaded, and run a route benchmark against configured scenarios.</p>
        </div>
        <div className="view-hero-actions">
          <button type="button" className="action-secondary" onClick={() => void loadMetrics()} disabled={loadingMetrics}><RefreshCw size={15} /> Refresh reports</button>
          <button type="button" className="action-primary" onClick={handleRunBenchmark} disabled={runningExp}><Play size={15} /> {runningExp ? 'Running…' : 'Run route benchmark'}</button>
        </div>
      </section>

      {metricsError && <div className="view-inline-error" role="alert"><AlertCircle size={16} /><div><strong>Saved metrics unavailable</strong><p>{metricsError}</p></div></div>}
      {error && <div className="view-inline-error" role="alert"><AlertCircle size={16} /><div><strong>Benchmark failed</strong><p>{error}</p></div></div>}

      <section className="evaluation-section-grid">
        <article className="surface-card evaluation-model-panel">
          <div className="surface-card-heading">
            <div><p className="eyebrow">MODEL A · CLASSIFICATION</p><h3>Congestion classifier</h3><p className="surface-subtitle">Forecasts LOW / MEDIUM / HIGH traffic class for the next 15-minute window.</p></div>
            <span className={`status-tag ${congestionModel?.is_loaded ? 'tag-good' : 'tag-muted'}`}>{!modelsStatus ? 'STATUS UNKNOWN' : congestionModel?.is_loaded ? 'LOADED' : 'NOT LOADED'}</span>
          </div>
          <div className="model-fact-grid">
            <div><span>Algorithm</span><strong>{congestionModel?.name || 'Random Forest'}</strong></div>
            <div><span>Version</span><strong>{congestionModel?.version || '—'}</strong></div>
            <div><span>Features</span><strong>{congestionModel?.feature_count ?? '—'}</strong></div>
            <div><span>Artifact SHA-256</span><strong title={congestionModel?.sha256 || 'Not reported'}>{congestionModel?.sha256 ? `${congestionModel.sha256.slice(0, 12)}…` : 'Not reported'}</strong></div>
          </div>
          <div className="evaluation-metric-grid">
            <MetricCard label="Test accuracy" value={formatPercent(classifierAccuracy)} note="From returned evaluation data" />
            <MetricCard label="Balanced accuracy" value={formatPercent(classifierBalanced)} note="Class-balanced performance" />
            <MetricCard label="Macro-F1" value={formatNumber(classifierF1, 4)} note="Equal weight per class" />
          </div>
        </article>

        <article className="surface-card evaluation-model-panel">
          <div className="surface-card-heading">
            <div><p className="eyebrow">MODEL B · REGRESSION</p><h3>Segment ETA regressor</h3><p className="surface-subtitle">Estimates the traversal time of each airway segment in minutes.</p></div>
            <span className={`status-tag ${etaModel?.is_loaded ? 'tag-good' : 'tag-muted'}`}>{!modelsStatus ? 'STATUS UNKNOWN' : etaModel?.is_loaded ? 'LOADED' : 'NOT LOADED'}</span>
          </div>
          <div className="model-fact-grid">
            <div><span>Algorithm</span><strong>{etaModel?.name || 'Random Forest regressor'}</strong></div>
            <div><span>Version</span><strong>{etaModel?.version || '—'}</strong></div>
            <div><span>Features</span><strong>{etaModel?.feature_count ?? '—'}</strong></div>
            <div><span>Artifact SHA-256</span><strong title={etaModel?.sha256 || 'Not reported'}>{etaModel?.sha256 ? `${etaModel.sha256.slice(0, 12)}…` : 'Not reported'}</strong></div>
          </div>
          <div className="evaluation-metric-grid">
            <MetricCard label="Test MAE" value={etaMae === null ? '—' : `${formatNumber(etaMae, 3)} min`} note="Lower is better" />
            <MetricCard label="Test RMSE" value={etaRmse === null ? '—' : `${formatNumber(etaRmse, 3)} min`} note="Penalizes larger errors" />
            <MetricCard label="R² score" value={formatNumber(etaR2, 4)} note="Fit versus mean baseline" />
          </div>
        </article>
      </section>

      <section className="surface-card candidate-benchmarks-panel">
        <div className="surface-card-heading">
          <div><p className="eyebrow">BASELINE COMPARISON</p><h3>Candidate model results</h3><p className="surface-subtitle">Tables are built from candidate-level records returned by the saved metrics API; no sample scores are inserted by the frontend.</p></div>
          <FlaskConicalIcon />
        </div>
        <div className="candidate-tables-grid">
          <div>
            <h4>Congestion classification</h4>
            {classificationCandidates.length ? (
              <div className="overflow-x-auto"><table className="candidate-comparison-table w-full text-left text-xs">
                <thead><tr><th>Candidate</th><th>Accuracy</th><th>Balanced accuracy</th><th>Macro-F1</th></tr></thead>
                <tbody>{classificationCandidates.map((row, index) => <tr key={`${row.name}-${index}`}><td>{row.name}</td><td>{formatPercent(row.accuracy)}</td><td>{formatPercent(row.balancedAccuracy)}</td><td>{formatNumber(row.macroF1, 4)}</td></tr>)}</tbody>
              </table></div>
            ) : <p className="candidate-empty-note">No candidate-level classifier table was found in the metrics response. Inspect the raw report below for the available fields.</p>}
          </div>
          <div>
            <h4>Segment ETA regression</h4>
            {regressionCandidates.length ? (
              <div className="overflow-x-auto"><table className="candidate-comparison-table w-full text-left text-xs">
                <thead><tr><th>Candidate</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
                <tbody>{regressionCandidates.map((row, index) => <tr key={`${row.name}-${index}`}><td>{row.name}</td><td>{row.mae === null ? '—' : `${formatNumber(row.mae)} min`}</td><td>{row.rmse === null ? '—' : `${formatNumber(row.rmse)} min`}</td><td>{formatNumber(row.r2, 4)}</td></tr>)}</tbody>
              </table></div>
            ) : <p className="candidate-empty-note">No candidate-level regressor table was found in the metrics response. Inspect the raw report below for the available fields.</p>}
          </div>
        </div>
      </section>

      <section className="surface-card evaluation-integrity-panel">
        <div className="surface-card-heading">
          <div><p className="eyebrow">REPRODUCIBILITY & INTEGRITY</p><h3>What this page can verify</h3></div>
          <ShieldCheckIcon />
        </div>
        <div className="integrity-check-grid">
          <div className="integrity-check"><CheckCircle2 size={17} /><span><strong>Artifact load status</strong><small>{modelsStatus ? 'Read from /api/models/status' : 'Model status API has not returned data'}</small></span></div>
          <div className="integrity-check"><CheckCircle2 size={17} /><span><strong>Metric source</strong><small>{reportAvailable ? 'Loaded from /api/models/metrics' : 'No report payload is available yet'}</small></span></div>
          <div className="integrity-check"><Activity size={17} /><span><strong>Evaluation split</strong><small>Confirm the chronological split details in the saved training report.</small></span></div>
        </div>
        {loadingMetrics && <div className="loading-strip"><span className="loading-spinner" /> Loading model evaluation reports…</div>}
        {reportAvailable && (
          <details className="raw-report-details">
            <summary><FileJson size={15} /> Inspect raw metrics response</summary>
            <pre>{JSON.stringify(metrics, null, 2)}</pre>
          </details>
        )}
        {!reportAvailable && !loadingMetrics && !metricsError && <p className="empty-report-note">The endpoint returned no metrics payload. Run model training and confirm the report files are exposed by the backend before presenting scores.</p>}
      </section>

      {experimentResult && (
        <section className="surface-card">
          <div className="surface-card-heading"><div><p className="eyebrow">ON-DEMAND ROUTING EXPERIMENT</p><h3>Benchmark result</h3></div><span className="status-tag tag-good">RESPONSE RECEIVED</span></div>
          <pre className="experiment-json">{JSON.stringify(experimentResult, null, 2)}</pre>
        </section>
      )}
    </div>
  );
};

const ShieldCheckIcon = () => <CheckCircle2 className="text-emerald-300" size={19} />;
const FlaskConicalIcon = () => <Activity className="text-cyan-300" size={19} />;
