import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Cpu, 
  Layers, 
  TrendingUp, 
  Activity,
  AlertCircle,
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { ModelsStatusResponse, CongestionCell } from '../types';
import { api } from '../services/api';
import { findMetric, formatNumber, formatPercent } from '../services/metricUtils';

interface CongestionIntelligenceViewProps {
  modelsStatus: ModelsStatusResponse | null;
}

export const CongestionIntelligenceView: React.FC<CongestionIntelligenceViewProps> = ({ modelsStatus }) => {
  const [cells, setCells] = useState<CongestionCell[]>([]);
  const [featureImportance, setFeatureImportance] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCell, setSelectedCell] = useState<string | null>(null);
  const [dataError, setDataError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [cellResult, importanceResult] = await Promise.allSettled([
          api.getCongestionCells(),
          api.getFeatureImportance(),
        ]);
        if (cellResult.status === 'fulfilled' && Array.isArray(cellResult.value)) {
          setCells(cellResult.value);
        } else {
          setCells([]);
          setDataError('Congestion cell data could not be loaded from the backend.');
        }
        if (importanceResult.status === 'fulfilled') {
          const payload = importanceResult.value as { congestion?: unknown[] } | unknown[];
          const rows = Array.isArray(payload) ? payload : Array.isArray(payload?.congestion) ? payload.congestion : [];
          setFeatureImportance(rows.map((item) => {
            const row = item && typeof item === 'object' ? item as Record<string, unknown> : {};
            const feature = row.feature ?? row.feature_name ?? row.name ?? row.key;
            const importance = row.importance ?? row.value ?? row.score;
            return { feature: String(feature ?? 'Unknown feature'), importance: Number(importance) };
          }).filter((row) => row.feature !== 'Unknown feature' && Number.isFinite(row.importance))); 
        } else {
          setFeatureImportance([]);
          setDataError((current) => current || 'Feature-importance report could not be loaded.');
        }
      } catch (e) {
        setDataError(e instanceof Error ? e.message : 'Congestion data could not be loaded.');
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  const congModel = modelsStatus?.congestion_model;
  const modelMetrics = congModel?.metrics || {};
  const accuracy = findMetric(modelMetrics, ['test_accuracy', 'accuracy', 'accuracy_score']);
  const balancedAccuracy = findMetric(modelMetrics, ['balanced_accuracy', 'balanced_acc']);
  const macroF1 = findMetric(modelMetrics, ['macro_f1', 'f1_macro', 'macro_f1_score']);
  const metricSampleCount = findMetric(modelMetrics, ['test_samples', 'n_test_samples', 'test_size', 'test_records']);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg view-hero view-hero-green">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2 font-mono">
              <BarChart3 className="w-5 h-5 text-emerald-400" />
              CONGESTION INTELLIGENCE & FORECASTING
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Causal spatio-temporal machine-learning model forecasting traffic density for the next 15-minute window
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="px-2.5 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold">
              {congModel?.name ? `CLASSIFIER: ${congModel.name.toUpperCase()}` : 'CLASSIFIER: MODEL NOT REPORTED'}
            </span>
            <span className="px-2.5 py-1 rounded bg-aerospace-800 text-slate-300 border border-aerospace-700">
              WINDOW: 15 MIN
            </span>
          </div>
        </div>
      </div>

      {loading && <div className="loading-strip"><span className="loading-spinner" /> Loading congestion and feature-importance data…</div>}
      {/* Model Architecture Card & Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Model Card */}
        <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 space-y-4 shadow-lg">
          <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs font-semibold">
            <Cpu className="w-4 h-4" />
            <span>MODEL SPECIFICATION</span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1.5 border-b border-aerospace-800">
              <span className="text-slate-400">Model Name:</span>
              <span className="text-white font-bold">{congModel?.name || 'Not reported by API'}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-aerospace-800">
              <span className="text-slate-400">Version:</span>
              <span className="text-cyan-300">{congModel?.version || '—'}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-aerospace-800">
              <span className="text-slate-400">Feature Dimensions:</span>
              <span className="text-slate-200">{congModel?.feature_count != null ? `${congModel.feature_count} features` : '—'}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-aerospace-800">
              <span className="text-slate-400">Target Variable:</span>
              <span className="text-slate-200">Next 15m Class (3 tiers)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-aerospace-800">
              <span className="text-slate-400">Quantile Thresholds:</span>
              <span className="text-slate-200">LOW / MEDIUM / HIGH</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Artifact Checksum:</span>
              <span className="text-purple-300 truncate max-w-[150px]">{congModel?.sha256 ? `${congModel.sha256.slice(0, 12)}…` : 'Not reported'}</span>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 text-[11px] text-slate-300 leading-relaxed font-sans">
            <strong>Target Leakage Prevention:</strong> Input features strictly end at the prediction boundary 
            <code>floor(t, 15m)</code>. Next window targets are purged during cross-validation splits.
          </div>
        </div>

        {/* Evaluation Metrics Card */}
        <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 space-y-4 shadow-lg">
          <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-semibold">
            <Activity className="w-4 h-4" />
            <span>TEST EVALUATION METRICS</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800">
              <div className="text-slate-400 text-[10px]">RAW ACCURACY</div>
              <div className="text-xl font-bold text-emerald-400 mt-1">{formatPercent(accuracy)}</div>
              <div className="text-[10px] text-slate-400 mt-1">{metricSampleCount == null ? 'Sample count not reported' : `${formatNumber(metricSampleCount, 0)} test examples`}</div>
            </div>

            <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800">
              <div className="text-slate-400 text-[10px]">BALANCED ACCURACY</div>
              <div className="text-xl font-bold text-cyan-300 mt-1">{formatPercent(balancedAccuracy)}</div>
              <div className="text-[10px] text-slate-400 mt-1">Class-balanced score</div>
            </div>

            <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800">
              <div className="text-slate-400 text-[10px]">MACRO F1 SCORE</div>
              <div className="text-xl font-bold text-amber-400 mt-1">{formatNumber(macroF1, 4)}</div>
              <div className="text-[10px] text-slate-400 mt-1">From returned model report</div>
            </div>

            <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800">
              <div className="text-slate-400 text-[10px]">OOD DETECTION</div>
              <div className="text-xl font-bold text-purple-300 mt-1">{congModel?.is_loaded ? 'ACTIVE' : 'INACTIVE'}</div>
              <div className="text-[10px] text-slate-400 mt-1">Training bounds guard enabled</div>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 text-[11px] text-slate-300 leading-relaxed font-sans">
            <strong>Evaluation note:</strong> Metrics are shown only when they are returned by the model status payload. Open Model Evaluation to inspect the saved report and compare candidates.
          </div>
        </div>

        {/* Top Feature Importances */}
        <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 space-y-4 shadow-lg">
          <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs font-semibold">
            <TrendingUp className="w-4 h-4" />
            <span>TOP FEATURE IMPORTANCES</span>
          </div>

          {featureImportance.length > 0 ? <div className="h-52 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureImportance.slice(0, 6)}
                layout="vertical"
                margin={{ top: 0, right: 10, left: 35, bottom: 0 }}
              >
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'monospace' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f1c3d',
                    borderColor: '#1d3269',
                    borderRadius: '8px',
                    color: '#e2e8f0',
                    fontSize: '11px',
                  }}
                />
                <Bar dataKey="importance" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div> : <div className="empty-state feature-importance-empty"><TrendingUp size={21}/><h3>No feature-importance report</h3><p>The API did not return importance values. Check the model report or endpoint before presenting feature rankings.</p></div>}
          <p className="text-[10px] text-slate-400 font-mono">Importance scores are displayed as returned by the backend.</p>
        </div>
      </div>

      {/* Grid Cells Spatial Table */}
      <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            RESEARCH GRID CELLS & PREDICTED DENSITY
          </h3>
          <span className="text-xs font-mono text-slate-400">
            {cells.length} cells active along Delhi-Mumbai corridor
          </span>
        </div>

        <div className="overflow-x-auto max-h-72">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-aerospace-950 text-slate-400 sticky top-0 border-b border-aerospace-800">
              <tr>
                <th className="py-2.5 px-3">Cell ID</th>
                <th className="py-2.5 px-3">Center Coordinates</th>
                <th className="py-2.5 px-3">Active Aircraft</th>
                <th className="py-2.5 px-3">Predicted Density Tier</th>
                <th className="py-2.5 px-3">Congestion Score</th>
                <th className="py-2.5 px-3">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-aerospace-800 text-slate-300">
              {cells.length === 0 ? (
                <tr><td colSpan={6} className="py-8 text-center text-slate-500">No congestion-cell data returned. Check the backend response.</td></tr>
              ) : cells.map((c) => {
                const badgeColor =
                  c.predicted_class === 'HIGH'
                    ? 'bg-rose-950 text-rose-300 border-rose-800'
                    : c.predicted_class === 'MEDIUM'
                    ? 'bg-amber-950 text-amber-300 border-amber-800'
                    : 'bg-emerald-950 text-emerald-300 border-emerald-800';
                return (
                  <tr key={c.cell_id} onClick={() => setSelectedCell(c.cell_id)} className={`cursor-pointer hover:bg-aerospace-850/50 transition ${selectedCell === c.cell_id ? 'bg-cyan-950/25' : ''}`}>
                    <td className="py-2 px-3 font-bold text-white">{c.cell_id}</td>
                    <td className="py-2 px-3 text-slate-400">
                      {Number.isFinite(c.center_lat) ? c.center_lat.toFixed(1) : '—'}°, {Number.isFinite(c.center_lon) ? c.center_lon.toFixed(1) : '—'}°
                    </td>
                    <td className="py-2 px-3">{c.aircraft_count}</td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 rounded border text-[10px] font-bold ${badgeColor}`}>
                        {c.predicted_class}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-mono">{Number.isFinite(c.congestion_score) ? c.congestion_score.toFixed(2) : '—'}</td>
                    <td className="py-2 px-3 text-slate-400">
                      {c.confidence !== undefined && Number.isFinite(c.confidence) ? `${(c.confidence * 100).toFixed(0)}%` : 'N/A'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
