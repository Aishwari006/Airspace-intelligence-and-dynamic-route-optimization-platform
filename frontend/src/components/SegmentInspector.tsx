import React from 'react';
import { X, CloudRain, Activity, AlertTriangle, ShieldCheck, Compass, Info, Clock } from 'lucide-react';
import type { RouteSegment } from '../types';

interface SegmentInspectorProps {
  segment: RouteSegment | null;
  onClose: () => void;
}

export const SegmentInspector: React.FC<SegmentInspectorProps> = ({ segment, onClose }) => {
  if (!segment) return null;

  const wxSeverity = segment.weather_severity !== null ? (segment.weather_severity * 100).toFixed(0) : null;
  const cgScore = segment.congestion_score !== null ? (segment.congestion_score * 100).toFixed(0) : null;
  const uncPct = (segment.uncertainty * 100).toFixed(0);

  return (
    <div className="bg-aerospace-900 border border-aerospace-700/80 rounded-xl p-5 shadow-2xl space-y-4 font-mono text-xs text-slate-300">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-aerospace-800 pb-3">
        <div className="flex items-center gap-2">
          <Compass className="w-4 h-4 text-cyan-400" />
          <span className="font-bold text-white text-sm">SEGMENT INSPECTOR</span>
          <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 text-[11px]">
            {segment.from_node} → {segment.to_node}
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-aerospace-800 text-slate-400 hover:text-white transition"
          aria-label="Close Inspector"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        <div className="p-2.5 rounded-lg bg-aerospace-950 border border-aerospace-800">
          <div className="text-[10px] text-slate-400 uppercase">Distance</div>
          <div className="text-sm font-bold text-white mt-0.5">{segment.distance_km.toFixed(1)} km</div>
          <div className="text-[10px] text-slate-500 mt-1">Geodesic Arc</div>
        </div>

        <div className="p-2.5 rounded-lg bg-aerospace-950 border border-aerospace-800">
          <div className="text-[10px] text-slate-400 uppercase flex items-center gap-1">
            <Clock className="w-3 h-3 text-cyan-400" /> Traversal Time (ETA)
          </div>
          <div className="text-sm font-bold text-cyan-300 mt-0.5 flex items-baseline gap-1.5 flex-wrap">
            <span>{segment.time_min.toFixed(1)} min</span>
            {(() => {
              const refTime = (segment.distance_km / 800) * 60;
              const segDelta = segment.time_min - refTime;
              return segDelta > 0.1 ? (
                <span className="text-rose-400 text-[10px] font-semibold">(+{segDelta.toFixed(1)}m delay)</span>
              ) : segDelta < -0.1 ? (
                <span className="text-emerald-400 text-[10px] font-semibold">(-{Math.abs(segDelta).toFixed(1)}m early)</span>
              ) : (
                <span className="text-cyan-400 text-[10px]">(On Schedule)</span>
              );
            })()}
          </div>
          <div className="text-[10px] text-slate-500 mt-1 truncate" title={segment.time_source}>
            {segment.time_source}
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-aerospace-950 border border-aerospace-800">
          <div className="text-[10px] text-slate-400 uppercase flex items-center gap-1">
            <CloudRain className="w-3 h-3 text-amber-400" /> Weather Hazard
          </div>
          <div className="text-sm font-bold text-amber-300 mt-0.5">
            {wxSeverity !== null ? `${wxSeverity}%` : 'Unobserved'}
          </div>
          <div className="text-[10px] text-slate-500 mt-1 truncate" title={segment.weather_source}>
            {segment.weather_source}
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-aerospace-950 border border-aerospace-800">
          <div className="text-[10px] text-slate-400 uppercase flex items-center gap-1">
            <Activity className="w-3 h-3 text-rose-400" /> Traffic Density
          </div>
          <div className="text-sm font-bold text-rose-300 mt-0.5">
            {cgScore !== null ? `${cgScore}% (${segment.congestion_class || 'N/A'})` : 'Unobserved'}
          </div>
          <div className="text-[10px] text-slate-500 mt-1 truncate" title={segment.congestion_source}>
            {segment.congestion_source}
          </div>
        </div>
      </div>

      {/* Uncertainty & Closure Alerts */}
      <div className="space-y-2">
        {segment.uncertainty >= 0.3 && (
          <div className="p-2.5 rounded-lg bg-purple-950/40 border border-purple-800/60 text-purple-200 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-purple-400 shrink-0" />
            <span>
              <strong>Elevated Data Uncertainty ({uncPct}%):</strong> Unobserved inputs penalized; conservative proxy active.
            </span>
          </div>
        )}

        {segment.closed && (
          <div className="p-2.5 rounded-lg bg-red-950/60 border border-red-800/80 text-red-200 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
            <span>
              <strong>Hard Closure Constraint:</strong> Excluded from valid routes. Reason: {segment.closure_reasons.join(', ')}
            </span>
          </div>
        )}
      </div>

      {/* Dimensionless Cost Breakdown Table */}
      <div className="pt-2 border-t border-aerospace-800">
        <div className="text-[11px] text-slate-400 mb-2 flex items-center justify-between">
          <span>DIMENSIONLESS COST MODEL CONTRIBUTIONS</span>
          <span className="text-cyan-400 font-bold">Total Cost: {segment.cost_total.toFixed(4)}</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-[11px]">
          {Object.entries(segment.cost_components || {}).map(([factor, rawComp]) => {
            const weighted = segment.weighted_cost?.[factor] ?? 0;
            return (
              <div key={factor} className="p-2 rounded bg-aerospace-950/80 border border-aerospace-800/60">
                <div className="text-slate-400 capitalize">{factor}</div>
                <div className="font-semibold text-white mt-0.5">{rawComp.toFixed(3)}</div>
                <div className="text-[10px] text-cyan-400 mt-0.5">w: {weighted.toFixed(3)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

