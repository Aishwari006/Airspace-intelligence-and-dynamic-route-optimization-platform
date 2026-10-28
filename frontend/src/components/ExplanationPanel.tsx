import React, { useState } from 'react';
import { Sparkles, ShieldAlert, CheckCircle2, AlertTriangle, Send, Loader2, Info } from 'lucide-react';
import type { ExplanationResponse, RouteDetail } from '../types';
import { api } from '../services/api';

interface ExplanationPanelProps {
  route: RouteDetail | null;
  baselineRoute?: RouteDetail | null;
  objective: string;
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({
  route,
  baselineRoute,
  objective,
}) => {
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [userQuery, setUserQuery] = useState('');
  const [error, setError] = useState<string | null>(null);

  const fetchAIExplanation = async (queryText?: string) => {
    if (!route || !route.found) return;

    setLoading(true);
    setError(null);
    try {
      const payload = {
        task: 'route',
        context_data: {
          path: route.path,
          objective,
          totals: route.totals,
          cost_by_factor: route.cost_by_factor,
          baseline_totals: baselineRoute?.totals || route.totals,
          closed_avoided: [],
          provenance: route.provenance,
          warnings: route.totals?.mean_uncertainty && route.totals.mean_uncertainty > 0.3 ? ['weather_missing'] : [],
        },
        user_query: queryText || userQuery || undefined,
      };

      const res: ExplanationResponse = await api.getExplanation(payload);
      setExplanation(res);
    } catch (err: any) {
      setError(err.message || 'Explanation generation failed');
    } finally {
      setLoading(false);
    }
  };

  // If no LLM explanation yet, use the route's deterministic explanation array
  const defaultBullets = route?.explanation || [];

  return (
    <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4 font-mono text-xs text-slate-300">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-aerospace-800 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <span className="font-bold text-white text-sm">DECISION EXPLAINABILITY & PROVENANCE</span>
        </div>

        {explanation ? (
          <div className="flex items-center gap-2">
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                explanation.engine.toLowerCase().includes('gemini') || explanation.engine === 'phi3:latest'
                  ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                  : 'bg-amber-950 text-amber-300 border-amber-800'
              }`}
            >
              {explanation.engine.toLowerCase().includes('gemini')
                ? 'AI Synthesis (Google Gemini API)'
                : explanation.engine === 'phi3:latest'
                ? 'AI Synthesis (Phi-3 via Ollama)'
                : 'Deterministic Fallback Engine'}
            </span>
            <span className="text-[10px] text-slate-400">{explanation.latency_ms} ms</span>
          </div>
        ) : (
          <button
            onClick={() => fetchAIExplanation()}
            disabled={loading || !route?.found}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 text-[11px] font-bold transition disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            {loading ? 'Synthesizing...' : 'Query Gemini AI Reasoner'}
          </button>
        )}
      </div>

      {error && (
        <div className="p-2.5 rounded bg-red-950/40 border border-red-800/60 text-red-200 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Narrative Summary */}
      <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 text-slate-200 leading-relaxed">
        {explanation ? (
          <p className="text-xs">{explanation.summary}</p>
        ) : (
          <p className="text-xs text-slate-400">
            {defaultBullets[0] || 'Select an origin and destination to plan and explain routing trade-offs.'}
          </p>
        )}
      </div>

      {/* Supporting Factual Evidence */}
      <div className="space-y-2">
        <div className="text-[11px] text-slate-400 uppercase font-semibold flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-cyan-400" />
          <span>Verified Computational Facts:</span>
        </div>

        <div className="grid grid-cols-1 gap-2">
          {(explanation?.supporting_facts.length ? explanation.supporting_facts : defaultBullets.slice(1)).map(
            (fact, idx) => (
              <div
                key={idx}
                className="p-2.5 rounded-lg bg-aerospace-950/70 border border-aerospace-850 text-xs text-slate-300 flex items-start gap-2.5"
              >
                <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                <span>{fact}</span>
              </div>
            )
          )}
        </div>
      </div>

      {/* Ask Phi-3 Examiner Prompt Input */}
      <div className="pt-2 border-t border-aerospace-800">
        <label className="block text-[11px] text-slate-400 mb-1.5">Ask the Explanation Engine a specific question:</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={userQuery}
            onChange={(e) => setUserQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void fetchAIExplanation(userQuery);
            }}
            placeholder="e.g. Why was this path chosen over the direct flight?"
            className="flex-1 bg-aerospace-950 border border-aerospace-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
          />
          <button
            onClick={() => fetchAIExplanation(userQuery)}
            disabled={loading || !route?.found}
            className="px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-aerospace-950 font-bold text-xs transition disabled:opacity-50 flex items-center gap-1"
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
            Ask
          </button>
        </div>
      </div>

      {/* Academic Disclaimer */}
      <div className="text-[10px] text-slate-500 flex items-center gap-1 pt-1">
        <ShieldAlert className="w-3 h-3 text-slate-500 shrink-0" />
        <span>Explanations strictly reflect computed graph weights and ML predictions. Never treat as ATC clearance.</span>
      </div>
    </div>
  );
};

