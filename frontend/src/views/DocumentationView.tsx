import React, { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  Clock,
  CloudRain,
  Code,
  Compass,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  FolderGit2,
  Gauge,
  HelpCircle,
  Layers,
  Lightbulb,
  Map,
  Navigation,
  Play,
  Route,
  ShieldCheck,
  Sparkles,
  Terminal,
  UploadCloud,
  Zap,
} from 'lucide-react';

interface DocumentationViewProps {
  onNavigate?: (tab: string) => void;
}

export const DocumentationView: React.FC<DocumentationViewProps> = ({ onNavigate }) => {
  const [activeDocSection, setActiveDocSection] = useState<'playbook' | 'viva' | 'architecture'>('playbook');

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2 font-mono">
            <BookOpen className="w-5 h-5 text-cyan-400" />
            OPERATOR MANUAL, SYSTEM GUIDE & VIVA DEFENSE
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Complete step-by-step navigation instructions, workflow tutorials, and theoretical defense materials for Sky Force.
          </p>
        </div>

        {/* Section Navigation Pills */}
        <div className="flex flex-wrap items-center gap-1.5 p-1 bg-aerospace-950 border border-aerospace-800 rounded-lg self-start md:self-auto">
          <button
            type="button"
            onClick={() => setActiveDocSection('playbook')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold font-mono transition-all flex items-center gap-1.5 cursor-pointer ${
              activeDocSection === 'playbook'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Compass size={13} />
            <span>Operator Playbook</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveDocSection('viva')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold font-mono transition-all flex items-center gap-1.5 cursor-pointer ${
              activeDocSection === 'viva'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Lightbulb size={13} />
            <span>Viva & Research Defense</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveDocSection('architecture')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold font-mono transition-all flex items-center gap-1.5 cursor-pointer ${
              activeDocSection === 'architecture'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Code size={13} />
            <span>Architecture & APIs</span>
          </button>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 1: OPERATOR PLAYBOOK & STEP-BY-STEP WORKFLOW                      */}
      {/* ========================================================================= */}
      {activeDocSection === 'playbook' && (
        <div className="space-y-6">
          {/* Quick-start walkthrough card */}
          <div className="bg-aerospace-900 border border-cyan-800/40 rounded-xl p-5 shadow-lg">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold font-mono text-white uppercase">
                Quick Start: End-to-End Demo in 3 Minutes
              </h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed mb-4">
              If you want to see the complete platform in action right away, follow this standard demonstration sequence:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 flex flex-col justify-between">
                <div>
                  <div className="text-cyan-400 font-mono font-bold mb-1">Step 1: Check Live Airspace</div>
                  <p className="text-slate-400 font-sans mb-3">
                    Go to <strong>Airspace Explorer</strong>. Switch the toggle from 2D to <strong>3D Perspective</strong> to see elevated flight trajectories and 2° extruded congestion columns across India.
                  </p>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('explorer')}
                    className="inline-flex items-center justify-between text-cyan-300 hover:text-cyan-100 text-[11px] font-semibold pt-2 border-t border-aerospace-800 cursor-pointer"
                  >
                    <span>Launch Airspace Explorer</span>
                    <ArrowRight size={12} />
                  </button>
                )}
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 flex flex-col justify-between">
                <div>
                  <div className="text-amber-400 font-mono font-bold mb-1">Step 2: Simulate Storm or Outage</div>
                  <p className="text-slate-400 font-sans mb-3">
                    Open <strong>Scenario Studio</strong>. Pick the <em>Monsoon Front</em> or <em>Delhi Hub Restriction</em> scenario. Observe the hard edge closures and storm exposure flags.
                  </p>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('scenarios')}
                    className="inline-flex items-center justify-between text-amber-300 hover:text-amber-100 text-[11px] font-semibold pt-2 border-t border-aerospace-800 cursor-pointer"
                  >
                    <span>Launch Scenario Studio</span>
                    <ArrowRight size={12} />
                  </button>
                )}
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 flex flex-col justify-between">
                <div>
                  <div className="text-emerald-400 font-mono font-bold mb-1">Step 3: Solve & Explain Route</div>
                  <p className="text-slate-400 font-sans mb-3">
                    Open <strong>Route Optimization</strong>. Choose <code>VIDP (Delhi)</code> to <code>VABB (Mumbai)</code>. Click <strong>Run Optimization</strong>, inspect the Pareto Frontier trade-offs, and read the AI Explanation!
                  </p>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('routing')}
                    className="inline-flex items-center justify-between text-emerald-300 hover:text-emerald-100 text-[11px] font-semibold pt-2 border-t border-aerospace-800 cursor-pointer"
                  >
                    <span>Launch Route Optimization</span>
                    <ArrowRight size={12} />
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Module-by-Module Operator Guides */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold font-mono text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              DETAILED OPERATING GUIDE BY SCREEN
            </h3>

            {/* Screen 1: Mission Control */}
            <div className="p-4 rounded-xl bg-aerospace-900 border border-aerospace-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-mono font-bold text-sm">
                  <Gauge className="w-4 h-4 text-cyan-400" />
                  <span>1. Mission Control (Overview)</span>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('overview')}
                    className="text-xs text-cyan-400 hover:underline inline-flex items-center gap-1 cursor-pointer"
                  >
                    <span>Go to tab</span>
                    <ChevronRight size={12} />
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-300">
                <strong>Purpose:</strong> System dashboard showing real-time backend connectivity, database status, ML inference readiness tags, and the top-level 2D network map.
              </p>
              <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                <li>Verify that both models (<em>Congestion Classifier</em> and <em>Segment ETA Regressor</em>) show green <strong>LOADED</strong> tags.</li>
                <li>Inspect overall node counts (19 Airports, 40 Waypoints, 276 Airway Segments).</li>
                <li>Use the 5-Step Pipeline buttons to navigate to any part of the workflow.</li>
              </ul>
            </div>

            {/* Screen 2: Dataset Studio & Lifecycle */}
            <div className="p-4 rounded-xl bg-aerospace-900 border border-aerospace-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-mono font-bold text-sm">
                  <UploadCloud className="w-4 h-4 text-cyan-400" />
                  <span>2. Dataset Studio & Dataset Lifecycle</span>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('studio')}
                    className="text-xs text-cyan-400 hover:underline inline-flex items-center gap-1 cursor-pointer"
                  >
                    <span>Go to tab</span>
                    <ChevronRight size={12} />
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-300">
                <strong>Purpose:</strong> Allows operators to ingest, map, and link raw custom CSV telemetry without altering global server state.
              </p>
              <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                <li><strong>Select Schema:</strong> Choose between <em>Aircraft Observations</em>, <em>Weather Observations</em>, or <em>Congestion Grid</em>.</li>
                <li><strong>Column Mapping:</strong> The visual mapper auto-detects latitude, longitude, timestamp, altitude, and ground speed columns.</li>
                <li><strong>Spatiotemporal Linking:</strong> Links weather observations to aircraft trajectories within 2° spatial grid cells and 15-minute time windows.</li>
                <li><strong>Dataset Explorer:</strong> Switch to the <em>Dataset Lifecycle</em> tab to inspect row counts, file hashes, and active dataset selection.</li>
              </ul>
            </div>

            {/* Screen 3: Airspace Explorer */}
            <div className="p-4 rounded-xl bg-aerospace-900 border border-aerospace-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-mono font-bold text-sm">
                  <Map className="w-4 h-4 text-cyan-400" />
                  <span>3. Airspace Explorer (2D & 3D Zero-Token Visualization)</span>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('explorer')}
                    className="text-xs text-cyan-400 hover:underline inline-flex items-center gap-1 cursor-pointer"
                  >
                    <span>Go to tab</span>
                    <ChevronRight size={12} />
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-300">
                <strong>Purpose:</strong> Interactive geospatial perspective viewer powered by MapLibre GL JS and deck.gl with zero external API key requirements.
              </p>
              <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                <li><strong>2D Mode:</strong> Leaflet canvas with custom dark basemap, airport markers, and airway corridors.</li>
                <li><strong>3D Mode:</strong> Extruded 3D columns: height indicates predicted congestion risk (0m to 25,000m), colored by severity (Green=Low, Amber=Med, Red=High).</li>
                <li><strong>Controls:</strong> Right-click + drag to tilt and rotate camera pitch/bearing in 3D. Scroll to zoom.</li>
              </ul>
            </div>

            {/* Screen 4: Route Optimization */}
            <div className="p-4 rounded-xl bg-aerospace-900 border border-cyan-800/60 space-y-2 bg-gradient-to-r from-aerospace-900 via-aerospace-850 to-aerospace-900">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-mono font-bold text-sm">
                  <Route className="w-4 h-4 text-cyan-400" />
                  <span>4. Route Optimization (Route Lab) — Core Engine</span>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('routing')}
                    className="text-xs text-cyan-400 hover:underline inline-flex items-center gap-1 cursor-pointer font-bold"
                  >
                    <span>Go to tab</span>
                    <ChevronRight size={12} />
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-300">
                <strong>Purpose:</strong> Solves multi-objective routing problems across Indian airspace airway networks using Dijkstra and A* search.
              </p>
              <div className="p-3 bg-aerospace-950 rounded-lg border border-aerospace-800 text-xs text-slate-300 space-y-2 font-mono">
                <div><span className="text-cyan-400 font-bold">1. Select Airports:</span> Origin = <code>VIDP (Delhi)</code>, Destination = <code>VABB (Mumbai)</code>.</div>
                <div><span className="text-cyan-400 font-bold">2. Objective Presets:</span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 mt-1 text-[11px] font-sans">
                    <span className="p-1 bg-aerospace-900 rounded border border-aerospace-800"><strong>Shortest:</strong> Pure geodesic km</span>
                    <span className="p-1 bg-aerospace-900 rounded border border-aerospace-800"><strong>Fastest:</strong> Min ML-predicted ETA</span>
                    <span className="p-1 bg-aerospace-900 rounded border border-aerospace-800"><strong>Weather:</strong> Storm penalties</span>
                    <span className="p-1 bg-aerospace-900 rounded border border-aerospace-800"><strong>Pareto:</strong> Non-dominated frontier</span>
                  </div>
                </div>
                <div><span className="text-cyan-400 font-bold">3. Inspect Results:</span>
                  <div className="text-slate-400 font-sans text-[11px] mt-0.5">
                    Click any waypoint segment along the route to inspect its along-track altitude profile, headwind/crosswind components, and percentage factor attribution.
                  </div>
                </div>
                <div><span className="text-cyan-400 font-bold">4. AI Explanation Panel:</span>
                  <div className="text-slate-400 font-sans text-[11px] mt-0.5">
                    Decomposes decisions into percentages (e.g. <em>Distance 45%, Storm Avoidance 35%, Congestion 20%</em>) with natural language justifications.
                  </div>
                </div>
              </div>
            </div>

            {/* Screen 5: Scenario Studio */}
            <div className="p-4 rounded-xl bg-aerospace-900 border border-aerospace-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-mono font-bold text-sm">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <span>5. Scenario Studio (Disruption Injection & Rerouting)</span>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('scenarios')}
                    className="text-xs text-cyan-400 hover:underline inline-flex items-center gap-1 cursor-pointer"
                  >
                    <span>Go to tab</span>
                    <ChevronRight size={12} />
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-300">
                <strong>Purpose:</strong> Tests airspace resiliency by injecting hard airway segment closures, storm cells, or capacity reductions.
              </p>
              <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                <li>Activate pre-configured scenarios: <em>Nominal Flow</em>, <em>Monsoon Thunderstorm Front</em>, or <em>Sector Capacity Reduction</em>.</li>
                <li>Observe the dynamic rerouting solver calculate diverging waypoints in under 15 milliseconds.</li>
              </ul>
            </div>

            {/* Screen 6: Congestion Intelligence & Evaluation */}
            <div className="p-4 rounded-xl bg-aerospace-900 border border-aerospace-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-mono font-bold text-sm">
                  <FlaskConical className="w-4 h-4 text-purple-400" />
                  <span>6. Congestion AI & Model Evaluation Center</span>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('evaluation')}
                    className="text-xs text-cyan-400 hover:underline inline-flex items-center gap-1 cursor-pointer"
                  >
                    <span>Go to tab</span>
                    <ChevronRight size={12} />
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-300">
                <strong>Purpose:</strong> Verifies Scikit-learn model performance against chronological out-of-sample test splits.
              </p>
              <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                <li><strong>Classifier:</strong> Accuracy (97.24%), Balanced Accuracy (88.81%), Macro-F1 (0.8961).</li>
                <li><strong>ETA Regressor:</strong> MAE (0.7357 min approx 44.1s), R^2 = 0.9545.</li>
                <li><strong>MDI & Permutation Importance:</strong> Shows causal traffic lags and along-track winds as top predictive features.</li>
              </ul>
            </div>

            {/* Screen 7: Data & QA Scrubber */}
            <div className="p-4 rounded-xl bg-aerospace-900 border border-aerospace-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-mono font-bold text-sm">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span>7. Data Quality & 9-Stage Scrubber QA</span>
                </div>
                {onNavigate && (
                  <button
                    type="button"
                    onClick={() => onNavigate('quality')}
                    className="text-xs text-cyan-400 hover:underline inline-flex items-center gap-1 cursor-pointer"
                  >
                    <span>Go to tab</span>
                    <ChevronRight size={12} />
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-300">
                <strong>Purpose:</strong> Validates raw telemetry across 9 sequential kinematic filters (altitude envelopes, acceleration thresholds, climb rates, and spatial FIR bounding).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SECTION 2: ACADEMIC VIVA & RESEARCH QUESTIONS                             */}
      {/* ========================================================================= */}
      {activeDocSection === 'viva' && (
        <div className="space-y-6">
          {/* 6 Research Hypotheses */}
          <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4">
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
              <Lightbulb className="w-4 h-4 text-amber-400" />
              6 RESEARCH QUESTIONS & EXPERIMENTAL FINDINGS
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1.5">
                <div className="text-cyan-400 font-mono font-bold">RQ1: Spatial Traffic Indicators</div>
                <p className="text-slate-300 leading-relaxed font-sans">
                  <strong>Finding:</strong> Converting 90s surveillance telemetry into half-open 2.0° grid cells with causal 
                  lagged counts produces highly informative indicators of density, capturing corridor choke points cleanly.
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1.5">
                <div className="text-cyan-400 font-mono font-bold">RQ2: Leakage-Free ML Forecasting</div>
                <p className="text-slate-300 leading-relaxed font-sans">
                  <strong>Finding:</strong> The Random Forest classifier achieves 97.2% test accuracy (Macro-F1 0.896, Balanced Accuracy 88.8%) 
                  for the next 15-minute window, significantly outperforming Zero-R baseline (0.312) under chronological partitioning.
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1.5">
                <div className="text-cyan-400 font-mono font-bold">RQ3: Multi-Objective Routing vs Distance</div>
                <p className="text-cyan-300 leading-relaxed font-sans">
                  <strong>Finding:</strong> Under weather storm zones or corridor closures, composite multi-factor routing 
                  successfully routes aircraft around hazards with minor distance increases (+3.8% to +7.2% length penalty).
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1.5">
                <div className="text-cyan-400 font-mono font-bold">RQ4: Natural Language Explainability</div>
                <p className="text-slate-300 leading-relaxed font-sans">
                  <strong>Finding:</strong> Decomposing route cost into dimensionless factors allows instant transparent 
                  explanations of trade-offs (e.g. "Selected route trades +45 km distance to avoid 180 km of severe weather").
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1.5">
                <div className="text-cyan-400 font-mono font-bold">RQ5: Dynamic Rerouting Resiliency</div>
                <p className="text-slate-300 leading-relaxed font-sans">
                  <strong>Finding:</strong> When hard segment closures occur after departure, the dynamic planner 
                  identifies the nearest viable diverging waypoint and calculates compliant alternatives in &lt;15 ms.
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1.5">
                <div className="text-cyan-400 font-mono font-bold">RQ6: Baseline Comparison Integrity</div>
                <p className="text-slate-300 leading-relaxed font-sans">
                  <strong>Finding:</strong> Comparing against classical baselines (Zero-R, Ridge, Dijkstra shortest distance) 
                  prevents overstating ML claims while proving clear statistical and operational utility.
                </p>
              </div>
            </div>
          </div>

          {/* Viva / Project Defense Questions */}
          <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4">
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
              <HelpCircle className="w-4 h-4 text-cyan-400" />
              EXAMINER VIVA DEFENSE GUIDE (Q&A)
            </h3>

            <div className="space-y-3 text-xs font-sans">
              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">
                <div className="text-white font-bold font-mono">
                  Q1: Why did you use graph search (Dijkstra / A*) instead of pure deep reinforcement learning for routing?
                </div>
                <p className="text-slate-300 leading-relaxed">
                  <strong>Answer:</strong> Graph search on candidate airway networks provides mathematical optimality, deterministic 
                  execution, and strict compliance with hard airspace closures. Deep RL can hallucinate infeasible waypoints and cannot 
                  guarantee safety constraints or triangle-inequality bounds.
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">
                <div className="text-white font-bold font-mono">
                  Q2: How do you prove your A* heuristic is admissible?
                </div>
                <p className="text-slate-300 leading-relaxed">
                  <strong>Answer:</strong> The great-circle Haversine formula strictly satisfies the spherical triangle inequality. 
                  The heuristic coefficient scales distance and time terms using the maximum network speed <code>v_max</code>:
                  <code>h(n) = [(w_d + ε)/ref_km + w_t/(v_max * ref_min)] * d_gc(n, goal)</code>. Because weather and congestion penalties 
                  are non-negative, <code>h(n) ≤ c*(n)</code> is guaranteed, ensuring A* never overestimates true cost.
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">
                <div className="text-white font-bold font-mono">
                  Q3: How do you prevent data leakage in your machine learning models?
                </div>
                <p className="text-slate-300 leading-relaxed">
                  <strong>Answer:</strong> We enforce strict chronological train/val/test splits (Days 1–4 train, 5–6 val, 7–8 test). 
                  Rolling statistics strictly use windows ending before or at the prediction timestamp. Targets are derived strictly 
                  from subsequent windows and purged across partition boundaries.
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">
                <div className="text-white font-bold font-mono">
                  Q4: How does Sky Force handle segment closures?
                </div>
                <p className="text-slate-300 leading-relaxed">
                  <strong>Answer:</strong> Closures are treated as hard graph constraints (edges are removed or excluded during cost evaluation), 
                  never as soft penalties. This ensures no generated route will ever traverse a closed or restricted corridor.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SECTION 3: SYSTEM ARCHITECTURE & API REFERENCE                            */}
      {/* ========================================================================= */}
      {activeDocSection === 'architecture' && (
        <div className="space-y-6">
          <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4">
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
              <Code className="w-4 h-4 text-cyan-400" />
              SYSTEM ARCHITECTURE OVERVIEW
            </h3>

            <div className="p-4 bg-aerospace-950 rounded-lg border border-aerospace-800 font-mono text-[11px] text-slate-300 leading-relaxed overflow-x-auto">
              <pre>{`┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SKY FORCE MISSION CONSOLE                             │
│                  React 19 + TypeScript + Vite + Tailwind CSS                    │
│      MapLibre GL + deck.gl (3D) • Leaflet (2D) • Recharts • Lucide Icons       │
└────────────────────────────────────────┬────────────────────────────────────────┘
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
│  • CPA Conflict Scan  │  • SHA-256 Registry      │  • Dynamic Reroute Deltas    │
└────────────────────────────────────────┬────────────────────────────────────────┘`}</pre>
            </div>
          </div>

          <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4">
            <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
              <Terminal className="w-4 h-4 text-cyan-400" />
              REST API CHEATSHEET
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">
                <div className="text-cyan-400 font-bold">POST /api/v1/routing/plan</div>
                <div className="text-slate-400 font-sans">Computes single optimal route between origin & destination under weights.</div>
              </div>
              <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">
                <div className="text-cyan-400 font-bold">POST /api/v1/routing/compare</div>
                <div className="text-slate-400 font-sans">Computes candidate routes and discovers the Pareto-optimal frontier.</div>
              </div>
              <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">
                <div className="text-cyan-400 font-bold">GET /api/v1/models/status</div>
                <div className="text-slate-400 font-sans">Returns active model versions, SHA-256 hashes, and loaded status.</div>
              </div>
              <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">
                <div className="text-cyan-400 font-bold">POST /api/v1/datasets/upload</div>
                <div className="text-slate-400 font-sans">Ingests multi-schema CSV files (telemetry, weather, or congestion).</div>
              </div>
            </div>

            <div className="pt-2">
              <a
                href="http://127.0.0.1:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-xs font-semibold text-cyan-400 hover:text-cyan-300"
              >
                <span>Open Interactive FastAPI Swagger Documentation</span>
                <ExternalLink size={13} />
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
