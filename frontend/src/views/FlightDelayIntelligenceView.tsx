import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  BarChart3,
  Calendar,
  CheckCircle2,
  Clock,
  Compass,
  Database,
  ExternalLink,
  FileText,
  Filter,
  Flame,
  HelpCircle,
  Info,
  Layers,
  MapPin,
  Plane,
  RefreshCw,
  Route,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Wind,
  Zap,
} from 'lucide-react';
import type {
  DelayDashboardResponse,
  FlightDetail,
  FlightSummary,
  OperationalStage,
  DataReadinessReport,
} from '../types';
import { api } from '../services/api';

interface FlightDelayIntelligenceViewProps {
  onInspectRoute?: (origin: string, destination: string) => void;
}

export const FlightDelayIntelligenceView: React.FC<FlightDelayIntelligenceViewProps> = ({
  onInspectRoute,
}) => {
  const [dashboard, setDashboard] = useState<DelayDashboardResponse | null>(null);
  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [selectedFlight, setSelectedFlight] = useState<FlightDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters for Attention Table
  const [filterSeverity, setFilterSeverity] = useState<'ALL' | 'MODERATE' | 'HIGH'>('ALL');
  const [filterAirport, setFilterAirport] = useState<string>('ALL');

  // Interactive Scenario Simulation States
  const [simInboundDelay, setSimInboundDelay] = useState<number>(0);
  const [simQueueAdditional, setSimQueueAdditional] = useState<number>(0);
  const [simWeatherSeverity, setSimWeatherSeverity] = useState<number>(0.0);
  const [simSlotDelay, setSimSlotDelay] = useState<number>(0);
  const [simulating, setSimulating] = useState(false);

  // Data Readiness Modal
  const [showReadinessModal, setShowReadinessModal] = useState(false);
  const [readinessReport, setReadinessReport] = useState<DataReadinessReport | null>(null);

  // Load Dashboard
  useEffect(() => {
    let cancelled = false;
    const fetchDashboard = async () => {
      setLoading(true);
      setError(null);
      try {
        const [dashData, readyData] = await Promise.all([
          api.getDelayDashboard(150),
          api.getDelayReadiness().catch(() => null),
        ]);
        if (cancelled) return;
        setDashboard(dashData);
        setReadinessReport(readyData);
        if (dashData.flights_requiring_attention.length > 0) {
          const firstId = dashData.flights_requiring_attention[0].flight_id;
          setSelectedFlightId(firstId);
          const detail = await api.getDelayFlightDetail(firstId);
          if (cancelled) return;
          setSelectedFlight(detail);
          setSimInboundDelay(Math.round(detail.inbound_delay_minutes || 0));
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Could not fetch delay intelligence data');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchDashboard();
    return () => {
      cancelled = true;
    };
  }, []);

  // Fetch flight detail when flight ID changes
  const handleSelectFlight = async (flightId: string) => {
    setSelectedFlightId(flightId);
    setDetailLoading(true);
    try {
      const detail = await api.getDelayFlightDetail(flightId);
      setSelectedFlight(detail);
      setSimInboundDelay(Math.round(detail.inbound_delay_minutes || 0));
      setSimQueueAdditional(0);
      setSimWeatherSeverity(detail.departure_weather_summary.severity_index || 0.0);
      setSimSlotDelay(0);
    } catch (err) {
      console.error('Failed to load detail for flight:', flightId, err);
    } finally {
      setDetailLoading(false);
    }
  };

  // Recalculate Scenario Simulation
  const handleRunSimulation = async () => {
    if (!selectedFlight) return;
    setSimulating(true);
    try {
      const recalculated = await api.predictDelayScenario({
        flight_id: selectedFlight.flight_id,
        inbound_delay_minutes: simInboundDelay,
        departure_queue_additional: simQueueAdditional,
        weather_severity_override: simWeatherSeverity > 0 ? simWeatherSeverity : undefined,
        slot_delay_minutes: simSlotDelay,
      });
      setSelectedFlight(recalculated);
    } catch (err) {
      console.error('Simulation calculation failed:', err);
    } finally {
      setSimulating(false);
    }
  };

  // Reset Simulation
  const handleResetSimulation = async () => {
    if (!selectedFlight) return;
    await handleSelectFlight(selectedFlight.flight_id);
  };

  // Filtered Attention Flights
  const filteredFlights = dashboard?.flights_requiring_attention.filter((f) => {
    if (filterSeverity !== 'ALL' && f.severity !== filterSeverity) return false;
    if (filterAirport !== 'ALL' && f.origin_iata !== filterAirport && f.destination_iata !== filterAirport) return false;
    return true;
  }) || [];

  return (
    <div className="space-y-6">
      {/* View Header */}
      <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono font-bold tracking-wider text-cyan-400 uppercase">OPERATIONS INTELLIGENCE</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-700/60 font-semibold font-mono">
              FLIGHT DELAY MODEL
            </span>
          </div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2 font-mono">
            <Clock className="w-5 h-5 text-cyan-400" />
            FLIGHT DEPARTURE DELAY & ROTATIONAL RISK INTELLIGENCE
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Flight-level departure delay forecasting, airframe rotation tracking, 8-stage operational progression, and what-if scenario simulation.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setShowReadinessModal(true)}
          className="self-start md:self-auto inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-cyan-950/80 border border-cyan-600/60 text-cyan-300 hover:text-white hover:bg-cyan-900/60 transition-all text-xs font-semibold cursor-pointer shadow-sm"
        >
          <ShieldCheck size={14} />
          <span>Data Readiness & Gap Report</span>
        </button>
      </div>

      {/* Top Operations KPI Metrics Bar */}
      {dashboard && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="p-3.5 rounded-xl bg-aerospace-900 border border-aerospace-800">
            <span className="text-[10px] font-mono text-slate-400 uppercase">Monitored Flights</span>
            <div className="text-xl font-bold font-mono text-white mt-0.5">{dashboard.total_flights_monitored}</div>
            <div className="text-[10px] text-slate-500">Active surveillance sample</div>
          </div>

          <div className="p-3.5 rounded-xl bg-aerospace-900 border border-emerald-900/40">
            <span className="text-[10px] font-mono text-emerald-400 uppercase">On-Time (&lt;15m)</span>
            <div className="text-xl font-bold font-mono text-emerald-300 mt-0.5">
              {dashboard.on_time_count}
              <span className="text-xs font-normal text-emerald-400/80 ml-1">
                ({Math.round((dashboard.on_time_count / dashboard.total_flights_monitored) * 100)}%)
              </span>
            </div>
            <div className="text-[10px] text-emerald-500/80">Nominal turnaround</div>
          </div>

          <div className="p-3.5 rounded-xl bg-aerospace-900 border border-amber-900/40">
            <span className="text-[10px] font-mono text-amber-400 uppercase">Moderate Delay</span>
            <div className="text-xl font-bold font-mono text-amber-300 mt-0.5">
              {dashboard.moderate_delay_count}
              <span className="text-xs font-normal text-amber-400/80 ml-1">
                ({Math.round((dashboard.moderate_delay_count / dashboard.total_flights_monitored) * 100)}%)
              </span>
            </div>
            <div className="text-[10px] text-amber-500/80">15 – 45 min impact</div>
          </div>

          <div className="p-3.5 rounded-xl bg-aerospace-900 border border-rose-900/40">
            <span className="text-[10px] font-mono text-rose-400 uppercase">Significant Delay</span>
            <div className="text-xl font-bold font-mono text-rose-300 mt-0.5">
              {dashboard.significant_delay_count}
              <span className="text-xs font-normal text-rose-400/80 ml-1">
                ({Math.round((dashboard.significant_delay_count / dashboard.total_flights_monitored) * 100)}%)
              </span>
            </div>
            <div className="text-[10px] text-rose-500/80">&gt; 45 min impact</div>
          </div>

          <div className="p-3.5 rounded-xl bg-aerospace-900 border border-cyan-900/40">
            <span className="text-[10px] font-mono text-cyan-400 uppercase">Avg Fleet Delay</span>
            <div className="text-xl font-bold font-mono text-cyan-300 mt-0.5">
              {dashboard.average_predicted_delay_min} min
            </div>
            <div className="text-[10px] text-cyan-500/80">Mean departure delta</div>
          </div>

          <div className="p-3.5 rounded-xl bg-aerospace-900 border border-purple-900/40">
            <span className="text-[10px] font-mono text-purple-400 uppercase">Active Airframes</span>
            <div className="text-xl font-bold font-mono text-purple-300 mt-0.5">
              {dashboard.model_readiness.fleet_size || 115}
            </div>
            <div className="text-[10px] text-purple-500/80">Rotational tracking</div>
          </div>
        </div>
      )}

      {/* Main Grid: Flights Requiring Attention + Deep Dive Investigation Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (5 Cols): Attention Table & Filters */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-4 shadow-lg space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-aerospace-800">
              <div className="flex items-center gap-2">
                <Flame className="w-4 h-4 text-rose-400" />
                <h3 className="text-xs font-bold font-mono text-white uppercase">
                  FLIGHTS REQUIRING ATTENTION ({filteredFlights.length})
                </h3>
              </div>
            </div>

            {/* Filter Controls */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <label className="text-[10px] font-mono text-slate-400 mb-1 block">SEVERITY</label>
                <select
                  value={filterSeverity}
                  onChange={(e) => setFilterSeverity(e.target.value as any)}
                  className="w-full px-2.5 py-1.5 rounded bg-aerospace-950 border border-aerospace-800 text-xs text-white"
                >
                  <option value="ALL">All Severities</option>
                  <option value="HIGH">High Delay (&gt;45m)</option>
                  <option value="MODERATE">Moderate Delay (15-45m)</option>
                </select>
              </div>

              <div>
                <label className="text-[10px] font-mono text-slate-400 mb-1 block">AIRPORT</label>
                <select
                  value={filterAirport}
                  onChange={(e) => setFilterAirport(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded bg-aerospace-950 border border-aerospace-800 text-xs text-white"
                >
                  <option value="ALL">All Airports</option>
                  <option value="DEL">Delhi (DEL)</option>
                  <option value="BOM">Mumbai (BOM)</option>
                  <option value="BLR">Bengaluru (BLR)</option>
                  <option value="HYD">Hyderabad (HYD)</option>
                  <option value="CCU">Kolkata (CCU)</option>
                </select>
              </div>
            </div>

            {/* Table of Flights */}
            <div className="max-h-[520px] overflow-y-auto space-y-2 pr-1">
              {filteredFlights.map((flight) => {
                const isSelected = flight.flight_id === selectedFlightId;
                const isHigh = flight.severity === 'HIGH';

                return (
                  <button
                    key={flight.flight_id}
                    type="button"
                    onClick={() => handleSelectFlight(flight.flight_id)}
                    className={`w-full text-left p-3 rounded-lg border transition-all cursor-pointer flex flex-col gap-1.5 ${
                      isSelected
                        ? 'bg-cyan-950/60 border-cyan-500 shadow-md ring-1 ring-cyan-500/50'
                        : 'bg-aerospace-950/70 border-aerospace-800 hover:border-slate-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-white">{flight.flight_number}</span>
                        <span className="font-mono text-[10px] text-cyan-400">{flight.aircraft_tail}</span>
                      </div>
                      <span
                        className={`text-[9px] font-mono px-2 py-0.5 rounded font-bold uppercase ${
                          isHigh
                            ? 'bg-rose-950 text-rose-300 border border-rose-800'
                            : 'bg-amber-950 text-amber-300 border border-amber-800'
                        }`}
                      >
                        +{Math.round(flight.predicted_delay_minutes)} min
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-xs text-slate-300">
                      <span className="font-mono font-bold text-slate-200">
                        {flight.origin_iata} ➔ {flight.destination_iata}
                      </span>
                      <span className="text-[11px] text-slate-400">
                        STD: {new Date(flight.scheduled_departure_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>

                    <div className="text-[10px] text-slate-500 flex items-center justify-between pt-1 border-t border-aerospace-800/60">
                      <span>Est. Pushback: {new Date(flight.estimated_departure_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      <span className="text-cyan-400 font-mono">Inspect ➔</span>
                    </div>
                    <div className="text-[10px] flex items-center justify-between pt-0.5">
                      <span className="text-cyan-300 font-semibold">
                        ETA: {flight.estimated_arrival_display ? flight.estimated_arrival_display.replace(' UTC', '') : (flight.estimated_arrival_utc ? new Date(flight.estimated_arrival_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '12:28 PM')}
                      </span>
                      <span className={flight.predicted_delay_minutes > 0 ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                        {flight.predicted_delay_minutes > 0
                          ? `(+${Math.round(flight.predicted_delay_minutes)} min delay)`
                          : flight.predicted_delay_minutes < 0
                          ? `(-${Math.round(Math.abs(flight.predicted_delay_minutes))} min early)`
                          : '(On Schedule)'}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Factor Breakdown Aggregate */}
          {dashboard && (
            <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-4 shadow-lg space-y-3">
              <h3 className="text-xs font-bold font-mono text-white uppercase flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
                <span>AGGREGATE DELAY CONTRIBUTIONS (FLEET SAMPLE)</span>
              </h3>
              <div className="space-y-2 text-xs">
                {Object.entries(dashboard.factor_breakdown).map(([factorName, totalMin]) => {
                  const maxVal = Math.max(...Object.values(dashboard.factor_breakdown), 1);
                  const pct = Math.round((totalMin / maxVal) * 100);
                  return (
                    <div key={factorName} className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-slate-300">{factorName}</span>
                        <span className="font-mono text-cyan-400 font-semibold">{Math.round(totalMin)} min</span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-aerospace-950 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right Column (7 Cols): Deep Dive Flight Inspector & Scenario Simulator */}
        <div className="lg:col-span-7 space-y-4">
          {detailLoading && (
            <div className="p-8 text-center bg-aerospace-900 border border-aerospace-800 rounded-xl text-xs text-slate-400">
              <RefreshCw className="w-5 h-5 text-cyan-400 animate-spin mx-auto mb-2" />
              <span>Loading detailed flight intelligence…</span>
            </div>
          )}

          {!detailLoading && selectedFlight && (
            <>
              {/* Selected Flight Summary Card */}
              <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-aerospace-800">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-base font-bold font-mono text-white">{selectedFlight.flight_number}</h3>
                      <span className="text-xs px-2 py-0.5 rounded bg-aerospace-950 text-cyan-300 font-mono border border-aerospace-800">
                        {selectedFlight.callsign}
                      </span>
                      <span className="text-xs text-slate-400">Tail: <strong className="text-white">{selectedFlight.aircraft_tail}</strong></span>
                    </div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      {selectedFlight.origin_name} ({selectedFlight.origin_iata}) ➔ {selectedFlight.destination_name} ({selectedFlight.destination_iata})
                    </div>
                  </div>

                  {onInspectRoute && (
                    <button
                      type="button"
                      onClick={() => onInspectRoute(selectedFlight.origin_id, selectedFlight.destination_id)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-950 border border-cyan-700/60 text-cyan-300 hover:text-white hover:bg-cyan-900/60 text-xs font-semibold transition-all cursor-pointer"
                    >
                      <Route size={13} />
                      <span>Route Lab Analysis</span>
                      <ArrowRight size={12} />
                    </button>
                  )}
                </div>

                {/* Grid of Key Timestamps & Differences */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3.5 rounded-lg bg-aerospace-950 border border-aerospace-800 text-xs">
                  <div>
                    <span className="text-slate-400 text-[10px]">SCHEDULED (STD)</span>
                    <div className="font-mono font-bold text-white mt-0.5">
                      {new Date(selectedFlight.scheduled_departure_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div className="text-[10px] text-slate-500">Timetable off-block</div>
                  </div>

                  <div>
                    <span className="text-cyan-400 text-[10px]">EST. GATE PUSHBACK</span>
                    <div className="font-mono font-bold text-cyan-300 mt-0.5">
                      {new Date(selectedFlight.estimated_gate_departure_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div className="text-[10px] text-cyan-500/80">+{Math.round(selectedFlight.predicted_delay_minutes)}m delay</div>
                  </div>

                  <div>
                    <span className="text-blue-400 text-[10px]">EST. RUNWAY TAKEOFF</span>
                    <div className="font-mono font-bold text-blue-300 mt-0.5">
                      {new Date(selectedFlight.estimated_takeoff_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div className="text-[10px] text-slate-500">Taxi: ~{Math.round(selectedFlight.taxi_time_minutes)} min</div>
                  </div>

                  <div>
                    <span className="text-amber-400 text-[10px]">EST. ARRIVAL (STA)</span>
                    <div className="font-mono font-bold text-amber-300 mt-0.5">
                      {new Date(selectedFlight.estimated_arrival_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div className="text-[10px] text-slate-500">En-route: ~{Math.round(selectedFlight.enroute_time_minutes)} min</div>
                  </div>
                </div>

                {/* Distinction Notice Banner */}
                <div className="p-3 rounded-lg bg-aerospace-950/70 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
                  <Info size={14} className="text-cyan-400 mt-0.5 shrink-0" />
                  <span>
                    <strong>Event Target Clarification:</strong> Departure delay reflects gate pushback delay relative to
                    scheduled off-block time. Runway take-off includes taxi queue sequencing. En-route flight plan optimization
                    modifies airborne traversal time (t_air), but does not resolve gate or boarding ground holds.
                  </span>
                </div>
              </div>

              {/* 8-Stage Operational Departure Progress Timeline */}
              <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-aerospace-800">
                  <h3 className="text-xs font-bold font-mono text-white uppercase flex items-center gap-2">
                    <Clock className="w-4 h-4 text-cyan-400" />
                    <span>8-STAGE OPERATIONAL DEPARTURE TIMELINE</span>
                  </h3>
                  <span className="text-[10px] font-mono text-slate-400">Chronological Lifecycle</span>
                </div>

                <div className="space-y-2.5">
                  {selectedFlight.timeline.map((stage) => {
                    const isDone = stage.status === 'COMPLETED';
                    const isDelayed = stage.status === 'DELAYED';
                    const isActive = stage.status === 'ACTIVE';

                    return (
                      <div
                        key={stage.stage_id}
                        className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800/80 flex items-start gap-3"
                      >
                        <span
                          className={`w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 ${
                            isDone
                              ? 'bg-emerald-400'
                              : isDelayed
                              ? 'bg-rose-400 animate-pulse'
                              : isActive
                              ? 'bg-cyan-400 animate-pulse'
                              : 'bg-slate-600'
                          }`}
                        />

                        <div className="flex-1 space-y-1">
                          <div className="flex flex-wrap items-center justify-between gap-1">
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-xs font-bold text-slate-400">Stage {stage.stage_number}:</span>
                              <span className="text-xs font-bold text-white">{stage.stage_name}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-[11px] text-cyan-300">{stage.time_display}</span>
                              <span
                                className={`text-[9px] font-mono px-1.5 py-0.5 rounded uppercase font-semibold ${
                                  stage.evidence_status === 'CONFIRMED'
                                    ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                                    : 'bg-blue-950 text-blue-300 border border-blue-800'
                                }`}
                              >
                                {stage.evidence_status}
                              </span>
                            </div>
                          </div>
                          <p className="text-slate-300 text-xs leading-relaxed font-sans">{stage.explanation}</p>
                          <div className="text-[10px] text-slate-500">Source: {stage.source}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Interactive Scenario Simulation Panel */}
              <div className="bg-gradient-to-r from-aerospace-900 via-aerospace-850 to-aerospace-900 border border-cyan-800/50 rounded-xl p-5 shadow-lg space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-aerospace-800">
                  <div className="flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-cyan-400" />
                    <h3 className="text-xs font-bold font-mono text-white uppercase">
                      WHAT-IF DISRUPTION SCENARIO SIMULATOR
                    </h3>
                  </div>
                  <span className="text-[10px] font-mono text-cyan-400">Dynamic Recalculation</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
                  {/* Slider 1: Inbound Delay Override */}
                  <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">Inbound Aircraft Arrival Delay:</span>
                      <span className="text-cyan-400 font-bold">{simInboundDelay} min</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="90"
                      step="5"
                      value={simInboundDelay}
                      onChange={(e) => setSimInboundDelay(Number(e.target.value))}
                      className="w-full accent-cyan-400"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>0m (On-time)</span>
                      <span>45m (Turnaround buffer limit)</span>
                      <span>90m (Severe)</span>
                    </div>
                  </div>

                  {/* Slider 2: Departure Queue Metering */}
                  <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">Departure Queue Addition:</span>
                      <span className="text-cyan-400 font-bold">+{simQueueAdditional} acft</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="15"
                      step="1"
                      value={simQueueAdditional}
                      onChange={(e) => setSimQueueAdditional(Number(e.target.value))}
                      className="w-full accent-cyan-400"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>0 (Nominal)</span>
                      <span>+8 (Peak bank)</span>
                      <span>+15 (Congested hold)</span>
                    </div>
                  </div>

                  {/* Slider 3: Weather Severity Override */}
                  <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">Local Weather Severity:</span>
                      <span className="text-amber-400 font-bold">{Math.round(simWeatherSeverity * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.1"
                      value={simWeatherSeverity}
                      onChange={(e) => setSimWeatherSeverity(Number(e.target.value))}
                      className="w-full accent-amber-400"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>VFR Clear</span>
                      <span>Showers</span>
                      <span>Monsoon Convective</span>
                    </div>
                  </div>

                  {/* Slider 4: Slot Delay */}
                  <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">Airspace Flow Slot Delay:</span>
                      <span className="text-purple-400 font-bold">+{simSlotDelay} min</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="40"
                      step="5"
                      value={simSlotDelay}
                      onChange={(e) => setSimSlotDelay(Number(e.target.value))}
                      className="w-full accent-purple-400"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>0m (Clear)</span>
                      <span>20m (Sector metered)</span>
                      <span>40m (Severe ground stop)</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 pt-2">
                  <button
                    type="button"
                    onClick={handleRunSimulation}
                    disabled={simulating}
                    className="flex-1 py-2 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-semibold flex items-center justify-center gap-2 transition-all cursor-pointer shadow-md disabled:opacity-50"
                  >
                    {simulating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                    <span>Recalculate Disruption Impact</span>
                  </button>

                  <button
                    type="button"
                    onClick={handleResetSimulation}
                    disabled={simulating}
                    className="px-4 py-2 rounded-lg bg-aerospace-950 border border-aerospace-800 hover:bg-aerospace-800 text-slate-300 text-xs font-semibold transition-all cursor-pointer"
                  >
                    Reset Baseline
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Data Readiness & Gap Analysis Modal */}
      {showReadinessModal && readinessReport && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-aerospace-900 border border-cyan-800/60 rounded-2xl p-6 max-w-2xl w-full max-h-[85vh] overflow-y-auto shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-aerospace-800">
              <div className="flex items-center gap-2 text-white font-mono font-bold text-sm">
                <ShieldCheck className="w-5 h-5 text-cyan-400" />
                <span>FLIGHT DELAY DATA READINESS & PRODUCTION GAP ANALYSIS</span>
              </div>
              <button
                type="button"
                onClick={() => setShowReadinessModal(false)}
                className="text-slate-400 hover:text-white text-xs font-mono px-2 py-1 rounded bg-aerospace-950 cursor-pointer"
              >
                ✕ Close
              </button>
            </div>

            <div className="space-y-4 text-xs">
              <div className="p-3 rounded-lg bg-cyan-950/40 border border-cyan-800/50 text-slate-200">
                <strong className="text-cyan-400">Current Status: {readinessReport.status}</strong>
                <p className="mt-1 leading-relaxed">{readinessReport.disclaimer}</p>
              </div>

              <div>
                <h4 className="font-mono font-bold text-slate-200 uppercase text-[11px] mb-2">
                  Implemented Features in Current Architecture:
                </h4>
                <ul className="space-y-1 list-disc list-inside text-slate-300 font-sans">
                  {readinessReport.current_capabilities.map((cap, i) => (
                    <li key={i}>{cap}</li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-mono font-bold text-amber-300 uppercase text-[11px] mb-2">
                  Missing Operational Labels Required for Supervised Real-World ML Training:
                </h4>
                <div className="space-y-2">
                  {readinessReport.missing_operational_labels.map((item, i) => (
                    <div key={i} className="p-2.5 rounded bg-aerospace-950 border border-aerospace-800 text-[11px]">
                      <div className="flex items-center justify-between">
                        <strong className="text-white font-mono">{item.field}</strong>
                        <span className="text-[10px] font-mono text-rose-400">{item.importance}</span>
                      </div>
                      <p className="text-slate-400 mt-1 font-sans">{item.reason}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="font-mono font-bold text-slate-200 uppercase text-[11px] mb-2">
                  Production Integration Roadmap:
                </h4>
                <ul className="space-y-1 list-disc list-inside text-slate-400 font-sans text-[11px]">
                  {readinessReport.requirements_for_production.map((req, i) => (
                    <li key={i}>{req}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
