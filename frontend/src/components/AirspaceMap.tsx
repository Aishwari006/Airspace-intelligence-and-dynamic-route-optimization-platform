import React, { useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Polygon, Tooltip, Popup } from 'react-leaflet';
import type { Airport, Waypoint, RouteEdge, RouteDetail, CongestionCell, RouteSegment } from '../types';
import { Eye, Layers, CloudRain } from 'lucide-react';

interface AirspaceMapProps {
  airports?: Airport[];
  waypoints?: Waypoint[];
  edges?: RouteEdge[];
  primaryRoute?: RouteDetail | null;
  secondaryRoute?: RouteDetail | null;
  congestionCells?: CongestionCell[];
  weatherCells?: any[];
  selectedSegmentId?: string | null;
  aircraft?: any[];
  center?: [number, number];
  zoom?: number;
  height?: string;
  onAirportClick?: (airport: Airport) => void;
  onSelectSegment?: (segment: RouteSegment | null) => void;
}

export const AirspaceMap: React.FC<AirspaceMapProps> = ({
  airports = [],
  waypoints = [],
  edges = [],
  primaryRoute = null,
  secondaryRoute = null,
  congestionCells = [],
  weatherCells = [],
  selectedSegmentId = null,
  aircraft = [],
  center = [21.5, 78.0], // Center over India
  zoom = 5,
  height = '580px',
  onAirportClick,
  onSelectSegment,
}) => {
  // Layer visibility toggles
  const [showAirports, setShowAirports] = useState(true);
  const [showWaypoints, setShowWaypoints] = useState(true);
  const [showAirways, setShowAirways] = useState(true);
  const [showCongestion, setShowCongestion] = useState(false);
  const [showWeather, setShowWeather] = useState(true);
  const [showAircraft, setShowAircraft] = useState(true);

  // Route edges may connect airports to intermediate waypoints, not only airport nodes.
  const nodeCoordinates = new Map<string, { latitude: number; longitude: number }>();
  airports.forEach((airport) => nodeCoordinates.set(airport.node_id, airport));
  waypoints.forEach((waypoint) => nodeCoordinates.set(waypoint.node_id, waypoint));

  const formatMapNumber = (value: unknown, decimals = 2): string => {

    const numeric = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(decimals) : '—';
  };

  const safeRouteGeometry = (route: RouteDetail | null) => (route?.geometry || []).filter(
    (point): point is [number, number] => Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]),
  );
  const primaryGeometry = safeRouteGeometry(primaryRoute);
  const secondaryGeometry = safeRouteGeometry(secondaryRoute);

  const getCellColor = (cls: string) => {
    switch (cls) {
      case 'HIGH':
        return '#ef4444';
      case 'MEDIUM':
        return '#f59e0b';
      case 'LOW':
      default:
        return '#10b981';
    }
  };

  return (
    <div className="relative w-full rounded-xl overflow-hidden border border-aerospace-800 shadow-2xl bg-aerospace-950">
      {/* Layer Toggles Control Overlay */}
      <div className="absolute top-4 right-4 z-[1000] bg-aerospace-900/90 backdrop-blur-md border border-aerospace-700/80 rounded-lg p-2.5 shadow-xl text-xs space-y-2">
        <div className="flex items-center gap-1.5 font-semibold text-slate-200 pb-1 border-b border-aerospace-800 font-mono">
          <Layers className="w-3.5 h-3.5 text-cyan-400" />
          <span>AIRSPACE LAYERS</span>
        </div>
        <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white">
          <input
            type="checkbox"
            checked={showAirports}
            onChange={(e) => setShowAirports(e.target.checked)}
            className="rounded bg-aerospace-950 border-aerospace-700 text-cyan-500 focus:ring-0"
          />
          <span>Airports ({airports.length})</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white">
          <input
            type="checkbox"
            checked={showWaypoints}
            onChange={(e) => setShowWaypoints(e.target.checked)}
            className="rounded bg-aerospace-950 border-aerospace-700 text-cyan-500 focus:ring-0"
          />
          <span>Waypoints ({waypoints.length})</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white">
          <input
            type="checkbox"
            checked={showAirways}
            onChange={(e) => setShowAirways(e.target.checked)}
            className="rounded bg-aerospace-950 border-aerospace-700 text-cyan-500 focus:ring-0"
          />
          <span>Airway Edges ({edges.length})</span>
        </label>
        {weatherCells && weatherCells.length > 0 && (
          <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white">
            <input
              type="checkbox"
              checked={showWeather}
              onChange={(e) => setShowWeather(e.target.checked)}
              className="rounded bg-aerospace-950 border-aerospace-700 text-amber-500 focus:ring-0"
            />
            <span className="flex items-center gap-1">
              <CloudRain className="w-3 h-3 text-amber-400" /> Weather Hazards
            </span>
          </label>
        )}
        <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white">
          <input
            type="checkbox"
            checked={showCongestion}
            onChange={(e) => setShowCongestion(e.target.checked)}
            className="rounded bg-aerospace-950 border-aerospace-700 text-cyan-500 focus:ring-0"
          />
          <span>Congestion Grid</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white">
          <input
            type="checkbox"
            checked={showAircraft}
            onChange={(e) => setShowAircraft(e.target.checked)}
            className="rounded bg-aerospace-950 border-aerospace-700 text-cyan-500 focus:ring-0"
          />
          <span>Surveillance ({aircraft.length})</span>
        </label>
      </div>

      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height, width: '100%' }}
        scrollWheelZoom={true}
      >
        {/* OpenStreetMap basemap: standard raster tiles styled for dark mode */}
        <TileLayer
          className="skyforce-dark-osm"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* 1. Convective Weather Hazards Overlay */}
        {showWeather &&
          weatherCells.map((wx, i) => {
            const lat = Number(wx.latitude);
            const lon = Number(wx.longitude);
            if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
            const precip = wx.precipitation_mm || 0;
            const gusts = wx.wind_gusts_10m_ms || 0;
            const sev = precip > 5 || gusts > 20 ? 'HIGH' : precip > 1 || gusts > 12 ? 'MEDIUM' : 'LOW';
            if (sev === 'LOW' && precip === 0) return null; // Only show active weather

            const color = sev === 'HIGH' ? '#ef4444' : '#f59e0b';
            return (
              <CircleMarker
                key={`wx-${wx.location_id || i}`}
                center={[lat, lon]}
                radius={sev === 'HIGH' ? 22 : 14}
                pathOptions={{
                  color,
                  fillColor: color,
                  fillOpacity: 0.35,
                  weight: 1.5,
                  dashArray: '3, 3',
                }}
              >
                <Tooltip sticky>
                  <div className="font-mono text-xs">
                    <strong className="text-amber-300">Convective Storm Cell</strong>
                    <div>Location: {wx.location_id}</div>
                    <div>Precipitation: {precip.toFixed(1)} mm/h</div>
                    <div>Wind Gusts: {gusts.toFixed(1)} m/s</div>
                    <div>Severity Rating: {sev}</div>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}

        {/* 2. Congestion Grid Cells */}
        {showCongestion &&
          congestionCells.map((c) => {
            if (!Array.isArray(c.bounds) || c.bounds.length !== 2 || !Array.isArray(c.bounds[0]) || !Array.isArray(c.bounds[1])) return null;
            const [[sLat, wLon], [nLat, eLon]] = c.bounds;
            if (![sLat, wLon, nLat, eLon].every((value) => Number.isFinite(value))) return null;
            const polyCoords: [number, number][] = [
              [sLat, wLon],
              [sLat, eLon],
              [nLat, eLon],
              [nLat, wLon],
            ];
            return (
              <Polygon
                key={c.cell_id}
                positions={polyCoords}
                pathOptions={{
                  color: getCellColor(c.predicted_class),
                  fillColor: getCellColor(c.predicted_class),
                  fillOpacity: 0.15,
                  weight: 1,
                  dashArray: '2, 4',
                }}
              >
                <Tooltip sticky>
                  <div className="font-mono text-xs">
                    <strong>Cell {c.cell_id}</strong>
                    <div>Density: {c.predicted_class}</div>
                    <div>Score: {formatMapNumber(c.congestion_score)}</div>
                    <div>Count: {c.aircraft_count}</div>
                  </div>
                </Tooltip>
              </Polygon>
            );
          })}

        {/* 3. Candidate Airway Edges */}
        {showAirways &&
          edges.map((e) => {
            const u = nodeCoordinates.get(e.source_node);
            const v = nodeCoordinates.get(e.target_node);
            if (!u || !v || !Number.isFinite(u.latitude) || !Number.isFinite(u.longitude) || !Number.isFinite(v.latitude) || !Number.isFinite(v.longitude)) return null;
            return (
              <Polyline
                key={e.edge_id}
                positions={[
                  [u.latitude, u.longitude],
                  [v.latitude, v.longitude],
                ]}
                pathOptions={{
                  color: '#475569',
                  weight: 1.2,
                  opacity: 0.7,
                }}
              >
                <Tooltip sticky>
                  <span className="font-mono text-[11px]">
                    {e.edge_id} ({formatMapNumber(e.distance_km, 0)} km)
                  </span>
                </Tooltip>
              </Polyline>
            );
          })}

        {/* 4. Secondary Route (Baseline) */}
        {secondaryGeometry.length > 1 && secondaryRoute && (
          <Polyline
            positions={secondaryGeometry}
            pathOptions={{
              color: '#f59e0b',
              weight: 3.5,
              opacity: 0.85,
              dashArray: '6, 6',
            }}
          >
            <Tooltip sticky>
              <div className="font-mono text-xs">
                <strong>{secondaryRoute.label || 'Baseline Route'}</strong>
                <div>Distance: {secondaryRoute.totals?.distance_km} km</div>
                <div>ETA: {secondaryRoute.totals?.estimated_time_min} min</div>
              </div>
            </Tooltip>
          </Polyline>
        )}

        {/* 5. Primary Route Interactive Segments (with click & highlight) */}
        {primaryRoute?.segments && primaryRoute.segments.length > 0 ? (
          primaryRoute.segments.map((seg) => {
            const u = nodeCoordinates.get(seg.from_node);
            const v = nodeCoordinates.get(seg.to_node);
            if (!u || !v) return null;
            const isSelected = seg.edge_id === selectedSegmentId;

            return (
              <Polyline
                key={`seg-${seg.edge_id}`}
                positions={[
                  [u.latitude, u.longitude],
                  [v.latitude, v.longitude],
                ]}
                pathOptions={{
                  color: isSelected ? '#ffffff' : '#00f0ff',
                  weight: isSelected ? 7 : 4.5,
                  opacity: isSelected ? 1.0 : 0.95,
                }}
                eventHandlers={{
                  click: () => onSelectSegment?.(seg),
                }}
              >
                <Tooltip sticky>
                  <div className="font-mono text-xs">
                    <strong>{seg.from_node} → {seg.to_node} ({seg.distance_km.toFixed(0)} km)</strong>
                    <div>ETA: {seg.time_min.toFixed(1)} min</div>
                    <div>Weather: {seg.weather_severity !== null ? `${(seg.weather_severity * 100).toFixed(0)}%` : 'None'}</div>
                    <div>Traffic: {seg.congestion_score !== null ? `${(seg.congestion_score * 100).toFixed(0)}%` : 'None'}</div>
                    <div className="text-[10px] text-cyan-300 mt-1">Click to inspect segment details</div>
                  </div>
                </Tooltip>
              </Polyline>
            );
          })
        ) : (
          primaryGeometry.length > 1 && primaryRoute && (
            <Polyline
              positions={primaryGeometry}
              pathOptions={{
                color: '#00f0ff',
                weight: 4.5,
                opacity: 0.95,
              }}
            >
              <Tooltip sticky>
                <div className="font-mono text-xs">
                  <strong>{primaryRoute.label || 'Active Route'}</strong>
                  <div>Distance: {primaryRoute.totals?.distance_km} km</div>
                  <div>ETA: {primaryRoute.totals?.estimated_time_min} min</div>
                </div>
              </Tooltip>
            </Polyline>
          )
        )}


        {/* 5. Intermediate airway waypoints */}
        {showWaypoints && waypoints.filter((waypoint) => Number.isFinite(waypoint.latitude) && Number.isFinite(waypoint.longitude)).map((waypoint) => (
          <CircleMarker
            key={waypoint.node_id}
            center={[waypoint.latitude, waypoint.longitude]}
            radius={2.8}
            pathOptions={{ color: '#90a6c1', fillColor: '#263c56', fillOpacity: 0.95, weight: 1.2 }}
          >
            <Tooltip direction="top" offset={[0, -4]}>
              <div className="font-mono text-xs"><strong>{waypoint.node_id}</strong><div>{waypoint.name}</div><div>Intermediate waypoint</div></div>
            </Tooltip>
          </CircleMarker>
        ))}

        {/* 6. Airports */}
        {showAirports &&
          airports.filter((a) => Number.isFinite(a.latitude) && Number.isFinite(a.longitude)).map((a) => (
            <CircleMarker
              key={a.node_id}
              center={[a.latitude, a.longitude]}
              radius={a.iata ? 5.5 : 3.5}
              pathOptions={{
                color: a.iata ? '#00f0ff' : '#64748b',
                fillColor: a.iata ? '#0b1329' : '#1e293b',
                fillOpacity: 0.9,
                weight: 1.8,
              }}
              eventHandlers={{
                click: () => onAirportClick?.(a),
              }}
            >
              <Tooltip direction="top" offset={[0, -5]} opacity={0.9}>
                <div className="font-mono text-xs">
                  <strong>{a.iata || a.node_id}</strong> - {a.name}
                </div>
              </Tooltip>
              <Popup>
                <div className="font-sans text-xs p-1 space-y-1">
                  <div className="font-bold text-cyan-400 font-mono text-sm">{a.node_id} ({a.iata || 'WP'})</div>
                  <div className="text-slate-200">{a.name}</div>
                  <div className="text-slate-400 font-mono text-[11px]">
                    Lat: {a.latitude.toFixed(4)}, Lon: {a.longitude.toFixed(4)}
                  </div>
                  {a.elevation_m !== undefined && Number.isFinite(a.elevation_m) && (
                    <div className="text-slate-400 text-[11px]">Elevation: {a.elevation_m}m</div>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          ))}

        {/* 7. Surveillance Aircraft */}
        {showAircraft &&
          aircraft.map((ac, idx) => {
            if (!Number.isFinite(ac.latitude) || !Number.isFinite(ac.longitude)) return null;
            return (
              <CircleMarker
                key={ac.observation_id || `ac-${idx}`}
                center={[ac.latitude, ac.longitude]}
                radius={3}
                pathOptions={{
                  color: '#10b981',
                  fillColor: '#10b981',
                  fillOpacity: 0.8,
                  weight: 1,
                }}
              >
                <Tooltip>
                  <div className="font-mono text-xs">
                    <strong>{ac.callsign || ac.source_aircraft_id}</strong>
                    <div>Speed: {Number.isFinite(ac.ground_speed_mps) ? `${(Number(ac.ground_speed_mps) * 3.6).toFixed(0)} km/h` : 'N/A'}</div>
                    <div>Alt: {Number.isFinite(ac.altitude_m) ? `${Number(ac.altitude_m).toFixed(0)} m` : 'N/A'}</div>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
      </MapContainer>
    </div>
  );
};
