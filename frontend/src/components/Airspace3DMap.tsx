import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as maplibregl from 'maplibre-gl';
import { setWorkerUrl } from 'maplibre-gl';
import mapLibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { PolygonLayer, PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import type { Airport, Waypoint, RouteEdge, RouteDetail, CongestionCell, RouteSegment } from '../types';
import { api } from '../services/api';
import { Layers, Compass, CloudRain, RotateCcw, AlertCircle, Info, Loader2 } from 'lucide-react';
// MapLibre GL JS v6 ships its worker as a separate ESM file. Vite must bundle
// that worker explicitly so MapLibre does not look for a missing sibling file.
// This must run at module initialization, before the first Map instance is made.
setWorkerUrl(mapLibreWorkerUrl);
interface Airspace3DMapProps {
  airports?: Airport[];
  waypoints?: Waypoint[];
  edges?: RouteEdge[];
  primaryRoute?: RouteDetail | null;
  secondaryRoute?: RouteDetail | null;
  // Accept either a cell array or the API envelope { cells: [...] }.
  congestionCells?: CongestionCell[] | { cells?: unknown[] } | null;
  weatherCells?: any[];
  selectedSegmentId?: string | null;
  height?: string;
  onSelectSegment?: (segment: RouteSegment | null) => void;
}
// OpenFreeMap Liberty: Public vector basemap style requiring ZERO API keys or tokens
const OPENFREEMAP_LIBERTY_STYLE = 'https://tiles.openfreemap.org/styles/liberty';
// First fallback: public CARTO dark raster tiles. This still requires internet, but
// uses a different host from OpenFreeMap and often works when the vector style is blocked.
const CARTO_RASTER_FALLBACK_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
      ],
      tileSize: 256,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    },
  },
  layers: [
    {
      id: 'carto-dark-layer',
      type: 'raster',
      source: 'carto-dark',
      minzoom: 0,
      maxzoom: 19,
    },
  ],
};
// Last resort: no remote requests at all. Deck.gl overlays can still render here.
const OFFLINE_DARK_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {},
  layers: [
    {
      id: 'offline-background',
      type: 'background',
      paint: { 'background-color': '#0b1220' },
    },
  ],
};
export const Airspace3DMap: React.FC<Airspace3DMapProps> = ({
  airports = [],
  waypoints = [],
  edges = [],
  primaryRoute = null,
  secondaryRoute = null,
  congestionCells: congestionCellsInput = [],
  weatherCells = [],
  selectedSegmentId = null,
  height = '560px',
  onSelectSegment,
}) => {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  // Map lifecycle and status
  const [mapLoading, setMapLoading] = useState(true);
  const [mapError, setMapError] = useState<string | null>(null);
  // Layer toggles
  const [showAirports, setShowAirports] = useState(true);
  const [showAirways, setShowAirways] = useState(true);
  const [showCongestion3D, setShowCongestion3D] = useState(true);
  const [showWeather3D, setShowWeather3D] = useState(true);
  const [showRoutes3D, setShowRoutes3D] = useState(true);
  // Hover tooltip state
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    object: any;
    layerType: 'segment' | 'congestion' | 'weather' | 'airport';
  } | null>(null);
  const nodeMap = useMemo(() => {
    const map = new Map<string, { latitude: number; longitude: number }>();
    airports.forEach((a) => map.set(a.node_id, a));
    waypoints.forEach((w) => map.set(w.node_id, w));
    return map;
  }, [airports, waypoints]);
  // The backend returns { cells: [...] }, while older frontend code expected an array.
  // Accept both forms and fetch the real cell predictions if the parent hasn't passed them.
  const providedCongestionCells = useMemo<any[]>(() => {
    if (Array.isArray(congestionCellsInput)) return congestionCellsInput as any[];
    const wrapped = (congestionCellsInput as any)?.cells;
    return Array.isArray(wrapped) ? wrapped : [];
  }, [congestionCellsInput]);
  const hasRenderableProvidedCells = useMemo(() => providedCongestionCells.some((cell: any) => {
    const hasGeometry =
      (Number.isFinite(Number(cell.center_lat ?? cell.centerLat ?? cell.latitude ?? cell.lat)) &&
        Number.isFinite(Number(cell.center_lon ?? cell.centerLon ?? cell.longitude ?? cell.lon))) ||
      (Number.isFinite(Number(cell.lat_min)) && Number.isFinite(Number(cell.lat_max)) &&
        Number.isFinite(Number(cell.lon_min)) && Number.isFinite(Number(cell.lon_max))) ||
      /^R\d{1,2}C\d{1,2}$/i.test(String(cell.cell_id ?? ''));
    const hasSignal = cell.congestion_score != null || cell.predicted_score != null || cell.score != null ||
      ['LOW', 'MEDIUM', 'HIGH'].includes(String(cell.predicted_class ?? cell.congestion_class ?? cell.class ?? '').toUpperCase());
    return hasGeometry && hasSignal;
  }), [providedCongestionCells]);
  const [fetchedCongestionCells, setFetchedCongestionCells] = useState<any[]>([]);
  useEffect(() => {
    if (hasRenderableProvidedCells) {
      setFetchedCongestionCells([]);
      return;
    }
    let cancelled = false;
    api.getCongestionCells()
      .then((response: any) => {
        const cells = Array.isArray(response)
          ? response
          : Array.isArray(response?.cells)
            ? response.cells
            : [];
        if (!cancelled) {
          setFetchedCongestionCells(cells);
          if (cells.length === 0) console.warn('Congestion API returned no spatial cells.');
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) console.warn('Could not load congestion cells for the 3D map:', error);
      });
    return () => { cancelled = true; };
  }, [hasRenderableProvidedCells]);
  // The congestion endpoint calls the model output `predicted_score`, but the map
  // historically used `congestion_score`. Use the true grid bounds when provided
  // (these are 2-degree research cells, not 0.9-degree visual approximations).
  const mapCongestionCells = useMemo<any[]>(() => {
    const source = hasRenderableProvidedCells
      ? providedCongestionCells
      : fetchedCongestionCells;
    return source.map((cell: any) => {
      // Uploaded pre-aggregated grid CSVs may only contain `cell_id` (e.g. R08C03).
      // Decode that ID using the backend's configured 2-degree grid when bounds are absent.
      const cellIdMatch = String(cell.cell_id ?? '').match(/^R(\d{1,2})C(\d{1,2})$/i);
      const row = cellIdMatch ? Number(cellIdMatch[1]) : NaN;
      const col = cellIdMatch ? Number(cellIdMatch[2]) : NaN;
      const decodedLatMin = Number.isFinite(row) ? 6 + row * 2 : undefined;
      const decodedLonMin = Number.isFinite(col) ? 66 + col * 2 : undefined;
      const latMin = Number(cell.lat_min ?? cell.bounds?.[0]?.[0] ?? decodedLatMin);
      const latMax = Number(cell.lat_max ?? cell.bounds?.[1]?.[0] ?? (decodedLatMin === undefined ? undefined : Math.min(decodedLatMin + 2, 34)));
      const lonMin = Number(cell.lon_min ?? cell.bounds?.[0]?.[1] ?? decodedLonMin);
      const lonMax = Number(cell.lon_max ?? cell.bounds?.[1]?.[1] ?? (decodedLonMin === undefined ? undefined : Math.min(decodedLonMin + 2, 94)));
      const latFromBounds = Number.isFinite(latMin) && Number.isFinite(latMax) ? (latMin + latMax) / 2 : undefined;
      const lonFromBounds = Number.isFinite(lonMin) && Number.isFinite(lonMax) ? (lonMin + lonMax) / 2 : undefined;
      const centerLat = Number(cell.center_lat ?? cell.centerLat ?? cell.latitude ?? cell.lat ?? latFromBounds);
      const centerLonValue = cell.center_lon ?? cell.centerLon ?? cell.longitude ?? cell.lon ?? lonFromBounds;
      const centerLon = centerLonValue === undefined || centerLonValue === null ? NaN : Number(centerLonValue);
      const rawScore = cell.congestion_score ?? cell.predicted_score ?? cell.score;
      const parsedScore = rawScore === null || rawScore === undefined ? NaN : Number(rawScore);
      const rawClass = String(cell.predicted_class ?? cell.congestion_class ?? cell.class ?? '').toUpperCase();
      const classScore = rawClass === 'HIGH' ? 0.9 : rawClass === 'MEDIUM' ? 0.55 : rawClass === 'LOW' ? 0.2 : NaN;
      const score = Number.isFinite(parsedScore) ? Math.max(0, Math.min(1, parsedScore)) : classScore;
      const displayClass = ['LOW', 'MEDIUM', 'HIGH'].includes(rawClass)
        ? rawClass
        : Number.isFinite(score)
          ? score >= 0.67 ? 'HIGH' : score >= 0.33 ? 'MEDIUM' : 'LOW'
          : '';
      return {
        ...cell,
        center_lat: centerLat,
        center_lon: centerLon,
        lat_min: Number.isFinite(latMin) ? latMin : centerLat - 1,
        lat_max: Number.isFinite(latMax) ? latMax : centerLat + 1,
        lon_min: Number.isFinite(lonMin) ? lonMin : centerLon - 1,
        lon_max: Number.isFinite(lonMax) ? lonMax : centerLon + 1,
        predicted_class: displayClass,
        congestion_score: score,
        aircraft_count: cell.aircraft_count ?? cell.observed_count ?? cell.count ?? null,
        has_congestion_signal: Number.isFinite(score) || ['LOW', 'MEDIUM', 'HIGH'].includes(rawClass),
      };
    }).filter((cell: any) =>
      Number.isFinite(cell.center_lat) &&
      Number.isFinite(cell.center_lon) &&
      cell.center_lat >= -90 && cell.center_lat <= 90 &&
      cell.center_lon >= -180 && cell.center_lon <= 180 &&
      cell.has_congestion_signal &&
      cell.lat_max > cell.lat_min &&
      cell.lon_max > cell.lon_min
    );
  }, [providedCongestionCells, hasRenderableProvidedCells, fetchedCongestionCells]);
  // Try OpenFreeMap first, then a different hosted raster provider, then a local dark canvas.
  // This avoids getting stuck forever when a style/tile host is blocked or unreachable.
  useEffect(() => {
    if (!mapContainerRef.current) return;
    setMapLoading(true);
    setMapError(null);
    let mapInstance: maplibregl.Map | null = null;
    let overlayInstance: MapboxOverlay | null = null;
    let disposed = false;
    let mapReady = false;
    let stage: 'openfreemap' | 'carto' | 'offline' = 'openfreemap';
    let openFreeMapTimer: ReturnType<typeof setTimeout> | null = null;
    let cartoTimer: ReturnType<typeof setTimeout> | null = null;
    let finalTimer: ReturnType<typeof setTimeout> | null = null;
    const clearTimers = () => {
      if (openFreeMapTimer) clearTimeout(openFreeMapTimer);
      if (cartoTimer) clearTimeout(cartoTimer);
      if (finalTimer) clearTimeout(finalTimer);
      openFreeMapTimer = null;
      cartoTimer = null;
      finalTimer = null;
    };
    const markReady = () => {
      if (disposed) return;
      mapReady = true;
      clearTimers();
      setMapLoading(false);
      if (stage === 'openfreemap') {
        setMapError(null);
      } else if (stage === 'carto') {
        setMapError('OpenFreeMap did not respond. Showing the backup CARTO basemap; scenario routes are still demo overlays.');
      } else {
        setMapError('Online basemaps did not respond. Demo overlays are running on the offline dark canvas.');
      }
    };
    const activateOfflineFallback = (reason: string) => {
      if (!mapInstance || stage === 'offline' || mapReady || disposed) return;
      stage = 'offline';
      if (cartoTimer) clearTimeout(cartoTimer);
      cartoTimer = null;
      setMapError(`Online basemaps did not load (${reason}). Switching to the offline demo canvas.`);
      try {
        mapInstance.once('idle', markReady);
        mapInstance.setStyle(OFFLINE_DARK_STYLE);
      } catch (error) {
        console.error('Unable to activate offline map canvas:', error);
        clearTimers();
        setMapLoading(false);
        setMapError('The map could not initialize. Try the 2D Tactical Map or check browser WebGL support.');
      }
    };
    const activateCartoFallback = (reason: string) => {
      if (!mapInstance || stage !== 'openfreemap' || mapReady || disposed) return;
      stage = 'carto';
      if (openFreeMapTimer) clearTimeout(openFreeMapTimer);
      openFreeMapTimer = null;
      setMapError(`OpenFreeMap did not load (${reason}). Trying a backup map provider...`);
      try {
        mapInstance.once('idle', markReady);
        mapInstance.setStyle(CARTO_RASTER_FALLBACK_STYLE);
        cartoTimer = setTimeout(() => {
          activateOfflineFallback('backup map request timed out after 8 seconds');
        }, 8000);
      } catch (error) {
        console.warn('CARTO fallback could not be initialized:', error);
        activateOfflineFallback('backup map initialization failed');
      }
    };
    try {
      mapInstance = new maplibregl.Map({
        container: mapContainerRef.current,
        style: OPENFREEMAP_LIBERTY_STYLE,
        center: [78.9629, 21.5937],
        zoom: 4.8,
        pitch: 48,
        bearing: -14,
        attributionControl: false,
      });
      overlayInstance = new MapboxOverlay({ layers: [] });
      mapInstance.addControl(overlayInstance as unknown as maplibregl.IControl);
      mapInstance.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
      mapInstance.on('load', markReady);
      mapInstance.on('error', (event) => {
        if (mapReady || disposed) return;
        const message = event?.error?.message || 'basemap request failed';
        // Ignore errors from stale OpenFreeMap requests after the style has switched.
        // The CARTO fallback has its own timeout before we go to the offline canvas.
        if (stage === 'openfreemap') {
          activateCartoFallback(message);
        }
      });
      // Some networks silently stall requests without emitting an error event.
      openFreeMapTimer = setTimeout(() => {
        activateCartoFallback('request timed out after 8 seconds');
      }, 8000);
      // Final guard: never leave the loading overlay visible indefinitely.
      finalTimer = setTimeout(() => {
        if (!mapReady && !disposed) {
          clearTimers();
          setMapLoading(false);
          setMapError('The map did not initialize. Check network access and WebGL, or switch to 2D Tactical Map.');
        }
      }, 20000);
      mapRef.current = mapInstance;
      overlayRef.current = overlayInstance;
    } catch (error: any) {
      console.error('Failed to initialize MapLibre map:', error);
      clearTimers();
      setMapError(error?.message || 'Failed to initialize 3D map');
      setMapLoading(false);
    }
    return () => {
      disposed = true;
      clearTimers();
      mapInstance?.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
  }, []);
  const handleResetCamera = () => {
    if (!mapRef.current) return;
    mapRef.current.flyTo({
      center: [78.9629, 21.5937],
      zoom: 4.8,
      pitch: 48,
      bearing: -14,
      essential: true,
    });
  };
  // Re-build deck.gl data-driven 3D layers when props or toggles change
  useEffect(() => {
    if (!overlayRef.current) return;
    const layers = [];
    // 1. Airways Corridor Base Layer
    if (showAirways && edges.length > 0) {
      const edgePaths = edges
        .map((e) => {
          const u = nodeMap.get(e.source_node);
          const v = nodeMap.get(e.target_node);
          if (!u || !v) return null;
          return {
            edge_id: e.edge_id,
            distance_km: e.distance_km,
            path: [
              [u.longitude, u.latitude, 300],
              [v.longitude, v.latitude, 300],
            ],
          };
        })
        .filter(Boolean);
      layers.push(
        new PathLayer({
          id: 'airways-layer-3d',
          data: edgePaths,
          getPath: (d: any) => d.path,
          getColor: [71, 85, 105, 90], // Slate-600 subtle airway corridor
          getWidth: 1.5,
          widthUnits: 'pixels',
          pickable: false,
        })
      );
    }
    // 2. 3D Congestion Columns Layer (Derived directly from measured density or predicted score)
    if (showCongestion3D && mapCongestionCells.length > 0) {
      const congestionPolys = mapCongestionCells.map((c: any) => {
        const wLon = c.lon_min;
        const eLon = c.lon_max;
        const sLat = c.lat_min;
        const nLat = c.lat_max;
        // Height is a visual representation of the model's normalized congestion score.
        const score = Number.isFinite(c.congestion_score) ? c.congestion_score : 0.2;
        const elevation = Math.max(700, score * 12000);
        let color = [16, 185, 129, 195]; // LOW: green
        if (c.predicted_class === 'HIGH' || score >= 0.67) {
          color = [244, 63, 94, 235]; // HIGH: vivid rose
        } else if (c.predicted_class === 'MEDIUM' || score >= 0.33) {
          color = [245, 158, 11, 225]; // MEDIUM: amber
        }
        return {
          cell_id: c.cell_id,
          polygon: [[wLon, sLat], [eLon, sLat], [eLon, nLat], [wLon, nLat]],
          elevation,
          color,
          cellData: c,
        };
      });
      layers.push(
        new PolygonLayer({
          id: 'congestion-3d-layer',
          data: congestionPolys,
          getPolygon: (d: any) => d.polygon,
          getElevation: (d: any) => d.elevation,
          getFillColor: (d: any) => d.color,
          getLineColor: [255, 255, 255, 50],
          getLineWidth: 1,
          lineWidthUnits: 'pixels',
          extruded: true,
          wireframe: true,
          pickable: true,
          onHover: (info: any) => {
            if (info.object) {
              setHoverInfo({
                x: info.x,
                y: info.y,
                object: info.object.cellData,
                layerType: 'congestion',
              });
            } else {
              setHoverInfo(null);
            }
          },
        })
      );
    }
    // 3. 3D Weather Storm Hazards Layer (Derived from actual precipitation & wind measurements)
    if (showWeather3D && weatherCells.length > 0) {
      const weatherPolys = weatherCells.map((w, idx) => {
        const delta = 0.4;
        const lat = w.latitude || w.center_lat || 20.0;
        const lon = w.longitude || w.center_lon || 78.0;
        const precip = w.precipitation_mm ?? (w.severity ? w.severity * 15 : 0);
        const sev = w.severity ?? Math.min(1.0, precip / 15.0);
        // Visual height scale: height directly scales with rainfall intensity (0 to 9,000m)
        const elevation = Math.max(400, sev * 9000);
        // Convective storm palette: cyan (light rain) to deep purple/magenta (severe storm)
        const r = Math.min(255, Math.floor(56 + sev * 150));
        const g = Math.max(20, Math.floor(189 - sev * 130));
        const b = 250;
        return {
          id: w.location_id || w.cell_id || `wx-${idx}`,
          polygon: [
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta],
          ],
          elevation,
          color: [r, g, b, 160],
          wxData: { ...w, computed_severity: sev, precip_val: precip },
        };
      });
      layers.push(
        new PolygonLayer({
          id: 'weather-3d-layer',
          data: weatherPolys,
          getPolygon: (d: any) => d.polygon,
          getElevation: (d: any) => d.elevation,
          getFillColor: (d: any) => d.color,
          getLineColor: [168, 85, 247, 90],
          getLineWidth: 1.5,
          lineWidthUnits: 'pixels',
          extruded: true,
          wireframe: true,
          pickable: true,
          onHover: (info: any) => {
            if (info.object) {
              setHoverInfo({
                x: info.x,
                y: info.y,
                object: info.object.wxData,
                layerType: 'weather',
              });
            } else {
              setHoverInfo(null);
            }
          },
        })
      );
    }
    // 4. Secondary (Baseline) Route in 3D (Derived from real backend baseline geometry)
    if (showRoutes3D && secondaryRoute && secondaryRoute.geometry && secondaryRoute.geometry.length > 1) {
      const baseCoords = secondaryRoute.geometry.map(([lat, lon]) => [lon, lat, 3800]); // Elevated 3,800m
      layers.push(
        new PathLayer({
          id: 'secondary-route-3d',
          data: [{ path: baseCoords, label: secondaryRoute.label, totals: secondaryRoute.totals }],
          getPath: (d: any) => d.path,
          getColor: [245, 158, 11, 210], // Amber baseline
          getWidth: 3.5,
          widthUnits: 'pixels',
          pickable: false,
        })
      );
    }
    // 5. Primary Optimized Route Segments in 3D (Interactive with Click Selection & Highlighting)
    if (showRoutes3D && primaryRoute?.segments && primaryRoute.segments.length > 0) {
      const segmentPaths = primaryRoute.segments
        .map((seg) => {
          const u = nodeMap.get(seg.from_node);
          const v = nodeMap.get(seg.to_node);
          if (!u || !v) return null;
          const isSelected = seg.edge_id === selectedSegmentId;
          return {
            segment: seg,
            isSelected,
            path: [
              [u.longitude, u.latitude, 16000], // Visual offset above congestion and weather columns
              [v.longitude, v.latitude, 16000],
            ],
            color: isSelected ? [255, 255, 255, 255] : [30, 144, 255, 255], // Bright blue; white when selected
            width: isSelected ? 8.5 : 5.0,
          };
        })
        .filter(Boolean);
      // Visual-only route casing: a dark outline makes the blue route stand out over colored columns.
      layers.push(
        new PathLayer({
          id: 'primary-route-casing-3d',
          data: segmentPaths,
          // A slight vertical offset avoids depth-fighting with the blue center line.
          getPath: (d: any) => d.path.map(([longitude, latitude]: [number, number, number]) => [longitude, latitude, 15800]),
          getColor: [3, 18, 40, 255],
          getWidth: (d: any) => d.isSelected ? 13 : 10,
          widthUnits: 'pixels',
          jointRounded: true,
          capRounded: true,
          pickable: false,
        })
      );

      layers.push(
        new PathLayer({
          id: 'primary-route-segments-3d',
          data: segmentPaths,
          getPath: (d: any) => d.path,
          getColor: (d: any) => d.color,
          getWidth: (d: any) => d.width,
          widthUnits: 'pixels',
          jointRounded: true,
          capRounded: true,
          pickable: true,
          onClick: (info: any) => {
            if (info.object && onSelectSegment) {
              onSelectSegment(info.object.segment);
            }
          },
          onHover: (info: any) => {
            if (info.object) {
              setHoverInfo({
                x: info.x,
                y: info.y,
                object: info.object.segment,
                layerType: 'segment',
              });
            } else {
              setHoverInfo(null);
            }
          },
        })
      );
    }
    // 6. Airports & Waypoints Discs & Floating 3D Text Labels
    if (showAirports && airports.length > 0) {
      const airportPoints = airports.map((a) => ({
        coordinates: [a.longitude, a.latitude, 100],
        name: a.name,
        code: a.node_id,
        iata: a.iata,
      }));
      layers.push(
        new ScatterplotLayer({
          id: 'airports-disc-3d',
          data: airportPoints,
          getPosition: (d: any) => d.coordinates,
          getRadius: 16000,
          getFillColor: [6, 182, 212, 180],
          getLineColor: [255, 255, 255, 240],
          getLineWidth: 2,
          stroked: true,
          pickable: true,
          onHover: (info: any) => {
            if (info.object) {
              setHoverInfo({
                x: info.x,
                y: info.y,
                object: info.object,
                layerType: 'airport',
              });
            } else {
              setHoverInfo(null);
            }
          },
        }),
        new TextLayer({
          id: 'airports-labels-3d',
          data: airportPoints,
          getPosition: (d: any) => [d.coordinates[0], d.coordinates[1], 1500],
          getText: (d: any) => d.code,
          getSize: 12,
          getColor: [255, 255, 255, 240],
          getTextAnchor: 'middle',
          getAlignmentBaseline: 'center',
          fontFamily: 'monospace',
          fontWeight: 700,
        })
      );
    }
    overlayRef.current.setProps({ layers });
  }, [
    showAirports,
    showAirways,
    showCongestion3D,
    showWeather3D,
    showRoutes3D,
    edges,
    mapCongestionCells,
    weatherCells,
    primaryRoute,
    secondaryRoute,
    selectedSegmentId,
    nodeMap,
    airports,
    onSelectSegment,
  ]);
  return (
    <div className="relative rounded-xl overflow-hidden border border-aerospace-800 bg-aerospace-950 shadow-2xl">
      {/* 3D WebGL Canvas */}
      <div ref={mapContainerRef} style={{ width: '100%', height }} />
      {/* Loading Overlay */}
      {mapLoading && (
        <div className="absolute inset-0 bg-aerospace-950/80 backdrop-blur-sm flex items-center justify-center z-30 font-mono text-xs text-cyan-400 gap-2">
          <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
          <span>INITIALIZING 3D AIRSPACE MAP...</span>
        </div>
      )}
      {!mapLoading && mapError && (
        <div className="absolute bottom-16 left-3 right-3 z-20 flex items-start gap-2 rounded-lg border border-amber-700/70 bg-aerospace-950/95 p-3 text-xs font-mono text-amber-200 shadow-xl">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{mapError}</span>
        </div>
      )}
      {/* Top Controls Overlay */}
      <div className="absolute top-3 left-3 z-10 flex flex-wrap items-center gap-2 bg-aerospace-900/90 backdrop-blur-md border border-aerospace-700/80 rounded-lg p-2 shadow-xl text-xs font-mono">
        <span className="text-cyan-400 font-bold flex items-center gap-1.5 px-2">
          <Compass className="w-3.5 h-3.5 animate-pulse" /> 3D VOLUMETRIC AIRSPACE
        </span>
        <button
          type="button"
          onClick={() => setShowCongestion3D(!showCongestion3D)}
          className={`px-2.5 py-1 rounded border transition flex items-center gap-1 ${
            showCongestion3D
              ? 'bg-rose-950/70 border-rose-600 text-rose-300'
              : 'bg-aerospace-950 border-aerospace-700 text-slate-400'
          }`}
        >
          <Layers className="w-3 h-3" />
          <span>3D Congestion ({mapCongestionCells.length})</span>
        </button>
        <button
          type="button"
          onClick={() => setShowWeather3D(!showWeather3D)}
          className={`px-2.5 py-1 rounded border transition flex items-center gap-1 ${
            showWeather3D
              ? 'bg-amber-950/70 border-amber-600 text-amber-300'
              : 'bg-aerospace-950 border-aerospace-700 text-slate-400'
          }`}
        >
          <CloudRain className="w-3 h-3" />
          <span>3D Weather</span>
        </button>
        <button
          type="button"
          onClick={() => setShowAirways(!showAirways)}
          className={`px-2.5 py-1 rounded border transition ${
            showAirways
              ? 'bg-cyan-950/70 border-cyan-600 text-cyan-300'
              : 'bg-aerospace-950 border-aerospace-700 text-slate-400'
          }`}
        >
          Airways
        </button>
        <button
          type="button"
          onClick={handleResetCamera}
          className="px-2.5 py-1 rounded bg-aerospace-800 hover:bg-aerospace-700 border border-aerospace-600 text-slate-200 transition flex items-center gap-1"
          title="Reset Orbit & Pitch"
        >
          <RotateCcw className="w-3 h-3" />
          <span>Reset Orbit</span>
        </button>
        <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800 text-cyan-300">
          OpenFreeMap Liberty · Zero API Key
        </span>
      </div>
      {/* Floating 3D Tooltip */}
      {hoverInfo && (
        <div
          className="absolute z-20 pointer-events-none bg-aerospace-900/95 backdrop-blur-md border border-cyan-700/80 rounded-lg p-3 shadow-2xl text-xs font-mono text-slate-200 max-w-xs"
          style={{ left: hoverInfo.x + 12, top: hoverInfo.y + 12 }}
        >
          {hoverInfo.layerType === 'segment' && (
            <div>
              <div className="text-cyan-300 font-bold border-b border-aerospace-800 pb-1 mb-1">
                ROUTE SEGMENT: {hoverInfo.object.from_node} → {hoverInfo.object.to_node}
              </div>
              <div>Distance: {hoverInfo.object.distance_km?.toFixed(1)} km</div>
              <div>Travel Time: {hoverInfo.object.estimated_time_min?.toFixed(1)} min</div>
              {hoverInfo.object.weather_severity !== null && (
                <div className="text-amber-400">
                  Weather Severity: {(hoverInfo.object.weather_severity * 100).toFixed(0)}%
                </div>
              )}
              {hoverInfo.object.congestion_score !== null && (
                <div className="text-rose-400">
                  Congestion: {(hoverInfo.object.congestion_score * 100).toFixed(0)}% ({hoverInfo.object.congestion_class})
                </div>
              )}
              <div className="text-[10px] text-slate-400 mt-1 italic">Click segment to inspect cost details</div>
            </div>
          )}
          {hoverInfo.layerType === 'congestion' && (
            <div>
              <div className="text-rose-300 font-bold border-b border-aerospace-800 pb-1 mb-1">
                CONGESTION SECTOR: {hoverInfo.object.cell_id}
              </div>
              <div>Predicted Density: <span className="font-bold">{hoverInfo.object.predicted_class}</span></div>
              <div>Congestion Score: {(hoverInfo.object.congestion_score * 100).toFixed(1)}%</div>
              <div>Observed Traffic: {hoverInfo.object.aircraft_count} flights</div>
            </div>
          )}
          {hoverInfo.layerType === 'weather' && (
            <div>
              <div className="text-amber-300 font-bold border-b border-aerospace-800 pb-1 mb-1">
                WEATHER HAZARD CELL: {hoverInfo.object.id}
              </div>
              <div>Precipitation: {hoverInfo.object.precip_val?.toFixed(1)} mm/h</div>
              <div>Severity Index: {(hoverInfo.object.computed_severity * 100).toFixed(0)}%</div>
              {hoverInfo.object.wind_gusts_10m_ms && (
                <div>Wind Gusts: {hoverInfo.object.wind_gusts_10m_ms?.toFixed(1)} m/s</div>
              )}
            </div>
          )}
          {hoverInfo.layerType === 'airport' && (
            <div>
              <div className="text-cyan-300 font-bold">{hoverInfo.object.code} ({hoverInfo.object.iata || 'WP'})</div>
              <div className="text-[11px] text-slate-300">{hoverInfo.object.name}</div>
            </div>
          )}
        </div>
      )}
      {/* Bottom Scale & Attribution Banner */}
      <div className="absolute bottom-2 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 p-2 rounded-lg bg-aerospace-900/85 backdrop-blur-md border border-aerospace-800 text-[10px] font-mono text-slate-400">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-slate-300">
            <span className="w-2.5 h-2.5 rounded bg-emerald-500 inline-block" /> Low Traffic
          </span>
          <span className="flex items-center gap-1 text-slate-300">
            <span className="w-2.5 h-2.5 rounded bg-amber-500 inline-block" /> Moderate Traffic
          </span>
          <span className="flex items-center gap-1 text-slate-300">
            <span className="w-2.5 h-2.5 rounded bg-rose-500 inline-block" /> Heavy Traffic
          </span>
          <span className="flex items-center gap-1 text-slate-300">
            <span className="w-2.5 h-2.5 rounded bg-purple-500 inline-block" /> Convective Rain
          </span>
        </div>
        <div className="flex items-center gap-3 text-slate-400">
          <span className="italic flex items-center gap-1">
            <Info className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            Elevation visualizes data density scale, not physical flight altitude.
          </span>
          <span className="border-l border-aerospace-700 pl-3">
            Map data © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer" className="hover:underline text-cyan-400">OpenStreetMap</a> · <a href="https://openfreemap.org" target="_blank" rel="noreferrer" className="hover:underline text-cyan-400">OpenFreeMap</a> · <a href="https://carto.com/attributions" target="_blank" rel="noreferrer" className="hover:underline text-cyan-400">CARTO</a>
          </span>
        </div>
      </div>
    </div>
  );
};
