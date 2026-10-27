import React, { useState, useEffect } from 'react';

import {

  UploadCloud,

  FileText,

  CheckCircle2,

  AlertTriangle,

  Play,




  Database,

  RefreshCw,



  CloudRain,

  Link2,


  Plane,

  BarChart3,

  Trash2,

} from 'lucide-react';

import { api } from '../services/api';

import type {

  DatasetSummaryItem,

  DatasetDetailResponse,

  DatasetUploadResponse,

  RunPipelineResponse,

  DatasetType,

  LinkWeatherResponse,

} from '../types';



interface DatasetStudioViewProps {

  activeDatasetId: string;

  onSelectActiveDataset: (id: string) => void;

  onNavigate?: (tab: string) => void;

}



const DATASET_TYPE_DETAILS: Record<

  DatasetType,

  { label: string; icon: any; desc: string; required: string[]; units: string }

> = {

  aircraft_observations: {

    label: 'Flight Surveillance',

    icon: Plane,

    desc: 'Individual aircraft positions, transponder ICAO24, kinematics and altitudes',

    required: ['source_aircraft_id', 'timestamp_utc', 'latitude', 'longitude'],

    units: 'Latitude/Longitude: deg, Altitude: m, Speed: m/s, Time: UTC',

  },

  weather_observations: {

    label: 'Weather Measurements',

    icon: CloudRain,

    desc: 'Precipitation, wind speeds, gusts, and cloud cover over coordinates or grid cells',

    required: ['latitude', 'longitude', 'valid_time_utc', 'precipitation_mm'],

    units: 'Precipitation: mm/h, Wind: m/s, Cloud Cover: %, Time: UTC',

  },

  congestion_cells: {

    label: 'Pre-aggregated Congestion',

    icon: BarChart3,

    desc: 'Sector cell traffic counts and measured/predicted density scores',

    required: ['cell_id', 'aircraft_count', 'congestion_score'],

    units: 'Cell ID: alphanumeric, Aircraft Count: integer >= 0, Score: 0.0 - 1.0',

  },

};



/** Header aliases used only for an early, friendly validation check. The backend
 * remains responsible for authoritative schema and value validation. */
const DATASET_HEADER_ALIASES: Record<DatasetType, Record<string, string[]>> = {
  aircraft_observations: {
    source_aircraft_id: [
      'source_aircraft_id', 'source_aircraft', 'aircraft_id', 'aircraftid',
      'icao24', 'icao_24', 'icao', 'transponder_id', 'hex', 'callsign', 'flight_id',
    ],
    timestamp_utc: [
      'timestamp_utc', 'timestamp', 'time_utc', 'utc_time', 'observed_at',
      'datetime', 'date_time', 'time', 'last_contact', 'lastcontact',
    ],
    latitude: ['latitude', 'lat', 'lat_deg', 'latitude_deg'],
    longitude: ['longitude', 'lon', 'lng', 'long', 'lon_deg', 'longitude_deg'],
  },
  weather_observations: {
    latitude: ['latitude', 'lat', 'lat_deg', 'latitude_deg'],
    longitude: ['longitude', 'lon', 'lng', 'long', 'lon_deg', 'longitude_deg'],
    valid_time_utc: [
      'valid_time_utc', 'valid_time', 'timestamp_utc', 'timestamp', 'time_utc',
      'utc_time', 'observed_at', 'datetime', 'date_time', 'time',
    ],
    precipitation_mm: [
      'precipitation_mm', 'precipitation', 'precip_mm', 'precipitation_mm_h',
      'rainfall_mm', 'rainfall', 'rain_rate', 'precipitation_rate', 'tp',
    ],
  },
  congestion_cells: {
    cell_id: ['cell_id', 'grid_cell_id', 'grid_id', 'sector_id', 'cell'],
    aircraft_count: ['aircraft_count', 'traffic_count', 'aircraft_total', 'count', 'n_aircraft'],
    congestion_score: ['congestion_score', 'score', 'density_score', 'congestion_probability'],
  },
};

const normalizeHeader = (header: string): string =>
  header
    .replace(/^\uFEFF/, '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

/** Split one CSV record while respecting commas inside quoted fields. */
const parseCsvRecord = (line: string): { values: string[]; unterminatedQuote: boolean } => {
  const values: string[] = [];
  let value = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      values.push(value.trim());
      value = '';
    } else {
      value += char;
    }
  }

  values.push(value.trim());
  return { values, unterminatedQuote: inQuotes };
};

/** Return a user-facing message when the selected file is not a usable dataset. */
const validateCsvBeforeUpload = async (
  file: File,
  datasetType: DatasetType,
): Promise<string | null> => {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    return 'Unsupported file format. Please upload a comma-separated .csv file.';
  }

  if (file.size === 0) {
    return 'The selected CSV file is empty. Choose a CSV file containing a header row and data records.';
  }

  let content: string;
  try {
    // The dataset is expected to be a CSV. Read it locally so common format
    // errors can be reported immediately, before an upload request is sent.
    // Only the first 512 KB are needed to validate the header and confirm that
    // at least one data row exists; avoid reading huge datasets into memory here.
    content = await file.slice(0, 512 * 1024).text();
  } catch {
    return 'The file could not be read. Export it as a UTF-8 CSV and try again.';
  }

  content = content.replace(/^\uFEFF/, '');
  const lines = content.split(/\r\n|\n|\r/);
  const headerLineIndex = lines.findIndex((line) => line.trim().length > 0);

  if (headerLineIndex < 0) {
    return 'The selected CSV contains no data. Please upload a non-empty dataset.';
  }

  const headerRecord = parseCsvRecord(lines[headerLineIndex]);
  if (headerRecord.unterminatedQuote) {
    return 'The CSV header contains an unclosed quote. Check the file formatting and export it again.';
  }

  const headers = headerRecord.values.map((header) => header.replace(/^\uFEFF/, '').trim());
  if (headers.length < 2) {
    return 'The file does not look like a comma-separated CSV. Please check the delimiter and header row.';
  }

  const emptyHeaderColumns = headers
    .map((header, index) => (header ? '' : String(index + 1)))
    .filter(Boolean);
  if (emptyHeaderColumns.length > 0) {
    return `The CSV has empty column names at position(s): ${emptyHeaderColumns.join(', ')}. Give every column a header.`;
  }

  const normalizedHeaders = headers.map(normalizeHeader);
  const seenHeaders = new Set<string>();
  const duplicateHeaders = new Set<string>();
  normalizedHeaders.forEach((header) => {
    if (seenHeaders.has(header)) duplicateHeaders.add(header);
    seenHeaders.add(header);
  });
  if (duplicateHeaders.size > 0) {
    return `The CSV contains duplicate column names: ${Array.from(duplicateHeaders).join(', ')}. Rename duplicate columns and upload it again.`;
  }

  const hasDataRow = lines
    .slice(headerLineIndex + 1)
    .some((line) => line.trim().length > 0);
  if (!hasDataRow) {
    return 'The CSV contains column headers but no data records. Add at least one data row and try again.';
  }

  const requiredAliases = DATASET_HEADER_ALIASES[datasetType];
  const missingFields = DATASET_TYPE_DETAILS[datasetType].required.filter((field) => {
    const aliases = requiredAliases[field] ?? [field];
    const accepted = new Set(aliases.map(normalizeHeader));
    return !normalizedHeaders.some((header) => accepted.has(header));
  });

  if (missingFields.length > 0) {
    const details = missingFields.map((field) => {
      const aliases = requiredAliases[field] ?? [field];
      return `${field} (accepted examples: ${aliases.slice(0, 4).join(', ')})`;
    });
    return [
      `This CSV does not match the selected schema: ${DATASET_TYPE_DETAILS[datasetType].label}.`,
      `Missing required field(s): ${details.join('; ')}.`,
      'Choose the correct dataset type or upload a CSV with the required columns. The file was not uploaded.',
    ].join(' ');
  }

  return null;
};

const getApiErrorMessage = (error: unknown): string => {
  const seen = new Set<unknown>();

  const extract = (value: unknown, depth = 0): string | null => {
    if (depth > 5 || value == null || seen.has(value)) return null;
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (!trimmed) return null;
      // Some fetch wrappers put the JSON response body directly in Error.message.
      if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
        try {
          const parsed: unknown = JSON.parse(trimmed);
          const parsedMessage = extract(parsed, depth + 1);
          if (parsedMessage) return parsedMessage;
        } catch {
          // Not JSON; display the original server message below.
        }
      }
      return trimmed;
    }
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (Array.isArray(value)) {
      const messages = value
        .map((item) => extract(item, depth + 1))
        .filter((item): item is string => Boolean(item));
      return messages.length ? messages.join('; ') : null;
    }
    if (typeof value !== 'object') return null;

    seen.add(value);
    const record = value as Record<string, unknown>;

    // FastAPI/Pydantic often returns detail as a string or a list containing loc/msg.
    const location = Array.isArray(record.loc)
      ? record.loc.map(String).join(' → ')
      : null;
    if (typeof record.msg === 'string' && record.msg.trim()) {
      return location ? `${location}: ${record.msg}` : record.msg;
    }

    const nestedCandidates = [
      record.detail,
      record.errors,
      record.issues,
      record.response,
      record.data,
      record.body,
      record.error,
      record.message,
    ];
    for (const candidate of nestedCandidates) {
      const message = extract(candidate, depth + 1);
      if (message) return message;
    }

    if (value instanceof Error && value.message.trim()) return value.message;
    return null;
  };

  return extract(error) || 'The server rejected the dataset upload. Check the selected schema and required columns, then try again.';
};


export const DatasetStudioView: React.FC<DatasetStudioViewProps> = ({

  activeDatasetId,

  onSelectActiveDataset,

}) => {

  const [datasets, setDatasets] = useState<DatasetSummaryItem[]>([]);

  const [selectedDataset, setSelectedDataset] = useState<DatasetDetailResponse | null>(null);

  const [uploadType, setUploadType] = useState<DatasetType>('aircraft_observations');

  const [uploading, setUploading] = useState(false);

  

  const [runningPipeline, setRunningPipeline] = useState(false);
  const [deletingDataset, setDeletingDataset] = useState(false);

  const [pipelineResult, setPipelineResult] = useState<RunPipelineResponse | null>(null);

  const [error, setError] = useState<string | null>(null);

  const [customMapping, setCustomMapping] = useState<Record<string, string>>({});



  // Weather linkage state

  const [selectedWeatherId, setSelectedWeatherId] = useState<string>('');

  const [linkingWeather, setLinkingWeather] = useState(false);

  const [linkResult, setLinkResult] = useState<LinkWeatherResponse | null>(null);



  const loadDatasets = async () => {

    try {

      const list = await api.getDatasets();

      setDatasets(list);

      const active = list.find((d) => d.id === activeDatasetId) || list[0];

      if (active) {

        const details = await api.getDataset(active.id);

        setSelectedDataset(details);

        setCustomMapping(details.schema_mapping || {});

      }

    } catch (err: any) {

      setError(getApiErrorMessage(err));

    }

  };



  useEffect(() => {

    void loadDatasets();

  }, [activeDatasetId]);



  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.currentTarget;
    const file = input.files?.[0];
    if (!file) return;

    setError(null);
    setPipelineResult(null);
    setLinkResult(null);

    // Validate basic file structure and schema before sending it to the backend.
    const formatError = await validateCsvBeforeUpload(file, uploadType);
    if (formatError) {
      setError(formatError);
      input.value = '';
      return;
    }

    setUploading(true);

    try {
      const rawResponse: unknown = await api.uploadDataset(file, uploadType);
      const response =
        rawResponse && typeof rawResponse === 'object'
          ? (rawResponse as Record<string, unknown>)
          : null;
      const datasetId = typeof response?.dataset_id === 'string'
        ? response.dataset_id.trim()
        : '';

      // Some API wrappers resolve non-2xx responses instead of throwing. Treat
      // an error payload or a missing dataset ID as a failed upload explicitly.
      if (!datasetId || response?.detail || response?.error || response?.errors) {
        throw new Error(getApiErrorMessage(rawResponse));
      }

      const result = rawResponse as DatasetUploadResponse;
      setCustomMapping(
        result.suggested_mapping && typeof result.suggested_mapping === 'object'
          ? result.suggested_mapping
          : {},
      );
      onSelectActiveDataset(datasetId);

      // Upload succeeded; refresh details separately so a follow-up details
      // request error is not misleadingly reported as a file-format rejection.
      try {
        const details = await api.getDataset(datasetId);
        setSelectedDataset(details);
        setCustomMapping(details.schema_mapping || result.suggested_mapping || {});
        await loadDatasets();
      } catch (detailsError: unknown) {
        setError(
          `The CSV was uploaded, but its dataset details could not be refreshed: ${getApiErrorMessage(detailsError)}`,
        );
      }
    } catch (uploadError: unknown) {
        setError(`Dataset upload failed: ${getApiErrorMessage(uploadError)}`);
    } finally {
      setUploading(false);
      // Permit selecting the same file again after a validation or upload failure.
      input.value = '';
    }
  };


  const handleRunPipeline = async () => {

    if (!selectedDataset) return;

    setRunningPipeline(true);

    setError(null);

    try {

      const stages =

        selectedDataset.dataset_type === 'weather_observations'

          ? ['validation', 'weather_indexing']

          : ['quality', 'segmentation', 'features', 'inference'];



      const res: RunPipelineResponse = await api.runDatasetPipeline(selectedDataset.id, {

        stages,

        custom_mapping: customMapping,

      });

      setPipelineResult(res);

      const updated = await api.getDataset(selectedDataset.id);

      setSelectedDataset(updated);

      await loadDatasets();

    } catch (err: any) {

      setError(`Pipeline execution failed: ${getApiErrorMessage(err)}`);

    } finally {

      setRunningPipeline(false);

    }

  };

  const handleDeleteDataset = async () => {
    if (!selectedDataset || selectedDataset.id === 'synthetic_demo_v1') return;
    if (!window.confirm(`Are you sure you want to delete dataset "${selectedDataset.filename}" (${selectedDataset.id})?`)) {
      return;
    }
    setDeletingDataset(true);
    setError(null);
    try {
      await api.deleteDataset(selectedDataset.id);
      if (activeDatasetId === selectedDataset.id) {
        onSelectActiveDataset('synthetic_demo_v1');
      }
      await loadDatasets();
    } catch (err: any) {
      setError(`Delete failed: ${getApiErrorMessage(err)}`);
    } finally {
      setDeletingDataset(false);
    }
  };



  const handleLinkWeather = async () => {

    if (!selectedDataset || !selectedWeatherId) return;

    setLinkingWeather(true);

    setError(null);

    setLinkResult(null);

    try {

      const res: LinkWeatherResponse = await api.linkWeather(selectedDataset.id, selectedWeatherId);

      setLinkResult(res);

      const updated = await api.getDataset(selectedDataset.id);

      setSelectedDataset(updated);

      await loadDatasets();

    } catch (err: any) {

      setError(`Weather linking failed: ${getApiErrorMessage(err)}`);

    } finally {

      setLinkingWeather(false);

    }

  };



  const handleSelectDatasetItem = async (id: string) => {

    try {

      const details = await api.getDataset(id);

      setSelectedDataset(details);

      setCustomMapping(details.schema_mapping || {});

      onSelectActiveDataset(id);

  
      setPipelineResult(null);

      setLinkResult(null);

    } catch (err: any) {

      setError(`Failed to switch dataset: ${getApiErrorMessage(err)}`);

    }

  };



  const availableWeatherDatasets = datasets.filter((d) => d.dataset_type === 'weather_observations');



  const currentTypeMeta =

    DATASET_TYPE_DETAILS[selectedDataset?.dataset_type || uploadType] ||

    DATASET_TYPE_DETAILS['aircraft_observations'];



  return (

    <div className="space-y-6 font-mono">

      {/* Top Header Card */}

      <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg flex flex-wrap items-center justify-between gap-4">

        <div>

          <div className="flex items-center gap-2">

            <Database className="w-5 h-5 text-cyan-400" />

            <h2 className="text-lg font-bold text-white uppercase tracking-wider">

              MULTI-SCHEMA DATASET STUDIO & LINKAGE

            </h2>

          </div>

          <p className="text-xs text-slate-400 mt-1">

            Ingest and normalize aircraft tracks, geographic weather grids, and congestion matrices with isolated data provenance.

          </p>

        </div>



        {/* Dataset Selector Dropdown */}

        <div className="flex items-center gap-3">

          <label className="text-xs text-slate-400">ACTIVE DATASET:</label>

          <select

            value={selectedDataset?.id || activeDatasetId}

            onChange={(e) => handleSelectDatasetItem(e.target.value)}

            className="bg-aerospace-950 border border-aerospace-700 rounded-lg px-3 py-1.5 text-xs text-cyan-300 font-bold focus:outline-none focus:border-cyan-500"

          >

            {datasets.map((d) => (

              <option key={d.id} value={d.id}>

                [{d.dataset_type?.slice(0, 7) || 'airc'}] {d.filename} ({d.row_count.toLocaleString()} rows){' '}

                {d.id === activeDatasetId ? '★ ACTIVE' : ''}

              </option>

            ))}

          </select>

        </div>

      </div>



      {error && (
        <div
          className="p-4 rounded-lg bg-red-950/70 border border-red-700 text-red-100 text-sm flex items-start gap-3"
          role="alert"
          aria-live="assertive"
        >
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div className="min-w-0">
            <p className="font-bold">Dataset validation / operation failed</p>
            <p className="mt-1 whitespace-pre-wrap break-words">{error}</p>
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="ml-auto text-xs underline underline-offset-2 text-red-200 hover:text-white"
            aria-label="Dismiss error message"
          >
            Dismiss
          </button>
        </div>
      )}



      {/* Upload Zone & Dataset Profiling Grid */}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Upload Box with Schema Selector */}

        <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4">

          <h3 className="text-xs font-bold text-white uppercase flex items-center gap-2 text-cyan-400">

            <UploadCloud className="w-4 h-4" />

            UPLOAD CSV BY SCHEMA TYPE

          </h3>



          {/* Dataset Type Selector */}

          <div className="space-y-1.5">

            <label className="text-[11px] text-slate-400 uppercase font-semibold">1. Choose Dataset Schema:</label>

            <div className="grid grid-cols-1 gap-2">

              {(Object.keys(DATASET_TYPE_DETAILS) as DatasetType[]).map((type) => {

                const item = DATASET_TYPE_DETAILS[type];

                const Icon = item.icon;

                const isSelected = uploadType === type;

                return (

                  <button

                    key={type}

                    type="button"

                    onClick={() => setUploadType(type)}

                    className={`p-2.5 rounded-lg border text-left transition flex items-start gap-2.5 ${

                      isSelected

                        ? 'bg-cyan-950/80 border-cyan-500 text-cyan-200'

                        : 'bg-aerospace-950/60 border-aerospace-800 text-slate-400 hover:border-aerospace-700'

                    }`}

                  >

                    <Icon className="w-4 h-4 mt-0.5 shrink-0 text-cyan-400" />

                    <div>

                      <div className="text-xs font-bold text-white">{item.label}</div>

                      <div className="text-[10px] text-slate-400 leading-snug">{item.desc}</div>

                    </div>

                  </button>

                );

              })}

            </div>

          </div>



          {/* Upload Dropzone */}

          <div className="space-y-1.5">

            <label className="text-[11px] text-slate-400 uppercase font-semibold">2. Select CSV File:</label>

            <label className="border-2 border-dashed border-aerospace-700 hover:border-cyan-500/80 rounded-xl p-5 flex flex-col items-center justify-center cursor-pointer bg-aerospace-950/50 hover:bg-aerospace-950 transition group">

              <UploadCloud className="w-7 h-7 text-slate-500 group-hover:text-cyan-400 transition" />

              <span className="text-xs font-semibold text-slate-300 mt-2">Select {DATASET_TYPE_DETAILS[uploadType].label} CSV</span>

              <span className="text-[10px] text-slate-500 mt-1">{DATASET_TYPE_DETAILS[uploadType].units}</span>

              <input

                type="file"

                accept=".csv,text/csv"

                onChange={handleFileUpload}

                disabled={uploading}

                className="hidden"

              />

            </label>

          </div>



          {uploading && (

            <div className="text-xs text-cyan-400 flex items-center gap-2">

              <RefreshCw className="w-3.5 h-3.5 animate-spin" />

              <span>Validating schema, parsing timestamps, and storing immutable copy...</span>

            </div>

          )}



          {/* Dataset Provenance Notice */}

          <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 text-[11px] text-slate-400 space-y-1">

            <div className="font-bold text-slate-300 flex items-center gap-1">

              <span className="w-2 h-2 rounded-full bg-emerald-400" />

              PROVENANCE ISOLATION

            </div>

            <p>

              Different schema rules apply for traffic vs weather. Uploaded weather datasets can be linked to traffic datasets to replace synthetic weather in routing.

            </p>

          </div>

        </div>



        {/* Selected Dataset Details & Schema Mapping */}

        <div className="lg:col-span-2 bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-4">

          <div className="flex items-center justify-between border-b border-aerospace-800 pb-3">

            <div>

              <div className="flex items-center gap-2">

                <h3 className="text-xs font-bold text-white uppercase flex items-center gap-2 text-cyan-400">

                  <FileText className="w-4 h-4" />

                  DATASET: {selectedDataset?.filename}

                </h3>

                <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 text-[10px] font-bold">

                  {selectedDataset?.dataset_type?.toUpperCase() || 'AIRCRAFT_OBSERVATIONS'}

                </span>

              </div>

              <p className="text-[11px] text-slate-400 mt-0.5">

                ID: {selectedDataset?.id} · Provenance: {selectedDataset?.provenance} · Uploaded:{' '}

                {selectedDataset?.uploaded_at_utc ? new Date(selectedDataset.uploaded_at_utc).toLocaleString() : 'N/A'}

              </p>

            </div>



            <div className="flex items-center gap-2">

              <button

                onClick={handleRunPipeline}

                disabled={runningPipeline || !selectedDataset || selectedDataset.id === 'synthetic_demo_v1'}

                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-aerospace-950 font-bold text-xs transition shadow-md shadow-cyan-500/20 disabled:opacity-50"

              >

                {runningPipeline ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current" />}

                {runningPipeline ? 'RUNNING PIPELINE...' : 'EXECUTE PIPELINE'}

              </button>

              {selectedDataset && selectedDataset.id !== 'synthetic_demo_v1' && (
                <button
                  type="button"
                  onClick={handleDeleteDataset}
                  disabled={deletingDataset || runningPipeline}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-950/70 hover:bg-red-900 border border-red-800 text-red-300 font-bold text-xs transition disabled:opacity-50"
                  title="Delete this uploaded dataset and its processed artifacts"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  {deletingDataset ? 'DELETING...' : 'DELETE'}
                </button>
              )}

            </div>

          </div>



          {/* Key Metrics Strip */}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">

            <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800">

              <div className="text-[10px] text-slate-400 uppercase">Records</div>

              <div className="text-base font-bold text-white mt-0.5">{selectedDataset?.row_count.toLocaleString()}</div>

            </div>

            <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800">

              <div className="text-[10px] text-slate-400 uppercase">Columns</div>

              <div className="text-base font-bold text-white mt-0.5">{selectedDataset?.col_count}</div>

            </div>

            <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800">

              <div className="text-[10px] text-slate-400 uppercase">File Size</div>

              <div className="text-base font-bold text-white mt-0.5">

                {selectedDataset ? (selectedDataset.file_size_bytes / (1024 * 1024)).toFixed(2) : '0'} MB

              </div>

            </div>

            <div className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800">

              <div className="text-[10px] text-slate-400 uppercase">Pipeline Status</div>

              <div className="text-base font-bold text-emerald-400 uppercase mt-0.5">{selectedDataset?.status}</div>

            </div>

          </div>



          {/* Schema Mapping Table */}

          <div className="space-y-2">

            <div className="text-xs font-bold text-slate-300 uppercase flex items-center justify-between">

              <span>Schema Recognition ({currentTypeMeta.label}):</span>

              <span className="text-[11px] text-slate-500">Units: {currentTypeMeta.units}</span>

            </div>



            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">

              {currentTypeMeta.required.map((canonical) => {

                const mapped = customMapping[canonical];

                return (

                  <div

                    key={canonical}

                    className={`p-2 rounded-lg border flex flex-col justify-between ${

                      mapped ? 'bg-aerospace-950 border-cyan-800/60' : 'bg-red-950/30 border-red-800/60'

                    }`}

                  >

                    <div className="text-[10px] text-slate-400 truncate" title={canonical}>

                      {canonical}

                    </div>

                    <div className="font-bold text-white truncate mt-1">

                      {mapped ? (

                        <span className="text-cyan-300 flex items-center gap-1">

                          <CheckCircle2 className="w-3 h-3 text-cyan-400 inline" /> {mapped}

                        </span>

                      ) : (

                        <span className="text-red-400">UNMAPPED</span>

                      )}

                    </div>

                  </div>

                );

              })}

            </div>

          </div>



          {/* Traffic and Weather Linking Section */}

          {selectedDataset?.dataset_type === 'aircraft_observations' && (

            <div className="p-4 rounded-xl bg-aerospace-950 border border-aerospace-800 space-y-3">

              <div className="flex items-center justify-between">

                <span className="text-xs font-bold text-white flex items-center gap-1.5 text-cyan-400 uppercase">

                  <Link2 className="w-4 h-4" />

                  LINK TRAFFIC OBSERVATIONS WITH MEASURED WEATHER

                </span>

                {selectedDataset.linked_weather_dataset_id && (

                  <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-300">

                    Linked to: {selectedDataset.linked_weather_dataset_id}

                  </span>

                )}

              </div>



              <p className="text-[11px] text-slate-400">

                Spatially and temporally associates flight tracks with an uploaded weather grid. Unmatched observations are marked with explicit missing-data uncertainty penalties.

              </p>



              <div className="flex flex-wrap items-center gap-3">

                <select

                  value={selectedWeatherId}

                  onChange={(e) => setSelectedWeatherId(e.target.value)}

                  className="bg-aerospace-900 border border-aerospace-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"

                >

                  <option value="">-- Select Measured Weather Grid --</option>

                  {availableWeatherDatasets.map((w) => (

                    <option key={w.id} value={w.id}>

                      {w.filename} ({w.row_count} records)

                    </option>

                  ))}

                </select>



                <button

                  type="button"

                  onClick={handleLinkWeather}

                  disabled={linkingWeather || !selectedWeatherId}

                  className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs transition disabled:opacity-50 flex items-center gap-1"

                >

                  {linkingWeather ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Link2 className="w-3.5 h-3.5" />}

                  <span>LINK DATASETS</span>

                </button>

              </div>



              {linkResult && (

                <div className="p-3 rounded-lg bg-cyan-950/60 border border-cyan-800 text-xs space-y-1 text-cyan-200">

                  <div className="font-bold flex items-center gap-1">

                    <CheckCircle2 className="w-4 h-4 text-cyan-400" />

                    LINKAGE COMPLETE: {linkResult.coverage_percentage}% WEATHER COVERAGE

                  </div>

                  <div>

                    Matched {linkResult.matched_rows.toLocaleString()} of {linkResult.total_traffic_rows.toLocaleString()} flight observations.

                  </div>

                  {linkResult.warnings.map((w, i) => (

                    <div key={i} className="text-amber-300 text-[11px]">

                      ⚠ {w}

                    </div>

                  ))}

                </div>

              )}

            </div>

          )}

        </div>

      </div>



      {/* Pipeline Execution Output Card */}

      {pipelineResult && (

        <div className="bg-aerospace-900 border border-aerospace-800 rounded-xl p-5 shadow-lg space-y-3">

          <div className="flex items-center justify-between border-b border-aerospace-800 pb-2">

            <h3 className="text-xs font-bold text-white uppercase flex items-center gap-2 text-cyan-400">

              <CheckCircle2 className="w-4 h-4" />

              PIPELINE EXECUTION REPORT (Total: {pipelineResult.total_elapsed_seconds}s)

            </h3>

            <span className="text-xs text-emerald-400 font-bold uppercase">{pipelineResult.status}</span>

          </div>



          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">

            {pipelineResult.stages.map((stage) => (

              <div key={stage.stage} className="p-3 rounded-lg bg-aerospace-950 border border-aerospace-800 space-y-1">

                <div className="flex items-center justify-between text-xs">

                  <span className="font-bold text-cyan-300 uppercase">{stage.stage}</span>

                  <span className="text-[10px] text-slate-500">{stage.elapsed_seconds}s</span>

                </div>

                <div className="text-[11px] text-slate-400">

                  {Object.entries(stage.output_summary).map(([k, v]) => (

                    <div key={k} className="truncate">

                      {k}: <strong className="text-white">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</strong>

                    </div>

                  ))}

                </div>

              </div>

            ))}

          </div>

        </div>

      )}

    </div>

  );

};
