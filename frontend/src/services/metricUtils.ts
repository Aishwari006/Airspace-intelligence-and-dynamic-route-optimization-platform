type JsonRecord = Record<string, unknown>;

export function asRecord(value: unknown): JsonRecord | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

const normalizeKey = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, '');

function searchForKey(source: unknown, target: string, depth: number): unknown {
  if (depth < 0 || source === null || typeof source !== 'object') return undefined;
  if (Array.isArray(source)) {
    for (const item of source) {
      const found = searchForKey(item, target, depth - 1);
      if (found !== undefined) return found;
    }
    return undefined;
  }

  const record = source as JsonRecord;
  for (const [key, value] of Object.entries(record)) {
    if (normalizeKey(key) === target && value !== null && value !== undefined) return value;
  }
  for (const value of Object.values(record)) {
    const found = searchForKey(value, target, depth - 1);
    if (found !== undefined) return found;
  }
  return undefined;
}

export function findMetric(source: unknown, aliases: string[]): number | null {
  for (const alias of aliases) {
    const value = searchForKey(source, normalizeKey(alias), 5);
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}

export function formatPercent(value: number | null): string {
  if (value === null) return '—';
  const percent = value <= 1 ? value * 100 : value;
  return `${percent.toFixed(1)}%`;
}

export function formatNumber(value: number | null, decimals = 3): string {
  return value === null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: decimals });
}

export function findNamedSection(source: unknown, hints: string[], depth = 5): JsonRecord | null {
  if (depth < 0 || source === null || typeof source !== 'object' || Array.isArray(source)) return null;
  const record = source as JsonRecord;
  for (const [key, value] of Object.entries(record)) {
    const normalized = normalizeKey(key);
    if (hints.some((hint) => normalized.includes(normalizeKey(hint)))) {
      const section = asRecord(value);
      if (section) return section;
    }
  }
  for (const value of Object.values(record)) {
    const found = findNamedSection(value, hints, depth - 1);
    if (found) return found;
  }
  return null;
}

export interface CandidateMetricRow {
  name: string;
  accuracy: number | null;
  balancedAccuracy: number | null;
  macroF1: number | null;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
}

export function findCandidateMetricRows(source: unknown, task: 'classification' | 'regression'): CandidateMetricRow[] {
  const visited = new Set<unknown>();
  const matchingArrays: CandidateMetricRow[][] = [];

  const inspect = (value: unknown, depth: number) => {
    if (depth < 0 || value === null || typeof value !== 'object' || visited.has(value)) return;
    visited.add(value);

    // Case 1: Array of candidate objects
    if (Array.isArray(value)) {
      const rows = value.flatMap((item): CandidateMetricRow[] => {
        const row = asRecord(item);
        if (!row) return [];
        const rawName = row.model_name ?? row.model ?? row.name ?? row.candidate ?? row.estimator ?? row.algorithm;
        if (typeof rawName !== 'string') return [];
        const parsed: CandidateMetricRow = {
          name: rawName,
          accuracy: findMetric(row, ['test_accuracy', 'accuracy', 'accuracy_score']),
          balancedAccuracy: findMetric(row, ['balanced_accuracy', 'balanced_acc']),
          macroF1: findMetric(row, ['macro_f1', 'f1_macro', 'macro_f1_score']),
          mae: findMetric(row, ['test_mae', 'mae_min', 'mae', 'mean_absolute_error']),
          rmse: findMetric(row, ['test_rmse', 'rmse_min', 'rmse', 'root_mean_squared_error']),
          r2: findMetric(row, ['test_r2', 'r2', 'r2_score', 'r_squared']),
        };
        const hasTaskMetrics = task === 'classification'
          ? parsed.accuracy !== null || parsed.balancedAccuracy !== null || parsed.macroF1 !== null
          : parsed.mae !== null || parsed.rmse !== null || parsed.r2 !== null;
        return hasTaskMetrics ? [parsed] : [];
      });
      if (rows.length) matchingArrays.push(rows);
      value.forEach((item) => inspect(item, depth - 1));
      return;
    }

    // Case 2: Object where keys are candidate names (e.g. { dummy_most_frequent: { test: {...} }, random_forest: { test: {...} } })
    const record = value as JsonRecord;
    const entries = Object.entries(record);
    const candidateEntries = entries.filter(([, val]) => {
      if (!val || typeof val !== 'object' || Array.isArray(val)) return false;
      const vRec = val as JsonRecord;
      return (
        'validation' in vRec ||
        'test' in vRec ||
        'accuracy' in vRec ||
        'macro_f1' in vRec ||
        'mae_min' in vRec ||
        'rmse_min' in vRec ||
        'r2' in vRec
      );
    });

    if (candidateEntries.length >= 2) {
      const rows = candidateEntries.flatMap(([rawName, candidateObj]): CandidateMetricRow[] => {
        const cRec = candidateObj as JsonRecord;
        const targetSplit = asRecord(cRec.test) || asRecord(cRec.validation) || cRec;
        const parsed: CandidateMetricRow = {
          name: rawName,
          accuracy: findMetric(targetSplit, ['test_accuracy', 'accuracy', 'accuracy_score']),
          balancedAccuracy: findMetric(targetSplit, ['balanced_accuracy', 'balanced_acc']),
          macroF1: findMetric(targetSplit, ['macro_f1', 'f1_macro', 'macro_f1_score']),
          mae: findMetric(targetSplit, ['test_mae', 'mae_min', 'mae', 'mean_absolute_error']),
          rmse: findMetric(targetSplit, ['test_rmse', 'rmse_min', 'rmse', 'root_mean_squared_error']),
          r2: findMetric(targetSplit, ['test_r2', 'r2', 'r2_score', 'r_squared']),
        };
        const hasTaskMetrics = task === 'classification'
          ? parsed.accuracy !== null || parsed.balancedAccuracy !== null || parsed.macroF1 !== null
          : parsed.mae !== null || parsed.rmse !== null || parsed.r2 !== null;
        return hasTaskMetrics ? [parsed] : [];
      });
      if (rows.length) matchingArrays.push(rows);
    }

    Object.values(record).forEach((item) => inspect(item, depth - 1));
  };

  inspect(source, 5);
  return matchingArrays.sort((a, b) => b.length - a.length)[0] || [];
}
