/**
 * Utility functions for Estimated Time of Arrival (ETA) calculation and display
 * with explicit bracketed delay (+time) or early arrival (-time) tags.
 */

export interface EtaInfo {
  clockTime: string;
  durationMinutes: number;
  deltaMinutes: number;
  badgeText: string;
  badgeColor: string;
  deltaType: 'delay' | 'early' | 'on-schedule';
}

/**
 * Formats arrival time and calculates delta against baseline or scheduled traversal.
 */
export function calculateEtaInfo(
  traversalMinutes: number,
  baselineMinutes?: number | null,
  baseDepartureUtc?: string | Date | null
): EtaInfo {
  const baseEpoch = baseDepartureUtc
    ? new Date(baseDepartureUtc).getTime()
    : new Date('2026-01-22T09:00:00Z').getTime();

  const arrivalEpoch = baseEpoch + traversalMinutes * 60 * 1000;
  const arrivalDate = new Date(arrivalEpoch);

  const clockTime = arrivalDate.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'UTC',
  }) + ' UTC';

  const baseMin =
    baselineMinutes !== undefined && baselineMinutes !== null && baselineMinutes > 0
      ? baselineMinutes
      : traversalMinutes;

  const delta = Math.round((traversalMinutes - baseMin) * 10) / 10;

  let deltaType: 'delay' | 'early' | 'on-schedule' = 'on-schedule';
  let badgeText = '(On Schedule)';
  let badgeColor = 'text-cyan-400 border-cyan-800 bg-cyan-950/40';

  if (delta > 0.1) {
    deltaType = 'delay';
    badgeText = `(+${delta.toFixed(1)} min delay)`;
    badgeColor = 'text-rose-400 border-rose-800 bg-rose-950/50';
  } else if (delta < -0.1) {
    deltaType = 'early';
    badgeText = `(-${Math.abs(delta).toFixed(1)} min early)`;
    badgeColor = 'text-emerald-400 border-emerald-800 bg-emerald-950/50';
  }

  return {
    clockTime,
    durationMinutes: traversalMinutes,
    deltaMinutes: delta,
    badgeText,
    badgeColor,
    deltaType,
  };
}

/**
 * Formats ETA and delay/early tag for a scheduled FlightSummary or FlightDetail.
 */
export function formatFlightEta(flight: {
  scheduled_departure_utc?: string;
  estimated_departure_utc?: string;
  scheduled_arrival_utc?: string;
  estimated_arrival_utc?: string;
  scheduled_arrival_display?: string;
  estimated_arrival_display?: string;
  arrival_delay_minutes?: number | null;
  predicted_delay_minutes?: number | null;
}): { clockTime: string; badgeText: string; badgeColor: string; deltaType: 'delay' | 'early' | 'on-schedule' } {
  let clockTime = flight.estimated_arrival_display || '';
  if (!clockTime && flight.estimated_arrival_utc) {
    clockTime =
      new Date(flight.estimated_arrival_utc).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'UTC',
      }) + ' UTC';
  }
  if (!clockTime && flight.scheduled_departure_utc) {
    const arr = new Date(
      new Date(flight.scheduled_departure_utc).getTime() +
        ((flight.predicted_delay_minutes || 0) + 90) * 60000
    );
    clockTime =
      arr.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }) + ' UTC';
  }

  const delayMin =
    flight.arrival_delay_minutes !== undefined && flight.arrival_delay_minutes !== null
      ? flight.arrival_delay_minutes
      : flight.predicted_delay_minutes || 0;

  const rounded = Math.round(delayMin * 10) / 10;

  if (rounded > 0.5) {
    return {
      clockTime,
      badgeText: `(+${Math.round(rounded)} min delay)`,
      badgeColor: 'text-rose-400 border-rose-800 bg-rose-950/60',
      deltaType: 'delay',
    };
  } else if (rounded < -0.5) {
    return {
      clockTime,
      badgeText: `(-${Math.round(Math.abs(rounded))} min early)`,
      badgeColor: 'text-emerald-400 border-emerald-800 bg-emerald-950/60',
      deltaType: 'early',
    };
  } else {
    return {
      clockTime,
      badgeText: '(On Schedule)',
      badgeColor: 'text-cyan-400 border-cyan-800 bg-cyan-950/40',
      deltaType: 'on-schedule',
    };
  }
}

