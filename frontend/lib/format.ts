/** Formatting utilities for numbers, percentages, and scientific notation. */

export function formatPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatSci(value: number | null): string {
  if (value === null || value === undefined) return "—";
  if (value === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(value)));
  const coeff = value / Math.pow(10, exp);
  if (Math.abs(coeff - 1) < 0.01) return `10^${exp}`;
  return `${coeff.toFixed(1)}×10^${exp}`;
}

export function formatUSD(value: number | null): string {
  if (value === null || value === undefined) return "—";
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
  return `$${value.toFixed(0)}`;
}

export function formatDelta(value: number, asPct = true): string {
  const sign = value >= 0 ? "+" : "";
  return asPct
    ? `${sign}${(value * 100).toFixed(1)}%`
    : `${sign}${value.toFixed(3)}`;
}

export function formatDurationMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function severityColor(severity: number): string {
  if (severity <= 2.0) return "#22c55e";
  if (severity <= 3.5) return "#f59e0b";
  return "#ef4444";
}

export function epistemicColor(tier: string): string {
  switch (tier) {
    case "GROUNDED":    return "#22c55e";
    case "DIRECTIONAL": return "#f59e0b";
    default:            return "#f87171";
  }
}

export function epistemicBg(tier: string): string {
  switch (tier) {
    case "GROUNDED":    return "#14532d";
    case "DIRECTIONAL": return "#78350f";
    default:            return "#450a0a";
  }
}
