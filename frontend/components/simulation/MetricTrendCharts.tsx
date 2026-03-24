"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatPct } from "@/lib/format";
import type { RoundSummary, ConfidenceBand } from "@/lib/types";

// ── Config ────────────────────────────────────────────────────────────────────

interface MetricConfig {
  key: string;
  label: string;
  sublabel: string;
  color: string;
  gradId: string;
}

const METRICS: MetricConfig[] = [
  {
    key:      "compliance_rate",
    label:    "Compliance Rate",
    sublabel: "Policy adherence over time",
    color:    "#8B1A1A",
    gradId:   "mcg-cr",
  },
  {
    key:      "relocation_rate",
    label:    "Relocation Rate",
    sublabel: "Jurisdiction flight over time",
    color:    "#C2410C",
    gradId:   "mcg-rr",
  },
  {
    key:      "ai_investment_index",
    label:    "AI Investment",
    sublabel: "Capital sentiment over time",
    color:    "#1D6344",
    gradId:   "mcg-inv",
  },
];

// ── Data builder ──────────────────────────────────────────────────────────────

interface MiniPoint {
  round: number;
  value: number;
  p10?: number;
  p90?: number;
}

function buildMiniData(
  roundSummaries: RoundSummary[],
  metricKey: string,
  bands?: ConfidenceBand[],
): MiniPoint[] {
  const bandMap = new Map(bands?.map((b) => [b.round, b]));
  return roundSummaries.map((r) => {
    const raw = (r as unknown as Record<string, number>)[metricKey] ?? 0;
    // investment arrives as 0-1 (already normalised in backend merge)
    const value = raw;
    const band = bandMap.get(r.round);
    return {
      round: r.round,
      value,
      ...(band ? { p10: band.p10, p90: band.p90 } : {}),
    };
  });
}

// ── Single mini chart ─────────────────────────────────────────────────────────

function MiniChart({
  metric,
  roundSummaries,
  bands,
}: {
  metric: MetricConfig;
  roundSummaries: RoundSummary[];
  bands?: ConfidenceBand[];
}) {
  const data     = buildMiniData(roundSummaries, metric.key, bands);
  const final    = data.at(-1)?.value ?? 0;
  const hasBands = data.some((d) => d.p10 !== undefined);

  return (
    <div className="card-warm flex flex-col gap-3 p-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="section-kicker text-[10px]">{metric.sublabel}</p>
          <h4 className="mt-0.5 text-sm font-semibold" style={{ color: "var(--ink-800)" }}>
            {metric.label}
          </h4>
          {hasBands && (
            <p className="mt-1 text-[10px]" style={{ color: "var(--ink-300)" }}>
              <span
                style={{
                  display: "inline-block",
                  width: 8,
                  height: 8,
                  background: `${metric.color}30`,
                  borderRadius: 2,
                  marginRight: 4,
                  verticalAlign: "middle",
                }}
              />
              9-run confidence band
            </p>
          )}
        </div>
        {/* Final value callout */}
        <div className="text-right">
          <p className="kicker text-[9px]">Final</p>
          <p
            className="metric-num text-xl font-bold"
            style={{ color: metric.color }}
          >
            {formatPct(final)}
          </p>
        </div>
      </div>

      {/* Chart */}
      <div
        className="overflow-hidden rounded-xl border"
        style={{ background: "var(--cream-100)", borderColor: "var(--border-warm)" }}
      >
        <ResponsiveContainer width="100%" height={148}>
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
            <defs>
              <linearGradient id={metric.gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={metric.color} stopOpacity={0.18} />
                <stop offset="95%" stopColor={metric.color} stopOpacity={0.02} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-warm)" />

            <XAxis
              dataKey="round"
              tick={{ fill: "var(--ink-400)", fontSize: 9 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              tick={{ fill: "var(--ink-400)", fontSize: 9 }}
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
              width={30}
            />

            <Tooltip
              contentStyle={{
                background:   "var(--cream-50)",
                border:       "1px solid var(--border-warm)",
                borderRadius: 12,
                boxShadow:    "var(--shadow-raised)",
                fontSize:     11,
              }}
              labelStyle={{ color: "var(--ink-700)", fontSize: 10 }}
              formatter={(v: number, name: string) => [
                formatPct(v),
                name === "value" ? metric.label : name === "p90" ? "p90 band" : "p10 band",
              ]}
            />

            {/* Confidence band — fill from p90 down, overpaint below p10 */}
            {hasBands && (
              <>
                <Area
                  type="monotone"
                  dataKey="p90"
                  stroke="none"
                  fill={`url(#${metric.gradId})`}
                  isAnimationActive={false}
                  connectNulls
                  legendType="none"
                />
                <Area
                  type="monotone"
                  dataKey="p10"
                  stroke="none"
                  fill="var(--cream-100)"
                  isAnimationActive={false}
                  connectNulls
                  legendType="none"
                />
              </>
            )}

            {/* Main simulation line */}
            <Line
              type="monotone"
              dataKey="value"
              stroke={metric.color}
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
              activeDot={{ r: 4, fill: metric.color, stroke: "var(--cream-50)", strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ── Public component ──────────────────────────────────────────────────────────

interface Props {
  roundSummaries: RoundSummary[];
  /** Per-metric confidence bands from the evidence pack job (keyed by metric name). */
  bands?: Record<string, ConfidenceBand[]>;
}

export function MetricTrendCharts({ roundSummaries, bands }: Props) {
  const hasBands = !!bands;

  return (
    <div className="space-y-3">
      {/* Section header */}
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p className="section-kicker">Per-metric breakdown</p>
          <h3 className="mt-1 text-xl font-semibold" style={{ color: "var(--ink-900)" }}>
            Individual trajectories
          </h3>
          <p className="mt-1 text-sm" style={{ color: "var(--ink-400)" }}>
            {hasBands
              ? "Shaded areas show p10–p90 spread across 9 ensemble runs."
              : "Run confidence bands to see 9-run severity sweep."}
          </p>
        </div>
        {hasBands && (
          <span
            className="rounded-full px-3 py-1 text-xs font-semibold"
            style={{
              background: "rgba(29,99,68,0.10)",
              color:       "#1D6344",
              border:      "1px solid rgba(29,99,68,0.20)",
            }}
          >
            ✓ 9-run bands applied
          </span>
        )}
      </div>

      {/* 3-column grid of mini charts */}
      <div className="grid gap-3 sm:grid-cols-3">
        {METRICS.map((metric) => (
          <MiniChart
            key={metric.key}
            metric={metric}
            roundSummaries={roundSummaries}
            bands={bands?.[metric.key]}
          />
        ))}
      </div>
    </div>
  );
}
