"use client";

import { useState } from "react";
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
import { buildChartData } from "@/lib/chart-helpers";
import { CHART_COLORS } from "@/lib/constants";
import { formatPct } from "@/lib/format";
import type { RoundSummary, ConfidenceBand } from "@/lib/types";

type MetricKey =
  | "compliance_rate"
  | "relocation_rate"
  | "ai_investment_index"
  | "enforcement_contact_rate";

const METRICS: { key: MetricKey; label: string; color: string; note: string }[] = [
  { key: "compliance_rate",          label: "Compliance", color: "#8B1A1A", note: "Policy adherence"   },
  { key: "relocation_rate",          label: "Relocation", color: "#C2410C", note: "Jurisdiction flight" },
  { key: "ai_investment_index",      label: "Investment", color: "#1D6344", note: "Capital sentiment"   },
  { key: "enforcement_contact_rate", label: "Enforcement",color: "#1E3A8A", note: "Regulatory contact"  },
];

interface Props {
  roundSummaries: RoundSummary[];
  bands?: ConfidenceBand[];
}

export function RoundSummaryChart({ roundSummaries, bands }: Props) {
  const [active, setActive] = useState<Set<MetricKey>>(
    new Set<MetricKey>(["compliance_rate", "relocation_rate", "ai_investment_index", "enforcement_contact_rate"]),
  );

  const data = buildChartData(roundSummaries, bands);

  function toggle(key: MetricKey) {
    setActive((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="section-kicker">Trajectory view</p>
          <h3 className="mt-2 text-2xl font-semibold" style={{ color: "var(--ink-900)" }}>
            Metric shifts across {roundSummaries.length} rounds
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--ink-400)]">
            Toggle the lines to compare compliance, relocation, investment, and enforcement
            pressure across the scenario timeline.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {METRICS.map(({ key, label, color, note }) => {
            const isActive = active.has(key);
            return (
              <button
                key={key}
                onClick={() => toggle(key)}
                className="rounded-2xl border px-3 py-2 text-left transition-all duration-200"
                style={
                  isActive
                    ? {
                        borderColor: `${color}55`,
                        background: `${color}18`,
                        color,
                      }
                    : {
                        borderColor: "var(--border-warm)",
                        background:  "var(--cream-200)",
                        color:       "var(--ink-400)",
                      }
                }
              >
                <div className="flex items-center gap-1">
                  {isActive && (
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                      <path d="M1.5 5L4 7.5L8.5 2.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                  <span className="text-xs font-semibold">{label}</span>
                </div>
                <div className="text-[10px] opacity-80">{note}</div>
              </button>
            );
          })}
        </div>
      </div>

      <div
  className="overflow-hidden rounded-2xl border p-3 md:p-4"
  style={{ background: "var(--cream-100)", borderColor: "var(--border-warm)" }}
>
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={data} margin={{ top: 12, right: 16, bottom: 0, left: -18 }}>
            <defs>
              <linearGradient id="compliance-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#8B1A1A" stopOpacity={0.18} />
                <stop offset="95%" stopColor="#8B1A1A" stopOpacity={0}    />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-warm)" />
            <XAxis
              dataKey="round"
              tick={{ fill: "var(--ink-400)", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tickFormatter={(v) => formatPct(v, 0)}
              tick={{ fill: "var(--ink-400)", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{
                background:   "var(--cream-50)",
                border:       "1px solid var(--border-warm)",
                borderRadius: 16,
                boxShadow:    "var(--shadow-raised)",
              }}
              labelStyle={{ color: "var(--ink-700)", fontSize: 11 }}
              itemStyle={{ fontSize: 12 }}
              formatter={(value: number) => formatPct(value)}
            />

            {bands && active.has("compliance_rate") && (
              <Area
                type="monotone"
                dataKey="band_p90"
                stroke="none"
                fill="url(#compliance-fill)"
                connectNulls
                isAnimationActive={false}
              />
            )}

            {METRICS.filter((m) => active.has(m.key)).map(({ key, color }) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={color}
                dot={false}
                strokeWidth={2.75}
                isAnimationActive={false}
                activeDot={{ r: 5, fill: color, stroke: "var(--cream-50)", strokeWidth: 2 }}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
