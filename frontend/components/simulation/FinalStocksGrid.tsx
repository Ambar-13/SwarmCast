"use client";

import { AnimatedCounter } from "@/components/ui/AnimatedCounter";
import { formatDelta, formatPct } from "@/lib/format";

interface StatCardProps {
  label: string;
  value: number;
  formatter: (v: number) => string;
  note: string;
  delta?: number | null;
  index: number;
}

function StatCard({ label, value, formatter, note, delta, index }: StatCardProps) {
  const positive = delta !== null && delta !== undefined && delta >= 0;

  return (
    <div
      className="card-warm p-4"
      style={{
        animation:      "slideUpFade 380ms ease both",
        animationDelay: `${index * 55}ms`,
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="kicker">{label}</p>
        {delta !== null && delta !== undefined && (
          <span
            className="metric-num flex-shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-semibold"
            style={{
              color:       positive ? "#15803D"                 : "#DC2626",
              borderColor: positive ? "rgba(21,128,61,0.22)"   : "rgba(220,38,38,0.22)",
              background:  positive ? "#f0fdf4"                 : "#fef2f2",
            }}
          >
            {formatDelta(delta)}
          </span>
        )}
      </div>
      <p
        className="metric-num mt-3 text-[30px] font-bold leading-none"
        style={{ color: "var(--ink-900)" }}
      >
        <AnimatedCounter value={value} formatter={formatter} />
      </p>
      <p className="mt-2 text-sm leading-5" style={{ color: "var(--ink-400)" }}>
        {note}
      </p>
    </div>
  );
}

interface Props {
  finalStocks: Record<string, number>;
  finalPopulationSummary: Record<string, number>;
  compareStocks?: Record<string, number>;
  comparePopulation?: Record<string, number>;
}

export function FinalStocksGrid({
  finalStocks,
  finalPopulationSummary,
  compareStocks,
  comparePopulation,
}: Props) {
  const delta = (
    key: string,
    base: Record<string, number>,
    comp?: Record<string, number>,
  ) => (comp ? (comp[key] ?? 0) - (base[key] ?? 0) : null);

  return (
    <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
      <StatCard
        index={0}
        label="Compliance rate"
        value={finalPopulationSummary.compliance_rate ?? 0}
        formatter={formatPct}
        note="Share of regulated entities in compliance."
        delta={delta("compliance_rate", finalPopulationSummary, comparePopulation)}
      />
      <StatCard
        index={1}
        label="Relocation rate"
        value={finalPopulationSummary.relocation_rate ?? 0}
        formatter={formatPct}
        note="Fraction that shifted jurisdiction."
        delta={delta("relocation_rate", finalPopulationSummary, comparePopulation)}
      />
      <StatCard
        index={2}
        label="Evasion rate"
        value={finalPopulationSummary.evasion_rate ?? 0}
        formatter={formatPct}
        note="Population share sustaining evasion."
        delta={delta("evasion_rate", finalPopulationSummary, comparePopulation)}
      />
      <StatCard
        index={3}
        label="Public trust"
        value={finalStocks.public_trust ?? 0}
        formatter={(v) => v.toFixed(1)}
        note="Aggregate trust stock at close."
        delta={
          compareStocks
            ? (compareStocks.public_trust ?? 0) - (finalStocks.public_trust ?? 0)
            : null
        }
      />
      <StatCard
        index={4}
        label="Domestic firms"
        value={finalStocks.domestic_companies ?? 0}
        formatter={(v) => Math.round(v).toLocaleString()}
        note="Domestic company count remaining."
        delta={
          compareStocks
            ? (compareStocks.domestic_companies ?? 0) -
              (finalStocks.domestic_companies ?? 0)
            : null
        }
      />
    </section>
  );
}
