"use client";

import { AnimatedCounter } from "@/components/ui/AnimatedCounter";
import { TiltCard } from "@/components/ui/TiltCard";
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
    <TiltCard
      className="card-warm p-4"
      maxTilt={5}
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
    </TiltCard>
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

  // Detect floor conditions to show model-limitation disclosure
  const trustFloor = (finalStocks.public_trust ?? 1) === 0;
  const investFloor = (finalStocks.ai_investment_index ?? 1) === 0;
  const showFloorNote = trustFloor || investFloor;

  return (
    <section className="space-y-3">
    {showFloorNote && (
      <div
        className="flex items-start gap-2.5 rounded-xl px-3.5 py-3 text-xs leading-5"
        style={{
          background:   "rgba(180,30,40,0.05)",
          border:       "1px solid rgba(180,30,40,0.15)",
          color:        "var(--ink-500)",
        }}
      >
        <span style={{ color: "var(--crimson-700)", fontSize: 14, lineHeight: 1 }}>⚠</span>
        <span>
          <strong style={{ color: "var(--ink-700)" }}>Model floor reached</strong>
          {trustFloor && investFloor && " — public trust and investment both hit 0.0. "}
          {trustFloor && !investFloor && " — public trust hit 0.0. "}
          {!trustFloor && investFloor && " — AI investment hit 0.0. "}
          This is the stock depletion model hitting its lower bound over many rounds at high severity,
          not a literal finding. In practice {trustFloor ? "trust" : ""}{trustFloor && investFloor ? " and " : ""}{investFloor ? "investment" : ""} would
          floor above zero (safety advocates support bans; below-threshold firms still attract capital).
          {(finalPopulationSummary.evasion_rate ?? 0) === 0 && " Compute-reporting evasion is also likely underestimated — the model's evasion pathway uses a different trigger condition."}
        </span>
      </div>
    )}
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
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
    </div>
    </section>
  );
}
