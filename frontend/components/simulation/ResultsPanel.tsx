import { FinalStocksGrid } from "@/components/simulation/FinalStocksGrid";
import { JurisdictionFlowDiagram } from "@/components/simulation/JurisdictionFlowDiagram";
import { NetworkStatsPanel } from "@/components/simulation/NetworkStatsPanel";
import { PopulationSummaryBar } from "@/components/simulation/PopulationSummaryBar";
import { RoundSummaryChart } from "@/components/simulation/RoundSummaryChart";
import { SMMTable } from "@/components/simulation/SMMTable";
import { AnimatedCounter } from "@/components/ui/AnimatedCounter";
import { formatDurationMs, formatPct } from "@/lib/format";
import type { ConfidenceBand, SimulateResponse } from "@/lib/types";

interface Props {
  result: SimulateResponse;
  comparisonResult?: SimulateResponse;
  complianceBands?: ConfidenceBand[];
}

export function ResultsPanel({ result, comparisonResult, complianceBands }: Props) {
  const pop    = result.final_population_summary;
  const stocks = result.final_stocks;

  return (
    <div className="space-y-4">
      {/* ── Verdict header ──────────────────────────────────────── */}
      <div
        className="card-raised overflow-hidden p-6 md:p-8"
        style={{ animation: "slideUpFade 380ms ease both" }}
      >
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-2xl space-y-3">
            <p className="kicker">Simulation verdict</p>
            <h2
              className="text-3xl font-bold leading-tight"
              style={{ color: "var(--ink-900)" }}
            >
              {result.policy_name}
            </h2>
            <p
              className="text-base leading-7"
              style={{ color: "var(--ink-500)" }}
            >
              Final compliance settled at{" "}
              <span
                className="metric-num font-bold"
                style={{ color: "var(--ink-900)" }}
              >
                {formatPct(pop.compliance_rate ?? 0)}
              </span>
              , with relocation at{" "}
              <span
                className="metric-num font-bold"
                style={{ color: "var(--ink-900)" }}
              >
                {formatPct(pop.relocation_rate ?? 0)}
              </span>
              . Public trust finished at{" "}
              <span
                className="metric-num font-bold"
                style={{ color: "var(--ink-900)" }}
              >
                {(stocks.public_trust ?? 0).toFixed(1)}
              </span>
              , AI investment at{" "}
              <span
                className="metric-num font-bold"
                style={{ color: "var(--ink-900)" }}
              >
                {(stocks.ai_investment_index ?? 0).toFixed(1)}
              </span>
              .
            </p>
          </div>

          {/* 2 key callout metrics */}
          <div className="flex gap-3 flex-shrink-0">
            {[
              {
                label: "Public trust",
                value: stocks.public_trust ?? 0,
                fmt:   (v: number) => v.toFixed(1),
              },
              {
                label: "Investment",
                value: stocks.ai_investment_index ?? 0,
                fmt:   (v: number) => v.toFixed(1),
              },
            ].map(({ label, value, fmt }) => (
              <div
                key={label}
                className="card-warm min-w-[88px] px-4 py-3 text-center"
              >
                <p className="kicker text-[10px]">{label}</p>
                <p
                  className="metric-num mt-2 text-2xl font-bold"
                  style={{ color: "var(--ink-900)" }}
                >
                  <AnimatedCounter value={value} formatter={fmt} />
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── 5 stat cards ────────────────────────────────────────── */}
      <FinalStocksGrid
        finalStocks={stocks}
        finalPopulationSummary={pop}
        compareStocks={comparisonResult?.final_stocks}
        comparePopulation={comparisonResult?.final_population_summary}
      />

      {/* ── Population breakdown bar ────────────────────────────── */}
      <PopulationSummaryBar
        complianceRate={pop.compliance_rate ?? 0}
        relocationRate={pop.relocation_rate ?? 0}
        evasionRate={pop.evasion_rate ?? 0}
        everLobbyedRate={pop.ever_lobbied_rate ?? 0}
      />

      {/* ── Chart + SMM table ───────────────────────────────────── */}
      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="card-warm p-5 md:p-6">
          <RoundSummaryChart
            roundSummaries={result.round_summaries}
            bands={complianceBands}
          />
        </div>
        <div className="card-warm p-5 md:p-6">
          <SMMTable
            moments={result.simulated_moments}
            distanceToGdpr={result.smm_distance_to_gdpr}
          />
        </div>
      </div>

      {/* ── Network + Jurisdiction ──────────────────────────────── */}
      <div className="grid gap-4 xl:grid-cols-[2fr_3fr]">
        <NetworkStatsPanel
          networkStatistics={result.network_statistics}
          networkHubs={result.network_hubs}
        />
        <JurisdictionFlowDiagram
          jurisdictionSummary={result.jurisdiction_summary}
        />
      </div>

      {/* ── Run metadata footer ─────────────────────────────────── */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border px-4 py-3 text-sm"
        style={{
          borderColor: "var(--border-warm)",
          background:  "var(--cream-100)",
          color:       "var(--ink-400)",
        }}
      >
        <span>
          Ran in{" "}
          <span
            className="metric-num font-semibold"
            style={{ color: "var(--ink-700)" }}
          >
            {formatDurationMs(result.run_metadata.duration_ms)}
          </span>
        </span>
        <span>
          seed{" "}
          <span
            className="metric-num font-semibold"
            style={{ color: "var(--ink-700)" }}
          >
            {result.run_metadata.seed}
          </span>
        </span>
        <span>
          n=
          <span
            className="metric-num font-semibold"
            style={{ color: "var(--ink-700)" }}
          >
            {result.run_metadata.n_population.toLocaleString()}
          </span>
        </span>
      </div>
    </div>
  );
}
