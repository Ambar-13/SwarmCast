"use client";

import { AppShell } from "@/components/layout/AppShell";
import { runInjection } from "@/lib/api-client";
import { useInfluenceStore } from "@/lib/store";
import { formatPct, formatDelta } from "@/lib/format";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { Activity } from "lucide-react";

function ResilienceScoreCard({ score }: { score: number }) {
  const scoreColor =
    score >= 0.7
      ? "var(--success)"
      : score >= 0.4
        ? "var(--warning)"
        : "var(--danger)";
  const label = score >= 0.7 ? "Resistant" : score >= 0.4 ? "Moderate" : "Vulnerable";
  const cardClass = score < 0.4 ? "card-tinted" : "card-raised";

  return (
    <div className={`${cardClass} p-6 text-center space-y-2`}>
      <p className="kicker">Network resilience</p>
      <p
        className="metric-num text-5xl font-bold"
        style={{ color: scoreColor }}
      >
        {score.toFixed(3)}
      </p>
      <p className="text-sm font-medium" style={{ color: scoreColor }}>
        {label}
      </p>
      <p className="text-[10px]" style={{ color: "var(--ink-400)" }}>
        1 = highly resistant · 0 = star topology (one hub dominates)
      </p>
    </div>
  );
}

export default function InfluencePage() {
  const store = useInfluenceStore();

  async function handleRun() {
    store.setLoading(true);
    store.setError(null);
    try {
      const result = await runInjection({
        policy_name: store.policyName,
        policy_description: store.policyDescription,
        policy_severity: store.policySeverity,
        n_population: store.nPopulation,
        num_rounds: store.numRounds,
        seed: store.seed,
        injection: store.injection,
      });
      store.setResult(result);
    } catch (e) {
      store.setError(e instanceof Error ? e.message : String(e));
    }
  }

  const chartData = store.result
    ? store.result.round_compliance_baseline.map((v, i) => ({
        round: i + 1,
        baseline: v,
        injected: store.result!.round_compliance_injected[i] ?? 0,
      }))
    : [];

  const injectionStart = store.injection.injection_start_round;

  return (
    <AppShell>
      <div className="flex h-full">
        {/* Left panel */}
        <div
          className="w-80 flex-shrink-0 p-4 space-y-4 overflow-y-auto"
          style={{ borderRight: "1px solid var(--border-warm)" }}
        >
          <div className="flex items-center gap-3">
            <div
              className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl"
              style={{ background: "var(--cream-200)", border: "1px solid var(--border-warm)" }}
            >
              <Activity size={17} style={{ color: "var(--ink-500)" }} />
            </div>
            <div>
              <h1 className="text-base font-bold" style={{ color: "var(--ink-900)" }}>
                Influence scenario
              </h1>
              <p className="text-xs" style={{ color: "var(--ink-400)" }}>
                Test how resilient the policy network is to adversarial belief injection.
              </p>
            </div>
          </div>

          {/* Policy */}
          <div className="card-warm p-4 space-y-3">
            <p className="kicker">Policy</p>
            <input
              className="input-warm w-full"
              placeholder="Policy name"
              value={store.policyName}
              onChange={(e) =>
                store.setPolicy(e.target.value, store.policyDescription, store.policySeverity)
              }
            />
            <div className="flex gap-2 items-center">
              <label className="text-xs w-16 flex-shrink-0" style={{ color: "var(--ink-500)" }}>
                Severity
              </label>
              <input
                type="number"
                className="input-warm flex-1"
                min={1}
                max={5}
                step={0.01}
                value={store.policySeverity}
                onChange={(e) =>
                  store.setPolicy(
                    store.policyName,
                    store.policyDescription,
                    parseFloat(e.target.value) || 1,
                  )
                }
              />
            </div>
          </div>

          {/* Simulation params */}
          <div className="card-warm p-4 space-y-3">
            <p className="kicker">Simulation</p>
            <div className="grid grid-cols-3 gap-1.5">
              {[
                { label: "N", value: store.nPopulation, key: "nPopulation" as const, min: 100, max: 5000 },
                { label: "Rounds", value: store.numRounds, key: "numRounds" as const, min: 1, max: 32 },
                { label: "Seed", value: store.seed, key: "seed" as const, min: 0, max: undefined },
              ].map(({ label, value, key, min, max }) => (
                <div key={key}>
                  <p className="kicker text-[10px] mb-1">{label}</p>
                  <input
                    type="number"
                    className="input-warm w-full text-xs"
                    value={value}
                    min={min}
                    max={max}
                    onChange={(e) =>
                      store.setSimParams({ [key]: parseInt(e.target.value) || min })
                    }
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Injection config */}
          <div className="card-warm p-4 space-y-3">
            <p className="kicker">Injection parameters</p>

            <div>
              <label className="text-[10px]" style={{ color: "var(--ink-500)" }}>
                Rate (fraction of hubs per round):{" "}
                {(store.injection.injection_rate * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min={0.01}
                max={0.2}
                step={0.01}
                value={store.injection.injection_rate}
                onChange={(e) =>
                  store.setInjection({ injection_rate: parseFloat(e.target.value) })
                }
                className="w-full mt-1"
                style={{ accentColor: "var(--crimson-700)" }}
              />
            </div>

            <div>
              <label className="text-[10px]" style={{ color: "var(--ink-500)" }}>
                Magnitude (belief shift per injection):{" "}
                {store.injection.injection_magnitude.toFixed(2)}
              </label>
              <input
                type="range"
                min={0.01}
                max={0.3}
                step={0.01}
                value={store.injection.injection_magnitude}
                onChange={(e) =>
                  store.setInjection({ injection_magnitude: parseFloat(e.target.value) })
                }
                className="w-full mt-1"
                style={{ accentColor: "var(--crimson-700)" }}
              />
            </div>

            <div>
              <label className="text-[10px]" style={{ color: "var(--ink-500)" }}>
                Direction
              </label>
              <select
                value={store.injection.injection_direction}
                onChange={(e) =>
                  store.setInjection({ injection_direction: parseFloat(e.target.value) })
                }
                className="select-warm w-full mt-1"
              >
                <option value={1.0}>Anti-regulation (push burden narrative)</option>
                <option value={-1.0}>Pro-regulation (reduce burden narrative)</option>
              </select>
            </div>

            <div>
              <label className="text-[10px]" style={{ color: "var(--ink-500)" }}>
                Start round: {store.injection.injection_start_round}
              </label>
              <input
                type="range"
                min={1}
                max={store.numRounds}
                step={1}
                value={store.injection.injection_start_round}
                onChange={(e) =>
                  store.setInjection({ injection_start_round: parseInt(e.target.value) })
                }
                className="w-full mt-1"
                style={{ accentColor: "var(--crimson-700)" }}
              />
            </div>
          </div>

          <button
            onClick={handleRun}
            disabled={store.isLoading}
            className="btn-primary w-full"
          >
            {store.isLoading ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Running…
              </>
            ) : (
              "Run injection test"
            )}
          </button>

          {store.error && (
            <div
              className="rounded-xl border px-3 py-2.5 text-sm"
              style={{
                color:       "var(--danger)",
                borderColor: "rgba(220,38,38,0.2)",
                background:  "#fef2f2",
              }}
            >
              {store.error}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="flex-1 p-6 overflow-y-auto">
          {store.isLoading && (
            <div className="space-y-4 animate-pulse">
              <div
                className="h-32 rounded-xl"
                style={{ background: "var(--cream-200)" }}
              />
              <div
                className="h-56 rounded-xl"
                style={{ background: "var(--cream-200)" }}
              />
              <p
                className="text-xs text-center"
                style={{ color: "var(--ink-400)" }}
              >
                Running baseline + injected simulations — ~4–8 seconds…
              </p>
            </div>
          )}

          {!store.isLoading && store.result && (
            <div className="space-y-6">
              <ResilienceScoreCard score={store.result.resilience_score} />

              {/* Summary cards */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {[
                  { label: "Baseline compliance", value: formatPct(store.result.baseline_compliance) },
                  { label: "Injected compliance", value: formatPct(store.result.injected_compliance) },
                  { label: "Compliance delta", value: formatDelta(store.result.compliance_delta) },
                  { label: "Relocation delta", value: formatDelta(store.result.relocation_delta) },
                ].map(({ label, value }) => (
                  <div key={label} className="card-warm p-3">
                    <p className="kicker text-[10px]">{label}</p>
                    <p
                      className="metric-num text-lg font-semibold mt-0.5"
                      style={{ color: "var(--ink-900)" }}
                    >
                      {value}
                    </p>
                  </div>
                ))}
              </div>

              {/* Compliance trajectory chart */}
              <div className="card-warm p-4 space-y-3">
                <h3 className="kicker">
                  Compliance trajectory: baseline vs. injected
                </h3>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
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
                        borderRadius: 8,
                      }}
                      formatter={(v: number) => formatPct(v)}
                    />
                    <Legend wrapperStyle={{ fontSize: 11, color: "var(--ink-500)" }} />
                    <ReferenceLine
                      x={injectionStart}
                      stroke="var(--warning)"
                      strokeDasharray="4 4"
                      label={{
                        value: "Injection start",
                        fill:  "var(--warning)",
                        fontSize: 10,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="baseline"
                      name="Baseline"
                      stroke="#6366f1"
                      dot={false}
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="injected"
                      name="Injected"
                      stroke="#f97316"
                      dot={false}
                      strokeWidth={2}
                      strokeDasharray="5 3"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {!store.isLoading && !store.result && !store.error && (
            <div
              className="flex flex-col items-center justify-center h-full text-center gap-3"
              style={{ color: "var(--ink-400)" }}
            >
              <p className="text-sm">
                Configure injection parameters and click{" "}
                <strong style={{ color: "var(--ink-700)" }}>Run injection test</strong>
              </p>
              <p className="text-xs">
                Runs two matched simulations — baseline and injected — and measures network
                resilience.
              </p>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
