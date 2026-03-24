"use client";

import { AppShell } from "@/components/layout/AppShell";
import { EvidencePackSection } from "@/components/simulation/EvidencePackSection";
import { ResultsPanel } from "@/components/simulation/ResultsPanel";
import { SimConfigPanel } from "@/components/simulation/SimConfigPanel";
import { SimLoadingSkeleton } from "@/components/simulation/SimLoadingSkeleton";
import { PolicyParamPanel } from "@/components/policy/PolicyParamPanel";
import { PresetSelector } from "@/components/policy/PresetSelector";
import { simulate } from "@/lib/api-client";
import { useAnalyzeStore } from "@/lib/store";
import type { PresetPolicy } from "@/lib/types";
import { BarChart2, ArrowRight } from "lucide-react";

export default function AnalyzePage() {
  const store = useAnalyzeStore();

  function handlePreset(preset: PresetPolicy) {
    store.setFromSpec(preset.spec);
  }

  async function handleRun() {
    store.setLoading(true);
    store.setError(null);
    try {
      const result = await simulate({
        policy_name:        store.policyName,
        policy_description: store.policyDescription,
        policy_severity:    store.policySeverity,
        config:             store.simConfig,
      });
      store.setResult(result);
    } catch (e) {
      store.setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-[1560px] px-4 py-6 lg:px-8">
        {/* Page header */}
        <div className="mb-6 flex items-center gap-3">
          <div
            className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl"
            style={{ background: "var(--cream-200)", border: "1px solid var(--border-warm)" }}
          >
            <BarChart2 size={17} style={{ color: "var(--ink-500)" }} />
          </div>
          <div>
            <h1 className="text-xl font-bold" style={{ color: "var(--ink-900)" }}>
              Policy Simulator
            </h1>
            <p className="text-sm" style={{ color: "var(--ink-400)" }}>
              Configure a bill and launch a population simulation
            </p>
          </div>
        </div>

        {/* Two-panel layout */}
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start">
          {/* Left: policy authoring panel */}
          <aside className="w-full space-y-4 lg:w-[380px] lg:flex-shrink-0">
            <div className="card-warm p-5 space-y-4">
              <div>
                <p className="kicker mb-3">Scenario</p>
                <PresetSelector
                  onSelect={handlePreset}
                  selectedSlug={undefined}
                />
              </div>
              <div className="divider-warm" />
              <PolicyParamPanel
                name={store.policyName}
                description={store.policyDescription}
                severity={store.policySeverity}
                penaltyType="none"
                penaltyCapUsd={null}
                computeThresholdFlops={null}
                enforcementMechanism="none"
                gracePeriodMonths={0}
                scope="all"
                onChangeName={(v) =>
                  store.setPolicy(v, store.policyDescription, store.policySeverity)
                }
                onChangeDescription={(v) =>
                  store.setPolicy(store.policyName, v, store.policySeverity)
                }
                onChangeSeverity={(v) =>
                  store.setPolicy(store.policyName, store.policyDescription, v)
                }
              />
            </div>

            <SimConfigPanel
              config={store.simConfig}
              onChange={(p) => store.setSimConfig(p)}
            />

            <button
              onClick={handleRun}
              disabled={store.isLoading}
              className="btn-primary w-full py-3 text-base"
            >
              {store.isLoading ? "Simulating…" : "Run simulation"}
              <ArrowRight size={16} />
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

            {store.result && <EvidencePackSection />}
          </aside>

          {/* Right: results panel */}
          <section className="min-w-0 flex-1">
            {store.isLoading && <SimLoadingSkeleton />}

            {!store.isLoading && store.result && (
              <ResultsPanel
                result={store.result}
                complianceBands={store.evidencePackResult?.bands?.compliance_rate}
              />
            )}

            {!store.isLoading && !store.result && !store.error && (
              <div className="card-warm flex min-h-[500px] flex-col items-center justify-center gap-6 p-8 text-center">
                <div
                  className="flex h-16 w-16 items-center justify-center rounded-2xl"
                  style={{
                    background:  "var(--cream-200)",
                    border:      "1px solid var(--border-warm)",
                  }}
                >
                  <BarChart2 size={28} style={{ color: "var(--ink-400)" }} />
                </div>
                <div className="space-y-2">
                  <h3
                    className="text-xl font-semibold"
                    style={{ color: "var(--ink-900)" }}
                  >
                    Ready to simulate
                  </h3>
                  <p
                    className="max-w-sm text-sm leading-6"
                    style={{ color: "var(--ink-400)" }}
                  >
                    Configure parameters on the left and click{" "}
                    <strong style={{ color: "var(--ink-700)" }}>Run simulation</strong> to see
                    compliance trajectories, relocation pressure, and calibration moments.
                  </p>
                </div>
                <div className="grid w-full max-w-xs grid-cols-3 gap-2">
                  {[
                    ["Population", store.simConfig.n_population.toLocaleString()],
                    ["Rounds",     String(store.simConfig.num_rounds)],
                    ["Severity",   store.policySeverity.toFixed(1)],
                  ].map(([label, value]) => (
                    <div key={label} className="card-warm px-3 py-2.5 text-center">
                      <p className="kicker text-[10px]">{label}</p>
                      <p
                        className="metric-num mt-1 text-lg font-bold"
                        style={{ color: "var(--ink-900)" }}
                      >
                        {value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </AppShell>
  );
}
