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
import { ArrowRight, Play, GitCompareArrows } from "lucide-react";
import { useRouter } from "next/navigation";
import { ScrambleText } from "@/components/ui/ScrambleText";
import { AgentNetworkIdle } from "@/components/ui/AgentNetworkIdle";
import { useCompareStore, makeId } from "@/lib/store";

export default function AnalyzePage() {
  const store = useAnalyzeStore();
  const compareStore = useCompareStore();
  const router = useRouter();

  function handlePreset(preset: PresetPolicy) {
    store.setFromSpec(preset.spec);
  }

  function handleAddToCompare() {
    if (!store.result) return;
    compareStore.addSlotWithResult({
      id: makeId(),
      policyName: store.policyName,
      policyDescription: store.policyDescription,
      policySeverity: store.policySeverity,
      simConfig: store.simConfig,
      result: store.result,
    });
    router.push("/compare");
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
        <div className="mb-6">
          <h1
            className="text-3xl font-bold tracking-tight"
            style={{ color: "var(--ink-900)", letterSpacing: "-0.03em" }}
          >
            <ScrambleText text="Policy Simulator" duration={800} delay={80} />
          </h1>
          <p className="mt-1.5 text-sm" style={{ color: "var(--ink-400)" }}>
            Configure a regulation and run a population-level ABM simulation
          </p>
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

            {store.result && (
              <button
                onClick={handleAddToCompare}
                className="btn-secondary w-full py-2.5"
              >
                <GitCompareArrows size={15} />
                Add to Compare
              </button>
            )}

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
              <div
                className="card-warm flex min-h-[520px] flex-col items-center justify-center gap-5 overflow-hidden p-8 text-center"
                style={{ animation: "slideUpFade 400ms ease both" }}
              >
                {/* Animated agent network — main visual */}
                <div style={{ animation: "floatSlow 5s ease-in-out infinite" }}>
                  <AgentNetworkIdle />
                </div>

                {/* Text */}
                <div className="space-y-2">
                  <h3
                    className="text-2xl font-bold"
                    style={{ color: "var(--ink-900)", letterSpacing: "-0.03em" }}
                  >
                    Ready to simulate
                  </h3>
                  <p
                    className="mx-auto max-w-[300px] text-sm leading-6"
                    style={{ color: "var(--ink-400)" }}
                  >
                    Configure a regulation on the left, then run the simulation to see
                    how{" "}
                    <span style={{ color: "var(--ink-700)", fontWeight: 500 }}>
                      {store.simConfig.n_population.toLocaleString()} agents
                    </span>{" "}
                    respond across {store.simConfig.num_rounds} decision rounds.
                  </p>
                </div>

                {/* Mini params preview */}
                <div className="flex gap-3">
                  {[
                    ["Population", store.simConfig.n_population.toLocaleString()],
                    ["Rounds",     String(store.simConfig.num_rounds)],
                    ["Severity",   store.policySeverity.toFixed(1)],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="rounded-xl border px-4 py-2.5 text-center"
                      style={{ background: "var(--cream-200)", borderColor: "var(--border-warm)" }}
                    >
                      <p className="kicker text-[10px]">{label}</p>
                      <p
                        className="metric-num mt-1 text-base font-bold"
                        style={{ color: "var(--ink-900)" }}
                      >
                        {value}
                      </p>
                    </div>
                  ))}
                </div>

                {/* CTA arrow pointing left toward run button */}
                <p className="flex items-center gap-1.5 text-xs" style={{ color: "var(--ink-300)" }}>
                  <Play size={11} style={{ color: "var(--crimson-700)" }} />
                  <span>Click <strong style={{ color: "var(--ink-500)" }}>Run simulation</strong> to begin</span>
                </p>
              </div>
            )}
          </section>
        </div>
      </div>
    </AppShell>
  );
}
