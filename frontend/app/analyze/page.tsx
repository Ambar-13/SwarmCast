"use client";

import { AppShell } from "@/components/layout/AppShell";
import { EvidencePackSection } from "@/components/simulation/EvidencePackSection";
import { ResultsPanel } from "@/components/simulation/ResultsPanel";
import { SimConfigPanel } from "@/components/simulation/SimConfigPanel";
import { PolicyParamPanel } from "@/components/policy/PolicyParamPanel";
import { PresetSelector } from "@/components/policy/PresetSelector";
import { simulate } from "@/lib/api-client";
import { useAnalyzeStore } from "@/lib/store";
import type { PresetPolicy } from "@/lib/types";
import { ArrowRight, Brain, GitCompareArrows, CheckCircle2, KeyRound, Play, Zap } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { ScrambleText } from "@/components/ui/ScrambleText";
import { AgentNetworkIdle } from "@/components/ui/AgentNetworkIdle";
import { useCompareStore, makeId } from "@/lib/store";


// ── Simulation-started banner ─────────────────────────────────────────────────

function StartBanner({ policyName, useSwarm }: { policyName: string; useSwarm: boolean }) {
  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        left: 24,
        zIndex: 9999,
        width: 288,
        animation: "slideInLeft 300ms cubic-bezier(0.22,1,0.36,1) both",
      }}
    >
      <div className="card-warm shadow-xl overflow-hidden" style={{ border: "1px solid var(--border-warm)" }}>
        <div style={{ height: 3, background: "linear-gradient(90deg, var(--crimson-600), var(--crimson-900))" }} />
        <div className="flex items-center gap-3 px-4 py-3">
          <svg className="animate-spin h-4 w-4 flex-shrink-0" viewBox="0 0 24 24" fill="none"
            style={{ color: "var(--crimson-700)" }}>
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <div className="min-w-0">
            <p className="text-xs font-semibold truncate" style={{ color: "var(--ink-700)" }}>
              Simulating…
            </p>
            <p className="text-[10px] truncate" style={{ color: "var(--ink-400)" }}>
              {policyName}{useSwarm ? " · swarm elicitation" : ""}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Toast ─────────────────────────────────────────────────────────────────────

interface ToastData {
  policyName: string;
  compliance: number;
  elapsedMs: number;
  vectorMs: number;
}

function SimToast({ data, onDismiss }: { data: ToastData; onDismiss: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDismiss, 5000);
    return () => clearTimeout(t);
  }, [onDismiss]);

  const compliancePct = (data.compliance * 100).toFixed(1);
  const elapsedS = (data.elapsedMs / 1000).toFixed(1);
  const vectorMs = data.vectorMs < 1000
    ? `${data.vectorMs}ms`
    : `${(data.vectorMs / 1000).toFixed(1)}s`;

  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        right: 24,
        zIndex: 9999,
        width: 320,
        animation: "slideInRight 350ms cubic-bezier(0.22,1,0.36,1) both",
      }}
    >
      <div
        className="card-warm shadow-xl"
        style={{
          border: "1px solid var(--border-warm)",
          overflow: "hidden",
        }}
      >
        {/* Accent bar */}
        <div style={{ height: 3, background: "linear-gradient(90deg, var(--crimson-900), var(--crimson-600))" }} />

        <div className="flex items-start gap-3 p-4">
          <div
            className="mt-0.5 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full"
            style={{ background: "rgba(180,30,40,0.1)" }}
          >
            <CheckCircle2 size={16} style={{ color: "var(--crimson-700)" }} />
          </div>

          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold truncate" style={{ color: "var(--ink-900)" }}>
              Simulation complete
            </p>
            <p className="mt-0.5 text-xs truncate" style={{ color: "var(--ink-500)" }}>
              {data.policyName}
            </p>

            <div className="mt-2.5 flex items-center gap-3">
              <div>
                <p className="kicker text-[9px]">Compliance</p>
                <p className="metric-num text-base font-bold" style={{ color: "var(--ink-900)" }}>
                  {compliancePct}%
                </p>
              </div>
              <div
                className="h-8 w-px"
                style={{ background: "var(--border-warm)" }}
              />
              <div>
                <p className="kicker text-[9px]">Total time</p>
                <p className="metric-num text-base font-bold" style={{ color: "var(--ink-900)" }}>
                  {elapsedS}s
                </p>
              </div>
              <div
                className="h-8 w-px"
                style={{ background: "var(--border-warm)" }}
              />
              <div>
                <div className="flex items-center gap-1">
                  <Zap size={9} style={{ color: "var(--crimson-700)" }} />
                  <p className="kicker text-[9px]">Engine</p>
                </div>
                <p className="metric-num text-base font-bold" style={{ color: "var(--crimson-700)" }}>
                  {vectorMs}
                </p>
              </div>
            </div>
          </div>

          <button
            onClick={onDismiss}
            className="flex-shrink-0 text-lg leading-none"
            style={{ color: "var(--ink-300)", lineHeight: 1 }}
          >
            ×
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function AnalyzePage() {
  const store = useAnalyzeStore();
  const compareStore = useCompareStore();
  const router = useRouter();
  const [useSwarm, setUseSwarm] = useState(true);
  const [openaiKey, setOpenaiKey] = useState("");
  const [toast, setToast] = useState<ToastData | null>(null);

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
    setToast(null);

    const wallStart = Date.now();

    try {
      const result = await simulate({
        policy_name:           store.policyName,
        policy_description:    store.policyDescription,
        policy_severity:       store.policySeverity,
        config:                store.simConfig,
        use_swarm_elicitation: useSwarm,
        openai_api_key:        openaiKey.trim() || undefined,
      });

      const wallMs = Date.now() - wallStart;
      store.setResult(result);

      // Show success toast. `elapsed_ms` from backend is the pure engine time;
      // wallMs is wall-clock including animation pad + network.
      setToast({
        policyName: store.policyName,
        compliance: result.final_population_summary.compliance_rate ?? 0,
        elapsedMs:  wallMs,
        vectorMs:   result.run_metadata.duration_ms,
      });
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

            {/* ── Swarm elicitation + API key ─────────────────────── */}
            <div className="card-warm overflow-hidden">
              {/* Swarm toggle row */}
              <div className="px-4 py-3">
                <label className="flex cursor-pointer items-start gap-3">
                  <div className="relative mt-0.5 flex-shrink-0">
                    <input
                      type="checkbox"
                      checked={useSwarm}
                      onChange={(e) => setUseSwarm(e.target.checked)}
                      className="peer sr-only"
                    />
                    <div
                      className="h-4 w-8 rounded-full transition-colors"
                      style={{ background: useSwarm ? "#d97706" : "var(--cream-400)" }}
                    />
                    <div
                      className="absolute top-0.5 h-3 w-3 rounded-full bg-white shadow transition-transform"
                      style={{ left: "2px", transform: useSwarm ? "translateX(16px)" : "translateX(0)" }}
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <Brain size={12} style={{ color: "var(--crimson-700)", flexShrink: 0 }} />
                      <p className="text-sm font-medium" style={{ color: "var(--ink-700)" }}>
                        Swarm elicitation
                      </p>
                      <span
                        className="rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-widest"
                        style={{
                          background: "rgba(245,158,11,0.12)",
                          color:       "#b45309",
                          border:      "1px solid rgba(245,158,11,0.25)",
                        }}
                      >
                        SWARM-ELICITED
                      </span>
                    </div>
                    <p className="mt-0.5 text-xs leading-4" style={{ color: "var(--ink-400)" }}>
                      Run 23 LLM personas before simulation to calibrate behavioral priors. ~5–10s extra.
                    </p>
                  </div>
                </label>
              </div>

              {/* API key field — shown when swarm is enabled */}
              {useSwarm && (
                <div
                  className="border-t px-4 pb-3 pt-3"
                  style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}
                >
                  <label className="block">
                    <div className="mb-1 flex items-center gap-1.5">
                      <KeyRound size={11} style={{ color: "var(--ink-400)" }} />
                      <span className="text-xs font-medium" style={{ color: "var(--ink-600)" }}>
                        OpenAI API key
                      </span>
                    </div>
                    <p className="mb-1.5 text-[11px] leading-4" style={{ color: "var(--ink-400)" }}>
                      Leave blank if already set in your .env file. Enter a key here to override it.
                    </p>
                    <input
                      type="password"
                      value={openaiKey}
                      onChange={(e) => setOpenaiKey(e.target.value)}
                      placeholder="sk-…"
                      className="input-warm w-full font-mono text-xs"
                      autoComplete="off"
                      spellCheck={false}
                    />
                  </label>
                </div>
              )}
            </div>

            <button
              onClick={handleRun}
              disabled={store.isLoading}
              className="btn-primary w-full py-3 text-base"
            >
              {store.isLoading ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Simulating…
                </>
              ) : (
                <>
                  Run simulation
                  <ArrowRight size={16} />
                </>
              )}
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

            {/* key=runId forces EvidencePackSection to fully remount on each new
                simulation, resetting its internal job/status/progress state so
                "Build →" reappears instead of staying "✓ Applied" forever */}
            {store.result && (
              <EvidencePackSection key={store.runId} />
            )}
          </aside>

          {/* Right: results panel */}
          <section className="min-w-0 flex-1">
            {!store.isLoading && store.result && (
              <ResultsPanel
                result={store.result}
                allBands={store.evidencePackResult?.bands}
              />
            )}

            {!store.isLoading && !store.result && (
              <div
                className="card-warm flex min-h-[520px] flex-col items-center justify-center gap-5 overflow-hidden p-8 text-center"
                style={{ animation: "slideUpFade 400ms ease both" }}
              >
                {/* Error banner — shown when a run fails */}
                {store.error && (
                  <div
                    className="w-full max-w-sm rounded-xl border px-4 py-3 text-sm text-left"
                    style={{
                      color:       "var(--danger)",
                      borderColor: "rgba(220,38,38,0.25)",
                      background:  "#fef2f2",
                    }}
                  >
                    <p className="font-semibold mb-0.5">Simulation failed</p>
                    <p className="text-xs leading-5 opacity-80">{store.error}</p>
                  </div>
                )}

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
                    {store.error ? "Ready to retry" : "Ready to simulate"}
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

                {/* CTA */}
                {!store.error && (
                  <p className="flex items-center gap-1.5 text-xs" style={{ color: "var(--ink-300)" }}>
                    <Play size={11} style={{ color: "var(--crimson-700)" }} />
                    <span>Click <strong style={{ color: "var(--ink-500)" }}>Run simulation</strong> to begin</span>
                  </p>
                )}
              </div>
            )}
          </section>
        </div>
      </div>

      {/* Simulation-started banner — fixed bottom-left, shown while loading */}
      {store.isLoading && (
        <StartBanner policyName={store.policyName} useSwarm={useSwarm} />
      )}

      {/* Success toast — fixed bottom-right */}
      {toast && (
        <SimToast data={toast} onDismiss={() => setToast(null)} />
      )}
    </AppShell>
  );
}
