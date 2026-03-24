"use client";

import { useState, Suspense } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { comparePolicies } from "@/lib/api-client";
import { useCompareStore, makeId } from "@/lib/store";
import { DEFAULT_SIM_CONFIG } from "@/lib/constants";
import { formatPct } from "@/lib/format";
import { RoundSummaryChart } from "@/components/simulation/RoundSummaryChart";
import { X, Plus, GitCompareArrows } from "lucide-react";

const SLOT_COLORS = ["#6366f1", "#f97316", "#22c55e", "#a855f7"];

function CompareContent() {
  const store = useCompareStore();
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newSev, setNewSev] = useState(2.0);

  function addSlot() {
    if (!newName.trim()) return;
    store.addSlot({
      id: makeId(),
      policyName: newName,
      policyDescription: newDesc || newName,
      policySeverity: newSev,
      simConfig: { ...DEFAULT_SIM_CONFIG },
    });
    setNewName("");
    setNewDesc("");
    setNewSev(2.0);
  }

  async function handleRun() {
    if (store.slots.length < 2) return;
    store.setLoading(true);
    store.setError(null);
    try {
      const { results } = await comparePolicies({
        policies: store.slots.map((s) => ({
          policy_name: s.policyName,
          policy_description: s.policyDescription,
          policy_severity: s.policySeverity,
          config: s.simConfig,
        })),
      });
      store.setResults(results);
    } catch (e) {
      store.setError(e instanceof Error ? e.message : String(e));
    }
  }

  const hasResults = store.slots.some((s) => s.result !== null);

  return (
    <AppShell>
      <div className="mx-auto max-w-[1560px] px-4 py-6 lg:px-8 space-y-6">
        {/* Page header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl"
              style={{ background: "var(--cream-200)", border: "1px solid var(--border-warm)" }}
            >
              <GitCompareArrows size={17} style={{ color: "var(--ink-500)" }} />
            </div>
            <div>
              <h1 className="text-xl font-bold" style={{ color: "var(--ink-900)" }}>
                Compare policies
              </h1>
              <p className="text-sm" style={{ color: "var(--ink-400)" }}>
                Add 2–4 policies to compare side-by-side.
              </p>
            </div>
          </div>
          <button
            onClick={handleRun}
            disabled={store.slots.length < 2 || store.isLoading}
            className="btn-primary"
          >
            {store.isLoading ? "Running…" : "Run compare"}
          </button>
        </div>

        {/* Add policy form */}
        <div className="card-warm p-4 space-y-3">
          <p className="kicker">Add policy</p>
          <div className="flex gap-2">
            <input
              className="input-warm flex-1"
              placeholder="Policy name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
            <input
              className="input-warm w-24"
              type="number"
              min={1}
              max={5}
              step={0.1}
              placeholder="Severity"
              value={newSev}
              onChange={(e) => setNewSev(parseFloat(e.target.value) || 2)}
            />
            <button
              onClick={addSlot}
              disabled={!newName.trim() || store.slots.length >= 4}
              className="btn-secondary"
            >
              <Plus size={14} />
            </button>
          </div>
        </div>

        {/* Slots */}
        {store.slots.length > 0 && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {store.slots.map((slot, i) => (
              <div
                key={slot.id}
                className="card-warm p-4 space-y-3"
                style={{ borderColor: SLOT_COLORS[i] + "44" }}
              >
                <div className="flex items-center justify-between">
                  <span
                    className="text-sm font-medium truncate"
                    style={{ color: SLOT_COLORS[i] }}
                  >
                    {slot.policyName}
                  </span>
                  <button
                    onClick={() => store.removeSlot(slot.id)}
                    style={{ color: "var(--ink-400)" }}
                  >
                    <X size={13} />
                  </button>
                </div>
                <p className="text-xs" style={{ color: "var(--ink-500)" }}>
                  severity {slot.policySeverity.toFixed(2)}
                </p>
                {slot.result && (
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span style={{ color: "var(--ink-500)" }}>Compliance</span>
                      <span className="font-mono" style={{ color: "var(--ink-900)" }}>
                        {formatPct(slot.result.final_population_summary.compliance_rate ?? 0)}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span style={{ color: "var(--ink-500)" }}>Relocation</span>
                      <span className="font-mono" style={{ color: "var(--ink-900)" }}>
                        {formatPct(slot.result.final_population_summary.relocation_rate ?? 0)}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span style={{ color: "var(--ink-500)" }}>Evasion</span>
                      <span className="font-mono" style={{ color: "var(--ink-900)" }}>
                        {formatPct(slot.result.final_population_summary.evasion_rate ?? 0)}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Comparison charts */}
        {hasResults && (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {store.slots.map(
              (slot, i) =>
                slot.result && (
                  <div key={slot.id} className="card-warm p-4 space-y-2">
                    <p className="text-sm font-medium" style={{ color: SLOT_COLORS[i] }}>
                      {slot.policyName}
                    </p>
                    <RoundSummaryChart roundSummaries={slot.result.round_summaries} />
                  </div>
                ),
            )}
          </div>
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
      </div>
    </AppShell>
  );
}

export default function ComparePage() {
  return (
    <Suspense>
      <CompareContent />
    </Suspense>
  );
}
