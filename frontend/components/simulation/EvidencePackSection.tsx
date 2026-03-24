"use client";

import { useState, useEffect } from "react";
import { startEvidencePack, pollEvidencePack } from "@/lib/api-client";
import { useAnalyzeStore } from "@/lib/store";
import type { EvidencePackRequest } from "@/lib/types";

export function EvidencePackSection() {
  const store = useAnalyzeStore();
  const [jobId, setJobId]     = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus]   = useState<string | null>(null);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    if (!jobId || status === "complete" || status === "error") return;
    const interval = setInterval(async () => {
      try {
        const job = await pollEvidencePack(jobId);
        setProgress(job.progress);
        setStatus(job.status);
        if (job.status === "complete" && job.result) {
          store.setEvidencePackResult(job.result);
          clearInterval(interval);
        } else if (job.status === "error") {
          setError(job.error ?? "Unknown error");
          clearInterval(interval);
        }
      } catch {
        clearInterval(interval);
      }
    }, 1200);
    return () => clearInterval(interval);
  }, [jobId, status, store]);

  async function handleStart() {
    setError(null);
    setProgress(0);
    setStatus("queued");
    const req: EvidencePackRequest = {
      policy_name:        store.policyName,
      policy_description: store.policyDescription,
      base_severity:      store.policySeverity,
      sim_config: {
        n_population:        store.simConfig.n_population,
        num_rounds:          store.simConfig.num_rounds,
        seed:                store.simConfig.seed,
        compute_cost_factor: store.simConfig.compute_cost_factor,
      },
      ensemble_size: 3,
    };
    try {
      const { job_id } = await startEvidencePack(req);
      setJobId(job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }

  const hasResult = Boolean(store.evidencePackResult);

  return (
    <div className="card-warm p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold" style={{ color: "var(--ink-700)" }}>
            Confidence bands
          </p>
          <p className="mt-0.5 text-xs leading-5" style={{ color: "var(--ink-400)" }}>
            9 ensemble runs (3 severity × 3 seeds) applied to the chart.
          </p>
        </div>
        {!hasResult && status !== "running" && status !== "queued" && (
          <button onClick={handleStart} className="btn-primary flex-shrink-0 px-3 py-1.5 text-xs">
            Build →
          </button>
        )}
        {hasResult && (
          <span className="text-xs font-medium" style={{ color: "var(--success)" }}>
            ✓ Applied
          </span>
        )}
      </div>

      {(status === "running" || status === "queued") && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-[11px]" style={{ color: "var(--ink-400)" }}>
            <span>{status === "queued" ? "Queued…" : "Running ensemble…"}</span>
            <span>{(progress * 100).toFixed(0)}%</span>
          </div>
          <div
            className="h-1.5 overflow-hidden rounded-full"
            style={{ background: "var(--cream-300)" }}
          >
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width:      `${progress * 100}%`,
                background: "var(--crimson-700)",
              }}
            />
          </div>
        </div>
      )}

      {error && (
        <p className="text-xs" style={{ color: "var(--danger)" }}>
          {error}
        </p>
      )}
    </div>
  );
}
