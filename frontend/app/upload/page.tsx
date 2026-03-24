"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { AppShell } from "@/components/layout/AppShell";
import { simulateUpload } from "@/lib/api-client";
import { PolicyParamPanel } from "@/components/policy/PolicyParamPanel";
import { ResultsPanel } from "@/components/simulation/ResultsPanel";
import { EpistemicBadge } from "@/components/simulation/EpistemicBadge";
import { SimLoadingSkeleton } from "@/components/simulation/SimLoadingSkeleton";
import type { UploadResponse, EpistemicTier } from "@/lib/types";
import { Upload } from "lucide-react";

const ALLOWED_EXTS = [".pdf", ".txt", ".md", ".docx"];
const MAX_BYTES = 20 * 1024 * 1024;

function TraceabilityRow({
  label,
  value,
  confidence,
  tier,
  source,
}: {
  label: string;
  value: unknown;
  confidence: number;
  tier: EpistemicTier;
  source: string;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className={`card-warm py-2 last:border-0 ${tier === "ASSUMED" ? "opacity-60" : ""}`}
      style={{ borderBottom: "1px solid var(--border-warm)" }}
    >
      <div
        className="flex items-center gap-2 cursor-pointer"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="text-xs w-40 flex-shrink-0" style={{ color: "var(--ink-500)" }}>
          {label}
        </span>
        <span className="text-xs font-mono flex-1" style={{ color: "var(--ink-900)" }}>
          {String(value ?? "—")}
        </span>
        <span
          className="text-[10px] font-mono w-14 text-right"
          style={{ color: "var(--ink-400)" }}
        >
          {(confidence * 100).toFixed(0)}%
        </span>
        <EpistemicBadge tier={tier} small />
      </div>
      {open && source && (
        <div
          className="mt-1.5 ml-40 text-[11px] italic pl-3 py-1 rounded-r"
          style={{
            color:        "var(--ink-500)",
            background:   "var(--cream-200)",
            borderLeft:   "2px solid var(--border-warm)",
          }}
        >
          {source}
        </div>
      )}
    </div>
  );
}

export default function UploadPage() {
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-4o");

  const onDrop = useCallback(
    async (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;
      setLoading(true);
      setError(null);
      setResult(null);
      try {
        const res = await simulateUpload(file, apiKey || undefined, model);
        setResult(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [apiKey, model],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "text/plain": [".txt"],
      "text/markdown": [".md"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    },
    maxSize: MAX_BYTES,
    multiple: false,
  });

  const ext = result?.extraction;

  return (
    <AppShell>
      <div className="flex h-full">
        {/* Left panel */}
        <div
          className="w-80 flex-shrink-0 p-4 space-y-4 overflow-y-auto"
          style={{ borderRight: "1px solid var(--border-warm)" }}
        >
          <div>
            <h1 className="text-base font-bold" style={{ color: "var(--ink-900)" }}>
              Upload document
            </h1>
            <p className="text-xs mt-0.5" style={{ color: "var(--ink-400)" }}>
              PDF, txt, md, or docx → extracted parameters → simulation.
            </p>
          </div>

          {/* Drop zone */}
          <div
            {...getRootProps()}
            className="p-6 text-center cursor-pointer transition-colors"
            style={{
              background:   isDragActive ? "var(--cream-200)" : "var(--cream-100)",
              borderColor:  isDragActive ? "var(--crimson-700)" : "var(--border-warm)",
              borderStyle:  "dashed",
              borderWidth:  "2px",
              borderRadius: "16px",
            }}
          >
            <input {...getInputProps()} />
            <Upload size={24} className="mx-auto mb-2" style={{ color: "var(--ink-400)" }} />
            <p className="text-sm" style={{ color: "var(--ink-700)" }}>
              {isDragActive ? "Drop the file here" : "Drop a file or click to browse"}
            </p>
            <p className="text-[10px] mt-1" style={{ color: "var(--ink-400)" }}>
              {ALLOWED_EXTS.join(", ")} · max 20 MB
            </p>
          </div>

          {/* API key */}
          <div className="space-y-2">
            <label className="kicker">API key (optional — for LLM extraction)</label>
            <input
              type="password"
              className="input-warm w-full"
              placeholder="sk-…"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="select-warm w-full"
            >
              <option>gpt-4o</option>
              <option>gpt-4o-mini</option>
              <option>gpt-4-turbo</option>
              <option>claude-3-5-sonnet-20241022</option>
            </select>
            <p className="text-[10px]" style={{ color: "var(--ink-400)" }}>
              Without a key, regex extraction runs as fallback (lower confidence, no hallucination).
            </p>
          </div>

          {/* Extracted spec */}
          {result && (
            <div className="space-y-3">
              <p className="kicker">Extracted policy</p>
              <PolicyParamPanel
                name={result.spec.name}
                description={result.spec.description}
                severity={result.spec.severity}
                penaltyType={result.spec.penalty_type}
                penaltyCapUsd={result.spec.penalty_cap_usd}
                computeThresholdFlops={result.spec.compute_threshold_flops}
                enforcementMechanism={result.spec.enforcement_mechanism}
                gracePeriodMonths={result.spec.grace_period_months}
                scope={result.spec.scope}
                readOnly
              />
              <p className="text-[10px]" style={{ color: "var(--ink-400)" }}>
                {result.confidence_summary}
              </p>
            </div>
          )}

          {error && (
            <div
              className="rounded-xl border px-3 py-2.5 text-sm"
              style={{
                color:       "var(--danger)",
                borderColor: "rgba(220,38,38,0.2)",
                background:  "#fef2f2",
              }}
            >
              {error}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="flex-1 p-6 overflow-y-auto">
          {loading && <SimLoadingSkeleton />}

          {!loading && result && (
            <div className="space-y-6">
              {/* Traceability accordion */}
              {ext && (
                <div
                  className="card-warm overflow-hidden"
                >
                  <div
                    className="px-4 py-3"
                    style={{
                      background:  "var(--cream-200)",
                      borderBottom: "1px solid var(--border-warm)",
                    }}
                  >
                    <h3 className="kicker">
                      Traceability — {result.document_name}
                    </h3>
                  </div>
                  <div className="px-4 py-2">
                    {(
                      [
                        ["Policy name",       ext.policy_name],
                        ["Penalty type",      ext.penalty_type],
                        ["Penalty cap (USD)", ext.penalty_cap_usd],
                        ["Compute threshold", ext.compute_threshold_flops],
                        ["Enforcement",       ext.enforcement_mechanism],
                        ["Grace period",      ext.grace_period_months],
                        ["Scope",             ext.scope],
                        ["Jurisdiction",      ext.source_jurisdiction],
                        ["SME provisions",    ext.has_sme_provisions],
                        ["Frontier labs",     ext.has_frontier_lab_focus],
                      ] as [string, typeof ext.policy_name][]
                    ).map(([label, field]) => (
                      <TraceabilityRow
                        key={label}
                        label={label}
                        value={field.value}
                        confidence={field.confidence}
                        tier={field.epistemic_tag}
                        source={field.source_passage}
                      />
                    ))}
                  </div>
                </div>
              )}

              <ResultsPanel result={result.result} />
            </div>
          )}

          {!loading && !result && !error && (
            <div
              className="flex flex-col items-center justify-center h-full text-center gap-3"
              style={{ color: "var(--ink-400)" }}
            >
              <Upload size={48} className="opacity-30" />
              <p className="text-sm">
                Drop a policy document to extract parameters and run a simulation.
              </p>
              <p className="text-xs">
                Supports PDF, plain text, Markdown, and Word documents.
              </p>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
