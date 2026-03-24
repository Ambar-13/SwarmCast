"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Brain, AlertTriangle } from "lucide-react";
import type { SwarmResult, SwarmTypeResult, SwarmConfidence } from "@/lib/types";

// ── GDPR baseline months per type ────────────────────────────────────────────
const GDPR_MONTHS: Record<string, number> = {
  large_company: 10,
  mid_company: 21,
  startup: 33,
  frontier_lab: 4.5,
  investor: 15,
};

const TYPE_LABELS: Record<string, string> = {
  large_company: "Large / Frontier",
  mid_company: "Mid-Tier",
  startup: "Startups",
  investor: "Investors",
};

const CONFIDENCE_STYLES: Record<
  SwarmConfidence,
  { bg: string; text: string; border: string; label: string }
> = {
  high: {
    bg: "rgba(34,197,94,0.10)",
    text: "#16a34a",
    border: "rgba(34,197,94,0.25)",
    label: "High confidence — adjustments applied",
  },
  medium: {
    bg: "rgba(245,158,11,0.10)",
    text: "#d97706",
    border: "rgba(245,158,11,0.25)",
    label: "Medium confidence — adjustments applied",
  },
  low: {
    bg: "rgba(248,113,113,0.10)",
    text: "#dc2626",
    border: "rgba(248,113,113,0.20)",
    label: "Low confidence — fell back to GDPR baseline",
  },
};

function ConfidenceBadge({ confidence }: { confidence: SwarmConfidence }) {
  const s = CONFIDENCE_STYLES[confidence];
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-widest"
      style={{ background: s.bg, color: s.text, border: `1px solid ${s.border}` }}
    >
      {confidence}
    </span>
  );
}

function PressureBar({
  dist,
}: {
  dist: Record<string, number>;
}) {
  const total = (dist.low ?? 0) + (dist.medium ?? 0) + (dist.high ?? 0);
  if (total === 0) return null;
  const pLow = (dist.low ?? 0) / total;
  const pMed = (dist.medium ?? 0) / total;
  const pHigh = (dist.high ?? 0) / total;

  return (
    <div className="flex items-center gap-2">
      <div
        className="flex h-2 w-full overflow-hidden rounded-full"
        style={{ background: "var(--cream-300)" }}
      >
        <div style={{ width: `${pLow * 100}%`, background: "#22c55e" }} />
        <div style={{ width: `${pMed * 100}%`, background: "#f59e0b" }} />
        <div style={{ width: `${pHigh * 100}%`, background: "#ef4444" }} />
      </div>
      <span className="text-[10px] whitespace-nowrap" style={{ color: "var(--ink-400)" }}>
        {dist.high ?? 0}H / {dist.medium ?? 0}M / {dist.low ?? 0}L
      </span>
    </div>
  );
}

function TypeCard({ tr }: { tr: SwarmTypeResult }) {
  const [expanded, setExpanded] = useState(false);
  const style = CONFIDENCE_STYLES[tr.confidence];
  const gdprMonths = GDPR_MONTHS[tr.agent_type] ?? 20;
  const projMonths = gdprMonths * tr.mean_compliance_factor;
  const factorLabel =
    tr.mean_compliance_factor > 1
      ? `${tr.mean_compliance_factor.toFixed(2)}× longer`
      : tr.mean_compliance_factor < 1
        ? `${tr.mean_compliance_factor.toFixed(2)}× shorter`
        : "same as GDPR";
  const typeLabel = TYPE_LABELS[tr.agent_type] ?? tr.agent_type;

  return (
    <div
      className="rounded-xl border overflow-hidden"
      style={{ borderColor: "var(--border-warm)", background: "var(--cream-100)" }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-semibold truncate" style={{ color: "var(--ink-800)" }}>
            {typeLabel}
          </span>
          <span className="text-xs" style={{ color: "var(--ink-400)" }}>
            {tr.n_agents} agents
          </span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <ConfidenceBadge confidence={tr.confidence} />
        </div>
      </div>

      {/* Metrics grid */}
      <div
        className="grid grid-cols-2 gap-3 border-t px-4 py-3 sm:grid-cols-4"
        style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}
      >
        {/* Compliance factor */}
        <div>
          <p className="kicker text-[10px]">Compliance factor</p>
          <p
            className="metric-num mt-1 text-lg font-bold"
            style={{
              color:
                tr.mean_compliance_factor > 1.3
                  ? "#dc2626"
                  : tr.mean_compliance_factor < 0.8
                    ? "#16a34a"
                    : "var(--ink-800)",
            }}
          >
            {tr.mean_compliance_factor.toFixed(2)}×
          </p>
          <p className="text-[10px] mt-0.5" style={{ color: "var(--ink-400)" }}>
            {factorLabel}
          </p>
        </div>

        {/* Projected months */}
        <div>
          <p className="kicker text-[10px]">Projected compliance</p>
          <p className="metric-num mt-1 text-lg font-bold" style={{ color: "var(--ink-800)" }}>
            {projMonths.toFixed(0)} mo
          </p>
          <p className="text-[10px] mt-0.5" style={{ color: "var(--ink-400)" }}>
            vs GDPR {gdprMonths} mo
          </p>
        </div>

        {/* λ multiplier */}
        <div>
          <p className="kicker text-[10px]">λ adjustment</p>
          <p
            className="metric-num mt-1 text-lg font-bold"
            style={{
              color: tr.applied_lambda_multiplier !== 1.0 ? style.text : "var(--ink-400)",
            }}
          >
            {tr.applied_lambda_multiplier !== 1.0
              ? `${tr.applied_lambda_multiplier.toFixed(2)}×`
              : "—"}
          </p>
          <p className="text-[10px] mt-0.5" style={{ color: "var(--ink-400)" }}>
            compliance speed
          </p>
        </div>

        {/* Threshold shift */}
        <div>
          <p className="kicker text-[10px]">Reloc. threshold</p>
          <p
            className="metric-num mt-1 text-lg font-bold"
            style={{
              color: tr.applied_threshold_shift !== 0 ? style.text : "var(--ink-400)",
            }}
          >
            {tr.applied_threshold_shift !== 0
              ? `${tr.applied_threshold_shift > 0 ? "+" : ""}${tr.applied_threshold_shift.toFixed(1)}`
              : "—"}
          </p>
          <p className="text-[10px] mt-0.5" style={{ color: "var(--ink-400)" }}>
            burden units
          </p>
        </div>
      </div>

      {/* Relocation pressure bar + dominant action */}
      <div
        className="flex items-center gap-4 border-t px-4 py-2.5"
        style={{ borderColor: "var(--border-warm)" }}
      >
        <div className="flex-1 min-w-0">
          <p className="kicker text-[10px] mb-1">Relocation pressure</p>
          <PressureBar dist={tr.relocation_pressure_dist} />
        </div>
        <div className="flex-shrink-0 text-right">
          <p className="kicker text-[10px] mb-1">Dominant action</p>
          <span
            className="text-xs font-semibold capitalize"
            style={{ color: "var(--crimson-700)" }}
          >
            {tr.dominant_action}
          </span>
        </div>
      </div>

      {/* Collapsible agent list */}
      <div className="border-t" style={{ borderColor: "var(--border-warm)" }}>
        <button
          onClick={() => setExpanded((x) => !x)}
          className="flex w-full items-center gap-1.5 px-4 py-2 text-xs transition-colors"
          style={{ color: "var(--ink-400)" }}
        >
          {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          {expanded ? "Hide" : "Show"} agent responses
        </button>
        {expanded && (
          <div
            className="space-y-2 border-t px-4 pb-3 pt-2"
            style={{ borderColor: "var(--border-warm)" }}
          >
            {tr.agents.map((a, i) => (
              <div
                key={i}
                className="rounded-lg border px-3 py-2"
                style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}
              >
                <div className="flex items-center justify-between gap-2 mb-1">
                  <p
                    className="text-[11px] font-medium leading-tight"
                    style={{ color: "var(--ink-700)" }}
                  >
                    {a.persona}
                  </p>
                  <span
                    className="flex-shrink-0 text-[10px] font-semibold"
                    style={{ color: "var(--ink-500)" }}
                  >
                    {a.compliance_factor.toFixed(2)}× · {a.primary_action}
                  </span>
                </div>
                {a.reasoning && (
                  <p
                    className="text-[10px] leading-4"
                    style={{ color: "var(--ink-400)" }}
                  >
                    {a.reasoning}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function SwarmElicitationPanel({ swarm }: { swarm: SwarmResult }) {
  const [panelOpen, setPanelOpen] = useState(true);

  return (
    <div
      className="rounded-2xl border overflow-hidden"
      style={{ borderColor: "var(--border-warm)", background: "var(--cream-100)" }}
    >
      {/* Panel header */}
      <button
        onClick={() => setPanelOpen((x) => !x)}
        className="flex w-full items-center justify-between gap-3 px-5 py-4"
      >
        <div className="flex items-center gap-2">
          <Brain size={16} style={{ color: "var(--crimson-700)" }} />
          <span
            className="text-sm font-semibold"
            style={{ color: "var(--ink-800)" }}
          >
            Swarm Elicitation
          </span>
          {/* Epistemic badge */}
          <span
            className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest"
            style={{
              background: "rgba(245,158,11,0.12)",
              color: "#b45309",
              border: "1px solid rgba(245,158,11,0.3)",
            }}
          >
            SWARM-ELICITED
          </span>
          <span className="text-xs" style={{ color: "var(--ink-400)" }}>
            {swarm.n_total_agents} agents · {swarm.elapsed_seconds.toFixed(1)}s · {swarm.llm_model}
          </span>
        </div>
        {panelOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>

      {panelOpen && (
        <div
          className="border-t"
          style={{ borderColor: "var(--border-warm)" }}
        >
          {/* Epistemic warning */}
          <div
            className="flex items-start gap-2.5 px-5 py-3"
            style={{ background: "rgba(245,158,11,0.06)", borderBottom: "1px solid rgba(245,158,11,0.15)" }}
          >
            <AlertTriangle size={13} className="flex-shrink-0 mt-0.5" style={{ color: "#d97706" }} />
            <p className="text-[11px] leading-5" style={{ color: "#92400e" }}>
              <strong>Epistemic note:</strong> These are LLM priors filtered through training data — commentary about regulation, not measured firm behaviour. They adjust calibrated GDPR-fitted parameters within bounded ranges (±30% on λ, ±15 burden units). Low-confidence types fall back to GDPR baselines unchanged.
            </p>
          </div>

          {/* Type cards */}
          <div className="space-y-3 p-4">
            {swarm.type_results.map((tr) => (
              <TypeCard key={tr.agent_type} tr={tr} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
