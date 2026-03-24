"use client";

import { useState } from "react";
import { EpistemicBadge } from "@/components/simulation/EpistemicBadge";
import { formatPct } from "@/lib/format";
import type { SimulatedMoments, EpistemicTier } from "@/lib/types";

const MOMENT_ROWS: {
  key: keyof Omit<SimulatedMoments, "n_runs">;
  label: string;
  tier: EpistemicTier;
  note?: string;
}[] = [
  { key: "compliance_rate_y1",   label: "Compliance rate (year 1)",          tier: "GROUNDED",     note: "Mean compliance_rate averaged over rounds 1–4. Higher than the round-1 snapshot in the chart because early adopters join during the year." },
  { key: "relocation_rate",      label: "Relocation rate",                   tier: "DIRECTIONAL"  },
  { key: "lobbying_rate",        label: "Lobbying rate",                     tier: "DIRECTIONAL"  },
  { key: "sme_compliance_24mo",  label: "SME compliance (24 months)",        tier: "ASSUMED",      note: "Mean compliance_rate for SME-type agents averaged over rounds 1–8 (2 years of quarterly cycles)." },
  { key: "large_compliance_24mo",label: "Large firm compliance (24 months)", tier: "DIRECTIONAL",  note: "Mean compliance_rate for large-company agents averaged over rounds 1–8." },
  { key: "enforcement_rate",     label: "Enforcement action rate (year 1)",  tier: "DIRECTIONAL",  note: "Mean enforcement_contact_rate averaged over rounds 1–4." },
];

interface Props {
  moments: SimulatedMoments;
  distanceToGdpr: number | null;
}

function InfoButton({ note }: { note: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 16,
          height: 16,
          borderRadius: "50%",
          border: "1.5px solid var(--border-warm)",
          background: open ? "var(--cream-300)" : "var(--cream-200)",
          color: "var(--ink-500)",
          fontSize: 10,
          fontWeight: 700,
          fontStyle: "normal",
          fontFamily: "serif",
          cursor: "help",
          flexShrink: 0,
          lineHeight: 1,
          userSelect: "none",
          transition: "background 0.12s",
        }}
      >
        i
      </span>
      {open && (
        <span
          style={{
            position: "absolute",
            bottom: "calc(100% + 6px)",
            left: "50%",
            transform: "translateX(-50%)",
            width: 220,
            background: "var(--cream-50, #faf8f4)",
            border: "1px solid var(--border-warm)",
            borderRadius: 10,
            padding: "8px 10px",
            fontSize: 11,
            lineHeight: "1.55",
            color: "var(--ink-600, #4b5563)",
            boxShadow: "0 4px 16px rgba(0,0,0,0.10)",
            zIndex: 50,
            pointerEvents: "none",
            whiteSpace: "normal",
          }}
        >
          {note}
        </span>
      )}
    </span>
  );
}

export function SMMTable({ moments, distanceToGdpr }: Props) {
  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="section-kicker">Calibration moments</p>
          <h3 className="mt-2 text-2xl font-semibold" style={{ color: "var(--ink-900)" }}>Simulated moments</h3>
          <p className="mt-2 max-w-xl text-sm leading-6 text-[var(--ink-400)]">
            Compare the output moments and their epistemic footing instead of burying confidence
            assumptions in a footnote.
          </p>
        </div>
        {distanceToGdpr !== null && (
          <div className="card-warm px-4 py-3">
            <p className="kicker text-[10px]">SMM distance</p>
            <p className="metric-num mt-1 text-xl font-semibold" style={{ color: "var(--ink-900)" }}>
              {distanceToGdpr.toFixed(3)}
            </p>
          </div>
        )}
      </div>

      <div className="overflow-hidden rounded-2xl border" style={{ borderColor: "var(--border-warm)", background: "var(--cream-100)" }}>
        <table className="w-full text-sm">
          <thead style={{ background: "var(--cream-200)" }}>
            <tr className="border-b" style={{ borderColor: "var(--border-warm)" }}>
              <th className="px-4 py-3 text-left text-[10px] font-medium uppercase tracking-[0.22em] text-[var(--ink-500)]">
                Moment
              </th>
              <th className="px-4 py-3 text-right text-[10px] font-medium uppercase tracking-[0.22em] text-[var(--ink-500)]">
                Simulated
              </th>
              <th className="px-4 py-3 text-right text-[10px] font-medium uppercase tracking-[0.22em] text-[var(--ink-500)]">
                Tier
              </th>
            </tr>
          </thead>
          <tbody>
            {MOMENT_ROWS.map(({ key, label, tier, note }) => (
              <tr
                key={key}
                className="border-b last:border-0"
                style={{ borderColor: "var(--border-warm)" }}
              >
                <td className="px-4 py-3 text-sm" style={{ color: tier === "ASSUMED" ? "var(--ink-300)" : "var(--ink-500)" }}>
                  <span className="inline-flex items-center gap-1.5">
                    {label}
                    {note && <InfoButton note={note} />}
                  </span>
                </td>
                <td className="metric-value px-4 py-3 text-right text-base font-semibold" style={{ color: tier === "ASSUMED" ? "var(--ink-400)" : "var(--ink-900)" }}>
                  {formatPct(moments[key] as number)}
                </td>
                <td className="px-4 py-3 text-right">
                  <EpistemicBadge tier={tier} small />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div
          className="border-t px-4 py-3 text-xs leading-5"
          style={{ borderColor: "var(--border-warm)", color: "var(--ink-400)" }}
        >
          <span style={{ color: "#15803d", fontWeight: 600 }}>GROUNDED</span> — directly calibrated against empirical data (GDPR year-1 compliance from DLA Piper 2020 survey, n=200 firms).<br />
          <span style={{ color: "#b45309", fontWeight: 600 }}>DIRECTIONAL</span> — model output points in the empirically documented direction but no direct calibration target exists for AI governance.<br />
          <span style={{ color: "#b91c1c", fontWeight: 600 }}>ASSUMED</span> — structural assumption with no comparable empirical base; treat as scenario input, not prediction.<br />
          Hover the ⓘ next to a moment label to see its exact measurement window. "Year 1" = mean over rounds 1–4; the trajectory chart shows per-round snapshots, so those values will differ.
        </div>
      </div>
    </div>
  );
}
