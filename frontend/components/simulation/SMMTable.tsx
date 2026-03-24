import { EpistemicBadge } from "@/components/simulation/EpistemicBadge";
import { formatPct } from "@/lib/format";
import type { SimulatedMoments, EpistemicTier } from "@/lib/types";

const MOMENT_ROWS: {
  key: keyof Omit<SimulatedMoments, "n_runs">;
  label: string;
  tier: EpistemicTier;
}[] = [
  { key: "compliance_rate_y1", label: "Compliance rate (year 1)", tier: "GROUNDED" },
  { key: "relocation_rate", label: "Relocation rate", tier: "DIRECTIONAL" },
  { key: "lobbying_rate", label: "Lobbying rate", tier: "DIRECTIONAL" },
  { key: "sme_compliance_24mo", label: "SME compliance (24 months)", tier: "ASSUMED" },
  { key: "large_compliance_24mo", label: "Large firm compliance (24 months)", tier: "DIRECTIONAL" },
  { key: "enforcement_rate", label: "Enforcement action rate (year 1)", tier: "DIRECTIONAL" },
];

interface Props {
  moments: SimulatedMoments;
  distanceToGdpr: number | null;
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
            {MOMENT_ROWS.map(({ key, label, tier }) => (
              <tr
                key={key}
                className={`border-b last:border-0 ${tier === "ASSUMED" ? "opacity-65" : ""}`}
                style={{ borderColor: "var(--border-warm)" }}
              >
                <td className="px-4 py-3 text-sm" style={{ color: "var(--ink-500)" }}>{label}</td>
                <td className="metric-value px-4 py-3 text-right text-base font-semibold" style={{ color: "var(--ink-900)" }}>
                  {formatPct(moments[key] as number)}
                </td>
                <td className="px-4 py-3 text-right">
                  <EpistemicBadge tier={tier} small />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
