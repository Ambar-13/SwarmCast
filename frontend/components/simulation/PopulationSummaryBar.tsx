import { formatPct } from "@/lib/format";

interface Props {
  complianceRate: number;
  relocationRate: number;
  evasionRate: number;
  everLobbyedRate: number;
}

interface Segment {
  label: string;
  value: number;
  color: string;
  bg: string;
  border: string;
}

export function PopulationSummaryBar({
  complianceRate,
  relocationRate,
  evasionRate,
  everLobbyedRate,
}: Props) {
  const segments: Segment[] = [
    {
      label: "Compliant",
      value: complianceRate,
      color: "#15803D",
      bg: "#f0fdf4",
      border: "rgba(21,128,61,0.22)",
    },
    {
      label: "Relocated",
      value: relocationRate,
      color: "#C2410C",
      bg: "#fff7ed",
      border: "rgba(194,65,12,0.22)",
    },
    {
      label: "Evading",
      value: evasionRate,
      color: "#DC2626",
      bg: "#fef2f2",
      border: "rgba(220,38,38,0.22)",
    },
    {
      label: "Lobbied",
      value: everLobbyedRate,
      color: "#1E3A8A",
      bg: "#eff6ff",
      border: "rgba(30,58,138,0.22)",
    },
  ];

  return (
    <div className="card-warm p-5 space-y-4">
      <div>
        <p className="kicker">Population breakdown</p>
        <p className="mt-1.5 text-lg font-semibold" style={{ color: "var(--ink-900)" }}>
          End-state behaviour distribution
        </p>
      </div>

      {/* Stacked bar */}
      <div
        className="flex h-8 w-full overflow-hidden rounded-xl"
        style={{ background: "var(--cream-300)" }}
      >
        {segments.map((seg) =>
          seg.value > 0.001 ? (
            <div
              key={seg.label}
              className="h-full transition-all duration-700"
              style={{ width: `${seg.value * 100}%`, background: seg.color }}
              title={`${seg.label}: ${formatPct(seg.value)}`}
            />
          ) : null,
        )}
      </div>

      {/* Legend grid */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {segments.map((seg) => (
          <div
            key={seg.label}
            className="rounded-xl border px-3 py-2.5"
            style={{ background: seg.bg, borderColor: seg.border }}
          >
            <p
              className="text-[10px] font-semibold uppercase tracking-[0.18em]"
              style={{ color: seg.color }}
            >
              {seg.label}
            </p>
            <p
              className="metric-num mt-1 text-xl font-bold"
              style={{ color: seg.color }}
            >
              {formatPct(seg.value)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
