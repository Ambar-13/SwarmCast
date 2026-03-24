import { AnimatedCounter } from "@/components/ui/AnimatedCounter";

interface Props {
  networkStatistics: Record<string, number>;
  networkHubs: Record<string, unknown>[];
}

export function NetworkStatsPanel({ networkStatistics, networkHubs }: Props) {
  const meanDegree  = networkStatistics.mean_degree            ?? 0;
  const clustering  = networkStatistics.clustering_coefficient ?? 0;
  const density     = networkStatistics.density                ?? 0;

  const stats = [
    { label: "Mean degree",  value: meanDegree, fmt: (v: number) => v.toFixed(1)  },
    { label: "Clustering",   value: clustering, fmt: (v: number) => v.toFixed(3)  },
    { label: "Density",      value: density,    fmt: (v: number) => v.toFixed(3)  },
  ];

  return (
    <div className="card-warm p-5 space-y-4">
      <div>
        <p className="kicker">Network topology</p>
        <p className="mt-1.5 text-lg font-semibold" style={{ color: "var(--ink-900)" }}>
          Influence graph structure
        </p>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {stats.map(({ label, value, fmt }) => (
          <div
            key={label}
            className="rounded-xl border px-3 py-3 text-center"
            style={{ background: "var(--cream-200)", borderColor: "var(--border-warm)" }}
          >
            <p className="kicker text-[10px]">{label}</p>
            <p
              className="metric-num mt-2 text-2xl font-bold"
              style={{ color: "var(--ink-900)" }}
            >
              <AnimatedCounter value={value} formatter={fmt} />
            </p>
          </div>
        ))}
      </div>

      {networkHubs.length > 0 && (
        <div>
          <p className="kicker mb-2">Top hubs</p>
          <div className="space-y-1">
            {networkHubs.slice(0, 3).map((hub, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded-lg border px-3 py-2 text-sm"
                style={{ borderColor: "var(--border-warm)", background: "var(--cream-100)" }}
              >
                <span style={{ color: "var(--ink-700)" }}>
                  Hub {String(hub.node ?? i)}
                </span>
                <span
                  className="metric-num font-semibold"
                  style={{ color: "var(--ink-500)" }}
                >
                  deg {String(hub.degree ?? "—")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
