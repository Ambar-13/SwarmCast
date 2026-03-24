export function SimLoadingSkeleton() {
  return (
    <div
      className="card-warm flex min-h-[600px] flex-col gap-5 p-6 md:p-8"
      style={{ animation: "slideUpFade 200ms ease both" }}
    >
      <div className="space-y-5">
        {/* Header */}
        <div className="space-y-2.5">
          <div
            className="h-3 w-20 rounded-full"
            style={{ background: "var(--cream-300)" }}
          />
          <div
            className="h-7 w-64 rounded-xl"
            style={{ background: "var(--cream-300)" }}
          />
          <div
            className="h-4 w-full max-w-sm rounded-full"
            style={{ background: "var(--cream-200)" }}
          />
          <div
            className="h-4 w-3/4 max-w-xs rounded-full"
            style={{ background: "var(--cream-200)" }}
          />
        </div>

        {/* Stat cards row */}
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="h-28 rounded-2xl"
              style={{ background: "var(--cream-200)" }}
            />
          ))}
        </div>

        {/* Stacked bar skeleton */}
        <div
          className="h-20 rounded-2xl"
          style={{ background: "var(--cream-200)" }}
        />

        {/* Charts row */}
        <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
          <div
            className="h-80 rounded-2xl"
            style={{ background: "var(--cream-200)" }}
          />
          <div
            className="h-80 rounded-2xl"
            style={{ background: "var(--cream-200)" }}
          />
        </div>
      </div>

      <p className="text-center text-sm" style={{ color: "var(--ink-400)" }}>
        Simulating scenario dynamics…
      </p>
    </div>
  );
}
