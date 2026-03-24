"use client";

import { useEffect, useState } from "react";

// ── Deterministic LCG — pure integer arithmetic, identical on server + client ─

function lcgNext(s: number): number {
  return (Math.imul(1664525, s) + 1013904223) >>> 0;
}
function lcgSeq(seed: number, n: number): number[] {
  const out: number[] = [];
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) { s = lcgNext(s); out.push(s); }
  return out;
}

// ── Network layout — built once at module scope (no hydration drift) ──────────

const W = 480;
const H = 300;
const N_NODES = 52;
const MARGIN = 30;

type AgentState = "idle" | "comply" | "relocate" | "lobby" | "evade";

interface NetworkNode {
  id: number;
  x: number;
  y: number;
  r: number;
  state: AgentState;
  activateRound: number;
}
interface NetworkEdge { a: number; b: number; }

const STATE_COLOR: Record<AgentState, string> = {
  comply:   "#22c55e",
  relocate: "#f97316",
  lobby:    "#a855f7",
  evade:    "#ef4444",
  idle:     "#cbd5e1",
};

const STATES: AgentState[] = [
  "comply", "comply", "comply", "comply",
  "lobby",  "lobby",
  "relocate", "relocate",
  "evade",
  "idle",
];

const { NODES, EDGES } = (() => {
  const seq = lcgSeq(0xc0ffee42, N_NODES * 3 + 60);
  let si = 0;
  const r = () => seq[si++] ?? 0;

  const nodes: NetworkNode[] = [];

  // 6 hub nodes clustered in the centre region
  const hubCx = [220, 265, 195, 290, 245, 215];
  const hubCy = [148, 138, 112, 168, 178, 155];
  for (let i = 0; i < 6; i++) {
    nodes.push({
      id: i, x: hubCx[i], y: hubCy[i], r: 6,
      state: STATES[i % STATES.length],
      activateRound: 1 + (i % 3),
    });
  }

  // Remaining nodes scattered across canvas
  for (let i = 6; i < N_NODES; i++) {
    const x = MARGIN + (r() % (W - MARGIN * 2));
    const y = MARGIN + (r() % (H - MARGIN * 2));
    nodes.push({
      id: i, x, y, r: i < 16 ? 4.5 : 3.5,
      state: STATES[i % STATES.length],
      activateRound: 1 + Math.floor(((i - 6) / (N_NODES - 6)) * 13),
    });
  }

  // Edges: k-nearest neighbours (integer distance² avoids float drift)
  const edges: NetworkEdge[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < N_NODES; i++) {
    const ni = nodes[i];
    const dists = nodes
      .map((nj, j) => ({ j, d2: Math.round((ni.x - nj.x) ** 2 + (ni.y - nj.y) ** 2) }))
      .filter((o) => o.j !== i)
      .sort((a, b) => a.d2 - b.d2);

    const k = i < 6 ? 6 : 3;
    for (let ei = 0; ei < k && ei < dists.length; ei++) {
      const j = dists[ei].j;
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (!seen.has(key)) { seen.add(key); edges.push({ a: i, b: j }); }
    }
  }

  return { NODES: nodes, EDGES: edges };
})();

// ── Stage labels ──────────────────────────────────────────────────────────────

const SIMULATE_STAGES = [
  "Seeding agents across jurisdiction…",
  "Building influence network (scale-free)…",
  "Computing initial compliance states…",
  "Propagating peer influence…",
  "Running decision rounds…",
  "Measuring relocation pressure…",
  "Propagating network effects…",
  "Computing simulated moments…",
  "Calibrating GDPR distance…",
  "Finalizing statistics…",
];

const UPLOAD_STAGES = [
  "Parsing document structure…",
  "Extracting policy parameters…",
  "Running LLM provision analysis…",
  "Resolving confidence scores…",
  "Seeding simulation population…",
  "Running decision rounds…",
  "Computing final statistics…",
  "Finalizing…",
];

// ── Props & Component ─────────────────────────────────────────────────────────

export interface SimLoadingSkeletonProps {
  mode?: "simulate" | "upload";
  numRounds?: number;
  nPopulation?: number;
}

export function SimLoadingSkeleton({
  mode = "simulate",
  numRounds = 16,
  nPopulation = 1000,
}: SimLoadingSkeletonProps) {
  const [currentRound, setCurrentRound] = useState(0);
  const [fakePercent, setFakePercent] = useState(0);

  useEffect(() => {
    if (mode === "simulate") {
      const msPerRound = Math.max(80, 2800 / numRounds);
      const t = setInterval(() => {
        setCurrentRound((r) => { if (r >= numRounds) { clearInterval(t); return r; } return r + 1; });
      }, msPerRound);
      return () => clearInterval(t);
    } else {
      const t = setInterval(() => {
        setFakePercent((p) => {
          if (p >= 93) { clearInterval(t); return p; }
          const step = p < 60 ? 1.2 : p < 80 ? 0.7 : 0.25;
          return Math.min(93, p + step);
        });
      }, 85);
      return () => clearInterval(t);
    }
  }, [mode, numRounds]);

  const progress =
    mode === "simulate" ? currentRound / numRounds : fakePercent / 100;

  const stages = mode === "simulate" ? SIMULATE_STAGES : UPLOAD_STAGES;
  const stageLabel = stages[Math.min(Math.floor(progress * stages.length), stages.length - 1)];

  // Which nodes are active this round
  const activeThreshold =
    mode === "simulate" ? currentRound : Math.round(progress * 13) + 1;
  const activeIds = new Set(
    NODES.filter((n) => n.activateRound <= activeThreshold).map((n) => n.id)
  );

  return (
    <div
      className="card-warm flex min-h-[580px] flex-col items-center justify-center gap-5 p-6"
      style={{ animation: "fadeIn 220ms ease both" }}
    >
      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="text-center space-y-0.5">
        <p
          className="text-[11px] font-bold uppercase tracking-widest"
          style={{ color: "var(--crimson-700)", letterSpacing: "0.14em" }}
        >
          {mode === "simulate" ? "Simulation running" : "Extracting policy"}
        </p>
        {mode === "simulate" && (
          <p className="metric-num text-xs tabular-nums" style={{ color: "var(--ink-400)" }}>
            ROUND {String(currentRound).padStart(2, "0")} / {String(numRounds).padStart(2, "0")}
          </p>
        )}
      </div>

      {/* ── Agent network graph ─────────────────────────────────── */}
      <div
        className="w-full overflow-hidden rounded-2xl"
        style={{
          background:   "var(--cream-50)",
          border:       "1px solid var(--border-warm)",
          maxWidth:     540,
        }}
      >
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
          {/* Subtle grid */}
          <defs>
            <pattern id="sg" width="24" height="24" patternUnits="userSpaceOnUse">
              <path d="M24 0L0 0 0 24" fill="none" stroke="var(--cream-300)" strokeWidth="0.4" />
            </pattern>
          </defs>
          <rect width={W} height={H} fill="url(#sg)" />

          {/* Dormant edges */}
          {EDGES.map((e, i) => (
            <line
              key={`d-${i}`}
              x1={NODES[e.a].x} y1={NODES[e.a].y}
              x2={NODES[e.b].x} y2={NODES[e.b].y}
              stroke="var(--cream-300)"
              strokeWidth="1"
              opacity={0.6}
            />
          ))}

          {/* Live edges — both endpoints active */}
          {EDGES.filter((e) => activeIds.has(e.a) && activeIds.has(e.b)).map((e, i) => {
            const ca = STATE_COLOR[NODES[e.a].state];
            const cb = STATE_COLOR[NODES[e.b].state];
            return (
              <line
                key={`l-${i}`}
                x1={NODES[e.a].x} y1={NODES[e.a].y}
                x2={NODES[e.b].x} y2={NODES[e.b].y}
                stroke={ca === cb ? ca : "#94a3b8"}
                strokeWidth="1.5"
                opacity={0.6}
                style={{ transition: "opacity 0.5s ease" }}
              />
            );
          })}

          {/* Nodes */}
          {NODES.map((n) => {
            const active = activeIds.has(n.id);
            const col = active ? STATE_COLOR[n.state] : "var(--cream-300)";
            return (
              <g key={n.id}>
                {active && (
                  <circle
                    cx={n.x} cy={n.y}
                    r={n.r + 5}
                    fill={col}
                    opacity={0.15}
                    style={{ transition: "all 0.5s ease" }}
                  />
                )}
                <circle
                  cx={n.x} cy={n.y}
                  r={active ? n.r : n.r * 0.7}
                  fill={col}
                  style={{
                    transition: "fill 0.45s ease",
                    filter: active ? `drop-shadow(0 0 3px ${col})` : "none",
                  }}
                />
              </g>
            );
          })}

          {/* Activation pulse ring radiating from the network centre */}
          {currentRound > 0 && mode === "simulate" && (
            <circle
              key={`pulse-${currentRound}`}
              cx={W / 2} cy={H / 2}
              r={18}
              fill="none"
              stroke="var(--crimson-700)"
              strokeWidth="1.5"
              opacity={0.35}
              style={{
                transformOrigin: `${W / 2}px ${H / 2}px`,
                animation: "pulseRing 1.1s ease-out both",
              }}
            />
          )}

          {/* Agent count watermark */}
          <text
            x={W - 8} y={H - 8}
            textAnchor="end"
            fontSize={9}
            fill="var(--ink-300)"
            fontFamily="monospace"
          >
            {activeIds.size}/{N_NODES} active
          </text>
        </svg>
      </div>

      {/* ── Progress bar ────────────────────────────────────────── */}
      <div className="w-full max-w-sm space-y-2">
        <div
          className="h-1.5 w-full overflow-hidden rounded-full"
          style={{ background: "var(--cream-300)" }}
        >
          <div
            className="h-full rounded-full"
            style={{
              width:      `${Math.round(progress * 100)}%`,
              background: "linear-gradient(90deg, var(--crimson-900), var(--crimson-600))",
              transition: "width 0.35s ease",
              boxShadow:  "0 0 6px rgba(180,30,40,0.4)",
            }}
          />
        </div>
        <p
          className="text-center text-xs"
          key={stageLabel}
          style={{ color: "var(--ink-400)", animation: "slideUpFade 280ms ease both" }}
        >
          {stageLabel}
        </p>
      </div>

      {/* ── Config footer ────────────────────────────────────────── */}
      <div className="flex items-center gap-5">
        {[
          [mode === "simulate" ? "Agents" : "Mode",
           mode === "simulate" ? nPopulation.toLocaleString() : "LLM + Sim"],
          ["Rounds", mode === "simulate" ? String(numRounds) : "—"],
          ["Jurisdiction", "EU"],
        ].map(([label, value]) => (
          <div key={label} className="text-center">
            <p className="kicker text-[9px]">{label}</p>
            <p className="metric-num text-sm font-semibold" style={{ color: "var(--ink-700)" }}>
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* ── Legend ───────────────────────────────────────────────── */}
      <div className="flex items-center gap-4">
        {(
          [["#22c55e","Comply"],["#a855f7","Lobby"],["#f97316","Relocate"],["#ef4444","Evade"]] as const
        ).map(([color, label]) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full" style={{ background: color }} />
            <span className="text-[10px]" style={{ color: "var(--ink-400)" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
