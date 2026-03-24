"use client";

/**
 * Animated SVG illustrating an agent network in its "ready" state.
 * Used as the empty state before a simulation is run on /analyze.
 *
 * Purely decorative — no data. Nodes pulse, edges animate a travelling dash,
 * and one hub node glows crimson (the "policy" node) to anchor the eye.
 */

import type { CSSProperties } from "react";

// Fixed node layout (normalized 0–1 space, rendered in 320×220 viewport)
const W = 320;
const H = 220;

interface Node {
  id: string;
  x: number; // 0–1
  y: number; // 0–1
  r: number; // radius
  kind: "hub" | "large" | "mid" | "small";
  label?: string;
  delay: number; // animation delay in seconds
}

interface Edge {
  from: string;
  to: string;
  delay: number;
}

const NODES: Node[] = [
  // Hub — the policy node
  { id: "policy", x: 0.50, y: 0.42, r: 11, kind: "hub",   label: "Policy", delay: 0 },
  // Large companies
  { id: "lc1",    x: 0.20, y: 0.20, r: 7,  kind: "large",              delay: 0.1 },
  { id: "lc2",    x: 0.78, y: 0.18, r: 7,  kind: "large",              delay: 0.2 },
  // Mid companies
  { id: "mc1",    x: 0.12, y: 0.62, r: 5,  kind: "mid",                delay: 0.15 },
  { id: "mc2",    x: 0.36, y: 0.78, r: 5,  kind: "mid",                delay: 0.25 },
  { id: "mc3",    x: 0.65, y: 0.75, r: 5,  kind: "mid",                delay: 0.3 },
  { id: "mc4",    x: 0.88, y: 0.55, r: 5,  kind: "mid",                delay: 0.35 },
  // Small nodes
  { id: "s1",     x: 0.08, y: 0.35, r: 3.5, kind: "small",             delay: 0.05 },
  { id: "s2",     x: 0.32, y: 0.12, r: 3.5, kind: "small",             delay: 0.12 },
  { id: "s3",     x: 0.62, y: 0.10, r: 3.5, kind: "small",             delay: 0.18 },
  { id: "s4",     x: 0.92, y: 0.30, r: 3.5, kind: "small",             delay: 0.22 },
  { id: "s5",     x: 0.82, y: 0.82, r: 3.5, kind: "small",             delay: 0.28 },
  { id: "s6",     x: 0.18, y: 0.85, r: 3.5, kind: "small",             delay: 0.32 },
];

const EDGES: Edge[] = [
  { from: "policy", to: "lc1",  delay: 0    },
  { from: "policy", to: "lc2",  delay: 0.1  },
  { from: "policy", to: "mc1",  delay: 0.2  },
  { from: "policy", to: "mc3",  delay: 0.15 },
  { from: "policy", to: "mc4",  delay: 0.25 },
  { from: "lc1",    to: "s1",   delay: 0.3  },
  { from: "lc1",    to: "s2",   delay: 0.35 },
  { from: "lc1",    to: "mc2",  delay: 0.4  },
  { from: "lc2",    to: "s3",   delay: 0.3  },
  { from: "lc2",    to: "s4",   delay: 0.35 },
  { from: "mc1",    to: "s6",   delay: 0.45 },
  { from: "mc3",    to: "mc2",  delay: 0.5  },
  { from: "mc3",    to: "s5",   delay: 0.4  },
  { from: "mc4",    to: "s5",   delay: 0.45 },
  { from: "mc4",    to: "s4",   delay: 0.5  },
];

const NODE_MAP = new Map(NODES.map((n) => [n.id, n]));

function px(n: Node) { return n.x * W; }
function py(n: Node) { return n.y * H; }

function edgeLength(e: Edge): number {
  const a = NODE_MAP.get(e.from)!;
  const b = NODE_MAP.get(e.to)!;
  // Round to 2dp so server and client render identical strings.
  return Math.round(Math.hypot(px(b) - px(a), py(b) - py(a)) * 100) / 100;
}

const KIND_FILL: Record<Node["kind"], string> = {
  hub:   "var(--crimson-700)",
  large: "var(--cream-300)",
  mid:   "var(--cream-300)",
  small: "var(--cream-200)",
};

const KIND_STROKE: Record<Node["kind"], string> = {
  hub:   "var(--crimson-600)",
  large: "var(--border-strong)",
  mid:   "var(--border-warm)",
  small: "var(--border-warm)",
};

export function AgentNetworkIdle({ style }: { style?: CSSProperties }) {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      style={{ maxWidth: 360, display: "block", overflow: "visible", ...style }}
      aria-hidden
    >
      <defs>
        {/* Glow for hub node */}
        <filter id="hub-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3.5" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>

        {/* Subtle drop shadow for large nodes */}
        <filter id="node-shadow" x="-40%" y="-40%" width="180%" height="180%">
          <feDropShadow dx="0" dy="1" stdDeviation="2" floodColor="rgba(26,18,8,0.12)" />
        </filter>

        {/* All keyframes in a single <style> — multiple insertions each force a full style recalc */}
        <style>{`
          @keyframes pulse-node{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.12);opacity:.85}}
          @keyframes pulse-halo{0%,100%{transform:scale(1);opacity:.18}50%{transform:scale(1.4);opacity:.05}}
          ${EDGES.map((e) => {
            const len  = edgeLength(e);
            const dash = Math.max(len * 0.22, 8);
            return `@keyframes dash-${e.from}-${e.to}{from{stroke-dashoffset:${len + dash}}to{stroke-dashoffset:0}}`;
          }).join("")}
        `}</style>
      </defs>

      {/* ── Edges ─────────────────────────────────────────────── */}
      {EDGES.map((e) => {
        const a = NODE_MAP.get(e.from)!;
        const b = NODE_MAP.get(e.to)!;
        const len  = edgeLength(e);
        const dash = Math.max(len * 0.22, 8);
        const gap  = len - dash;
        return (
          <line
            key={`${e.from}-${e.to}`}
            x1={px(a)} y1={py(a)}
            x2={px(b)} y2={py(b)}
            stroke="var(--border-strong)"
            strokeWidth={1}
            strokeOpacity={0.55}
            strokeDasharray={`${dash} ${gap}`}
            style={{
              animation: `dash-${e.from}-${e.to} 3.2s ${e.delay + 0.4}s linear infinite`,
            }}
          />
        );
      })}

      {/* ── Nodes ─────────────────────────────────────────────── */}
      {NODES.map((n) => {
        const cx = px(n);
        const cy = py(n);
        const isHub = n.kind === "hub";
        const pulseStyle: CSSProperties = {
          transformOrigin: `${cx}px ${cy}px`,
          animation: `pulse-node ${isHub ? 2.0 : 2.8}s ${n.delay}s ease-in-out infinite`,
        };

        return (
          <g key={n.id}>
            {/* Halo ring behind hub */}
            {isHub && (
              <circle
                cx={cx} cy={cy}
                r={n.r + 7}
                fill="none"
                stroke="var(--crimson-700)"
                strokeWidth={1}
                strokeOpacity={0.18}
                style={{
                  transformOrigin: `${cx}px ${cy}px`,
                  animation: `pulse-halo 2.4s ${n.delay}s ease-in-out infinite`,
                }}
              />
            )}
            <circle
              cx={cx} cy={cy}
              r={n.r}
              fill={KIND_FILL[n.kind]}
              stroke={KIND_STROKE[n.kind]}
              strokeWidth={isHub ? 1.5 : 1}
              filter={isHub ? "url(#hub-glow)" : n.kind === "large" ? "url(#node-shadow)" : undefined}
              style={pulseStyle}
            />
            {/* Hub label */}
            {isHub && (
              <text
                x={cx} y={cy + n.r + 12}
                textAnchor="middle"
                fontSize={9}
                fontWeight={600}
                letterSpacing="0.12em"
                fill="var(--crimson-700)"
                style={{ textTransform: "uppercase", fontFamily: "var(--font-inter), sans-serif" }}
              >
                Policy
              </text>
            )}
          </g>
        );
      })}

    </svg>
  );
}
