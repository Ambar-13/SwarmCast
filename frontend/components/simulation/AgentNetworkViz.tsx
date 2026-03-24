"use client";

import { useState } from "react";
import type { SwarmResult, SwarmAgentResponse } from "@/lib/types";

// ── Colours ───────────────────────────────────────────────────────────────────

const ACTION_COLOR: Record<string, string> = {
  comply:   "#22c55e",
  lobby:    "#8b5cf6",
  relocate: "#f97316",
  evade:    "#ef4444",
};

const ACTION_LABEL: Record<string, string> = {
  comply:   "Comply",
  lobby:    "Lobby",
  relocate: "Relocate",
  evade:    "Evade",
};

const TYPE_LABEL: Record<string, string> = {
  large_company: "Large Cos",
  mid_company:   "Mid-Tier",
  startup:       "Startups",
  investor:      "Investors",
  frontier_lab:  "Frontier Labs",
};

// ── SVG layout helpers ────────────────────────────────────────────────────────

const SVG_W = 560;
const SVG_H = 320;
const CX = SVG_W / 2;
const CY = SVG_H / 2;
const R_HUB  = 105; // type hub ring radius
const R_NODE =  52; // persona node radius from hub

function hubPos(idx: number, total: number) {
  const angle = -Math.PI / 2 + (2 * Math.PI * idx) / total;
  return { x: CX + R_HUB * Math.cos(angle), y: CY + R_HUB * Math.sin(angle), angle };
}

function agentPos(hubX: number, hubY: number, hubAngle: number, idx: number, count: number) {
  const spread = Math.PI * 0.7;
  const a = hubAngle - spread / 2 + (count > 1 ? (spread * idx) / (count - 1) : 0);
  return { x: hubX + R_NODE * Math.cos(a), y: hubY + R_NODE * Math.sin(a) };
}

// ── LCG helpers (identical in Node.js + browser — no Math.random) ─────────────
function lcgNext(s: number): number {
  return (Math.imul(1664525, s) + 1013904223) >>> 0;
}
function lcgShuffle<T>(arr: T[]): T[] {
  const out = [...arr];
  let s = 0x9e3779b9;
  for (let i = out.length - 1; i > 0; i--) {
    s = lcgNext(s);
    const j = s % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ── Persona network (tab A) ───────────────────────────────────────────────────

function PersonaNetwork({
  swarm,
}: {
  swarm: SwarmResult;
}) {
  const [selected, setSelected] = useState<SwarmAgentResponse | null>(null);

  const groups = swarm.type_results.filter((t) => t.agents?.length);

  const hubs = groups.map((g, i) => ({
    ...hubPos(i, groups.length),
    group: g,
    label: TYPE_LABEL[g.agent_type] ?? g.agent_type,
  }));

  const nodes = hubs.flatMap((hub, hi) =>
    (hub.group.agents ?? []).map((agent, ai) => ({
      agent,
      ...agentPos(hub.x, hub.y, hub.angle, ai, hub.group.agents.length),
      hi,
    }))
  );

  return (
    <div className="flex flex-col gap-4 lg:flex-row">
      {/* SVG */}
      <div
        className="min-w-0 flex-1 overflow-hidden rounded-2xl border"
        style={{ background: "var(--cream-100)", borderColor: "var(--border-warm)" }}
      >
        <svg
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          style={{ width: "100%", height: "auto", display: "block" }}
          aria-label="Swarm persona network"
        >
          {/* Center → hub edges */}
          {hubs.map((h, i) => (
            <line key={i} x1={CX} y1={CY} x2={h.x} y2={h.y}
              stroke="var(--border-strong)" strokeWidth={1} strokeOpacity={0.35} />
          ))}

          {/* Hub → agent edges */}
          {nodes.map((n, i) => (
            <line key={i}
              x1={hubs[n.hi].x} y1={hubs[n.hi].y} x2={n.x} y2={n.y}
              stroke="var(--border-warm)" strokeWidth={1} strokeOpacity={0.5} />
          ))}

          {/* Policy center */}
          <circle cx={CX} cy={CY} r={15} fill="var(--crimson-700)" />
          <text x={CX} y={CY + 25} textAnchor="middle"
            fontSize={8} fontWeight={700} fill="var(--crimson-700)"
            style={{ letterSpacing: "0.12em", textTransform: "uppercase", fontFamily: "var(--font-inter), sans-serif" }}>
            POLICY
          </text>

          {/* Type hubs */}
          {hubs.map((h, i) => (
            <g key={i}>
              <circle cx={h.x} cy={h.y} r={8}
                fill="var(--cream-300)" stroke="var(--border-strong)" strokeWidth={1.5} />
              <text x={h.x} y={h.y + 19} textAnchor="middle"
                fontSize={7.5} fontWeight={600} fill="var(--ink-400)"
                style={{ fontFamily: "var(--font-inter), sans-serif" }}>
                {h.label}
              </text>
            </g>
          ))}

          {/* Persona nodes */}
          {nodes.map((n, i) => {
            const color = ACTION_COLOR[n.agent.primary_action] ?? "#6b7280";
            const isActive = selected?.persona === n.agent.persona;
            return (
              <g key={i} style={{ cursor: "pointer" }}
                onClick={() => setSelected(isActive ? null : n.agent)}
              >
                {isActive && (
                  <circle cx={n.x} cy={n.y} r={11}
                    fill="none" stroke={color} strokeWidth={2} strokeOpacity={0.4} />
                )}
                <circle cx={n.x} cy={n.y} r={isActive ? 7.5 : 5.5}
                  fill={color} fillOpacity={isActive ? 1 : 0.7}
                  stroke={isActive ? "var(--cream-50)" : "none"}
                  strokeWidth={isActive ? 1.5 : 0}
                />
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 px-4 pb-3">
          {Object.entries(ACTION_LABEL).map(([key, label]) => (
            <div key={key} className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full flex-shrink-0"
                style={{ background: ACTION_COLOR[key] }} />
              <span className="text-[10px]" style={{ color: "var(--ink-400)" }}>{label}</span>
            </div>
          ))}
          <span className="ml-auto text-[10px]" style={{ color: "var(--ink-300)" }}>
            Click a node to read reasoning
          </span>
        </div>
      </div>

      {/* Reasoning side panel */}
      <div
        className="w-full rounded-2xl border p-4 lg:w-72 lg:flex-shrink-0"
        style={{
          background: "var(--cream-50)",
          borderColor: selected ? "var(--border-strong)" : "var(--border-warm)",
          transition: "border-color 200ms",
        }}
      >
        {selected ? (
          <div className="space-y-3" style={{ animation: "fadeIn 200ms ease both" }}>
            <div>
              <p className="kicker text-[9px]">
                {TYPE_LABEL[selected.agent_type] ?? selected.agent_type}
              </p>
              <h4 className="mt-1 text-sm font-bold" style={{ color: "var(--ink-900)" }}>
                {selected.persona}
              </h4>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <span
                  className="rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wide"
                  style={{
                    background: ACTION_COLOR[selected.primary_action] + "22",
                    color: ACTION_COLOR[selected.primary_action],
                  }}
                >
                  {ACTION_LABEL[selected.primary_action]}
                </span>
                <span className="text-[10px]" style={{ color: "var(--ink-400)" }}>
                  compliance ×{selected.compliance_factor.toFixed(2)}
                </span>
                <span
                  className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
                  style={{ background: "var(--cream-300)", color: "var(--ink-500)" }}
                >
                  {selected.relocation_pressure} reloc pressure
                </span>
              </div>
            </div>

            <div className="divider-warm" />

            <div>
              <p className="kicker text-[9px] mb-1.5">LLM reasoning</p>
              <p className="text-[11px] leading-[1.65]" style={{ color: "var(--ink-600)" }}>
                {selected.reasoning}
              </p>
            </div>

            <button
              onClick={() => setSelected(null)}
              className="text-xs"
              style={{ color: "var(--ink-300)" }}
            >
              ✕ dismiss
            </button>
          </div>
        ) : (
          <div className="flex h-full min-h-[160px] flex-col items-center justify-center gap-2 text-center">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
              stroke="var(--border-strong)" strokeWidth="1.5" strokeLinecap="round">
              <circle cx="12" cy="12" r="10"/>
              <circle cx="12" cy="8" r="1.5" fill="var(--border-strong)" stroke="none"/>
              <path d="M12 11v5"/>
            </svg>
            <p className="text-xs" style={{ color: "var(--ink-400)" }}>
              Select a node to read<br />the persona's reasoning
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Population dot cloud (tab B) ──────────────────────────────────────────────

const BEHAVIOR_LEGEND = [
  { key: "comply",   label: "Compliant",  color: ACTION_COLOR.comply   },
  { key: "lobby",    label: "Lobbying",   color: ACTION_COLOR.lobby    },
  { key: "relocate", label: "Relocated",  color: ACTION_COLOR.relocate },
  { key: "evade",    label: "Evading",    color: ACTION_COLOR.evade    },
];

function PopulationModel({
  pop,
  nPopulation,
}: {
  pop: Record<string, number>;
  nPopulation: number;
}) {
  const N = 400;
  const compliance  = pop["compliance_rate"]  ?? 0;
  const relocation  = pop["relocation_rate"]  ?? 0;
  const evasion     = pop["evasion_rate"]     ?? 0;
  // lobbying is anything not in the three above
  const lobbying    = Math.max(0, 1 - compliance - relocation - evasion);

  const nComply   = Math.round(compliance * N);
  const nRelocate = Math.round(relocation * N);
  const nEvade    = Math.round(evasion    * N);
  const nLobby    = N - nComply - nRelocate - nEvade;

  const dots: string[] = lcgShuffle([
    ...Array(nComply).fill(ACTION_COLOR.comply),
    ...Array(Math.max(0, nRelocate)).fill(ACTION_COLOR.relocate),
    ...Array(Math.max(0, nEvade)).fill(ACTION_COLOR.evade),
    ...Array(Math.max(0, nLobby)).fill(ACTION_COLOR.lobby),
  ]);

  return (
    <div className="space-y-4">
      {/* Dot cloud — full width, compact */}
      <div
        className="overflow-hidden rounded-2xl border p-4"
        style={{ background: "var(--cream-100)", borderColor: "var(--border-warm)" }}
      >
        <p className="kicker text-[10px] mb-3">
          {N} representative actors · scaled from {nPopulation.toLocaleString()}
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
          {dots.map((color, i) => (
            <span
              key={i}
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: color,
                opacity: 0.85,
              }}
            />
          ))}
        </div>
        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-4">
          {BEHAVIOR_LEGEND.map(({ key, label, color }) => (
            <div key={key} className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full flex-shrink-0" style={{ background: color }} />
              <span className="text-[10px] font-medium" style={{ color: "var(--ink-600)" }}>
                {label}
              </span>
              <span className="text-[10px]" style={{ color: "var(--ink-400)" }}>
                {((key === "comply" ? compliance : key === "relocate" ? relocation : key === "evade" ? evasion : lobbying) * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* How it works — below, 3-column on large screens */}
      <div
        className="rounded-2xl border p-4"
        style={{ background: "var(--cream-50)", borderColor: "var(--border-warm)" }}
      >
        <p className="kicker text-[9px] mb-3">How the model works</p>
        <div className="grid gap-4 sm:grid-cols-3 text-[11px] leading-[1.65]" style={{ color: "var(--ink-500)" }}>
          <p>
            Each simulated actor starts with a <strong style={{ color: "var(--ink-700)" }}>behavioral prior</strong> —
            a compliance threshold drawn from real-world distributions (GDPR 2020, OECD PMR).
            Swarm elicitation adjusts these priors using LLM personas calibrated to the specific policy.
          </p>
          <p>
            Each <strong style={{ color: "var(--ink-700)" }}>round</strong> (decision cycle),
            actors observe their neighbours' actions, receive enforcement signals,
            and weigh compliance costs against penalties. The Hegselmann-Krause opinion
            dynamics model governs how beliefs spread across the network.
          </p>
          <p>
            Actors with high relocation pressure and low domestic advantage
            <strong style={{ color: "var(--ink-700)" }}> exit the jurisdiction</strong>.
            Those at the compliance threshold may <strong style={{ color: "var(--ink-700)" }}>lobby</strong>
            {" "}to shift the regulatory baseline rather than comply outright.
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Public export ─────────────────────────────────────────────────────────────

interface Props {
  swarmResult: SwarmResult | null;
  pop: Record<string, number>;
  nPopulation: number;
}

export function AgentNetworkViz({ swarmResult, pop, nPopulation }: Props) {
  const hasSwarm = !!(swarmResult?.type_results?.some((t) => t.agents?.length));
  const [tab, setTab] = useState<"personas" | "population">(
    hasSwarm ? "personas" : "population"
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="section-kicker">Agent network</p>
          <h3 className="mt-2 text-2xl font-semibold" style={{ color: "var(--ink-900)" }}>
            Simulated actors
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-6" style={{ color: "var(--ink-400)" }}>
            {tab === "personas"
              ? "Each node is a named LLM persona. Click to read their full reasoning on this policy."
              : "Each dot is one simulated actor. Color shows their end-state behaviour."}
          </p>
        </div>

        {/* Tab pills */}
        <div className="flex gap-2">
          {(
            [
              { key: "personas",   label: "Swarm personas",  disabled: !hasSwarm },
              { key: "population", label: "Population model", disabled: false     },
            ] as { key: "personas" | "population"; label: string; disabled: boolean }[]
          ).map(({ key, label, disabled }) => (
            <button
              key={key}
              onClick={() => !disabled && setTab(key)}
              disabled={disabled}
              className="rounded-full px-4 py-1.5 text-xs font-semibold transition-all"
              style={
                disabled
                  ? { background: "var(--cream-200)", color: "var(--ink-300)", cursor: "not-allowed" }
                  : tab === key
                  ? { background: "var(--ink-900)", color: "var(--cream-50)" }
                  : { background: "var(--cream-300)", color: "var(--ink-500)" }
              }
            >
              {label}
              {disabled && <span className="ml-1 opacity-60">(run with swarm)</span>}
            </button>
          ))}
        </div>
      </div>

      {tab === "personas" && hasSwarm && (
        <PersonaNetwork swarm={swarmResult!} />
      )}

      {tab === "population" && (
        <PopulationModel pop={pop} nPopulation={nPopulation} />
      )}
    </div>
  );
}
