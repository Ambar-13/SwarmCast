"use client";

import { useMemo } from "react";

interface Props {
  jurisdictionSummary: Record<string, { company_count?: number; burden?: number }>;
  sourceJurisdiction?: string;
}

interface DestNode {
  label: string;
  count: number;
  x: number;
  y: number;
  r: number;
  color: string;
}

const DEST_COLORS: Record<string, string> = {
  US:        "#1E3A8A",
  UK:        "#15803D",
  Singapore: "#B45309",
  UAE:       "#6B21A8",
  EU:        "#8B1A1A",
};
const DEFAULT_COLOR = "#6B5744";

export function JurisdictionFlowDiagram({
  jurisdictionSummary,
  sourceJurisdiction = "EU",
}: Props) {
  const destinations = useMemo(() => {
    return Object.entries(jurisdictionSummary)
      .map(([key, val]) => ({
        label: key
          .replace(/^dest_/, "")
          .replace(/_companies$/, "")
          .replace(/_/g, " "),
        count: val.company_count ?? 0,
      }))
      .filter((d) => d.count > 0 && d.label !== sourceJurisdiction)
      .sort((a, b) => b.count - a.count);
  }, [jurisdictionSummary, sourceJurisdiction]);

  if (destinations.length === 0) return null;

  const W       = 480;
  const padding = 48;
  const H       = Math.max(200, destinations.length * 70 + padding * 2);
  const srcX    = 72;
  const srcY    = H / 2;
  const destX   = W - 72;
  const maxCount = Math.max(...destinations.map((d) => d.count), 1);

  const nodes: DestNode[] = destinations.map((d, i) => {
    const step = (H - padding * 2) / Math.max(destinations.length - 1, 1);
    return {
      label: d.label,
      count: d.count,
      x:     destX,
      y:     destinations.length === 1 ? H / 2 : padding + step * i,
      r:     12 + (d.count / maxCount) * 18,
      color: DEST_COLORS[d.label] ?? DEFAULT_COLOR,
    };
  });

  return (
    <div className="card-warm p-5 space-y-3">
      <div>
        <p className="kicker">Jurisdiction flow</p>
        <p className="mt-1.5 text-lg font-semibold" style={{ color: "var(--ink-900)" }}>
          Relocation destinations from {sourceJurisdiction}
        </p>
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          style={{ maxHeight: 300 }}
          aria-label="Jurisdiction flow diagram"
        >
          {/* Bezier flow paths */}
          {nodes.map((node) => {
            const strokeW = Math.max(2, (node.count / maxCount) * 12);
            const cp1x    = srcX  + (destX - srcX) * 0.4;
            const cp2x    = destX - (destX - srcX) * 0.4;
            const d       = `M ${srcX} ${srcY} C ${cp1x} ${srcY}, ${cp2x} ${node.y}, ${node.x - node.r} ${node.y}`;
            return (
              <path
                key={node.label}
                d={d}
                fill="none"
                stroke={node.color}
                strokeWidth={strokeW}
                strokeOpacity={0.35}
                strokeLinecap="round"
              />
            );
          })}

          {/* Source node */}
          <circle
            cx={srcX}
            cy={srcY}
            r={26}
            fill="var(--crimson-50)"
            stroke="var(--crimson-700)"
            strokeWidth={1.5}
          />
          <text
            x={srcX}
            y={srcY - 5}
            textAnchor="middle"
            fontSize={10}
            fontWeight={700}
            fill="var(--crimson-700)"
          >
            {sourceJurisdiction}
          </text>
          <text
            x={srcX}
            y={srcY + 8}
            textAnchor="middle"
            fontSize={9}
            fill="var(--ink-500)"
          >
            source
          </text>

          {/* Destination nodes */}
          {nodes.map((node) => (
            <g key={node.label}>
              <circle
                cx={node.x}
                cy={node.y}
                r={node.r}
                fill={`${node.color}18`}
                stroke={node.color}
                strokeWidth={1.5}
              />
              <text
                x={node.x}
                y={node.y - 3}
                textAnchor="middle"
                fontSize={10}
                fontWeight={700}
                fill={node.color}
              >
                {node.label}
              </text>
              <text
                x={node.x}
                y={node.y + 10}
                textAnchor="middle"
                fontSize={9}
                fill="var(--ink-500)"
              >
                {node.count}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
