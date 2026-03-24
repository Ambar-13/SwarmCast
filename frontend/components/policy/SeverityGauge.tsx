"use client";

import { severityColor } from "@/lib/format";

interface Props {
  severity: number; // 1.0–5.0
  size?: number;
}

/** Pure SVG radial gauge, 220-degree span, three-zone color, no chart library. */
export function SeverityGauge({ severity, size = 140 }: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const r  = (size / 2) * 0.7;

  // Arc spans 220° starting at 160° (bottom-left) going clockwise
  const START_DEG = 160;
  const SPAN_DEG  = 220;

  function polar(deg: number): [number, number] {
    const rad = (deg * Math.PI) / 180;
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
  }

  function describeArc(startDeg: number, endDeg: number): string {
    const [x1, y1] = polar(startDeg);
    const [x2, y2] = polar(endDeg);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  }

  // Three zone arcs: low (1–2), mid (2–3.5), high (3.5–5)
  const zones = [
    { from: 0, to: (1 / 4) * SPAN_DEG, color: "#22c55e" },     // 1–2
    { from: (1 / 4) * SPAN_DEG, to: (2.5 / 4) * SPAN_DEG, color: "#f59e0b" }, // 2–3.5
    { from: (2.5 / 4) * SPAN_DEG, to: SPAN_DEG, color: "#ef4444" },            // 3.5–5
  ];

  // Needle angle
  const fraction = (severity - 1) / 4; // 0–1
  const needleDeg = START_DEG + fraction * SPAN_DEG;
  const [nx, ny] = polar(needleDeg);

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-label={`Severity gauge: ${severity.toFixed(2)}`}
    >
      {/* Background track */}
      <path
        d={describeArc(START_DEG, START_DEG + SPAN_DEG)}
        fill="none"
        stroke="#e8e0d4"
        strokeWidth={size * 0.06}
        strokeLinecap="round"
      />

      {/* Zone arcs */}
      {zones.map((z) => (
        <path
          key={z.color}
          d={describeArc(START_DEG + z.from, START_DEG + z.to)}
          fill="none"
          stroke={z.color}
          strokeWidth={size * 0.055}
          strokeLinecap="butt"
          opacity={0.5}
        />
      ))}

      {/* Filled value arc */}
      <path
        d={describeArc(START_DEG, needleDeg)}
        fill="none"
        stroke={severityColor(severity)}
        strokeWidth={size * 0.055}
        strokeLinecap="round"
      />

      {/* Needle */}
      <line
        x1={cx}
        y1={cy}
        x2={nx}
        y2={ny}
        stroke="#44403c"
        strokeWidth={2}
        strokeLinecap="round"
      />
      <circle cx={cx} cy={cy} r={size * 0.04} fill="#faf7f2" stroke="#44403c" strokeWidth={1.5} />

      {/* Label */}
      <text
        x={cx}
        y={cy + size * 0.22}
        textAnchor="middle"
        fill={severityColor(severity)}
        fontSize={size * 0.14}
        fontFamily="var(--font-jetbrains-mono), monospace"
        fontWeight="600"
      >
        {severity.toFixed(2)}
      </text>
      <text
        x={cx}
        y={cy + size * 0.33}
        textAnchor="middle"
        fill="#78716c"
        fontSize={size * 0.075}
      >
        severity
      </text>
    </svg>
  );
}
