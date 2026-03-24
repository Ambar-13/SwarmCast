"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { SimConfigRequest } from "@/lib/types";

interface Props {
  config: SimConfigRequest;
  onChange: (patch: Partial<SimConfigRequest>) => void;
}

function Row({
  label,
  description,
  children,
}: {
  label: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="grid gap-2 rounded-xl border px-3 py-3 md:grid-cols-[1fr_auto] md:items-center"
      style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}
    >
      <div>
        <p className="text-sm font-medium" style={{ color: "var(--ink-700)" }}>
          {label}
        </p>
        <p className="mt-0.5 text-xs leading-5" style={{ color: "var(--ink-400)" }}>
          {description}
        </p>
      </div>
      {children}
    </div>
  );
}

function NumInput({
  value,
  min,
  max,
  step,
  onChange,
}: {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="input-warm w-28"
    />
  );
}

export function SimConfigPanel({ config, onChange }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="card-warm overflow-hidden">
      <button
        className="flex w-full items-center gap-2 px-4 py-3 text-sm font-medium transition-colors"
        style={{ color: open ? "var(--ink-700)" : "var(--ink-400)" }}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        Advanced simulation config
      </button>
      {open && (
        <div
          className="space-y-2 border-t px-4 pb-4 pt-3"
          style={{ borderColor: "var(--border-warm)" }}
        >
          <Row label="Population size" description="Number of simulated actors.">
            <NumInput
              value={config.n_population}
              min={100}
              max={20000}
              step={100}
              onChange={(v) => onChange({ n_population: v })}
            />
          </Row>
          <Row label="Rounds" description="Decision cycles the simulation runs.">
            <NumInput
              value={config.num_rounds}
              min={1}
              max={64}
              step={1}
              onChange={(v) => onChange({ num_rounds: v })}
            />
          </Row>
          <Row label="Seed" description="Deterministic seed for reproducibility.">
            <NumInput
              value={config.seed}
              min={0}
              step={1}
              onChange={(v) => onChange({ seed: v })}
            />
          </Row>
          <Row
            label="Spillover factor"
            description="How strongly effects propagate across the system."
          >
            <NumInput
              value={config.spillover_factor}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => onChange({ spillover_factor: v })}
            />
          </Row>
          <Row
            label="Compute cost factor"
            description="Relative compute burden on covered entities."
          >
            <NumInput
              value={config.compute_cost_factor}
              min={0}
              max={10}
              step={0.1}
              onChange={(v) => onChange({ compute_cost_factor: v })}
            />
          </Row>
          <Row
            label="HK epsilon"
            description="Sensitivity term in population response dynamics."
          >
            <NumInput
              value={config.hk_epsilon}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => onChange({ hk_epsilon: v })}
            />
          </Row>
        </div>
      )}
    </div>
  );
}
