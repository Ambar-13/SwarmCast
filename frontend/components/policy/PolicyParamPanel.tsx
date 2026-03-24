"use client";

import { SeverityGauge } from "@/components/policy/SeverityGauge";
import { ENFORCEMENT_LABELS, PENALTY_LABELS, SCOPE_LABELS } from "@/lib/constants";
import { formatSci, formatUSD } from "@/lib/format";
import type { EnforcementType, PenaltyType, ScopeType } from "@/lib/types";

interface Props {
  name: string;
  description: string;
  severity: number;
  penaltyType: PenaltyType;
  penaltyCapUsd: number | null;
  computeThresholdFlops: number | null;
  enforcementMechanism: EnforcementType;
  gracePeriodMonths: number;
  scope: ScopeType;
  readOnly?: boolean;
  onChangeName?: (v: string) => void;
  onChangeDescription?: (v: string) => void;
  onChangeSeverity?: (v: number) => void;
  onChangePenaltyType?: (v: PenaltyType) => void;
  onChangePenaltyCapUsd?: (v: number | null) => void;
  onChangeComputeThresholdFlops?: (v: number | null) => void;
  onChangeEnforcementMechanism?: (v: EnforcementType) => void;
  onChangeGracePeriodMonths?: (v: number) => void;
  onChangeScope?: (v: ScopeType) => void;
}

function Label({ children }: { children: React.ReactNode }) {
  return <label className="kicker mb-1.5 block">{children}</label>;
}

function InputBase({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`input-warm ${className ?? ""}`}
    />
  );
}

function SelectBase({ className, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={`select-warm ${className ?? ""}`}
    />
  );
}

function StaticValue({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="rounded-xl border px-3 py-2 text-sm"
      style={{
        background: "var(--cream-100)",
        borderColor: "var(--border-warm)",
        color: "var(--ink-700)",
      }}
    >
      {children}
    </div>
  );
}

export function PolicyParamPanel({
  name,
  description,
  severity,
  penaltyType,
  penaltyCapUsd,
  computeThresholdFlops,
  enforcementMechanism,
  gracePeriodMonths,
  scope,
  readOnly = false,
  onChangeName,
  onChangeDescription,
  onChangeSeverity,
  onChangePenaltyType,
  onChangeEnforcementMechanism,
  onChangeGracePeriodMonths,
  onChangeScope,
}: Props) {
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-[116px_minmax(0,1fr)]">
        <div
          className="flex items-center justify-center rounded-2xl border p-2"
          style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}
        >
          <SeverityGauge severity={severity} size={104} />
        </div>

        <div className="space-y-3">
          <div>
            <Label>Policy name</Label>
            {readOnly ? (
              <StaticValue>{name}</StaticValue>
            ) : (
              <InputBase
                value={name}
                onChange={(e) => onChangeName?.(e.target.value)}
                placeholder="e.g. California SB-53"
              />
            )}
          </div>

          {!readOnly && (
            <div>
              <Label>Description</Label>
              <textarea
                value={description}
                onChange={(e) => onChangeDescription?.(e.target.value)}
                rows={3}
                className="input-warm w-full resize-none"
              />
            </div>
          )}

          <div>
            <Label>Severity (1–5)</Label>
            {readOnly ? (
              <StaticValue>{severity.toFixed(2)}</StaticValue>
            ) : (
              <InputBase
                type="number"
                min={1}
                max={5}
                step={0.01}
                value={severity}
                onChange={(e) => onChangeSeverity?.(parseFloat(e.target.value) || 1)}
              />
            )}
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <Label>Penalty type</Label>
          {readOnly ? (
            <StaticValue>{PENALTY_LABELS[penaltyType] ?? penaltyType}</StaticValue>
          ) : (
            <SelectBase value={penaltyType} onChange={(e) => onChangePenaltyType?.(e.target.value as PenaltyType)}>
              {Object.entries(PENALTY_LABELS).map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </SelectBase>
          )}
        </div>

        <div>
          <Label>Enforcement</Label>
          {readOnly ? (
            <StaticValue>{ENFORCEMENT_LABELS[enforcementMechanism] ?? enforcementMechanism}</StaticValue>
          ) : (
            <SelectBase
              value={enforcementMechanism}
              onChange={(e) => onChangeEnforcementMechanism?.(e.target.value as EnforcementType)}
            >
              {Object.entries(ENFORCEMENT_LABELS).map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </SelectBase>
          )}
        </div>

        <div>
          <Label>Scope</Label>
          {readOnly ? (
            <StaticValue>{SCOPE_LABELS[scope] ?? scope}</StaticValue>
          ) : (
            <SelectBase value={scope} onChange={(e) => onChangeScope?.(e.target.value as ScopeType)}>
              {Object.entries(SCOPE_LABELS).map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </SelectBase>
          )}
        </div>

        <div>
          <Label>Grace period (months)</Label>
          {readOnly ? (
            <StaticValue>{gracePeriodMonths}</StaticValue>
          ) : (
            <InputBase
              type="number"
              min={0}
              max={60}
              value={gracePeriodMonths}
              onChange={(e) => onChangeGracePeriodMonths?.(parseInt(e.target.value) || 0)}
            />
          )}
        </div>

        <div>
          <Label>Penalty cap (USD)</Label>
          <StaticValue>{penaltyCapUsd !== null ? formatUSD(penaltyCapUsd) : "—"}</StaticValue>
        </div>

        <div>
          <Label>Compute threshold</Label>
          <StaticValue>{formatSci(computeThresholdFlops)} FLOPS</StaticValue>
        </div>
      </div>
    </div>
  );
}
