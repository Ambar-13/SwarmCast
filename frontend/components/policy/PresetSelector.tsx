"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchPresets } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { PresetPolicy } from "@/lib/types";

interface Props {
  onSelect: (preset: PresetPolicy) => void;
  selectedSlug?: string;
}

export function PresetSelector({ onSelect, selectedSlug }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.presets(),
    queryFn: fetchPresets,
    staleTime: Infinity,
  });

  if (isLoading) {
    return <div className="kicker animate-pulse">Loading presets…</div>;
  }
  if (error || !data) {
    return (
      <div className="text-xs font-medium" style={{ color: "var(--danger)" }}>
        Failed to load presets
      </div>
    );
  }

  return (
    <div>
      <label className="kicker mb-1.5 block">Load a preset</label>
      <select
        className="select-warm"
        value={selectedSlug ?? ""}
        onChange={(e) => {
          const preset = data.presets.find((p) => p.slug === e.target.value);
          if (preset) onSelect(preset);
        }}
      >
        <option value="">Select a starting policy…</option>
        {data.presets.map((p) => (
          <option key={p.slug} value={p.slug}>
            {p.label}
          </option>
        ))}
      </select>
    </div>
  );
}
