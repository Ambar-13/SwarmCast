import type { EpistemicTier } from "@/lib/types";

const TIER_STYLES: Record<EpistemicTier, { color: string; bg: string; border: string }> = {
  GROUNDED:    { color: "#15803D", bg: "#f0fdf4", border: "rgba(21,128,61,0.22)"  },
  DIRECTIONAL: { color: "#B45309", bg: "#fffbeb", border: "rgba(180,83,9,0.22)"   },
  ASSUMED:     { color: "#DC2626", bg: "#fef2f2", border: "rgba(220,38,38,0.22)"  },
};

interface Props {
  tier: EpistemicTier;
  small?: boolean;
}

export function EpistemicBadge({ tier, small = false }: Props) {
  const s = TIER_STYLES[tier];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-semibold ${
        small ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs"
      }`}
      style={{ color: s.color, background: s.bg, borderColor: s.border }}
    >
      <span
        className="h-1.5 w-1.5 flex-shrink-0 rounded-full"
        style={{ background: s.color }}
      />
      {tier}
    </span>
  );
}
