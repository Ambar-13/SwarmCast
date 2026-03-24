import type { RoundSummary, ConfidenceBand } from "@/lib/types";

export interface ChartDataPoint {
  round: number;
  compliance_rate: number;
  relocation_rate: number;
  ai_investment_index: number;
  enforcement_contact_rate: number;
  // Optional confidence bands (from evidence pack)
  compliance_p10?: number;
  compliance_p90?: number;
}

export function buildChartData(
  roundSummaries: RoundSummary[],
  bands?: ConfidenceBand[],
): ChartDataPoint[] {
  const bandMap = new Map(bands?.map((b) => [b.round, b]));
  return roundSummaries.map((r) => {
    const band = bandMap.get(r.round);
    return {
      round: r.round,
      compliance_rate: r.compliance_rate,
      relocation_rate: r.relocation_rate,
      ai_investment_index: r.ai_investment_index,
      enforcement_contact_rate: r.enforcement_contact_rate,
      ...(band
        ? { compliance_p10: band.p10, compliance_p90: band.p90 }
        : {}),
    };
  });
}

/** Normalize ai_investment_index (can be 0–100) to 0–1 for chart consistency. */
export function normalizeInvestmentIndex(value: number): number {
  return value / 100;
}
