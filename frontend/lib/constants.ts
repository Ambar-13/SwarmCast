import type { SimConfigRequest } from "@/lib/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const JURISDICTIONS = [
  "EU", "US", "UK", "Singapore", "UAE",
  "China", "Canada", "Japan", "Switzerland", "Australia", "India",
  "Russia", "South Korea", "France", "Germany",
] as const;

export const PENALTY_LABELS: Record<string, string> = {
  none: "None",
  voluntary: "Voluntary",
  civil: "Civil fine",
  civil_heavy: "Civil (heavy)",
  criminal: "Criminal",
};

export const ENFORCEMENT_LABELS: Record<string, string> = {
  none: "None",
  self_report: "Self-report",
  third_party_audit: "Third-party audit",
  government_inspect: "Government inspection",
  criminal_invest: "Criminal investigation",
};

export const SCOPE_LABELS: Record<string, string> = {
  all: "All developers",
  large_developers_only: "Large developers only",
  frontier_only: "Frontier labs only",
  voluntary: "Voluntary",
};

export const DEFAULT_SIM_CONFIG: SimConfigRequest = {
  n_population: 1000,
  num_rounds: 16,
  spillover_factor: 0.5,
  seed: 42,
  use_network: true,
  use_vectorized: true,
  source_jurisdiction: "EU",
  destination_jurisdictions: ["US", "UK", "Singapore", "UAE", "Canada", "Japan", "Switzerland", "Australia", "India"],
  relocation_temperature: 0.1,
  adversarial_injection_rate: 0.0,
  adversarial_injection_direction: 1.0,
  adversarial_injection_magnitude: 0.08,
  adversarial_injection_start_round: 1,
  compute_cost_factor: 1.0,
  hk_epsilon: 1.0,
  type_distribution: null,
};

export const CHART_COLORS = {
  compliance: "#6366f1",
  relocation: "#f97316",
  investment: "#06b6d4",
  enforcement: "#a855f7",
};

export const METRIC_LABELS: Record<string, string> = {
  compliance_rate: "Compliance rate",
  relocation_rate: "Relocation rate",
  ai_investment_index: "AI investment index",
  enforcement_contact_rate: "Enforcement contact rate",
};
