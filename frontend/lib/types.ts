/**
 * Shared TypeScript types — single source of truth.
 * Mirrors the FastAPI Pydantic schemas exactly.
 */

export type EpistemicTier = "GROUNDED" | "DIRECTIONAL" | "ASSUMED";
export type Jurisdiction = "EU" | "US" | "UK" | "Singapore" | "UAE";

export type PenaltyType =
  | "none"
  | "voluntary"
  | "civil"
  | "civil_heavy"
  | "criminal";

export type EnforcementType =
  | "none"
  | "self_report"
  | "third_party_audit"
  | "government_inspect"
  | "criminal_invest";

export type ScopeType =
  | "all"
  | "large_developers_only"
  | "frontier_only"
  | "voluntary";

// ── Policy ────────────────────────────────────────────────────────────────────

export interface PolicySpec {
  name: string;
  description: string;
  severity: number; // 1.0–5.0
  justification: string[];
  penalty_type: PenaltyType;
  penalty_cap_usd: number | null;
  compute_threshold_flops: number | null;
  enforcement_mechanism: EnforcementType;
  grace_period_months: number;
  scope: ScopeType;
  recommended_n_population: number;
  recommended_num_rounds: number;
  recommended_severity_sweep: number[];
  compute_cost_factor: number;
}

export interface PresetPolicy {
  slug: string;
  label: string;
  spec: PolicySpec;
}

// ── Simulation ────────────────────────────────────────────────────────────────

export interface SimConfigRequest {
  n_population: number;
  num_rounds: number;
  spillover_factor: number;
  seed: number;
  use_network: boolean;
  use_vectorized: boolean;
  source_jurisdiction: string;
  destination_jurisdictions: string[];
  relocation_temperature: number;
  adversarial_injection_rate: number;
  adversarial_injection_direction: number;
  adversarial_injection_magnitude: number;
  adversarial_injection_start_round: number;
  compute_cost_factor: number;
  hk_epsilon: number;
  type_distribution: Record<string, number> | null;
}

// ── Swarm elicitation ─────────────────────────────────────────────────────────

export type SwarmConfidence = "high" | "medium" | "low";
export type SwarmPrimaryAction = "comply" | "relocate" | "evade" | "lobby";
export type SwarmRelocationPressure = "low" | "medium" | "high";

export interface SwarmAgentResponse {
  persona: string;
  agent_type: string;
  compliance_factor: number;
  primary_action: SwarmPrimaryAction;
  relocation_pressure: SwarmRelocationPressure;
  reasoning: string;
}

export interface SwarmTypeResult {
  agent_type: string;
  n_agents: number;
  mean_compliance_factor: number;
  compliance_factor_cv: number;
  dominant_action: SwarmPrimaryAction;
  relocation_pressure_dist: Record<SwarmRelocationPressure, number>;
  confidence: SwarmConfidence;
  applied_lambda_multiplier: number;
  applied_threshold_shift: number;
  agents: SwarmAgentResponse[];
}

export interface ParameterAdjustments {
  lambda_multipliers: Record<string, number>;
  threshold_shifts: Record<string, number>;
}

export interface SwarmResult {
  epistemic_tag: "SWARM-ELICITED";
  policy_name: string;
  n_total_agents: number;
  type_results: SwarmTypeResult[];
  parameter_adjustments: ParameterAdjustments;
  llm_model: string;
  elapsed_seconds: number;
  warning: string;
}

// ── Simulation ────────────────────────────────────────────────────────────────

export interface SimulateRequest {
  policy_name: string;
  policy_description: string;
  policy_severity: number;
  config: SimConfigRequest;
  use_swarm_elicitation?: boolean;
  openai_api_key?: string;
}

export interface RoundSummary {
  round: number;
  compliance_rate: number;
  relocation_rate: number;
  ai_investment_index: number;
  enforcement_contact_rate: number;
}

export interface SimulatedMoments {
  lobbying_rate: number;
  compliance_rate_y1: number;
  relocation_rate: number;
  sme_compliance_24mo: number;
  large_compliance_24mo: number;
  enforcement_rate: number;
  n_runs: number;
}

export interface RunMetadata {
  duration_ms: number;
  seed: number;
  n_population: number;
}

export interface SimulateResponse {
  policy_name: string;
  policy_severity: number;
  round_summaries: RoundSummary[];
  stock_history: Record<string, number>[];
  final_stocks: Record<string, number>;
  final_population_summary: Record<string, number>;
  network_statistics: Record<string, number>;
  network_hubs: Record<string, unknown>[];
  simulated_moments: SimulatedMoments;
  smm_distance_to_gdpr: number | null;
  jurisdiction_summary: Record<string, { company_count?: number; burden?: number }>;
  run_metadata: RunMetadata;
  event_log: unknown[];
  swarm_result: SwarmResult | null;
}

// ── Upload / extraction ───────────────────────────────────────────────────────

export interface ExtractedField {
  value: unknown;
  confidence: number;
  source_passage: string;
  extraction_method: string;
  epistemic_tag: EpistemicTier;
}

export interface ExtractionResult {
  policy_name: ExtractedField;
  policy_description: ExtractedField;
  penalty_type: ExtractedField;
  penalty_cap_usd: ExtractedField;
  compute_threshold_flops: ExtractedField;
  enforcement_mechanism: ExtractedField;
  grace_period_months: ExtractedField;
  scope: ExtractedField;
  source_jurisdiction: ExtractedField;
  has_sme_provisions: ExtractedField;
  has_frontier_lab_focus: ExtractedField;
  has_research_exemptions: ExtractedField;
  has_investor_provisions: ExtractedField;
  estimated_n_regulated: ExtractedField;
  key_provisions: [string, string][];
  extraction_method_used: string;
  model_used: string | null;
  unresolved_provisions: string[];
}

export interface UploadResponse {
  document_name: string;
  spec: PolicySpec;
  extraction: ExtractionResult;
  warnings: string[];
  elapsed_seconds: number;
  result: SimulateResponse;
  confidence_summary: string;
}

// ── Injection ─────────────────────────────────────────────────────────────────

export interface InjectionConfig {
  injection_rate: number;
  injection_direction: number;
  injection_magnitude: number;
  injection_start_round: number;
}

export interface InjectRequest {
  policy_name: string;
  policy_description: string;
  policy_severity: number;
  n_population: number;
  num_rounds: number;
  seed: number;
  injection: InjectionConfig;
}

export interface InjectionResult {
  baseline_compliance: number;
  injected_compliance: number;
  compliance_delta: number;
  baseline_relocation: number;
  injected_relocation: number;
  relocation_delta: number;
  resilience_score: number;
  injection_params: Record<string, unknown>;
  round_compliance_baseline: number[];
  round_compliance_injected: number[];
}

// ── Evidence pack (jobs) ──────────────────────────────────────────────────────

export interface ConfidenceBand {
  round: number;
  mean: number;
  p10: number;
  p90: number;
}

export interface EvidencePackResult {
  severity_levels: number[];
  bands: Record<string, ConfidenceBand[]>;
}

export type JobStatusValue = "queued" | "running" | "complete" | "error";

export interface JobStatus {
  job_id: string;
  status: JobStatusValue;
  progress: number;
  result: EvidencePackResult | null;
  error: string | null;
}

export interface EvidencePackRequest {
  policy_name: string;
  policy_description: string;
  base_severity: number;
  sim_config: Partial<SimConfigRequest>;
  ensemble_size: number;
}

// ── Compare ───────────────────────────────────────────────────────────────────

export interface CompareRequest {
  policies: SimulateRequest[];
}

export interface CompareResponse {
  results: SimulateResponse[];
}
