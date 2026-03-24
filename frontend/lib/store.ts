"use client";

import { create } from "zustand";
import type {
  PolicySpec,
  SimConfigRequest,
  SimulateResponse,
  InjectionResult,
  InjectionConfig,
  EvidencePackResult,
} from "@/lib/types";
import { DEFAULT_SIM_CONFIG } from "@/lib/constants";

// ── Analyze page store ────────────────────────────────────────────────────────

interface AnalyzeState {
  policyName: string;
  policyDescription: string;
  policySeverity: number;
  simConfig: SimConfigRequest;
  result: SimulateResponse | null;
  isLoading: boolean;
  error: string | null;
  evidencePackResult: EvidencePackResult | null;
  /** Increments each time a new result arrives — used to remount EvidencePackSection */
  runId: number;

  setPolicy: (name: string, description: string, severity: number) => void;
  setFromSpec: (spec: PolicySpec) => void;
  /** Atomically populate from an upload result — sets spec + pre-computed result. */
  setFromSpecWithResult: (spec: PolicySpec, result: SimulateResponse) => void;
  setSimConfig: (patch: Partial<SimConfigRequest>) => void;
  setResult: (result: SimulateResponse) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setEvidencePackResult: (result: EvidencePackResult | null) => void;
}

export const useAnalyzeStore = create<AnalyzeState>((set) => ({
  policyName: "California SB-53 (2025)",
  policyDescription:
    "Requires frontier AI developers to implement safety protocols, conduct impact assessments, and report incidents to the state.",
  policySeverity: 2.39,
  simConfig: { ...DEFAULT_SIM_CONFIG },
  result: null,
  isLoading: false,
  error: null,
  evidencePackResult: null,
  runId: 0,

  setPolicy: (policyName, policyDescription, policySeverity) =>
    set({ policyName, policyDescription, policySeverity }),

  setFromSpec: (spec) =>
    set({
      policyName: spec.name,
      policyDescription: spec.description,
      policySeverity: spec.severity,
      simConfig: {
        ...DEFAULT_SIM_CONFIG,
        compute_cost_factor: spec.compute_cost_factor,
        n_population: spec.recommended_n_population,
        num_rounds: spec.recommended_num_rounds,
      },
    }),

  setFromSpecWithResult: (spec, result) =>
    set((s) => ({
      policyName: spec.name,
      policyDescription: spec.description,
      policySeverity: spec.severity,
      simConfig: {
        ...DEFAULT_SIM_CONFIG,
        compute_cost_factor: spec.compute_cost_factor,
        n_population: spec.recommended_n_population,
        num_rounds: spec.recommended_num_rounds,
      },
      result,
      isLoading: false,
      error: null,
      evidencePackResult: null,
      runId: s.runId + 1,
    })),

  setSimConfig: (patch) =>
    set((s) => ({ simConfig: { ...s.simConfig, ...patch } })),

  // Reset evidencePackResult and bump runId so EvidencePackSection remounts fresh
  setResult: (result) =>
    set((s) => ({
      result,
      isLoading: false,
      error: null,
      evidencePackResult: null,
      runId: s.runId + 1,
    })),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),
  setEvidencePackResult: (evidencePackResult) => set({ evidencePackResult }),
}));

// ── Compare page store ────────────────────────────────────────────────────────

export interface CompareSlot {
  id: string;
  policyName: string;
  policyDescription: string;
  policySeverity: number;
  simConfig: SimConfigRequest;
  result: SimulateResponse | null;
}

interface CompareState {
  slots: CompareSlot[];
  isLoading: boolean;
  error: string | null;
  addSlot: (slot: Omit<CompareSlot, "result">) => void;
  /** Push a slot that already has a result (from the Analyze page). */
  addSlotWithResult: (slot: CompareSlot) => void;
  removeSlot: (id: string) => void;
  setResults: (results: SimulateResponse[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

function makeId(): string {
  return Math.random().toString(36).slice(2, 9);
}

export const useCompareStore = create<CompareState>((set) => ({
  slots: [],
  isLoading: false,
  error: null,

  addSlot: (slot) =>
    set((s) => ({ slots: [...s.slots, { ...slot, result: null }] })),

  addSlotWithResult: (slot) =>
    set((s) => {
      // Don't add duplicates by policyName
      if (s.slots.some((sl) => sl.policyName === slot.policyName)) return s;
      // Cap at 4 slots
      if (s.slots.length >= 4) return s;
      return { slots: [...s.slots, slot] };
    }),

  removeSlot: (id) =>
    set((s) => ({ slots: s.slots.filter((sl) => sl.id !== id) })),

  setResults: (results) =>
    set((s) => ({
      slots: s.slots.map((sl, i) => ({
        ...sl,
        result: results[i] ?? sl.result,
      })),
      isLoading: false,
      error: null,
    })),

  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),
}));

export { makeId };

// ── Influence page store ──────────────────────────────────────────────────────

interface InfluenceState {
  policyName: string;
  policyDescription: string;
  policySeverity: number;
  nPopulation: number;
  numRounds: number;
  seed: number;
  injection: InjectionConfig;
  result: InjectionResult | null;
  isLoading: boolean;
  error: string | null;

  setPolicy: (name: string, description: string, severity: number) => void;
  /** Sync policy from Analyze store — also clears previous injection result */
  syncPolicy: (name: string, description: string, severity: number) => void;
  setInjection: (patch: Partial<InjectionConfig>) => void;
  setSimParams: (patch: { nPopulation?: number; numRounds?: number; seed?: number }) => void;
  setResult: (result: InjectionResult) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useInfluenceStore = create<InfluenceState>((set) => ({
  policyName: "California SB-53 (2025)",
  policyDescription:
    "Requires frontier AI developers to implement safety protocols, conduct impact assessments, and report incidents to the state.",
  policySeverity: 2.39,
  nPopulation: 1000,
  numRounds: 8,
  seed: 42,
  injection: {
    injection_rate: 0.05,
    injection_direction: 1.0,
    injection_magnitude: 0.08,
    injection_start_round: 1,
  },
  result: null,
  isLoading: false,
  error: null,

  setPolicy: (policyName, policyDescription, policySeverity) =>
    set({ policyName, policyDescription, policySeverity }),

  syncPolicy: (policyName, policyDescription, policySeverity) =>
    set({ policyName, policyDescription, policySeverity, result: null }),

  setInjection: (patch) =>
    set((s) => ({ injection: { ...s.injection, ...patch } })),

  setSimParams: (patch) =>
    set((s) => ({
      nPopulation: patch.nPopulation ?? s.nPopulation,
      numRounds: patch.numRounds ?? s.numRounds,
      seed: patch.seed ?? s.seed,
    })),

  setResult: (result) => set({ result, isLoading: false, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),
}));
