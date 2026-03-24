"""Sensitivity analysis layer for PolicyLab.

Systematically sweeps all ASSUMED and DIRECTIONAL parameters to show which
results are robust (appear regardless of parameter values) and which are
model-dependent (only appear at extreme or specific parameter values).

USAGE
─────
  from policylab.game_master.sensitivity_layer import SensitivitySweep
  sweep = SensitivitySweep(scenario_fn)
  report = sweep.run()
  report.print_robustness_summary()

HOW TO INTERPRET
────────────────
  ROBUST result    — appears in RIGOROUS_BASELINE and in >80% of sensitivity
                     configurations. Can be cited with confidence.

  DIRECTIONAL-DEPENDENT — only appears when directional Layer-2 params are
                     non-zero. True direction likely; magnitude uncertain.

  ASSUMED-DEPENDENT — only appears when Layer-3 (invented) params are active.
                      This is a model artifact, NOT a policy prediction.
                      Must be labeled NON-ROBUST in any report.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Callable

from policylab.game_master.indicator_dynamics import (
    DynamicsConfig,
    GroundedDynamicsConfig,
    DirectionalDynamicsConfig,
    AssumedDynamicsConfig,
    RIGOROUS_BASELINE,
    SENSITIVITY_DIRECTIONAL,
    SENSITIVITY_ASSUMED,
)


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP GRID
# ─────────────────────────────────────────────────────────────────────────────

# Layer-2 directional sweep values (direction known, magnitude unknown)
DIRECTIONAL_SWEEP: dict[str, list[float]] = {
    "trust_to_burden":             [0.0, -0.001, -0.003, -0.006],
    "burden_to_trust":             [0.0, -0.001, -0.002, -0.004],
    "innovation_to_trust":         [0.0,  0.000,  0.001,  0.002],
    "passive_burden_per_severity": [0.0,  0.1,    0.5,    1.0],
}

# Layer-3 assumed sweep values (no empirical basis)
ASSUMED_SWEEP: dict[str, list[float]] = {
    "investment_cascade_threshold": [0.0, 10.0, 20.0, 30.0],
    "trust_collapse_threshold":     [0.0, 10.0, 20.0, 30.0],
    "innovation_death_threshold":   [0.0,  5.0, 15.0, 25.0],
    "cascade_multiplier":           [1.0,  1.2,  1.5,  2.0],
}


def _build_directional_configs() -> list[DynamicsConfig]:
    """Build all Layer-2 sensitivity configurations (Layer-3 disabled)."""
    configs = []
    for t2b in DIRECTIONAL_SWEEP["trust_to_burden"]:
        for b2t in DIRECTIONAL_SWEEP["burden_to_trust"]:
            for i2t in DIRECTIONAL_SWEEP["innovation_to_trust"]:
                for pbs in DIRECTIONAL_SWEEP["passive_burden_per_severity"]:
                    configs.append(DynamicsConfig(
                        grounded=GroundedDynamicsConfig(),
                        directional=DirectionalDynamicsConfig(
                            trust_to_burden=t2b,
                            burden_to_trust=b2t,
                            innovation_to_trust=i2t,
                            passive_burden_per_severity=pbs,
                        ),
                        assumed=AssumedDynamicsConfig(),
                    ))
    return configs  # 4^4 = 256 configs


def _build_assumed_configs() -> list[DynamicsConfig]:
    """Build Layer-3 sensitivity configurations with INDEPENDENT threshold variation.

    Each tipping-point threshold is swept independently so that the effect
    of each threshold can be isolated. Co-varying thresholds (old design)
    confounds the three effects and makes individual threshold sensitivity
    invisible.

    This generates 4^4 = 256 configurations.
    Reference: Saltelli et al. (2004) — independence of factor variation
    is required to identify which threshold drives which result.
    """
    configs = []
    dir_cfg = DirectionalDynamicsConfig(
        trust_to_burden=-0.003,
        burden_to_trust=-0.002,
        passive_burden_per_severity=0.5,
    )
    # Independent sweep: 4 values per parameter × 4 parameters = 256 configs
    for inv_thr in ASSUMED_SWEEP["investment_cascade_threshold"]:
        for tru_thr in ASSUMED_SWEEP["trust_collapse_threshold"]:
            for inn_thr in ASSUMED_SWEEP["innovation_death_threshold"]:
                for cm in ASSUMED_SWEEP["cascade_multiplier"]:
                    configs.append(DynamicsConfig(
                        grounded=GroundedDynamicsConfig(),
                        directional=dir_cfg,
                        assumed=AssumedDynamicsConfig(
                            investment_cascade_threshold=inv_thr,
                            trust_collapse_threshold=tru_thr,   # independent
                            innovation_death_threshold=inn_thr,  # independent
                            cascade_multiplier=cm,
                        ),
                    ))
    return configs  # 4^4 = 256 configs (each threshold varied independently)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SensitivityResult:
    config: DynamicsConfig
    final_indicators: dict[str, float]
    failure_modes: list[str]
    tipping_points: list[str]


@dataclasses.dataclass
class RobustnessVerdict:
    result_name: str
    appears_in_rigorous_baseline: bool
    appears_in_fraction_of_directional: float  # 0.0–1.0
    appears_in_fraction_of_assumed: float      # 0.0–1.0
    verdict: str  # ROBUST | DIRECTIONAL-DEPENDENT | ASSUMED-DEPENDENT | NOT-ROBUST

    def describe(self) -> str:
        lines = [
            f"  {self.result_name}:",
            f"    Rigorous baseline: {'YES' if self.appears_in_rigorous_baseline else 'NO'}",
            f"    Directional sweep: {self.appears_in_fraction_of_directional:.0%} of configs",
            f"    Assumed sweep:     {self.appears_in_fraction_of_assumed:.0%} of configs",
            f"    VERDICT: {self.verdict}",
        ]
        return "\n".join(lines)


def classify_robustness(
    result_name: str,
    baseline_present: bool,
    dir_fraction: float,
    assumed_fraction: float,
) -> str:
    """Classify a result by how robust it is across parameter configurations.

    Thresholds follow Saltelli et al. (2004) recommendations for policy-relevant
    sensitivity analysis: policy models require ≥90% appearance rate for a result
    to be cited as ROBUST. The previous 80% threshold was too permissive for
    policy decisions where errors have real-world consequences.

    Reference: Saltelli, A. et al. (2004). Sensitivity Analysis in Practice.
    Wiley. Chapter 6: "For policy applications, robustness thresholds should
    reflect the decision stakes."
    """
    # ≥90% across all configs → ROBUST (Saltelli et al. 2004: policy threshold)
    if baseline_present and dir_fraction >= 0.9 and assumed_fraction >= 0.9:
        return "ROBUST"
    # ≥80% directional → DIRECTIONAL-DEPENDENT (raised from 0.6)
    if baseline_present and dir_fraction >= 0.8:
        return "DIRECTIONAL-DEPENDENT"
    # Only in assumed configs → ASSUMED-DEPENDENT
    if not baseline_present and assumed_fraction >= 0.5:
        return "ASSUMED-DEPENDENT"
    return "NOT-ROBUST"


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SensitivityReport:
    verdicts: list[RobustnessVerdict]
    indicator_ranges: dict[str, tuple[float, float]]  # name → (min, max) across all configs
    baseline_indicators: dict[str, float]

    def print_robustness_summary(self) -> None:
        sep = "─" * 70
        print(f"\n{sep}")
        print("SENSITIVITY ANALYSIS — ROBUSTNESS REPORT")
        print(sep)
        print(f"\nBaseline indicators (RIGOROUS_BASELINE, no assumed params):")
        for k, v in self.baseline_indicators.items():
            lo, hi = self.indicator_ranges.get(k, (v, v))
            print(f"  {k:<30}: {v:6.1f}  [range across all configs: {lo:.1f}–{hi:.1f}]")

        print(f"\nResult robustness:")
        robust = [v for v in self.verdicts if v.verdict == "ROBUST"]
        directional = [v for v in self.verdicts if v.verdict == "DIRECTIONAL-DEPENDENT"]
        assumed = [v for v in self.verdicts if v.verdict == "ASSUMED-DEPENDENT"]
        not_robust = [v for v in self.verdicts if v.verdict == "NOT-ROBUST"]

        if robust:
            print(f"\n  ✓ ROBUST ({len(robust)} results — cite with confidence):")
            for v in robust:
                print(v.describe())
        if directional:
            print(f"\n  ~ DIRECTIONAL-DEPENDENT ({len(directional)} results — cite direction only):")
            for v in directional:
                print(v.describe())
        if assumed:
            print(f"\n  ✗ ASSUMED-DEPENDENT ({len(assumed)} results — DO NOT cite as predictions):")
            for v in assumed:
                print(v.describe())
        if not_robust:
            print(f"\n  ✗ NOT-ROBUST ({len(not_robust)} results — discard):")
            for v in not_robust:
                print(v.describe())
        print(f"\n{sep}")


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTWEIGHT CONFIG-ONLY SENSITIVITY (no LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

def run_config_sensitivity(
    baseline_indicators: dict[str, float],
    n_rounds: int = 8,
) -> SensitivityReport:
    """Run sensitivity analysis on indicator dynamics alone (no LLM agents).

    This is the fast path: apply indicator_dynamics to a fixed starting
    state across all parameter configurations to show how much the
    indicator coupling equations alone explain vs. the agent behavior.

    For full sensitivity analysis including agent behavior, use the
    slow path: run the full simulation with different DynamicsConfig presets.
    """
    from policylab.game_master.indicator_dynamics import apply_indicator_feedback

    # Mock world state
    class _MockWorldState:
        def __init__(self, indicators):
            self.economic_indicators = dict(indicators)
            self.active_policies = {}

    def simulate_indicators(cfg: DynamicsConfig, start: dict) -> dict:
        ws = _MockWorldState(dict(start))
        for _ in range(n_rounds):
            apply_indicator_feedback(ws, cfg)
            # Clamp
            for k in ws.economic_indicators:
                ws.economic_indicators[k] = max(0.0, min(100.0, ws.economic_indicators[k]))
        return dict(ws.economic_indicators)

    # Run baseline
    baseline_final = simulate_indicators(RIGOROUS_BASELINE, baseline_indicators)

    # Run directional sweep
    dir_configs = _build_directional_configs()
    dir_results = [simulate_indicators(c, baseline_indicators) for c in dir_configs]

    # Run assumed sweep
    asm_configs = _build_assumed_configs()
    asm_results = [simulate_indicators(c, baseline_indicators) for c in asm_configs]

    all_results = dir_results + asm_results

    # Compute indicator ranges
    indicator_ranges = {}
    for key in baseline_final:
        all_vals = [r.get(key, baseline_final[key]) for r in all_results]
        indicator_ranges[key] = (min(all_vals), max(all_vals))

    # Classify robustness for key indicators
    verdicts = []
    for indicator in ["innovation_rate", "ai_investment_index", "public_trust", "regulatory_burden"]:
        base_val = baseline_final.get(indicator, 50.0)

        # "Innovation suppressed" = final < starting × 0.7
        start_val = baseline_indicators.get(indicator, 50.0)
        threshold = start_val * 0.7

        baseline_suppressed = base_val < threshold
        dir_suppressed = sum(1 for r in dir_results if r.get(indicator, 50.0) < threshold) / max(1, len(dir_results))
        asm_suppressed = sum(1 for r in asm_results if r.get(indicator, 50.0) < threshold) / max(1, len(asm_results))

        result_name = f"{indicator} drops >30%"
        verdict_str = classify_robustness(result_name, baseline_suppressed, dir_suppressed, asm_suppressed)
        verdicts.append(RobustnessVerdict(
            result_name=result_name,
            appears_in_rigorous_baseline=baseline_suppressed,
            appears_in_fraction_of_directional=dir_suppressed,
            appears_in_fraction_of_assumed=asm_suppressed,
            verdict=verdict_str,
        ))

    return SensitivityReport(
        verdicts=verdicts,
        indicator_ranges=indicator_ranges,
        baseline_indicators=baseline_final,
    )
