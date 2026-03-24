"""Advanced stress test using PolicyLab v2 hybrid simulation.

10x better than v1 stress test because:
  1. 100 heterogeneous population agents (vs 6)
  2. Calibrated behavioral responses (vs LLM-only)
  3. Proper stock-flow accounting (vs indicator ratchet)
  4. International spillovers (vs innovation destroyed)
  5. 16-round horizon (vs 8)
  6. Network influence propagation (vs flat all-to-all)
  7. SMM validation against GDPR/EU AI Act data
  8. Counterfactual with genuine population dynamics

USAGE
─────
  from policylab.v2.stress_test_v2 import HybridStressTest

  tester = HybridStressTest(n_population=100, num_rounds=16)
  report = tester.run(
      policy_name="Total AI Development Moratorium",
      policy_description="...",
      n_ensemble=5,
  )
  report.print_summary()
"""

from __future__ import annotations

import dataclasses
import os
import json
from datetime import datetime
from typing import Any

import numpy as np

from policylab.v2.simulation.hybrid_loop import (
    HybridSimConfig,
    HybridSimResult,
    run_hybrid_simulation,
)
from policylab.v2.calibration.smm_framework import (
    TargetMoments,
    smm_objective,
)
from policylab.game_master.severity import classify_severity
from policylab.disclaimers import indicator_disclaimer


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_severity(policy_description: str) -> float:
    """Detect policy severity from description text."""
    desc_lower = policy_description.lower()
    if any(w in desc_lower for w in ["moratorium", "ban", "prohibition", "dissolution", "imprisonment"]):
        return 5.0
    elif any(w in desc_lower for w in ["criminal", "mandatory", "shutdown", "suspend"]):
        return 4.0
    elif any(w in desc_lower for w in ["mandatory", "require", "comply", "audit"]):
        return 3.0
    elif any(w in desc_lower for w in ["register", "report", "disclose"]):
        return 2.0
    else:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE OF HYBRID RUNS
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class HybridEnsembleReport:
    policy_name: str
    policy_severity: float
    n_runs: int
    results: list[HybridSimResult]

    def _mean_final(self, key: str) -> float:
        vals = [r.final_stocks.get(key, 0.0) for r in self.results]
        return float(np.mean(vals)) if vals else 0.0

    def _std_final(self, key: str) -> float:
        vals = [r.final_stocks.get(key, 0.0) for r in self.results]
        return float(np.std(vals)) if vals else 0.0

    def mean_compliance_trajectory(self) -> list[float]:
        if not self.results:
            return []
        n_rounds = len(self.results[0].round_summaries)
        trajectories = [r.compliance_trajectory() for r in self.results]
        return [np.mean([t[i] for t in trajectories if i < len(t)])
                for i in range(n_rounds)]

    def mean_relocation_trajectory(self) -> list[float]:
        if not self.results:
            return []
        n_rounds = len(self.results[0].round_summaries)
        trajectories = [r.relocation_trajectory() for r in self.results]
        return [np.mean([t[i] for t in trajectories if i < len(t)])
                for i in range(n_rounds)]

    def summary(self) -> str:
        sep = "═" * 68
        inner_sep = "─" * 68
        lines = [
            f"\n{sep}",
            f"V2 HYBRID STRESS TEST: {self.policy_name}",
            f"Severity: {self.policy_severity:.0f}/5 | "
            f"Ensemble: {self.n_runs} runs | "
            f"Population: {self.results[0].config.n_population if self.results else 0} agents",
            sep,
            "",
            "━━━ STOCK TRAJECTORIES (mean ± std across runs) ━━━",
            "",
            f"  {'Indicator':<28} {'Final mean':>12}  {'± std':>8}  "
            f"{'Unit':>20}",
            "  " + "─" * 70,
        ]
        indicators = [
            ("innovation_rate", "% TFP growth/yr",
             lambda v: f"{v/50*2.5:.2f}%"),
            ("ai_investment_index", "$B AI R&D/yr",
             lambda v: f"${v/50*50:.0f}B"),
            ("regulatory_burden", "% R&D overhead",
             lambda v: f"{v:.0f}%"),
            ("public_trust", "% pop. trust AI",
             lambda v: f"{v:.0f}%"),
            ("domestic_companies", "companies remaining",
             lambda v: f"{v:.0f}/100"),
        ]
        for key, unit, fmt in indicators:
            m = self._mean_final(key)
            s = self._std_final(key)
            lines.append(
                f"  {key:<28} {m:>12.1f}  ±{s:>6.1f}  {unit:>20}  [{fmt(m)}]"
            )

        lines += [
            "",
            "━━━ CALIBRATED COMPLIANCE TRAJECTORY ━━━",
            "  (Weibull model fitted to DLA Piper 2020 GDPR data)",
            "",
        ]
        compliance_traj = self.mean_compliance_trajectory()
        for i, rate in enumerate(compliance_traj):
            bar = "█" * int(rate * 30)
            lines.append(f"  Round {i+1:2d} ({(i+1)*3:2d}mo): {bar:<30} {rate:.1%}")

        lines += [
            "",
            "━━━ RELOCATION TRAJECTORY (with 2-4 round pipeline delay) ━━━",
            "",
        ]
        reloc_traj = self.mean_relocation_trajectory()
        for i, rate in enumerate(reloc_traj):
            bar = "█" * int(rate * 30)
            lines.append(f"  Round {i+1:2d} ({(i+1)*3:2d}mo): {bar:<30} {rate:.1%}")

        lines += [
            "",
            "━━━ SMM CALIBRATION DISTANCE (vs GDPR/EU AI Act moments) ━━━",
            "",
        ]
        smm_distances = [r.smm_distance_to_gdpr for r in self.results
                         if r.smm_distance_to_gdpr is not None]
        if smm_distances:
            target = TargetMoments()
            lines += [
                f"  Mean SMM distance: {np.mean(smm_distances):.4f}",
                f"  (0.0 = perfect calibration; >1.0 = poor fit)",
                "",
                f"  Target moments for comparison:",
                f"    lobbying:   {target.m1_large_company_lobbying_rate:.2f}  "
                f"(source: {target.m1_source})",
                f"    relocation: {target.m3_relocation_threat_rate:.2f}  "
                f"(source: EU AI Act Transparency Register)",
                f"    compliance: {target.m5_large_compliance_24mo:.2f}  "
                f"(source: {target.m5_source})",
                "",
                f"  Simulated (mean across runs):",
            ]
            # Compute means
            mean_lobby = np.mean([r.simulated_moments.lobbying_rate for r in self.results])
            mean_reloc = np.mean([r.simulated_moments.relocation_rate for r in self.results])
            mean_comp = np.mean([r.simulated_moments.compliance_rate_y1 for r in self.results])
            lines += [
                f"    lobbying:   {mean_lobby:.2f}",
                f"    relocation: {mean_reloc:.2f}",
                f"    compliance: {mean_comp:.2f}",
            ]

        # Failure modes
        # Thresholds below are [ASSUMED] heuristics, not empirically grounded cutoffs.
        #   final_reloc > 30%: corresponds to the rough threshold above which
        #     policy economists describe capital flight as "significant" (no single
        #     published cutoff; this is a rule-of-thumb from practitioner literature).
        #   final_comp < 50%: majority non-compliance at end of 16-round horizon
        #     is a pragmatic indicator of policy unenforcability.
        #   final_evade > 15%: one-in-six agents actively evading suggests
        #     enforcement is structurally overwhelmed.
        # These should be treated as alerting thresholds for further investigation,
        # not as definitive failure criteria.
        lines += [
            "",
            inner_sep,
            "FAILURE MODES (population-level):",
            "  [ASSUMED] Thresholds: reloc>30%, compliance<50%, evasion>15% are heuristics.",
            inner_sep,
        ]
        final_reloc = np.mean([r.final_population_summary.get("relocation_rate", 0)
                                for r in self.results])
        final_comp = np.mean([r.final_population_summary.get("compliance_rate", 0)
                               for r in self.results])
        final_evade = np.mean([r.final_population_summary.get("evasion_rate", 0)
                                for r in self.results])

        if final_reloc > 0.30:
            lines.append(f"  [HIGH] CAPITAL FLIGHT: {final_reloc:.1%} of companies relocated")
        if final_comp < 0.50:
            lines.append(f"  [HIGH] COMPLIANCE FAILURE: only {final_comp:.1%} fully compliant")
        if final_evade > 0.15:
            lines.append(f"  [MEDIUM] REGULATORY EVASION: {final_evade:.1%} evading")

        lines += [
            "",
            inner_sep,
            indicator_disclaimer(),
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────

class HybridStressTest:
    """Policy stress test using PolicyLab v2 hybrid simulation."""

    def __init__(
        self,
        n_population: int = 100,
        num_rounds: int = 16,
        spillover_factor: float = 0.5,
        output_dir: str = "./results/v2_stress_test",
    ):
        self.n_population = n_population
        self.num_rounds = num_rounds
        self.spillover_factor = spillover_factor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        policy_name: str,
        policy_description: str,
        policy_severity: float | None = None,
        n_ensemble: int = 5,
    ) -> HybridEnsembleReport:
        """Run n_ensemble hybrid simulations and aggregate results into an ensemble report.

        If policy_severity is None, severity is auto-detected from the description
        text via _detect_severity(). This is a keyword-match heuristic (ban/moratorium
        → 5, criminal/shutdown → 4, mandatory/comply → 3, etc.). Callers with known
        severity should pass it explicitly to avoid mis-classification.

        Only the first run in the ensemble uses verbose=True — subsequent runs run
        silently. This means detailed round-by-round output is shown once (to
        verify the simulation is behaving correctly) without flooding the console
        for every seed.

        Each run uses a different seed (42, 43, 44, …) so that stochastic variation
        in agent assignment, network topology, and behavioral draws is sampled
        independently. The ensemble report aggregates across all runs, providing
        mean ± std for each indicator.

        Results are saved as JSON to output_dir for later inspection.
        """
        if policy_severity is None:
            policy_severity = _detect_severity(policy_description)

        print(f"\n{'='*68}")
        print(f"V2 HYBRID STRESS TEST: {policy_name}")
        print(f"Severity: {policy_severity:.0f}/5 | "
              f"N: {self.n_population} agents | "
              f"Rounds: {self.num_rounds} | "
              f"Ensemble: {n_ensemble}")
        print(f"{'='*68}")

        results = []
        for i in range(n_ensemble):
            seed = 42 + i
            config = HybridSimConfig(
                n_population=self.n_population,
                num_rounds=self.num_rounds,
                spillover_factor=self.spillover_factor,
                seed=seed,
                verbose=(i == 0),  # only verbose for first run
            )
            print(f"\n  Run {i+1}/{n_ensemble} (seed={seed})...", end=" ", flush=True)
            import time; t0 = time.time()
            result = run_hybrid_simulation(
                policy_name=policy_name,
                policy_description=policy_description,
                policy_severity=policy_severity,
                config=config,
            )
            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s  "
                  f"compliance={result.final_population_summary.get('compliance_rate',0):.1%} "
                  f"reloc={result.final_population_summary.get('relocation_rate',0):.1%}")
            results.append(result)

        report = HybridEnsembleReport(
            policy_name=policy_name,
            policy_severity=policy_severity,
            n_runs=n_ensemble,
            results=results,
        )

        # Save
        path = os.path.join(
            self.output_dir,
            f"v2_{policy_name.lower().replace(' ','_')}_{datetime.now():%Y%m%d_%H%M%S}.json",
        )
        try:
            with open(path, "w") as f:
                json.dump({
                    "policy_name": policy_name,
                    "policy_severity": policy_severity,
                    "n_ensemble": n_ensemble,
                    "final_stocks": [r.final_stocks for r in results],
                    "final_populations": [r.final_population_summary for r in results],
                }, f, indent=2, default=str)
            print(f"\n  Report saved to {path}")
        except Exception as e:
            print(f"\n  Warning: could not save report: {e}")

        return report
