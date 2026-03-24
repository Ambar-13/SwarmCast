"""PolicyLab v2 analysis tools.

Three modules in one file:
  1. CounterfactualV2    — clean no-policy baseline for v2 hybrid simulation
  2. PolicyComparator    — rank N policies by impact severity
  3. SensitivityV2       — parameter sweep over assumed coefficients

All three are architecture-correct for v2 (proper stock-flow, calibrated agents).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. COUNTERFACTUAL — clean no-regulation baseline
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CounterfactualResultV2:
    """Delta between a policy scenario and the clean no-regulation baseline."""
    policy_name: str
    policy_severity: float

    # Treated scenario
    treatment_final: dict
    treatment_reloc: float
    treatment_compliance: float

    # Clean baseline
    baseline_final: dict
    baseline_reloc: float
    baseline_compliance: float

    # Deltas (treatment - baseline)
    delta: dict

    def summary(self) -> str:
        sep = "─" * 68
        lines = [
            f"\n{sep}",
            f"COUNTERFACTUAL ANALYSIS — {self.policy_name}",
            f"(policy outcome vs clean no-regulation baseline)",
            sep,
            "",
            f"{'Indicator':<28} {'Baseline':>10}  {'Policy':>10}  {'Delta':>10}  {'Unit':>16}",
            "  " + "─" * 72,
        ]
        units = {
            "innovation_rate": "% TFP/yr",
            "ai_investment_index": "$B/yr",
            "regulatory_burden": "% R&D overhead",
            "public_trust": "% trust AI",
            "domestic_companies": "firms remain",
        }
        for key, unit in units.items():
            base = self.baseline_final.get(key, 0)
            treat = self.treatment_final.get(key, 0)
            delta = treat - base
            sign = "+" if delta > 0 else ""
            lines.append(
                f"  {key:<28} {base:>10.1f}  {treat:>10.1f}  "
                f"{sign}{delta:>9.1f}  {unit:>16}"
            )
        lines += [
            "",
            f"  relocation_rate:  baseline={self.baseline_reloc:.1%}  "
            f"policy={self.treatment_reloc:.1%}  "
            f"delta={self.treatment_reloc - self.baseline_reloc:+.1%}",
            "",
            "  INTERPRETATION:",
            "    Negative innovation/investment deltas = policy caused decline.",
            "    Positive burden delta = policy added compliance overhead.",
            "    Relocation delta = policy-attributable capital flight.",
            f"    [EPISTEMIC] Baseline assumes neutral world. "
            f"Any remaining difference is policy effect.",
            sep,
        ]
        return "\n".join(lines)


def run_counterfactual_v2(
    policy_name: str,
    policy_description: str,
    policy_severity: float,
    n_population: int = 100,
    num_rounds: int = 16,
    n_ensemble: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> CounterfactualResultV2:
    """Run treatment + clean baseline, return delta.

    The baseline uses a genuinely neutral world premise: severity=0.01 with
    a description that explicitly describes normal commercial operation ("No
    major new regulation has been proposed or enacted"). This is critical.
    Any policy text in the baseline contaminates the result — agents respond
    to the description linguistically as well as to the severity number, so
    even a neutral-sounding description that mentions "regulation" will shift
    agent beliefs and compliance behaviour relative to a truly clean baseline.
    The description is therefore written to contain no regulatory framing at all.

    Same population seed, same network, same initial conditions — only the
    policy stimulus and its burden are absent. The delta (treatment - baseline)
    isolates the policy's causal effect.

    This is the v2-correct version of the counterfactual fix from v1.
    """
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation
    from policylab.v2.population.agents import generate_population

    def _run_ensemble(is_baseline: bool, label: str) -> tuple[dict, float, float]:
        """Run n_ensemble simulations, return mean final stocks + behaviour."""
        all_finals = []
        all_reloc = []
        all_comp = []
        if verbose:
            print(f"  [{label}] Running {n_ensemble} runs...")
        for i in range(n_ensemble):
            config = HybridSimConfig(
                n_population=n_population,
                num_rounds=num_rounds,
                verbose=False,
                seed=seed + i,
            )
            # Baseline: severity 0 → no burden added, no relocation incentive
            eff_severity = 0.01 if is_baseline else policy_severity
            eff_name = "[Baseline — No Policy]" if is_baseline else policy_name
            eff_desc = (
                "The AI sector operates normally. No major new regulation has been "
                "proposed or enacted. Standard commercial law applies. Companies "
                "compete freely. Researchers publish and build without restriction."
                if is_baseline else policy_description
            )
            r = run_hybrid_simulation(eff_name, eff_desc, eff_severity, config)
            all_finals.append(r.final_stocks)
            all_reloc.append(r.final_population_summary.get("relocation_rate", 0))
            all_comp.append(r.final_population_summary.get("compliance_rate", 0))

        mean_final = {
            k: float(np.mean([f.get(k, 0) for f in all_finals]))
            for k in all_finals[0].keys()
        }
        return mean_final, float(np.mean(all_reloc)), float(np.mean(all_comp))

    treatment_final, t_reloc, t_comp = _run_ensemble(False, "POLICY")
    baseline_final, b_reloc, b_comp = _run_ensemble(True, "BASELINE")

    delta = {k: treatment_final.get(k, 0) - baseline_final.get(k, 0)
             for k in treatment_final}

    return CounterfactualResultV2(
        policy_name=policy_name,
        policy_severity=policy_severity,
        treatment_final=treatment_final,
        treatment_reloc=t_reloc,
        treatment_compliance=t_comp,
        baseline_final=baseline_final,
        baseline_reloc=b_reloc,
        baseline_compliance=b_comp,
        delta=delta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. POLICY COMPARATOR — rank N policies by impact severity
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ComparisonPolicy:
    """Lightweight policy descriptor for compare_policies().

    Distinct from policylab.v2.policy.parser.PolicySpec (which has 10+ fields
    and carries the full provenance chain). Use ComparisonPolicy when you just
    want to compare a few named scenarios without running the full parser.
    """
    name: str
    description: str
    severity: float | None = None

# Alias for backwards compatibility — existing code using PolicySpec from
# this module continues to work.
PolicySpec = ComparisonPolicy


@dataclasses.dataclass
class PolicyRanking:
    """Ranked list of policies by impact severity."""
    policies: list[ComparisonPolicy]
    results: list[dict]      # one per policy, keyed by indicator
    rankings: dict[str, list[str]]  # indicator → policy names in descending impact

    def summary(self) -> str:
        sep = "═" * 68
        lines = [
            f"\n{sep}",
            "POLICY COMPARISON RANKING",
            f"{len(self.policies)} policies compared across key governance indicators",
            sep,
            "",
        ]

        # For each key indicator, show ranking
        indicators = [
            ("relocation_rate", "Relocation (capital flight)", True, "%"),
            ("regulatory_burden", "Regulatory burden", True, "pts"),
            ("delta_innovation", "Innovation impact (delta)", True, "pts"),
            ("delta_investment", "Investment impact (delta)", True, "pts"),
        ]
        for key, label, higher_is_worse, unit in indicators:
            lines.append(f"  {label}:")
            sorted_results = sorted(
                self.results,
                key=lambda r: abs(r.get(key, 0)),
                reverse=True,
            )
            for rank, r in enumerate(sorted_results, 1):
                val = r.get(key, 0)
                # relocation_rate is stored as fraction 0-1; display as percentage
                if key == "relocation_rate":
                    display_val = val * 100
                    display_unit = "%"
                else:
                    display_val = val
                    display_unit = unit
                sign = "+" if display_val > 0 and not higher_is_worse else ""
                sev_tag = " ← most severe" if rank == 1 else ""
                lines.append(
                    f"    {rank}. {r['name']:<35} {sign}{display_val:+.1f}{display_unit}{sev_tag}"
                )
            lines.append("")

        lines += [
            sep,
            "OVERALL RANKING (composite impact score = sum of |deltas|):",
            "",
        ]
        scored = sorted(
            self.results,
            key=lambda r: (
                abs(r.get("delta_innovation", 0)) +
                abs(r.get("delta_investment", 0)) +
                r.get("relocation_rate", 0) * 100 +
                r.get("regulatory_burden", 0)
            ),
            reverse=True,
        )
        for rank, r in enumerate(scored, 1):
            score = (
                abs(r.get("delta_innovation", 0)) +
                abs(r.get("delta_investment", 0)) +
                r.get("relocation_rate", 0) * 100 +
                r.get("regulatory_burden", 0)
            )
            lines.append(
                f"  {rank}. [{r['severity']:.0f}/5] {r['name']:<40} score={score:.0f}"
            )
        lines += [
            "",
            "  Score = |Δinnovation| + |Δinvestment| + relocation*100 + burden",
            "  [ORDINAL] Rankings are robust. Exact scores are not calibrated magnitudes.",
            sep,
        ]
        return "\n".join(lines)


def compare_policies(
    policies: list[PolicySpec],
    n_population: int = 100,
    num_rounds: int = 16,
    n_ensemble: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> PolicyRanking:
    """Compare N policies by running a counterfactual for each against a shared baseline.

    A single baseline ensemble (severity=0.01, neutral description) is computed
    once and reused for all policies. This is essential for cross-policy
    comparability: if each policy had its own baseline with slightly different
    stochastic draws, the deltas (treatment - baseline) would reflect noise
    differences as well as policy differences, making ranking unreliable.

    Composite impact score (for overall ranking):
      score = |Δinnovation| + |Δinvestment| + relocation_rate × 100 + burden

    The relocation_rate is multiplied by 100 to put it on the same scale as
    the index-based innovation and investment deltas. This is an ordinal
    comparison only — the absolute scores are not calibrated magnitudes.
    [ORDINAL] Rankings are robust; exact score values should not be interpreted
    as proportional measures of policy harm.
    """
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation

    if verbose:
        print(f"\n{'='*68}")
        print(f"POLICY COMPARISON: {len(policies)} policies")
        print(f"{'='*68}")

    # Compute shared baseline once
    if verbose:
        print("\n  Computing shared no-regulation baseline...")
    all_baseline = []
    for i in range(n_ensemble):
        config = HybridSimConfig(
            n_population=n_population, num_rounds=num_rounds,
            verbose=False, seed=seed + i,
        )
        r = run_hybrid_simulation(
            "[Baseline]",
            "No new regulation. Normal AI sector operation.",
            0.01, config,
        )
        all_baseline.append(r)
    baseline_final = {
        k: float(np.mean([r.final_stocks.get(k, 0) for r in all_baseline]))
        for k in all_baseline[0].final_stocks
    }
    baseline_reloc = float(np.mean([
        r.final_population_summary.get("relocation_rate", 0) for r in all_baseline
    ]))

    # Run each policy
    results = []
    for pol in policies:
        if verbose:
            print(f"\n  Policy: {pol.name} (severity={pol.severity or 'auto'})")
        sev = pol.severity
        if sev is None:
            from policylab.v2.stress_test_v2 import _detect_severity
            sev = _detect_severity(pol.description)

        pol_finals = []
        pol_relocs = []
        for i in range(n_ensemble):
            config = HybridSimConfig(
                n_population=n_population, num_rounds=num_rounds,
                verbose=False, seed=seed + i,
            )
            r = run_hybrid_simulation(pol.name, pol.description, sev, config)
            pol_finals.append(r.final_stocks)
            pol_relocs.append(r.final_population_summary.get("relocation_rate", 0))
            if verbose:
                print(f"    run {i+1}: reloc={pol_relocs[-1]:.0%}  "
                      f"burden={r.final_stocks.get('regulatory_burden',0):.0f}")

        mean_final = {
            k: float(np.mean([f.get(k, 0) for f in pol_finals]))
            for k in pol_finals[0]
        }
        results.append({
            "name": pol.name,
            "severity": sev,
            "relocation_rate": float(np.mean(pol_relocs)),
            "regulatory_burden": mean_final.get("regulatory_burden", 0),
            "delta_innovation": mean_final.get("innovation_rate", 0) - baseline_final.get("innovation_rate", 100),
            "delta_investment": mean_final.get("ai_investment_index", 0) - baseline_final.get("ai_investment_index", 100),
            "final_stocks": mean_final,
        })

    # Build rankings per indicator
    rankings = {
        "relocation": [r["name"] for r in sorted(results, key=lambda r: r["relocation_rate"], reverse=True)],
        "burden": [r["name"] for r in sorted(results, key=lambda r: r["regulatory_burden"], reverse=True)],
        "innovation_impact": [r["name"] for r in sorted(results, key=lambda r: abs(r["delta_innovation"]), reverse=True)],
    }

    return PolicyRanking(policies=policies, results=results, rankings=rankings)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SENSITIVITY ANALYSIS V2 — sweep assumed parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SensitivityResultV2:
    """Sensitivity of v2 outputs to assumed parameter values."""
    parameter_name: str
    values_tested: list[float]
    output_by_value: dict[float, dict]  # value → {indicator: mean}

    def robustness_classification(self, indicator: str) -> str:
        """Classify robustness by sign-consistency across the parameter grid: ≥90% = ROBUST, 70–90% = DIRECTIONAL-DEPENDENT, <70% = NON-ROBUST."""
        vals = [self.output_by_value[v].get(indicator, 0) for v in self.values_tested]
        if len(vals) < 2:
            return "UNKNOWN"

        indicator_range = max(vals) - min(vals)
        noise_threshold = max(0.02 * 100, 0.02 * indicator_range)  # 2% of range or 2pts abs

        n_pairs = len(vals) - 1
        if n_pairs == 0:
            return "UNKNOWN"

        # Count sign-consistent adjacent pairs, ignoring noise-level changes
        up = 0
        down = 0
        for i in range(n_pairs):
            diff = vals[i + 1] - vals[i]
            if abs(diff) < noise_threshold:
                continue  # too small to call direction
            if diff > 0:
                up += 1
            else:
                down += 1

        total_meaningful = up + down
        if total_meaningful == 0:
            return "ROBUST"  # all changes within noise — effectively flat = robust

        dominant = max(up, down)
        consistency = dominant / total_meaningful

        if consistency >= 0.90:
            return "ROBUST"
        elif consistency >= 0.70:
            return "DIRECTIONAL-DEPENDENT"
        else:
            return "NON-ROBUST"

    def summary(self) -> str:
        """Display sensitivity sweep results and robustness classification per indicator."""
        sep = "─" * 68
        lines = [
            f"\n{sep}",
            f"SENSITIVITY: {self.parameter_name}",
            f"Tested values: {self.values_tested}",
            sep,
        ]
        indicators = ["innovation_rate", "ai_investment_index",
                      "regulatory_burden", "relocation_rate"]
        header = f"  {'Value':>8} "
        for ind in indicators:
            header += f"  {ind[:10]:>12}"
        lines.append(header)
        lines.append("  " + "─" * (10 + 14 * len(indicators)))

        for val in self.values_tested:
            row = f"  {val:>8.3f} "
            for ind in indicators:
                v = self.output_by_value[val].get(ind, 0)
                row += f"  {v:>12.1f}"
            lines.append(row)

        lines.append("")
        lines.append("  ROBUSTNESS (Saltelli 2004 ≥90% = ROBUST):")
        for ind in indicators:
            cls = self.robustness_classification(ind)
            lines.append(f"    {ind:<28} {cls}")
        lines.append(sep)
        return "\n".join(lines)


def run_sensitivity_v2(
    policy_name: str,
    policy_description: str,
    policy_severity: float,
    parameter: str,
    values: list[float],
    n_population: int = 50,
    num_rounds: int = 8,
    n_ensemble: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> SensitivityResultV2:
    """Sweep one assumed parameter across a range of values and classify robustness.

    This is the primary tool for testing how sensitive conclusions are to the
    [ASSUMED] constants in the model. Every parameter marked [ASSUMED] in the
    codebase has a recommended sweep range below.

    Supported parameters and default sweep ranges:
      spillover_factor     — fraction of innovation remaining globally accessible
                             after relocation [0.3, 0.5, 0.7]. Tests whether
                             the conclusion that relocation destroys domestic
                             innovation holds across different assumptions about
                             how much relocated research remains globally accessible.
      ongoing_burden_rate  — [DIRECTIONAL] burden added per round per severity
                             unit [0.5, 1.5, 3.0]. Tests whether results depend
                             on how quickly burden accumulates under sustained policy.
      n_population         — number of agents [50, 100, 200]. Tests whether
                             population-level statistics are scale-stable (they
                             should be; if not, there is a model artefact).

    Returns SensitivityResultV2 with per-indicator robustness classification
    using the Saltelli (2004) sign-consistency test.
    """
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation
    import importlib

    if verbose:
        print(f"\nSensitivity sweep: {parameter} ∈ {values}")

    output_by_value = {}

    for val in values:
        if verbose:
            print(f"  {parameter}={val:.3f}...", end=" ", flush=True)

        run_results = []
        for i in range(n_ensemble):
            config = HybridSimConfig(
                n_population=int(val) if parameter == "n_population" else n_population,
                num_rounds=num_rounds,
                verbose=False,
                seed=seed + i,
                spillover_factor=val if parameter == "spillover_factor" else 0.5,
            )

            # Patch ongoing burden rate if testing that
            if parameter == "ongoing_burden_rate":
                import policylab.v2.simulation.hybrid_loop as hl_mod
                original = getattr(hl_mod, "_ONGOING_BURDEN_RATE", None)
                hl_mod._ONGOING_BURDEN_RATE = val

            r = run_hybrid_simulation(policy_name, policy_description, policy_severity, config)

            if parameter == "ongoing_burden_rate":
                if original is not None:
                    hl_mod._ONGOING_BURDEN_RATE = original

            run_results.append(r)

        mean_out = {
            k: float(np.mean([r.final_stocks.get(k, 0) for r in run_results]))
            for k in run_results[0].final_stocks
        }
        mean_out["relocation_rate"] = float(np.mean([
            r.final_population_summary.get("relocation_rate", 0) for r in run_results
        ]))
        output_by_value[val] = mean_out
        if verbose:
            print(f"reloc={mean_out['relocation_rate']:.0%}  "
                  f"inn={mean_out.get('innovation_rate',0):.0f}  "
                  f"bur={mean_out.get('regulatory_burden',0):.0f}")

    return SensitivityResultV2(
        parameter_name=parameter,
        values_tested=values,
        output_by_value=output_by_value,
    )
