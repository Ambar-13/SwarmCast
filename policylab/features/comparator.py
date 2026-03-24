"""Comparative analysis: counterfactual, sensitivity, and ablation tests."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from policylab.features.ensemble import EnsembleRunner, EnsembleReport
from policylab.game_master.resolution_config import ResolutionConfig


@dataclass
class CounterfactualResult:
    """Comparison between WITH-policy and WITHOUT-policy scenarios."""
    policy_name: str
    with_policy: EnsembleReport
    without_policy: EnsembleReport
    deltas: dict[str, float]

    def summary(self) -> str:
        """Return a formatted string showing per-indicator deltas between the with-policy and without-policy ensembles."""
        lines = [
            f"{'=' * 60}",
            f"COUNTERFACTUAL ANALYSIS: {self.policy_name}",
            f"{'=' * 60}",
            f"",
            f"INDICATOR DELTAS (WITH policy minus WITHOUT policy):",
            f"  Positive = policy INCREASES the indicator",
            f"  Negative = policy DECREASES the indicator",
            f"",
        ]

        for indicator, delta in sorted(self.deltas.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "+" if delta > 0 else "-" if delta < 0 else "="
            lines.append(f"  {indicator}: {delta:+.1f} {direction}")

        with_stats = self.with_policy.stats.get("indicator_distributions", {})
        without_stats = self.without_policy.stats.get("indicator_distributions", {})

        lines.append(f"\nDETAIL:")
        for indicator in self.deltas:
            w = with_stats.get(indicator, {})
            wo = without_stats.get(indicator, {})
            lines.append(
                f"  {indicator}: "
                f"WITH={w.get('mean', 0):.1f}±{w.get('stdev', 0):.1f} | "
                f"WITHOUT={wo.get('mean', 0):.1f}±{wo.get('stdev', 0):.1f}"
            )

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the counterfactual summary to stdout."""
        print(self.summary())


def run_counterfactual(
    with_policy_fn: Callable[[int], dict],
    without_policy_fn: Callable[[int], dict],
    policy_name: str,
    n_runs: int = 15,
    output_dir: str = "./results/counterfactual",
) -> CounterfactualResult:
    """Run the same scenario with and without a policy."""
    print(f"\n{'=' * 70}")
    print(f"COUNTERFACTUAL: {policy_name}")
    print(f"{'=' * 70}")

    print("\n--- WITH policy ---")
    runner_with = EnsembleRunner(n_runs=n_runs, output_dir=output_dir)
    with_report = runner_with.run(with_policy_fn, f"with_{policy_name}")

    print("\n--- WITHOUT policy ---")
    runner_without = EnsembleRunner(n_runs=n_runs, output_dir=output_dir)
    without_report = runner_without.run(without_policy_fn, f"without_{policy_name}")

    with_indicators = with_report.stats.get("indicator_distributions", {})
    without_indicators = without_report.stats.get("indicator_distributions", {})

    all_keys = set(with_indicators.keys()) | set(without_indicators.keys())
    deltas = {}
    for key in all_keys:
        w_mean = with_indicators.get(key, {}).get("mean", 0)
        wo_mean = without_indicators.get(key, {}).get("mean", 0)
        deltas[key] = w_mean - wo_mean

    result = CounterfactualResult(
        policy_name=policy_name,
        with_policy=with_report,
        without_policy=without_report,
        deltas=deltas,
    )

    return result


@dataclass
class SensitivityResult:
    """Outcome indicators measured at low, baseline, and high values of a single resolution parameter."""
    parameter_name: str
    baseline_value: float
    low_value: float
    high_value: float
    baseline_indicators: dict[str, float]
    low_indicators: dict[str, float]
    high_indicators: dict[str, float]
    sensitivity_scores: dict[str, float]


def run_sensitivity_analysis(
    scenario_fn_factory: Callable[[ResolutionConfig], Callable[[int], dict]],
    parameters_to_test: list[str] | None = None,
    n_runs: int = 10,
    variation_pct: float = 0.3,
    output_dir: str = "./results/sensitivity",
) -> list[SensitivityResult]:
    """Vary each resolution parameter and measure output changes."""
    import dataclasses
    base_config = ResolutionConfig()

    if parameters_to_test is None:
        parameters_to_test = [
            f.name for f in dataclasses.fields(ResolutionConfig)
            if isinstance(getattr(base_config, f.name), (int, float))
            and not f.name.startswith("failure_")
        ]

    print(f"\n{'=' * 70}")
    print(f"SENSITIVITY ANALYSIS")
    print(f"Testing {len(parameters_to_test)} parameters at +/-{variation_pct:.0%}")
    print(f"{'=' * 70}")

    results = []

    print("\n--- Baseline ---")
    baseline_fn = scenario_fn_factory(base_config)
    baseline_runner = EnsembleRunner(n_runs=n_runs, output_dir=output_dir)
    baseline_report = baseline_runner.run(baseline_fn, "baseline")
    baseline_indicators = {
        k: v.get("mean", 0)
        for k, v in baseline_report.stats.get("indicator_distributions", {}).items()
    }

    for param_name in parameters_to_test:
        base_value = getattr(base_config, param_name)
        if base_value == 0:
            continue

        low_value = base_value * (1 - variation_pct)
        high_value = base_value * (1 + variation_pct)

        print(f"\n--- {param_name}: {low_value:.1f} / {base_value:.1f} / {high_value:.1f} ---")

        low_config = copy.copy(base_config)
        setattr(low_config, param_name, low_value)
        low_fn = scenario_fn_factory(low_config)
        low_runner = EnsembleRunner(n_runs=n_runs, output_dir=output_dir)
        low_report = low_runner.run(low_fn, f"{param_name}_low")
        low_indicators = {
            k: v.get("mean", 0)
            for k, v in low_report.stats.get("indicator_distributions", {}).items()
        }

        high_config = copy.copy(base_config)
        setattr(high_config, param_name, high_value)
        high_fn = scenario_fn_factory(high_config)
        high_runner = EnsembleRunner(n_runs=n_runs, output_dir=output_dir)
        high_report = high_runner.run(high_fn, f"{param_name}_high")
        high_indicators = {
            k: v.get("mean", 0)
            for k, v in high_report.stats.get("indicator_distributions", {}).items()
        }

        sensitivity_scores = {}
        for indicator in baseline_indicators:
            baseline_val = baseline_indicators.get(indicator, 0)
            low_val = low_indicators.get(indicator, 0)
            high_val = high_indicators.get(indicator, 0)
            if baseline_val != 0:
                sensitivity_scores[indicator] = abs(high_val - low_val) / abs(baseline_val)
            else:
                sensitivity_scores[indicator] = abs(high_val - low_val)

        results.append(SensitivityResult(
            parameter_name=param_name,
            baseline_value=base_value,
            low_value=low_value,
            high_value=high_value,
            baseline_indicators=baseline_indicators,
            low_indicators=low_indicators,
            high_indicators=high_indicators,
            sensitivity_scores=sensitivity_scores,
        ))

    print(f"\n{'=' * 60}")
    print("SENSITIVITY RANKING (most influential parameters)")
    print(f"{'=' * 60}")

    ranked = sorted(
        results,
        key=lambda r: sum(r.sensitivity_scores.values()) / max(len(r.sensitivity_scores), 1),
        reverse=True,
    )
    for r in ranked:
        avg_sens = sum(r.sensitivity_scores.values()) / max(len(r.sensitivity_scores), 1)
        print(f"  {r.parameter_name}: avg_sensitivity={avg_sens:.3f}")
        for indicator, score in sorted(r.sensitivity_scores.items(), key=lambda x: -x[1])[:3]:
            print(f"    {indicator}: {score:.3f}")

    return results


def run_ablation_test(
    objectives_fn: Callable[[int], dict],
    personas_fn: Callable[[int], dict],
    n_runs: int = 15,
    output_dir: str = "./results/ablation",
) -> dict:
    """Compare constraint-based objectives vs simple persona prompts."""
    print(f"\n{'=' * 70}")
    print("ABLATION TEST: Objectives vs Personas")
    print(f"{'=' * 70}")

    print("\n--- With OBJECTIVES (game-theoretic) ---")
    obj_runner = EnsembleRunner(n_runs=n_runs, output_dir=output_dir)
    obj_report = obj_runner.run(objectives_fn, "objectives")

    print("\n--- With PERSONAS (simple descriptions) ---")
    persona_runner = EnsembleRunner(n_runs=n_runs, output_dir=output_dir)
    persona_report = persona_runner.run(personas_fn, "personas")

    obj_actions = obj_report.stats.get("action_frequency", {})
    persona_actions = persona_report.stats.get("action_frequency", {})

    obj_indicators = obj_report.stats.get("indicator_distributions", {})
    persona_indicators = persona_report.stats.get("indicator_distributions", {})

    action_divergence = {}
    all_action_types = set(obj_actions.keys()) | set(persona_actions.keys())
    for at in all_action_types:
        obj_count = obj_actions.get(at, 0)
        persona_count = persona_actions.get(at, 0)
        total = obj_count + persona_count
        if total > 0:
            action_divergence[at] = abs(obj_count - persona_count) / total

    indicator_divergence = {}
    for ind in set(obj_indicators.keys()) | set(persona_indicators.keys()):
        obj_mean = obj_indicators.get(ind, {}).get("mean", 0)
        persona_mean = persona_indicators.get(ind, {}).get("mean", 0)
        if obj_mean != 0:
            indicator_divergence[ind] = abs(obj_mean - persona_mean) / abs(obj_mean)
        else:
            indicator_divergence[ind] = abs(obj_mean - persona_mean)

    avg_action_div = (
        sum(action_divergence.values()) / len(action_divergence)
        if action_divergence else 0
    )
    avg_indicator_div = (
        sum(indicator_divergence.values()) / len(indicator_divergence)
        if indicator_divergence else 0
    )

    objectives_matter = avg_action_div > 0.15 or avg_indicator_div > 0.1

    result = {
        "verdict": "OBJECTIVES MATTER" if objectives_matter else "NO SIGNIFICANT DIFFERENCE",
        "avg_action_divergence": avg_action_div,
        "avg_indicator_divergence": avg_indicator_div,
        "action_divergence": action_divergence,
        "indicator_divergence": indicator_divergence,
        "objectives_report": obj_report.stats,
        "personas_report": persona_report.stats,
    }

    print(f"\n{'=' * 60}")
    print(f"ABLATION RESULT: {result['verdict']}")
    print(f"Action divergence: {avg_action_div:.3f} (>0.15 = significant)")
    print(f"Indicator divergence: {avg_indicator_div:.3f} (>0.10 = significant)")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"ablation_{datetime.now():%Y%m%d_%H%M%S}.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result
