"""Run scenarios N times with different seeds and aggregate statistics.

EnsembleRunner executes a scenario function under a varied temperature schedule and
Lempert et al. (2003) parameter perturbation to decompose output variance across
plausible structural assumptions, then aggregates behavioral and indicator statistics.
"""
from __future__ import annotations
from policylab.disclaimers import indicator_disclaimer


import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable

import numpy as np

from policylab.components.actions import ActionType
from policylab.components.governance_state import GovernanceWorldState


class EnsembleRunner:
    """Run a scenario function N times and aggregate results."""

    TEMPERATURE_SCHEDULE = [
        0.3, 0.5, 0.7, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0,
        0.3, 0.5, 0.7, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0,
        0.3, 0.5, 0.7, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0,
    ]

    # Structural parameter perturbation schedule for uncertainty quantification.
    # Each tuple is (param_name, multiplier) applied to ResolutionConfig.
    # Grounded in Lempert et al. (2003) scenario discovery methodology:
    # vary model parameters across plausible ranges to decompose output variance.
    PARAM_PERTURBATION_SCHEDULE = [
        # Runs 0-9: baseline parameters
        {},
        {}, {}, {}, {},
        # Runs 10-14: optimistic enforcement (detection easier)
        {"enforcement_base_prob_per_severity": 1.5, "evasion_max_detection_probability": 1.1},
        {"enforcement_base_prob_per_severity": 1.5, "evasion_max_detection_probability": 1.1},
        # Runs 15-19: pessimistic enforcement (regulatory capacity lower)
        {"enforcement_base_prob_per_severity": 0.6, "relocation_investment_cost": 0.7},
        {"enforcement_base_prob_per_severity": 0.6, "relocation_investment_cost": 0.7},
        {},
        # Runs 20-24: high industry resistance (lobbying more effective)
        {"lobbying_base_resistance": 0.6, "compliance_innovation_cost": 1.4},
        {"lobbying_base_resistance": 0.6, "compliance_innovation_cost": 1.4},
        {},
        # Runs 25-29: strong civil society (public statements more effective)
        {"public_statement_max_trust_shift": 1.5, "severity_evasion_detection_bonus": 1.3},
        {"public_statement_max_trust_shift": 1.5, "severity_evasion_detection_bonus": 1.3},
    ]

    def __init__(self, n_runs: int = 30, output_dir: str = "./results/ensemble"):
        """Configure the runner with the number of ensemble runs and the directory for saved reports."""
        self.n_runs = n_runs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        scenario_fn: Callable[[int, float, bool], dict],
        scenario_name: str = "unnamed",
    ) -> EnsembleReport:
        """Run scenario_fn(seed, temperature, shuffle_agents) n_runs times and aggregate."""
        print(f"\n{'=' * 70}")
        print(f"ENSEMBLE RUN: {scenario_name}")
        print(f"N = {self.n_runs} runs")
        print(f"{'=' * 70}")

        run_data: list[dict] = []

        for i in range(self.n_runs):
            seed = 42 + i
            temperature = (
                self.TEMPERATURE_SCHEDULE[i % len(self.TEMPERATURE_SCHEDULE)]
            )
            shuffle_agents = (i % 3 != 0)
            param_overrides = self.PARAM_PERTURBATION_SCHEDULE[
                i % len(self.PARAM_PERTURBATION_SCHEDULE)
            ]

            t0 = time.time()
            try:
                result = scenario_fn(seed, temperature, shuffle_agents, param_overrides)
                elapsed = time.time() - t0
                result["_seed"] = seed
                result["_run_index"] = i
                result["_temperature"] = temperature
                result["_agents_shuffled"] = shuffle_agents
                result["_param_overrides"] = param_overrides
                result["_elapsed_seconds"] = elapsed
                run_data.append(result)
                override_str = f", params={list(param_overrides.keys())}" if param_overrides else ""
                print(f"  Run {i + 1}/{self.n_runs} (seed={seed}, temp={temperature}{override_str}) — {elapsed:.1f}s")
            except Exception as e:
                print(f"  Run {i + 1}/{self.n_runs} (seed={seed}) — FAILED: {e}")
                run_data.append({"_seed": seed, "_run_index": i, "_error": str(e)})

        report = EnsembleReport(
            scenario_name=scenario_name,
            n_runs=self.n_runs,
            run_data=run_data,
        )
        report.compute_statistics()

        path = os.path.join(
            self.output_dir, f"ensemble_{scenario_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        report.save(path)
        print(f"\nEnsemble report saved to {path}")

        return report


class EnsembleReport:
    """Aggregated statistics across ensemble runs."""

    def __init__(
        self,
        scenario_name: str,
        n_runs: int,
        run_data: list[dict],
    ):
        """Store raw run data for a named scenario; call compute_statistics() to populate stats."""
        self.scenario_name = scenario_name
        self.n_runs = n_runs
        self.run_data = run_data
        self.stats: dict[str, Any] = {}

    def compute_statistics(self) -> None:
        """Compute aggregate statistics across all runs."""
        successful_runs = [r for r in self.run_data if "_error" not in r]
        n_success = len(successful_runs)

        if n_success == 0:
            self.stats = {"error": "All runs failed"}
            return

        action_counts: Counter = Counter()
        action_by_agent: dict[str, Counter] = defaultdict(Counter)
        for run in successful_runs:
            for item in run.get("results", []):
                if "action" not in item:
                    continue  # skip enforcement events
                action = item["action"]
                atype = action.get("action_type", "other")
                actor = action.get("actor", "unknown")
                action_counts[atype] += 1
                action_by_agent[actor][atype] += 1

        outcome_rates: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "fail": 0, "blocked": 0})
        for run in successful_runs:
            for item in run.get("results", []):
                if "action" not in item:
                    continue
                atype = item["action"].get("action_type", "other")
                outcome = item.get("outcome", {})
                if outcome.get("blocked_reason"):
                    outcome_rates[atype]["blocked"] += 1
                elif outcome.get("success"):
                    outcome_rates[atype]["success"] += 1
                else:
                    outcome_rates[atype]["fail"] += 1

        indicator_values: dict[str, list[float]] = defaultdict(list)
        for run in successful_runs:
            ws = run.get("final_world_state", {})
            for k, v in ws.get("economic_indicators", {}).items():
                if isinstance(v, (int, float)):
                    indicator_values[k].append(v)

        indicator_stats = {}
        for k, values in indicator_values.items():
            if values:
                indicator_stats[k] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "values": values,
                }

        event_frequencies: dict[str, int] = Counter()
        for run in successful_runs:
            events_seen = set()
            for item in run.get("results", []):
                if "action" not in item:
                    continue
                action = item["action"]
                atype = action.get("action_type", "other")
                actor = action.get("actor", "unknown")
                event_key = f"{actor}_{atype}"
                events_seen.add(event_key)
            for event in events_seen:
                event_frequencies[event] += 1

        relocation_runs = 0
        for run in successful_runs:
            for item in run.get("results", []):
                if "action" not in item:
                    continue
                if item["action"].get("action_type") == "relocate":
                    if item.get("outcome", {}).get("success"):
                        relocation_runs += 1
                        break

        compliance_dist: dict[str, Counter] = defaultdict(Counter)
        for run in successful_runs:
            ws = run.get("final_world_state", {})
            for agent, policies in ws.get("compliance_tracker", {}).items():
                for pid, status in policies.items():
                    compliance_dist[agent][status] += 1

        # Tipping point analysis
        tipping_counts: dict[str, int] = {}
        convergence_count = 0
        for run in successful_runs:
            for tp in run.get("tipping_points_fired", []):
                # Extract short label from message
                if "CAPITAL FLIGHT" in tp:
                    tipping_counts["capital_flight_cascade"] = tipping_counts.get("capital_flight_cascade", 0) + 1
                elif "TRUST COLLAPSE" in tp:
                    tipping_counts["trust_collapse"] = tipping_counts.get("trust_collapse", 0) + 1
                elif "INNOVATION DEATH" in tp:
                    tipping_counts["innovation_death"] = tipping_counts.get("innovation_death", 0) + 1
            if run.get("system_converged"):
                convergence_count += 1

        self.stats = {
            "n_successful_runs": n_success,
            "n_failed_runs": self.n_runs - n_success,
            "action_frequency": dict(action_counts),
            "action_by_agent": {k: dict(v) for k, v in action_by_agent.items()},
            "outcome_rates": {k: dict(v) for k, v in outcome_rates.items()},
            "indicator_distributions": {
                k: {kk: vv for kk, vv in v.items() if kk != "values"}
                for k, v in indicator_stats.items()
            },
            "indicator_raw": {k: v["values"] for k, v in indicator_stats.items()},
            "event_frequencies": dict(event_frequencies),
            "event_frequency_pct": {
                k: f"{v / n_success:.0%}" for k, v in event_frequencies.items()
            },
            "relocation_rate": f"{relocation_runs / n_success:.0%}",
            "relocation_count": relocation_runs,
            "compliance_distribution": {k: dict(v) for k, v in compliance_dist.items()},
            "tipping_point_rates": {
                k: f"{v}/{n_success} runs ({v/n_success:.0%})"
                for k, v in tipping_counts.items()
            },
            "system_converged_rate": f"{convergence_count}/{n_success} runs",
        }

    def summary(self) -> str:
        """Human-readable summary."""
        active_layers = "GROUNDED only (rigorous baseline)"  # default
        lines = [
            f"{'=' * 60}",
            f"ENSEMBLE REPORT: {self.scenario_name}",
            f"{'=' * 60}",
            f"Runs: {self.stats.get('n_successful_runs', 0)}/{self.n_runs} successful",
            "",
            "PRIMARY OUTPUT — BEHAVIORAL ANALYSIS",
            "(Agent action patterns. This is the most reliable output.)",
            "",
        ]

        lines.append("ACTION FREQUENCIES (across all runs):")
        for action, count in sorted(
            self.stats.get("action_frequency", {}).items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"  {action}: {count}")

        lines.append("\nOUTCOME RATES:")
        for action, rates in self.stats.get("outcome_rates", {}).items():
            total = sum(rates.values())
            if total > 0:
                success_pct = rates.get("success", 0) / total
                lines.append(
                    f"  {action}: {success_pct:.0%} success, "
                    f"{rates.get('blocked', 0)} blocked, "
                    f"{rates.get('fail', 0)} failed"
                )

        lines.append(
            "\nSECONDARY OUTPUT — INDICATOR TENDENCIES\n"
            "  Direction only. Do not treat as point estimates.\n"
            "  Grounded: burden→investment (OECD PMR), R&D→innovation (Ugur 2016).\n"
            "  Ungrounded: trust feedback, passive burden, tipping points.\n"
            "  FINAL ECONOMIC INDICATORS (mean ± stdev):"
        )
        for indicator, stats in self.stats.get("indicator_distributions", {}).items():
            lines.append(
                f"  {indicator}: {stats['mean']:.1f} ± {stats['stdev']:.1f} "
                f"[{stats['min']:.1f} — {stats['max']:.1f}]"
            )

        lines.append(f"\nKEY GOVERNANCE METRICS:")
        lines.append(f"  Relocation rate: {self.stats.get('relocation_rate', 'N/A')}")

        if self.stats.get("tipping_point_rates"):
            lines.append("")
            lines.append("TIPPING POINTS REACHED:")
            for tp, rate in self.stats["tipping_point_rates"].items():
                lines.append(f"  {tp}: {rate}")

        lines.append(f"  System convergence: {self.stats.get('system_converged_rate', 'N/A')}")

        lines.append(f"\nEVENT FREQUENCIES (% of runs where this happened):")
        for event, pct in sorted(
            self.stats.get("event_frequency_pct", {}).items(),
            key=lambda x: -int(x[1].rstrip("%")),
        )[:15]:
            lines.append(f"  {event}: {pct}")

        lines.append(indicator_disclaimer())
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Serialize scenario name, run count, and computed statistics to a JSON file at path."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        export = {
            "scenario_name": self.scenario_name,
            "n_runs": self.n_runs,
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
        }
        with open(path, "w") as f:
            json.dump(export, f, indent=2, default=str)

    def print_summary(self) -> None:
        """Print the ensemble summary to stdout."""
        print(self.summary())
