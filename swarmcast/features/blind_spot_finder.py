"""Find governance blind spots by running randomized scenario variations."""

from __future__ import annotations

import itertools
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from swarmcast.features.war_game import (
    GovernanceFramework,
    IncidentTemplate,
    INCIDENT_TEMPLATES,
    FRAMEWORK_MODERATE,
    WarGame,
)
from concordia.language_model import language_model as lm_lib


SEVERITY_LEVELS = ["low", "medium", "high", "catastrophic"]
CATEGORIES = ["misuse", "accident", "systemic", "adversarial"]
TIME_PRESSURES = ["none", "moderate", "extreme"]
ACTOR_CAPABILITIES = ["low", "medium", "high"]

# Map blind-spot dimensions to concrete simulation parameters
ROUNDS_MAP: dict[str, int] = {"none": 8, "moderate": 5, "extreme": 3}
CAPABILITY_MAP: dict[str, float] = {"low": 0.5, "medium": 1.0, "high": 2.0}


@dataclass
class ScenarioVariation:
    """A randomly sampled combination of incident, severity, time pressure, actor capability, and initial conditions."""
    base_incident: IncidentTemplate
    severity_override: str
    time_pressure: str
    actor_capability: str
    initial_trust: float
    initial_investment: float
    regulator_capacity_multiplier: float

    @property
    def name(self) -> str:
        return (
            f"{self.base_incident.name} "
            f"[{self.severity_override}/{self.time_pressure}pressure/"
            f"{self.actor_capability}cap]"
        )

    def to_dict(self) -> dict:
        return {
            "base_incident": self.base_incident.name,
            "severity": self.severity_override,
            "time_pressure": self.time_pressure,
            "actor_capability": self.actor_capability,
            "initial_trust": self.initial_trust,
            "initial_investment": self.initial_investment,
            "regulator_capacity_multiplier": self.regulator_capacity_multiplier,
        }


@dataclass
class BlindSpot:
    """A governance failure detected in a specific scenario variation, recording the failure type, severity, and final indicator values."""
    scenario: ScenarioVariation
    failure_type: str
    severity: str
    description: str
    final_indicators: dict[str, float]


class BlindSpotFinder:
    """Find governance blind spots by running many scenario variations."""

    def __init__(
        self,
        model: lm_lib.LanguageModel,
        embedder,
        n_scenarios: int = 50,
        n_ensemble_per_scenario: int = 5,
        num_rounds: int = 4,
        output_dir: str = "./results/blind_spots",
        failure_thresholds: dict[str, float] | None = None,
    ):
        """Configure the blind-spot finder. failure_thresholds defaults to standard governance targets if not provided."""
        self.model = model
        self.embedder = embedder
        self.n_scenarios = n_scenarios
        self.n_ensemble = n_ensemble_per_scenario
        self.num_rounds = num_rounds
        self.output_dir = output_dir
        self.failure_thresholds = failure_thresholds or {
            "trust": 30.0,
            "investment": 60.0,
            "innovation": 50.0,
            "relocation_rate": 0.5,
        }

    def find_blind_spots(
        self,
        framework: GovernanceFramework | None = None,
        seed: int = 42,
    ) -> BlindSpotReport:
        """Generate N scenario variations and find where the framework fails."""
        if framework is None:
            framework = FRAMEWORK_MODERATE

        rng = random.Random(seed)
        variations = self._generate_variations(rng)

        print(f"\n{'=' * 70}")
        print(f"BLIND SPOT FINDER")
        print(f"Framework: {framework.name}")
        print(f"Testing {len(variations)} scenario variations")
        print(f"{'=' * 70}")

        war_game = WarGame(
            model=self.model,
            embedder=self.embedder,
            n_ensemble=self.n_ensemble,
            num_rounds=self.num_rounds,
            output_dir=os.path.join(self.output_dir, "runs"),
        )

        all_results: list[dict] = []
        blind_spots: list[BlindSpot] = []

        for i, variation in enumerate(variations):
            print(f"\n  Scenario {i + 1}/{len(variations)}: {variation.name}")

            modified_incident = IncidentTemplate(
                name=variation.name,
                description=variation.base_incident.description,
                severity=variation.severity_override,
                category=variation.base_incident.category,
                initial_conditions={
                    "public_trust": variation.initial_trust,
                    "ai_investment_index": variation.initial_investment,
                },
            )

            modified_framework = GovernanceFramework(
                name=framework.name,
                description=framework.description,
                policies=framework.policies,
                regulator_resources={
                    k: v * variation.regulator_capacity_multiplier
                    for k, v in framework.regulator_resources.items()
                },
            )

            try:
                report = war_game.run_war_game(
                    incident=modified_incident,
                    frameworks=[modified_framework],
                    num_rounds_override=ROUNDS_MAP[variation.time_pressure],
                    company_resource_multiplier=CAPABILITY_MAP[variation.actor_capability],
                )

                fw_data = report.framework_results.get(framework.name)
                if fw_data:
                    stats = fw_data["ensemble"].stats
                    indicators = stats.get("indicator_distributions", {})

                    trust_mean = indicators.get("public_trust", {}).get("mean", 50)
                    invest_mean = indicators.get("ai_investment_index", {}).get("mean", 100)
                    innovation_mean = indicators.get("innovation_rate", {}).get("mean", 100)

                    failures = []
                    if trust_mean < self.failure_thresholds["trust"]:
                        failures.append(("trust_collapse", "high", f"Trust dropped to {trust_mean:.0f}"))
                    if invest_mean < self.failure_thresholds["investment"]:
                        failures.append(("investment_flight", "high", f"Investment dropped to {invest_mean:.0f}"))
                    if innovation_mean < self.failure_thresholds["innovation"]:
                        failures.append(("innovation_death", "high", f"Innovation dropped to {innovation_mean:.0f}"))

                    reloc = stats.get("relocation_count", 0)
                    n_runs = stats.get("n_successful_runs", 1)
                    if reloc / max(n_runs, 1) > self.failure_thresholds["relocation_rate"]:
                        failures.append(("mass_relocation", "critical", f"Relocation in {reloc}/{n_runs} runs"))

                    for ftype, fseverity, fdesc in failures:
                        blind_spots.append(BlindSpot(
                            scenario=variation,
                            failure_type=ftype,
                            severity=fseverity,
                            description=fdesc,
                            final_indicators={
                                "public_trust": trust_mean,
                                "ai_investment_index": invest_mean,
                                "innovation_rate": innovation_mean,
                            },
                        ))

                    all_results.append({
                        "variation": variation.to_dict(),
                        "indicators": {
                            k: v.get("mean", 0) for k, v in indicators.items()
                        },
                        "failures": [(f, s, d) for f, s, d in failures],
                    })

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({
                    "variation": variation.to_dict(),
                    "error": str(e),
                })

        report = BlindSpotReport(
            framework=framework,
            n_scenarios=len(variations),
            all_results=all_results,
            blind_spots=blind_spots,
        )

        report.save(os.path.join(
            self.output_dir,
            f"blind_spots_{framework.name.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M%S}.json",
        ))

        return report

    def _generate_variations(self, rng: random.Random) -> list[ScenarioVariation]:
        """Generate N random scenario variations."""
        variations = []
        for _ in range(self.n_scenarios):
            base = rng.choice(INCIDENT_TEMPLATES)
            variations.append(ScenarioVariation(
                base_incident=base,
                severity_override=rng.choice(SEVERITY_LEVELS),
                time_pressure=rng.choice(TIME_PRESSURES),
                actor_capability=rng.choice(ACTOR_CAPABILITIES),
                initial_trust=rng.uniform(20, 60),
                initial_investment=rng.uniform(60, 100),
                regulator_capacity_multiplier=rng.uniform(0.3, 1.5),
            ))
        return variations


class BlindSpotReport:
    """Report of governance blind spots found."""

    def __init__(
        self,
        framework: GovernanceFramework,
        n_scenarios: int,
        all_results: list[dict],
        blind_spots: list[BlindSpot],
    ):
        """Store the tested framework, scenario count, all simulation results, and the detected blind spots."""
        self.framework = framework
        self.n_scenarios = n_scenarios
        self.all_results = all_results
        self.blind_spots = blind_spots

    def summary(self) -> str:
        """Return a formatted string listing failure type distribution, most dangerous incidents, and the five worst blind spots."""
        lines = [
            f"{'=' * 60}",
            f"BLIND SPOT REPORT: {self.framework.name}",
            f"{'=' * 60}",
            f"Scenarios tested: {self.n_scenarios}",
            f"Blind spots found: {len(self.blind_spots)}",
            f"Failure rate: {len(self.blind_spots) / max(self.n_scenarios, 1):.0%}",
            "",
        ]

        type_counts = Counter(bs.failure_type for bs in self.blind_spots)
        lines.append("FAILURE TYPE DISTRIBUTION:")
        for ftype, count in type_counts.most_common():
            lines.append(f"  {ftype}: {count}")

        incident_counts = Counter(
            bs.scenario.base_incident.name for bs in self.blind_spots
        )
        lines.append("\nMOST DANGEROUS INCIDENTS:")
        for incident, count in incident_counts.most_common(5):
            lines.append(f"  {incident}: {count} failures")

        lines.append("\nFAILURE HEAT MAP (which dimensions cause failures):")

        severity_counts = Counter(bs.scenario.severity_override for bs in self.blind_spots)
        lines.append(f"  By severity: {dict(severity_counts)}")

        pressure_counts = Counter(bs.scenario.time_pressure for bs in self.blind_spots)
        lines.append(f"  By time pressure: {dict(pressure_counts)}")

        cap_counts = Counter(bs.scenario.actor_capability for bs in self.blind_spots)
        lines.append(f"  By actor capability: {dict(cap_counts)}")

        if self.blind_spots:
            avg_reg_cap = sum(
                bs.scenario.regulator_capacity_multiplier for bs in self.blind_spots
            ) / len(self.blind_spots)
            lines.append(f"  Avg regulator capacity multiplier in failures: {avg_reg_cap:.2f}x")

        lines.append(f"\n{'─' * 60}")
        lines.append("TOP 5 WORST BLIND SPOTS:")
        worst = sorted(
            self.blind_spots,
            key=lambda bs: bs.final_indicators.get("public_trust", 100),
        )[:5]
        for i, bs in enumerate(worst):
            lines.append(
                f"\n  {i + 1}. {bs.scenario.base_incident.name}"
                f"\n     Conditions: severity={bs.scenario.severity_override}, "
                f"pressure={bs.scenario.time_pressure}, "
                f"capability={bs.scenario.actor_capability}"
                f"\n     Failure: {bs.failure_type} — {bs.description}"
                f"\n     Final trust: {bs.final_indicators.get('public_trust', 0):.0f}"
            )

        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "framework": self.framework.name,
            "n_scenarios": self.n_scenarios,
            "n_blind_spots": len(self.blind_spots),
            "failure_rate": len(self.blind_spots) / max(self.n_scenarios, 1),
            "blind_spots": [
                {
                    "scenario": bs.scenario.to_dict(),
                    "failure_type": bs.failure_type,
                    "severity": bs.severity,
                    "description": bs.description,
                    "final_indicators": bs.final_indicators,
                }
                for bs in self.blind_spots
            ],
            "all_results": self.all_results,
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
