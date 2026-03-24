"""Stress-test proposed regulations through multi-agent simulation."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model as lm_lib
from swarmcast.disclaimers import indicator_disclaimer

from swarmcast.agents.governance_agents import (
    build_civil_society_agent,
    build_company_agent,
    build_government_agent,
    build_regulator_agent,
    build_bad_actor_agent,
)
from swarmcast.components.actions import ActionType
from swarmcast.components.governance_state import (
    GovernanceWorldState,
    Policy,
)
from swarmcast.components.objectives import (
    SAFETY_FIRST_CORP,
    CIVIL_SOCIETY,
    GOVERNMENT_US,
    GOVERNMENT_EU,
    REGULATOR,
    TECH_COMPANY_LARGE,
    TECH_COMPANY_STARTUP,
    BAD_ACTOR,
    Objective,
)
from swarmcast.game_master.resolution_engine import ResolutionEngine
from swarmcast.features.ensemble import EnsembleRunner, EnsembleReport


class StressTester:
    """Run multi-agent simulations to stress-test a proposed regulation across an ensemble of runs.

    Each run compares treatment (policy enacted) against a clean counterfactual baseline (no policy).
    Call stress_test() as the main entry point.
    """

    def __init__(
        self,
        model: lm_lib.LanguageModel,
        embedder,
        n_ensemble: int = 30,
        num_rounds: int = 8,
        output_dir: str = "./results/stress_test",
        revalidate: bool = False,
        dynamics_config=None,
    ):
        """
        Args:
            dynamics_config: DynamicsConfig controlling indicator coupling; defaults to RIGOROUS_BASELINE.
                Pass SENSITIVITY_ASSUMED only for exploratory analysis — results must NOT be cited as policy predictions.
        """
        from swarmcast.game_master.indicator_dynamics import (
            RIGOROUS_BASELINE, SENSITIVITY_ASSUMED, DynamicsConfig
        )
        self.model = model
        self.embedder = embedder
        self.n_ensemble = n_ensemble
        self.num_rounds = num_rounds
        self.output_dir = output_dir
        self.revalidate = revalidate
        self.dynamics_config = dynamics_config if dynamics_config is not None else RIGOROUS_BASELINE
        # Runtime guard: warn if non-rigorous config is active
        active = self.dynamics_config.active_layers()
        self._non_robust_mode = "ASSUMED" in active
        if self._non_robust_mode:
            import warnings
            warnings.warn(
                "[NON-ROBUST MODE] StressTester is using ASSUMED parameter layer. "
                "Results must NOT be cited as policy predictions. "
                "Use RIGOROUS_BASELINE for defensible output.",
                UserWarning,
                stacklevel=2,
            )

    def stress_test(
        self,
        policy_name: str,
        policy_description: str,
        regulated_entities: list[str],
        requirements: list[str],
        penalties: list[str],
        agent_configs: list[dict] | None = None,
        initial_state: dict[str, float] | None = None,
    ) -> StressTestReport:
        """Run a full ensemble stress test on the proposed regulation, including a clean counterfactual baseline.

        Returns a StressTestReport with failure mode analysis and counterfactual deltas.
        """
        if agent_configs is None:
            agent_configs = self._default_agent_configs()

        policy = Policy(
            id=policy_name.lower().replace(" ", "_"),
            name=policy_name,
            description=policy_description,
            regulated_entities=regulated_entities,
            requirements=requirements,
            penalties=penalties,
            # "enacted" from round 0: the stress-test simulates what happens
            # AFTER the policy passes, not the debate about whether it should.
            # This ensures the enforcement loop, passive burden, and compliance
            # tracking all fire from round 1 — not only if agents happen to
            # lobby it through during the simulation.
            status="enacted",
            enacted_round=0,
        )

        premise = self._build_premise(policy)

        def run_single(
            seed: int,
            temperature: float = 0.7,
            shuffle: bool = True,
            param_overrides: dict | None = None,
        ) -> dict:
            return self._run_single_simulation(
                seed=seed,
                policy=policy,
                premise=premise,
                agent_configs=agent_configs,
                temperature=temperature,
                shuffle=shuffle,
                initial_state=initial_state,
                param_overrides=param_overrides,
            )

        runner = EnsembleRunner(
            n_runs=self.n_ensemble,
            output_dir=self.output_dir,
        )
        ensemble = runner.run(run_single, scenario_name=f"stress_{policy.id}")

        # Counterfactual baseline: run same agents in a neutral world with NO policy.
        # This isolates the regulation's effect from initial-condition effects.
        #
        # ISOLATION DESIGN:
        #   - Premise is completely clean (no mention of the policy, see
        #     _build_baseline_premise). This is critical: even describing a
        #     "rejected" policy contaminates agent behavior because LLMs react
        #     to content, not framing. The previous "REJECTED by legislature"
        #     approach still exposed agents to "dissolution", "imprisonment",
        #     "500-person Bureau" — producing near-identical behavior to the
        #     treatment group.
        #   - Policy status is "repealed" so no enforcement loop fires.
        #   - Policy name and description are replaced with neutral placeholders.
        print("\n  [Counterfactual] Running clean neutral-world baseline for comparison...")
        baseline_ensemble = None
        try:
            # Build a genuinely null policy — no name, no description, no penalties
            null_policy = Policy(
                id="baseline_null",
                name="[No Policy — Baseline Condition]",
                description="No exceptional regulation in effect.",
                regulated_entities=[],
                requirements=[],
                penalties=[],
                status="repealed",
                enacted_round=-1,
            )
            baseline_premise = self._build_baseline_premise(policy)

            def run_counterfactual(
                seed: int,
                temperature: float = 0.7,
                shuffle: bool = True,
                param_overrides: dict | None = None,
            ) -> dict:
                """Clean no-regulation baseline with no policy content visible to agents."""
                return self._run_single_simulation(
                    seed=seed,
                    policy=null_policy,
                    premise=baseline_premise,
                    agent_configs=agent_configs,
                    temperature=temperature,
                    shuffle=shuffle,
                    initial_state=initial_state,
                    param_overrides=param_overrides,
                )

            baseline_runner = EnsembleRunner(
                n_runs=max(3, self.n_ensemble // 3),  # ≥3 runs for stability
                output_dir=os.path.join(self.output_dir, "baseline"),
            )
            baseline_ensemble = baseline_runner.run(
                run_counterfactual,
                scenario_name=f"baseline_{policy.id}",
            )
        except Exception as e:
            print(f"  [Counterfactual] Failed: {e} (proceeding without baseline)")

        failure_modes = self._analyze_failure_modes(ensemble)

        report = StressTestReport(
            policy=policy,
            ensemble=ensemble,
            failure_modes=failure_modes,
            baseline_ensemble=baseline_ensemble,
        )

        # Runtime guard: prefix report if non-rigorous config is active
        if getattr(self, '_non_robust_mode', False):
            report.non_robust_mode = True
            report.non_robust_label = ('[NON-ROBUST: SENSITIVITY_ASSUMED ACTIVE — '
                                       'results must NOT be cited as policy predictions]')
        report.save(os.path.join(
            self.output_dir,
            f"stress_report_{policy.id}_{datetime.now():%Y%m%d_%H%M%S}.json",
        ))

        return report

    def _run_single_simulation(
        self,
        seed: int,
        policy: Policy,
        premise: str,
        agent_configs: list[dict],
        temperature: float = 0.7,
        shuffle: bool = True,
        initial_state: dict[str, float] | None = None,
        param_overrides: dict | None = None,
    ) -> dict:
        """Run one simulation using the unified loop with the given seed, policy, and agent configs.

        Applies enforcement capacity hints extracted from the policy description to regulator resources before the loop starts.
        """
        from swarmcast.game_master.simulation_loop import run_simulation_loop

        from swarmcast.game_master.severity import classify_severity, extract_enforcement_capacity

        engine = ResolutionEngine(seed=seed)

        # Apply structural parameter overrides for uncertainty quantification.
        # These perturb model coefficients across their plausible ranges,
        # following Lempert et al. (2003) scenario discovery methodology.
        if param_overrides:
            import dataclasses as _dc
            current = engine.config
            fields = {f.name for f in _dc.fields(current)}
            valid = {k: getattr(current, k) * v
                     for k, v in param_overrides.items() if k in fields}
            if valid:
                engine.config = _dc.replace(current, **valid)
        world_state = GovernanceWorldState()
        if initial_state:
            world_state.economic_indicators.update(initial_state)
        policy_copy = Policy(**{k: v for k, v in policy.__dict__.items()})
        classify_severity(policy_copy, model=self.model)

        # Extract enforcement capacity hints from the policy description and
        # apply them to regulator resources and the enforcement grace period.
        # This makes "500-person Bureau" and "90-day deadline" mechanically real.
        enforcement_hints = extract_enforcement_capacity(
            (policy_copy.description or "") + " " + " ".join(policy_copy.requirements or []),
            staff_scaling=engine.config.enforcement_staff_scaling,
        )

        # Apply deadline → grace period. Overrides only when the description is
        # explicit; otherwise keep the severity-based default.
        if enforcement_hints["grace_rounds_override"] > 0:
            from swarmcast.game_master.resolution_config import ResolutionConfig
            import dataclasses
            engine.config = dataclasses.replace(
                engine.config,
                enforcement_grace_rounds_base=int(enforcement_hints["grace_rounds_override"]),
            )

        world_state.active_policies[policy.id] = policy_copy

        agents = {}
        agent_resources = {}
        agent_objectives = {}
        for config in agent_configs:
            name = config["name"]
            # Initialize resources BEFORE building so the resource_getter closure
            # can read a live reference to the same dict that the simulation mutates.
            resources = dict(config["resources"])
            # Apply enforcement capacity extracted from policy description to the
            # regulator. This makes "500-person Bureau" mechanically real:
            # the regulator actually starts with those staff rather than defaults.
            if enforcement_hints["staff_override"] > 0 and (
                "regulator" in name.lower() or "director" in name.lower()
                or "safety board" in name.lower() or "agency" in name.lower()
            ):
                resources["staff"] = max(resources.get("staff", 0), enforcement_hints["staff_override"])
                resources["budget"] = max(resources.get("budget", 0), enforcement_hints["staff_override"] * 0.4)
            agent_resources[name] = resources
            mem = basic_associative_memory.AssociativeMemoryBank(
                sentence_embedder=self.embedder
            )
            agent = config["builder_fn"](
                name=name.split(" (")[0],
                world_state=world_state,
                model=self.model,
                memory_bank=mem,
                resource_getter=lambda n=name: agent_resources[n],
                **config.get("kwargs", {}),
            )
            agents[name] = agent
            if "objective" in config:
                agent_objectives[name] = config["objective"]

        return run_simulation_loop(
            agents=agents,
            agent_resources=agent_resources,
            agent_objectives=agent_objectives,
            world_state=world_state,
            resolution_engine=engine,
            model=self.model,
            premise=premise,
            num_rounds=self.num_rounds,
            seed=seed,
            verbose=False,
            revalidate=self.revalidate,
            temperature=temperature,
            shuffle_agents=shuffle,
            dynamics_config=self.dynamics_config,
        )

    def _build_premise(self, policy: Policy) -> str:
        reqs = "\n".join(f"  - {r}" for r in policy.requirements)
        pens = "\n".join(f"  - {p}" for p in policy.penalties)
        return (
            f'A new regulation has been ENACTED into law: "{policy.name}"\n\n'
            f"Description: {policy.description}\n\n"
            f"Requirements (mandatory, effective immediately):\n{reqs}\n\n"
            f"Penalties for non-compliance:\n{pens}\n\n"
            f"This regulation IS NOW LAW. The question is not whether it will pass — "
            f"it already has. The question is how each stakeholder responds: "
            f"comply, evade, relocate, lobby for amendment, challenge in court, "
            f"or form coalitions. Each round, take ONE concrete strategic action. "
            f"Your resources are limited and actions have real consequences."
        )

    def _build_baseline_premise(self, policy: Policy) -> str:
        """Neutral world premise for counterfactual runs — no mention of the policy."""
        sector = "AI"
        jurisdiction = "the jurisdiction"

        return (
            f"You are operating in the {sector} technology sector in {jurisdiction}. "
            f"The regulatory environment is moderate — standard business laws apply, "
            f"and there are no exceptional new regulations on the table. "
            f"The government is focused on economic growth. Investors are active. "
            f"Companies are competing normally. Researchers are publishing and building.\n\n"
            f"No major new AI regulation has been proposed or enacted. "
            f"The sector operates under existing baseline rules only.\n\n"
            f"Each round, take ONE concrete strategic action to advance your organization's "
            f"goals. Your resources are limited and actions have real consequences."
        )

    def _default_agent_configs(self) -> list[dict]:
        """Return the default 6-agent configuration, including a contrarian industry actor that supports regulation.

        The contrarian tests the simulation against groupthink by pushing back on consensus positions
        """
        return [
            {
                "name": "Government Official",
                "builder_fn": build_government_agent,
                "kwargs": {"jurisdiction": "United States"},
                "resources": dict(GOVERNMENT_US.resources),
            },
            {
                "name": "Large AI Company",
                "builder_fn": build_company_agent,
                "kwargs": {"company_size": "large"},
                "resources": dict(TECH_COMPANY_LARGE.resources),
            },
            {
                "name": "AI Startup",
                "builder_fn": build_company_agent,
                "kwargs": {"company_size": "startup"},
                "resources": dict(TECH_COMPANY_STARTUP.resources),
            },
            {
                "name": "Regulator",
                "builder_fn": build_regulator_agent,
                "kwargs": {"agency": "the AI Oversight Agency"},
                "resources": dict(REGULATOR.resources),
            },
            {
                "name": "Civil Society Org",
                "builder_fn": build_civil_society_agent,
                "kwargs": {"organization": "the AI Rights Coalition"},
                "resources": dict(CIVIL_SOCIETY.resources),
            },
            {
                # CONTRARIAN AGENT — corrects LLM herd-behavior bias
                # OASIS (arXiv:2411.11581): LLM agents over-coordinate vs humans.
                # Governance models include industry actors that support regulation — not only opponents
                # This agent uses a PURPOSE-BUILT objective (SAFETY_FIRST_CORP) that
                # explicitly prohibits relocation and opposition, not just extra text.
                # The objective is the primary driver of LLM reasoning — extra_context
                # alone was insufficient (agent still relocated in 100% of runs).
                "name": "Safety-First AI Corp",
                "builder_fn": build_company_agent,
                "kwargs": {
                    "company_size": "large",
                    "extra_context": (
                        "CRITICAL IDENTITY: You are a large AI company that SUPPORTS "
                        "strong regulation as a competitive strategy. Your brand is built "
                        "on safety leadership. Relocating would destroy your market position. "
                        "Your ONLY strategic options are: comply early, lobby FOR stronger "
                        "rules, and publicly support the regulation. Do NOT relocate. "
                        "Do NOT lobby against the regulation. Do NOT evade."
                    ),
                },
                "objective": SAFETY_FIRST_CORP,
                "resources": dict(SAFETY_FIRST_CORP.resources),
                "_is_contrarian": True,
            },
        ]

    def _analyze_failure_modes(self, ensemble: EnsembleReport) -> list[dict]:
        """Identify and classify failure modes from ensemble statistics."""
        modes = []
        stats = ensemble.stats

        reloc_count = stats.get("relocation_count", 0)
        n = stats.get("n_successful_runs", 1)
        if reloc_count > 0:
            modes.append({
                "type": "capital_flight",
                "severity": "high" if reloc_count / n > 0.3 else "medium",
                "frequency": f"{reloc_count / n:.0%}",
                "description": (
                    f"In {reloc_count}/{n} runs ({reloc_count / n:.0%}), at least one "
                    f"company relocated operations to avoid regulation."
                ),
            })

        evasion_outcomes = stats.get("outcome_rates", {}).get("evade", {})
        total_evasions = sum(evasion_outcomes.values())
        if total_evasions > 0:
            success_rate = evasion_outcomes.get("success", 0) / total_evasions
            modes.append({
                "type": "regulatory_evasion",
                "severity": "high" if success_rate > 0.5 else "medium",
                "frequency": f"{success_rate:.0%} success rate",
                "description": (
                    f"Evasion attempts succeeded {success_rate:.0%} of the time "
                    f"({evasion_outcomes.get('success', 0)}/{total_evasions} attempts). "
                    f"Enforcement capacity may be insufficient."
                ),
            })

        enforce_outcomes = stats.get("outcome_rates", {}).get("enforce", {})
        blocked = enforce_outcomes.get("blocked", 0)
        if blocked > 0:
            modes.append({
                "type": "enforcement_gap",
                "severity": "high",
                "frequency": f"{blocked} blocked enforcement attempts",
                "description": (
                    f"The regulator was unable to enforce {blocked} times due to "
                    f"insufficient resources. The regulation may be an unfunded mandate."
                ),
            })

        innovation = stats.get("indicator_distributions", {}).get("innovation_rate", {})
        if innovation and innovation.get("mean", 100) < 80:
            modes.append({
                "type": "innovation_suppression",
                "severity": "high" if innovation["mean"] < 60 else "medium",
                "frequency": f"mean final: {innovation['mean']:.0f}/100",
                "description": (
                    f"Innovation rate dropped to {innovation['mean']:.0f} on average "
                    f"(from 100 baseline). The regulation may be too burdensome."
                ),
            })

        trust = stats.get("indicator_distributions", {}).get("public_trust", {})
        if trust and trust.get("mean", 50) < 40:
            modes.append({
                "type": "trust_erosion",
                "severity": "medium",
                "frequency": f"mean final: {trust['mean']:.0f}/100",
                "description": (
                    f"Public trust dropped to {trust['mean']:.0f} on average. "
                    f"Lobbying and regulatory battles may be undermining public confidence."
                ),
            })

        if not modes:
            modes.append({
                "type": "no_significant_failures",
                "severity": "low",
                "frequency": "N/A",
                "description": "No significant failure modes detected in ensemble runs.",
            })

        return modes


class StressTestReport:
    """Results of a regulation stress test."""

    def __init__(
        self,
        policy: Policy,
        ensemble: EnsembleReport,
        failure_modes: list[dict],
        baseline_ensemble: EnsembleReport | None = None,
    ):
        self.policy = policy
        self.ensemble = ensemble
        self.failure_modes = failure_modes
        self.baseline_ensemble = baseline_ensemble  # counterfactual (no-policy)

    def _counterfactual_delta(self, indicator: str) -> str | None:
        """Return a string showing policy effect vs no-policy baseline."""
        if self.baseline_ensemble is None:
            return None
        treated = (
            self.ensemble.stats
            .get("indicator_distributions", {})
            .get(indicator, {})
            .get("mean")
        )
        control = (
            self.baseline_ensemble.stats
            .get("indicator_distributions", {})
            .get(indicator, {})
            .get("mean")
        )
        if treated is None or control is None:
            return None
        delta = treated - control
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f} vs no-regulation baseline ({control:.1f})"

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"STRESS TEST REPORT: {self.policy.name}",
            f"{'=' * 60}",
            "",
            self.ensemble.summary(),
        ]

        # Counterfactual comparison
        if self.baseline_ensemble is not None:
            lines += [
                "",
                f"{'─' * 60}",
                "COUNTERFACTUAL ANALYSIS (policy vs no-regulation baseline):",
                f"{'─' * 60}",
            ]
            for ind in ["ai_investment_index", "innovation_rate", "public_trust", "regulatory_burden"]:
                delta_str = self._counterfactual_delta(ind)
                if delta_str:
                    lines.append(f"  {ind}: {delta_str}")
            lines.append("")
            lines.append("  Interpretation: negative deltas = policy caused this decline.")
            lines.append("  Positive deltas may indicate improved safety/trust.")

        lines += [
            "",
            f"{'─' * 60}",
            "FAILURE MODES DETECTED:",
            f"{'─' * 60}",
        ]
        for mode in self.failure_modes:
            lines.append(
                f"\n  [{mode['severity'].upper()}] {mode['type']}"
                f"\n  Frequency: {mode['frequency']}"
                f"\n  {mode['description']}"
            )

        lines.append(indicator_disclaimer())
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "policy": {
                "name": self.policy.name,
                "description": self.policy.description,
                "requirements": self.policy.requirements,
                "penalties": self.policy.penalties,
            },
            "failure_modes": self.failure_modes,
            "ensemble_stats": self.ensemble.stats,
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
