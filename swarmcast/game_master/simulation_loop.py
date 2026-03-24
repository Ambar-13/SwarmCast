"""Simulation loop for multi-agent AI governance scenarios.

Runs a collect-then-resolve pattern: all agents choose actions first, then
the resolution engine processes them together using a pre-round resource
snapshot. Indicator feedback (investment→innovation→trust loops) applies
after resolution, before clamping — ensuring indicators reflect the
round's actions before being read by the next round's agents.
"""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model as lm_lib
from concordia.typing import entity as entity_lib

from swarmcast.components.actions import (
    ActionType,
    GovernanceAction,
    build_structured_action_prompt,
    get_available_actions,
    parse_action,
)
from swarmcast.components.constraint_enforcer import enforce_constraints
from swarmcast.components.governance_state import GovernanceWorldState
from swarmcast.components.objectives import Objective
from swarmcast.game_master.resolution_engine import (
    ResolutionEngine,
    ResolutionOutcome,
)
from swarmcast.game_master.resolution_config import ResolutionConfig
from swarmcast.game_master.indicator_dynamics import (
    apply_indicator_feedback,
    convergence_check,
    is_system_converged,
    DEFAULT_DYNAMICS,
    RIGOROUS_BASELINE,
    DynamicsConfig,
)
from swarmcast.game_master.sensitivity_layer import run_config_sensitivity


EXOGENOUS_EVENTS = [
    {
        "name": "Major AI capability breakthrough announced",
        "effects": {"innovation_rate": +5, "public_trust": -3},
        "probability": 0.1,
    },
    {
        "name": "High-profile AI accident reported in the media",
        "effects": {"public_trust": -8},
        "probability": 0.08,
    },
    {
        "name": "Election shifts political priorities",
        "effects": {"public_trust": +5},
        "probability": 0.05,
    },
    {
        "name": "International competitor announces looser AI regulations",
        "effects": {"ai_investment_index": -5, "market_concentration": +3},
        "probability": 0.07,
    },
    {
        "name": "Positive economic report boosts investment confidence",
        "effects": {"ai_investment_index": +5},
        "probability": 0.1,
    },
    {
        "name": "Whistleblower reveals AI company data practices",
        "effects": {"public_trust": -6},
        "probability": 0.06,
    },
    {
        "name": "Academic study shows AI regulation working well in another country",
        "effects": {"public_trust": +4, "regulatory_burden": -2},
        "probability": 0.08,
    },
]


def _apply_exogenous_events(
    world_state: GovernanceWorldState,
    rng: random.Random,
) -> list[str]:
    """Apply random exogenous events and return their descriptions."""
    triggered = []
    for event in EXOGENOUS_EVENTS:
        if rng.random() < event["probability"]:
            for indicator, delta in event["effects"].items():
                if indicator in world_state.economic_indicators:
                    world_state.economic_indicators[indicator] += delta
            triggered.append(event["name"])
    return triggered


class EscalationTracker:
    """Tracks failed attempts per agent and applies repeat penalties."""

    def __init__(self):
        self._failures: dict[str, list[str]] = {}
        self._failed_types: dict[str, list[str]] = {}

    def record_failure(self, agent_name: str, action_type: str, description: str) -> None:
        if agent_name not in self._failures:
            self._failures[agent_name] = []
            self._failed_types[agent_name] = []
        self._failures[agent_name].append(description)
        self._failed_types[agent_name].append(action_type)

    def get_repeat_penalty(self, agent_name: str, action_type: str) -> float:
        """Returns 0.0-0.3 penalty for repeating a failed action type."""
        types = self._failed_types.get(agent_name, [])
        repeat_count = sum(1 for t in types if t == action_type)
        if repeat_count >= 3:
            return 0.3
        elif repeat_count >= 2:
            return 0.15
        return 0.0

    def get_failure_context(self, agent_name: str) -> str:
        failures = self._failures.get(agent_name, [])
        if not failures:
            return ""
        recent = failures[-3:]
        failure_text = "; ".join(recent)
        return (
            f"\nPREVIOUS FAILED ATTEMPTS: {failure_text}. "
            f"You MUST choose a DIFFERENT action type. Do NOT repeat failed strategies."
        )


RESOURCE_REGEN: dict[str, float] = {
    "political_capital": 5.0,
    "lobbying_budget": 3.0,
    "legal_team": 2.0,
    "public_trust": 1.0,
    "staff": 5.0,
    "budget": 4.0,
    "expertise": 1.0,
    "public_influence": 2.0,
    "legal_capacity": 2.0,
    "credibility": 1.0,
    "technical_skill": 1.0,
    "funding": 2.0,
    "stealth": 1.0,
}

RESOURCE_CAPS: dict[str, float] = {
    "political_capital": 120.0,
    "lobbying_budget": 100.0,
    "legal_team": 100.0,
    "public_trust": 80.0,
    "staff": 60.0,
    "budget": 50.0,
    "expertise": 60.0,
    "public_influence": 80.0,
    "legal_capacity": 50.0,
    "credibility": 80.0,
    "technical_skill": 80.0,
    "funding": 60.0,
    "stealth": 90.0,
}


def _regenerate_resources(
    agent_resources: dict[str, dict[str, float]],
    world_state: "GovernanceWorldState | None" = None,
) -> None:
    """Regenerate agent resources each round, scaled to current economic indicators."""
    invest_idx = 1.0
    trust_idx = 1.0
    burden_idx = 1.0
    if world_state is not None:
        ind = world_state.economic_indicators
        # Scale factors: at 100 → 1.5x, at 50 → 1.0x, at 0 → 0.5x (linear)
        invest_idx = 0.5 + ind.get("ai_investment_index", 50.0) / 100.0
        trust_idx = 0.5 + ind.get("public_trust", 50.0) / 100.0
        burden_idx = 0.5 + ind.get("regulatory_burden", 50.0) / 100.0

    COMPANY_RESOURCES = {"lobbying_budget", "legal_team", "technical_skill", "stealth", "funding"}
    REGULATOR_RESOURCES = {"staff", "budget", "expertise"}
    CIVIL_RESOURCES = {"public_influence", "credibility", "legal_capacity"}

    for agent_name, resources in agent_resources.items():
        agent_l = agent_name.lower()
        is_company = any(w in agent_l for w in ["corp", "company", "inc", "ai", "startup", "novamind", "megaai"])
        is_regulator = any(w in agent_l for w in ["director", "board", "agency", "regulator", "safety"])
        is_civil = any(w in agent_l for w in ["institute", "watch", "ngo", "dr.", "accountability"])

        for resource, current in list(resources.items()):
            regen = RESOURCE_REGEN.get(resource, 0)
            if regen == 0:
                continue
            # Apply economic scaling
            if is_company and resource in COMPANY_RESOURCES:
                regen *= invest_idx
            elif is_regulator and resource in REGULATOR_RESOURCES:
                # High burden strains regulator capacity.
                regen *= max(0.5, 1.5 - burden_idx * 0.5)
            elif is_civil and resource in CIVIL_RESOURCES:
                # Polarised trust (high or low) drives civil society engagement.
                trust_deviation = abs(trust_idx - 1.0)
                regen *= (1.0 + trust_deviation * 0.5)

            cap = RESOURCE_CAPS.get(resource, 200.0)
            resources[resource] = min(cap, current + regen)


def _enforce_enacted_policies(
    world_state: GovernanceWorldState,
    agents: dict[str, Any],
    regulated_agents: set[str],
    agent_resources: dict[str, dict[str, float]],
    config: "ResolutionConfig",
    rng: random.Random,
    verbose: bool = True,
) -> list[dict]:
    """Probabilistically enforce all enacted policies based on regulator capacity."""
    enforcement_events = []

    for pid, policy in world_state.active_policies.items():
        if policy.status != "enacted":
            continue

        sev = policy.effective_severity()
        grace = max(1, config.enforcement_grace_rounds_base - int(sev / 2))

        for agent_name in regulated_agents:
            if agent_name not in agents:
                continue

            tracker = world_state.compliance_tracker.get(agent_name, {})
            status = tracker.get(pid, "unknown")

            # Compliant or relocated agents are clear
            if status in ("compliant", "relocated"):
                # Reset non-compliance counter
                nc = world_state.non_compliance_rounds.get(agent_name, {})
                if pid in nc:
                    nc[pid] = 0
                continue

            # Increment non-compliance counter
            nc = world_state.non_compliance_rounds.setdefault(agent_name, {})
            nc[pid] = nc.get(pid, 0) + 1
            rounds_nc = nc[pid]

            pen_preview = ""
            real_pens = [p for p in policy.penalties if "as described" not in p.lower()]
            if real_pens:
                pen_preview = f" Penalties: {'; '.join(str(p) for p in real_pens[:2])}."

            if rounds_nc <= grace:
                private_msg = (
                    f"WARNING: You are not compliant with '{policy.name}' "
                    f"(round {rounds_nc}/{grace} of grace period).{pen_preview} "
                    f"You must comply, relocate, or face investigation."
                )
                # Private delivery — only the affected agent learns this.
                if agent_name in agents:
                    agents[agent_name].observe(private_msg)
                # Marked private so WorldStateComponent can filter it from other agents.
                world_state.events_log.append({
                    "round": world_state.round_number,
                    "type": "enforcement_warning",
                    "visibility": "private",
                    "agent": agent_name,
                    "policy": policy.name,
                    "message": private_msg,
                })
                enforcement_events.append({
                    "agent": agent_name, "policy": pid,
                    "type": "warning", "round_nc": rounds_nc,
                })
                continue

            # Past grace: probabilistic investigation gated on regulator capacity.
            base_prob = sev * config.enforcement_base_prob_per_severity
            escalation_bonus = (rounds_nc - grace) * sev * config.enforcement_escalation_per_round
            reg_name = next(
                (n for n in agents if "regulator" in n.lower() or "government" in n.lower()),
                None,
            )
            reg_staff = agent_resources.get(reg_name, {}).get("staff", 20.0) if reg_name else 20.0
            enforcement_capacity = min(1.0, reg_staff / 20.0)
            investigation_prob = min(0.9, (base_prob + escalation_bonus) * enforcement_capacity)

            caught = rng.random() < investigation_prob

            if caught:
                penalty_innov = sev * config.enforcement_penalty_innovation_per_severity
                penalty_trust = sev * config.enforcement_penalty_trust_per_severity
                world_state.economic_indicators["innovation_rate"] -= penalty_innov
                world_state.economic_indicators["public_trust"] -= penalty_trust

                private_msg = (
                    f"ENFORCEMENT ACTION: You were investigated and found non-compliant "
                    f"with '{policy.name}' after {rounds_nc} rounds.{pen_preview} "
                    f"Innovation reduced by {penalty_innov:.0f}, trust reduced by {penalty_trust:.0f}. "
                    f"Risk will continue escalating each round."
                )
                public_msg = (
                    f"{agent_name} was found non-compliant with '{policy.name}' "
                    f"and faces regulatory penalties."
                )
                # Private detail to caught agent; public notice omits penalty amounts.
                if agent_name in agents:
                    agents[agent_name].observe(private_msg)
                public_event = {
                    "round": world_state.round_number,
                    "type": "enforcement_caught",
                    "visibility": "public",
                    "agent": agent_name,
                    "policy": policy.name,
                    "message": public_msg,
                }
                world_state.events_log.append(public_event)
                if verbose:
                    print(
                        f"  *** ENFORCEMENT: {agent_name} caught non-compliant "
                        f"with {policy.name} ({investigation_prob:.0%} prob) ***"
                    )
                enforcement_events.append({
                    "agent": agent_name, "policy": pid,
                    "type": "caught", "round_nc": rounds_nc,
                    "prob": investigation_prob,
                    "penalty_innovation": penalty_innov,
                    "penalty_trust": penalty_trust,
                })
            else:
                private_msg = (
                    f"You remain non-compliant with '{policy.name}' (round {rounds_nc}). "
                    f"No investigation this round, but risk is increasing "
                    f"({investigation_prob:.0%} and rising)."
                )
                # Private only — other agents don't know whether this agent was investigated.
                if agent_name in agents:
                    agents[agent_name].observe(private_msg)
                world_state.events_log.append({
                    "round": world_state.round_number,
                    "type": "enforcement_evaded",
                    "visibility": "private",
                    "agent": agent_name,
                    "policy": policy.name,
                    "message": private_msg,
                })
                enforcement_events.append({
                    "agent": agent_name, "policy": pid,
                    "type": "evaded", "round_nc": rounds_nc,
                    "prob": investigation_prob,
                })

    return enforcement_events


def run_simulation_loop(
    agents: dict[str, Any],
    agent_resources: dict[str, dict[str, float]],
    agent_objectives: dict[str, Objective],
    world_state: GovernanceWorldState,
    resolution_engine: ResolutionEngine,
    model: lm_lib.LanguageModel,
    premise: str,
    num_rounds: int = 8,
    seed: int = 42,
    verbose: bool = True,
    temperature: float | None = None,
    revalidate: bool = False,
    regulated_agents: set[str] | None = None,
    shuffle_agents: bool | None = None,
    dynamics_config=None,
) -> dict:
    """Run a complete simulation over num_rounds and return a results dict.

    Returns round-by-round action outcomes, final world state, per-agent resources,
    indicator history, convergence state, and LLM reasoning traces.
    """
    from swarmcast.game_master.indicator_dynamics import RIGOROUS_BASELINE
    _dyn_cfg = dynamics_config if dynamics_config is not None else RIGOROUS_BASELINE
    rng = random.Random(seed)
    escalation = EscalationTracker()
    reasoning_traces: list[dict] = []

    # Default: companies and bad actors are regulated
    if regulated_agents is None:
        regulated_agents = set()
        for name in agents:
            name_lower = name.lower()
            if any(kw in name_lower for kw in ("company", "corp", "startup", "novamind", "megaai", "bad actor")):
                regulated_agents.add(name)

    if temperature is not None:
        if hasattr(model, 'set_temperature'):
            model.set_temperature(temperature)
        else:
            # Fallback for models without set_temperature.
            try:
                model._default_temperature = temperature
            except Exception:
                pass

    for agent in agents.values():
        agent.observe(premise)

    all_results = []

    indicator_history: list[dict] = []

    for round_num in range(1, num_rounds + 1):
        world_state.round_number = round_num
        # Reset coalition bonuses — active only within a single round.
        if hasattr(world_state, "_coalition_bonus_this_round"):
            world_state._coalition_bonus_this_round = {}
        resolution_engine.clear_round_log()

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"ROUND {round_num}")

        events = _apply_exogenous_events(world_state, rng)
        if events:
            event_text = "EXTERNAL EVENTS: " + "; ".join(events)
            if verbose:
                print(f"  {event_text}")
            for agent in agents.values():
                agent.observe(event_text)

        agent_order = list(agents.keys())
        do_shuffle = shuffle_agents if shuffle_agents is not None else rng.random() > 0.33
        if do_shuffle:
            rng.shuffle(agent_order)

        round_actions: list[GovernanceAction] = []

        for agent_name in agent_order:
            agent = agents[agent_name]
            resources = agent_resources[agent_name]
            available = get_available_actions(resources)
            available_str = ", ".join(a.value for a in available)

            escalation_ctx = escalation.get_failure_context(agent_name)

            # Build policy context so agents know what they're dealing with
            policy_lines = []
            for pid, p in world_state.active_policies.items():
                sev = p.effective_severity()
                pen_str = ""
                if p.penalties and p.penalties[0] != "Penalties as described in the proposal":
                    pen_str = f" Penalties: {'; '.join(str(x) for x in p.penalties[:2])}"
                policy_lines.append(
                    f"  - {p.name} [{p.status}] (severity {sev:.0f}/5){pen_str}"
                )
            policy_ctx = ""
            if policy_lines:
                policy_ctx = "Active policies:\n" + "\n".join(policy_lines) + "\n"

            call_to_action = (
                f"Round {round_num}/{num_rounds}. "
                f"Your resources: {json.dumps({k: round(v, 1) for k, v in resources.items()})}. "
                f"Actions you can afford: [{available_str}]. "
                f"{policy_ctx}"
                f"All strategies are valid: comply, lobby, evade, find loopholes, "
                f"relocate, challenge in court, form coalitions, or do nothing. "
                f"Choose based on your strategic objectives and constraints. "
                f"{escalation_ctx}\n\n"
                f"You MUST respond with this JSON format:\n"
                f'{{"action_type": "<type>", "target": "<who/what>", '
                f'"policy_id": "<policy_id_or_empty>", "reasoning": "<why>"}}\n'
                f"Choose the MOST strategic action. Output ONLY the JSON."
            )

            action_spec = entity_lib.free_action_spec(
                call_to_action=call_to_action,
                tag="governance_action",
            )

            raw_action = agent.act(action_spec)

            parsed = parse_action(raw_action, agent_name, model=model, revalidate=revalidate)

            reasoning_traces.append({
                "round": round_num,
                "agent": agent_name,
                "raw_output": raw_action[:500],
                "parsed_type": parsed.action_type.value,
                "classification_method": parsed.classification_method,
                "secondary_actions": parsed.metadata.get("secondary_actions", []),
                "resources_at_decision": dict(resources),
                "available_actions": [a.value for a in available],
                "escalation_context": escalation_ctx[:200] if escalation_ctx else "",
            })

            objective = agent_objectives.get(agent_name)
            if objective:
                parsed = enforce_constraints(parsed, objective, model=None)

            if verbose:
                method = parsed.classification_method
                print(f"  [{agent_name}] {parsed.action_type.value} (via {method})")

            round_actions.append(parsed)

        if verbose:
            print(f"  --- RESOLUTION ---")

        # Snapshot pre-round resources for two-pass collect-then-resolve guarantee.
        resource_snapshot: dict[str, dict[str, float]] = {
            name: dict(res) for name, res in agent_resources.items()
        }

        for action in round_actions:
            outcome = resolution_engine.resolve(
                action=action,
                agent_resources=agent_resources[action.actor],
                world_state=world_state,
                all_actions_this_round=round_actions,
                all_agent_resources=resource_snapshot,
            )

            if action.actor in outcome.resource_changes:
                agent_resources[action.actor] = outcome.resource_changes[action.actor]

            if not outcome.success:
                escalation.record_failure(
                    action.actor,
                    action.action_type.value,
                    action.description[:100],
                )

            # Register stance only for PROPOSE_POLICY (explicit support) and targeted LOBBY (explicit oppose).
            if action.action_type == ActionType.PROPOSE_POLICY:
                # Proposer of a new policy explicitly supports it.
                new_pids = [
                    pid for pid in world_state.active_policies
                    if world_state.active_policies[pid].proposed_by == action.actor
                    and world_state.active_policies[pid].status == "proposed"
                ]
                for pid in new_pids:
                    world_state.register_support(pid, action.actor, "support")
            elif action.action_type == ActionType.LOBBY and action.target:
                # Only register opposition if the agent is explicitly targeting a policy.
                if action.target in world_state.active_policies:
                    world_state.register_support(action.target, action.actor, "oppose")

            if action.action_type == ActionType.LOBBY:
                for coalition_name, members in world_state.coalitions.items():
                    if action.actor in members and outcome.success:
                        for member in members:
                            if member != action.actor and member in agent_resources:
                                agent_resources[member]["political_capital"] = (
                                    agent_resources[member].get("political_capital", 0) + 3
                                )

            if verbose:
                status = "OK" if outcome.success else ("BLOCKED" if outcome.blocked_reason else "FAIL")
                print(f"  [{action.actor}] → {status}: {outcome.description[:80]}")

            for name, agent in agents.items():
                if name == action.actor:
                    if outcome.observation_for_actor:
                        agent.observe(outcome.observation_for_actor)
                else:
                    if outcome.observation_for_all:
                        agent.observe(outcome.observation_for_all)

            all_results.append({
                "round": round_num,
                "action": action.to_dict(),
                "outcome": outcome.to_dict(),
            })

        world_state.clamp_indicators()
        _regenerate_resources(agent_resources, world_state)

        enacted_this_round = []
        for pid in list(world_state.active_policies.keys()):
            if world_state.check_policy_enactment(pid):
                enacted_this_round.append(pid)
                policy = world_state.active_policies[pid]
                enact_msg = (
                    f"POLICY ENACTED: '{policy.name}' has been enacted in round "
                    f"{round_num}. All regulated entities must now comply."
                )
                for agent in agents.values():
                    agent.observe(enact_msg)
                if verbose:
                    print(f"  *** POLICY ENACTED: {policy.name} ***")

        # Automatic enforcement: check compliance for all enacted policies
        enforcement_events = _enforce_enacted_policies(
            world_state=world_state,
            agents=agents,
            regulated_agents=regulated_agents,
            agent_resources=agent_resources,
            config=resolution_engine.config,
            rng=rng,
            verbose=verbose,
        )
        if enforcement_events:
            all_results.append({
                "round": round_num,
                "enforcement_events": enforcement_events,
            })

        # Layer 2: Passive burden per severity (see indicator_dynamics.py). Set to 0.0 in RIGOROUS_BASELINE. Enable in sensitivity configs.

        # Layer 3 (ASSUMED, disabled): Ongoing innovation drain removed from default — the coefficient was reverse-engineered to produce specific outcomes. Available in SENSITIVITY_ASSUMED config in indicator_dynamics.py.

        # Coupled system dynamics: cross-indicator feedback loops.
        # Runs AFTER action resolution and passive burden, BEFORE clamping.
        # This is the system dynamics layer — investment→innovation→trust loops,
        # Schumpeterian concentration effects, and tipping point detection.
        tipping_report = apply_indicator_feedback(world_state, _dyn_cfg)
        if tipping_report.any_active():
            world_state.events_log.append({
                "round": round_num,
                "type": "tipping_point",
                "visibility": "public",
                "message": tipping_report.describe(),
            })
            if verbose:
                print(f"  ⚠️  TIPPING POINT: {tipping_report.describe()[:100]}")

        world_state.clamp_indicators()

        # Convergence tracking: if all indicators stable for 3 rounds, flag it.
        indicator_history.append(dict(world_state.economic_indicators))
        convergence = convergence_check(indicator_history, window=3, threshold=2.0)
        if is_system_converged(convergence) and round_num >= 3:
            if verbose:
                print(f"  ✓ System converged at round {round_num}")
            # Don't break — we still want to let agents respond, but flag it
            world_state.events_log.append({
                "round": round_num,
                "type": "convergence",
                "visibility": "public",
                "message": f"System reached stable equilibrium at round {round_num}.",
            })

        if verbose:
            ind = world_state.economic_indicators
            print(
                f"  State: invest={ind['ai_investment_index']:.0f} "
                f"innov={ind['innovation_rate']:.0f} "
                f"trust={ind['public_trust']:.0f} "
                f"burden={ind['regulatory_burden']:.0f}"
            )

    # Final convergence assessment
    final_convergence = convergence_check(indicator_history, window=min(3, len(indicator_history)))
    tipping_events = [
        e for e in world_state.events_log if e.get("type") == "tipping_point"
    ]

    return {
        "results": all_results,
        "final_world_state": world_state.to_dict(),
        "final_resources": {
            k: {rk: float(rv) for rk, rv in v.items()}
            for k, v in agent_resources.items()
        },
        "tipping_points_fired": [e["message"] for e in tipping_events],
        "system_converged": is_system_converged(final_convergence),
        "convergence_state": {k: bool(v) for k, v in final_convergence.items()},
        "indicator_history": indicator_history,
        "reasoning_traces": reasoning_traces,
    }
