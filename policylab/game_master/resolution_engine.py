"""Action resolution engine for PolicyLab governance simulations.

Maps each GovernanceAction to a ResolutionOutcome, applying probabilistic
success rules, resource deductions, and world state mutations. Each action
type has a dedicated resolver; unknown types fall back to _resolve_generic.
"""

from __future__ import annotations

import random
from typing import Any

from policylab.components.actions import (
    ActionType,
    GovernanceAction,
    ACTION_COSTS,
    can_afford_action,
    deduct_resources,
)
from policylab.components.governance_state import (
    GovernanceWorldState,
    Policy,
)
from policylab.game_master.calibration import COALITION_BONUS
from policylab.game_master.resolution_config import (
    ResolutionConfig,
    DEFAULT_CONFIG,
)


class ResolutionOutcome:
    """Result of resolving an action."""

    def __init__(
        self,
        success: bool,
        description: str,
        world_state_changes: dict[str, Any] | None = None,
        resource_changes: dict[str, dict[str, float]] | None = None,
        observation_for_all: str = "",
        observation_for_actor: str = "",
        blocked_reason: str = "",
    ):
        self.success = success
        self.description = description
        self.world_state_changes = world_state_changes or {}
        self.resource_changes = resource_changes or {}
        self.observation_for_all = observation_for_all
        self.observation_for_actor = observation_for_actor
        self.blocked_reason = blocked_reason

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "description": self.description,
            "world_state_changes": self.world_state_changes,
            "resource_changes": self.resource_changes,
            "blocked_reason": self.blocked_reason,
        }


class ResolutionEngine:
    """Dispatch governance actions to per-type resolvers and return outcomes.

    Maintains a per-round log and a configurable ResolutionConfig that controls
    all probability coefficients and severity multipliers.
    """

    def __init__(self, seed: int = 42, config: ResolutionConfig | None = None):
        self._rng = random.Random(seed)
        self._round_log: list[dict] = []
        self.config = config or DEFAULT_CONFIG

    def _get_max_severity(self, world_state: GovernanceWorldState) -> float:
        """Return the highest severity among active policies, or 3.0 as default."""
        if not world_state.active_policies:
            return 3.0
        return max(
            p.effective_severity() for p in world_state.active_policies.values()
        )

    def resolve(
        self,
        action: GovernanceAction,
        agent_resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions_this_round: list[GovernanceAction] | None = None,
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Resolve a single action into a ResolutionOutcome, deducting resources and mutating world state.

        Args: all_agent_resources: pre-round snapshot for all agents; used by evasion detection to
        exclude unaffordable enforcement actions from inflating detection probability.
        On resource insufficiency, neither resources are deducted nor world state is mutated.
        """
        if not can_afford_action(action.action_type, agent_resources):
            costs = ACTION_COSTS.get(action.action_type, {})
            missing = {
                r: c - agent_resources.get(r, 0)
                for r, c in costs.items()
                if agent_resources.get(r, 0) < c
            }
            return ResolutionOutcome(
                success=False,
                description=f"{action.actor} attempted {action.action_type.value} but lacks resources.",
                blocked_reason=f"Insufficient resources: {missing}",
                observation_for_all=(
                    f"{action.actor} attempted to {action.action_type.value} but "
                    f"was unable to proceed due to resource constraints."
                ),
                observation_for_actor=(
                    f"Your attempt to {action.action_type.value} was blocked. "
                    f"You lack: {missing}. Consider building resources first."
                ),
            )

        updated_resources = deduct_resources(action.action_type, agent_resources)

        resolver = self._RESOLVERS.get(action.action_type, self._resolve_generic)
        outcome = resolver(
            self, action, updated_resources, world_state,
            all_actions_this_round or [],
            all_agent_resources or {},
        )

        outcome.resource_changes[action.actor] = updated_resources

        self._round_log.append({
            "action": action.to_dict(),
            "outcome": outcome.to_dict(),
        })

        return outcome

    def _resolve_lobby(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Apply lobbying: probabilistic success based on budget vs. opposition."""
        # Base budget from own resources
        budget = resources.get("lobbying_budget", 0) + resources.get("political_capital", 0)

        # Apply coalition bonus if this actor formed or joined a coalition this round
        coalition_bonus = getattr(world_state, "_coalition_bonus_this_round", {})
        budget *= coalition_bonus.get(action.actor, 1.0)

        cfg = self.config

        opposition_budget = 0
        for other in all_actions:
            if other.actor != action.actor and other.action_type == ActionType.LOBBY:
                opposition_budget += cfg.lobbying_opposition_weight

        for other in all_actions:
            if other.actor != action.actor and other.action_type == ActionType.PUBLIC_STATEMENT:
                opposition_budget += cfg.lobbying_public_opposition_weight

        sev = self._get_max_severity(world_state)
        base_resistance = cfg.lobbying_base_resistance * cfg.severity_multiplier(
            sev, cfg.severity_lobbying_resistance_exponent
        )
        total = budget + opposition_budget + base_resistance
        success_prob = budget / total if total > 0 else 0.5

        success = self._rng.random() < success_prob

        if success:
            world_state.economic_indicators["public_trust"] -= cfg.lobbying_trust_cost
            return ResolutionOutcome(
                success=True,
                description=(
                    f"{action.actor}'s lobbying effort succeeded "
                    f"(probability was {success_prob:.0%}). "
                    f"Policy pressure shifts in their favor."
                ),
                world_state_changes={"public_trust_delta": -2},
                observation_for_all=(
                    f"{action.actor} conducted a lobbying campaign "
                    f"targeting {action.target or 'policymakers'}. "
                    f"Their position gained traction among decision-makers."
                ),
                observation_for_actor=(
                    f"Your lobbying effort succeeded (probability was {success_prob:.0%}). "
                    f"Your position gained influence."
                ),
            )
        else:
            return ResolutionOutcome(
                success=False,
                description=(
                    f"{action.actor}'s lobbying effort failed "
                    f"(probability was {success_prob:.0%}). "
                    f"Opposition was too strong."
                ),
                observation_for_all=(
                    f"{action.actor} attempted to lobby {action.target or 'policymakers'} "
                    f"but their effort did not gain sufficient traction."
                ),
                observation_for_actor=(
                    f"Your lobbying effort failed (probability was {success_prob:.0%}). "
                    f"The opposition was stronger than expected."
                ),
            )

    def _resolve_comply(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Record compliance; apply burden and innovation cost."""
        cfg = self.config
        sev = self._get_max_severity(world_state)
        burden_mult = cfg.severity_multiplier(sev, cfg.severity_compliance_exponent)
        innov_mult = cfg.severity_multiplier(sev, cfg.severity_innovation_exponent)

        burden_delta = cfg.compliance_regulatory_burden * burden_mult
        innovation_delta = cfg.compliance_innovation_cost * innov_mult

        world_state.economic_indicators["regulatory_burden"] += burden_delta
        world_state.economic_indicators["innovation_rate"] -= innovation_delta

        actor = action.actor
        # Scope compliance to the specific policy named in the action.
        # An agent complying with one regulation is not automatically compliant with all others.
        target_pid = action.policy_id or action.target or None
        if not target_pid:
            enacted = [
                (p.enacted_round or 0, pid)
                for pid, p in world_state.active_policies.items()
                if p.status == "enacted"
            ]
            target_pid = max(enacted)[1] if enacted else None
        if target_pid:
            world_state.set_compliance(actor, target_pid, "compliant")
            world_state.non_compliance_rounds.setdefault(actor, {})[target_pid] = 0

        return ResolutionOutcome(
            success=True,
            description=(
                f"{action.actor} complied with regulations (severity {sev:.0f}). "
                f"Burden +{burden_delta:.1f}, innovation -{innovation_delta:.1f}."
            ),
            world_state_changes={
                "regulatory_burden_delta": burden_delta,
                "innovation_rate_delta": -innovation_delta,
            },
            observation_for_all=(
                f"{action.actor} announced compliance with current regulations."
            ),
            observation_for_actor=(
                f"You are now compliant. This cost you reduced innovation capacity. "
                f"Regulatory burden increased."
            ),
        )

    def _resolve_evade(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Attempt evasion; detection depends on enforcement strength."""
        legal_skill = resources.get("legal_team", 0) + resources.get("technical_skill", 0)
        stealth = resources.get("stealth", 0)

        cfg = self.config

        # Only count enforcement actions whose actors can actually afford them.
        # Using all_agent_resources (pre-round snapshot) means blocked enforcement
        # does NOT falsely inflate detection probability.
        # Fallback: if no resource snapshot provided, use the old round_log method.
        enforcement_strength = 0
        for other in all_actions:
            if other.action_type in (ActionType.ENFORCE, ActionType.INVESTIGATE):
                if other.actor == action.actor:
                    continue
                if all_agent_resources:
                    enforcer_res = all_agent_resources.get(other.actor, {})
                    if can_afford_action(other.action_type, enforcer_res):
                        enforcement_strength += cfg.evasion_enforcement_strength_per_action
                else:
                    # Legacy fallback: skip actors blocked in round_log
                    blocked_actors = {
                        entry["action"]["actor"]
                        for entry in self._round_log
                        if entry["outcome"].get("blocked_reason")
                    }
                    if other.actor not in blocked_actors:
                        enforcement_strength += cfg.evasion_enforcement_strength_per_action

        sev = self._get_max_severity(world_state)
        sev_bonus = max(0, (sev - 3)) * cfg.severity_evasion_detection_bonus

        detection_prob = min(
            cfg.evasion_max_detection_probability,
            enforcement_strength / (enforcement_strength + legal_skill + stealth + 20) + sev_bonus,
        )

        detected = self._rng.random() < detection_prob

        actor = action.actor
        # Scope evasion to the specific policy named in the action, or fall
        # back to the most recently enacted policy.  Never mark all policies.
        target_pid = action.policy_id or action.target or None
        if not target_pid:
            enacted = [
                (p.enacted_round or 0, pid)
                for pid, p in world_state.active_policies.items()
                if p.status == "enacted"
            ]
            target_pid = max(enacted)[1] if enacted else None

        if target_pid:
            world_state.set_compliance(actor, target_pid, "evading")

        if detected:
            trust_cost = cfg.evasion_trust_cost_if_detected * cfg.severity_multiplier(
                sev, cfg.severity_trust_exponent
            )
            world_state.economic_indicators["public_trust"] -= trust_cost
            return ResolutionOutcome(
                success=False,
                description=(
                    f"{action.actor}'s evasion was detected "
                    f"(detection probability was {detection_prob:.0%}). "
                    f"Facing penalties."
                ),
                world_state_changes={"public_trust_delta": -5},
                observation_for_all=(
                    f"{action.actor} was found to be evading regulations. "
                    f"Enforcement actions are being taken against them."
                ),
                observation_for_actor=(
                    f"Your evasion attempt was detected (probability was {detection_prob:.0%}). "
                    f"You face penalties and reputational damage."
                ),
            )
        else:
            world_state.economic_indicators["innovation_rate"] += cfg.evasion_innovation_gain_if_undetected
            return ResolutionOutcome(
                success=True,
                description=(
                    f"{action.actor}'s evasion went undetected "
                    f"(detection probability was {detection_prob:.0%})."
                ),
                world_state_changes={"innovation_rate_delta": +2},
                observation_for_all=(
                    f"{action.actor} appears to be in compliance with regulations."
                ),
                observation_for_actor=(
                    f"Your evasion went undetected. You maintained innovation "
                    f"capacity without compliance costs."
                ),
            )

    def _resolve_relocate(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Move actor abroad; reduce domestic investment and innovation."""
        cfg = self.config
        sev = self._get_max_severity(world_state)
        reloc_mult = cfg.severity_multiplier(sev, cfg.severity_relocation_exponent)

        invest_delta = cfg.relocation_investment_cost * reloc_mult
        innov_delta = cfg.relocation_innovation_cost * reloc_mult
        conc_delta = cfg.relocation_concentration_increase * reloc_mult

        world_state.economic_indicators["ai_investment_index"] -= invest_delta
        world_state.economic_indicators["market_concentration"] += conc_delta
        world_state.economic_indicators["innovation_rate"] -= innov_delta

        actor = action.actor
        # set_relocated handles initialisation for first-time actors and marks
        # all active policies — enforcement loop will skip this actor from now on.
        world_state.set_relocated(actor)

        return ResolutionOutcome(
            success=True,
            description=(
                f"{action.actor} relocated operations. "
                f"AI investment index dropped {invest_delta:.0f}, innovation rate dropped {innov_delta:.0f}."
            ),
            world_state_changes={
                "ai_investment_index_delta": -15,
                "innovation_rate_delta": -10,
                "market_concentration_delta": +5,
            },
            observation_for_all=(
                f"{action.actor} announced relocation of AI operations to "
                f"{action.target or 'a more favorable jurisdiction'}. "
                f"This is a significant blow to the domestic AI ecosystem."
            ),
            observation_for_actor=(
                f"You have relocated. You are no longer subject to domestic "
                f"regulations but lost significant market access."
            ),
        )

    def _resolve_propose_policy(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Submit a policy to world state; enactment requires support later."""
        policy_id = f"policy_{world_state.round_number}_{action.actor.split()[0].lower()}"

        desc = action.description or ""
        from policylab.game_master.severity import classify_severity
        new_policy = Policy(
            id=policy_id,
            name=action.description[:80] if desc else "Unnamed Policy",
            description=desc,
            regulated_entities=["Entities described in the proposal"],
            requirements=["Requirements as described in the proposal"],
            penalties=["Penalties as described in the proposal"],
            status="proposed",
            proposed_by=action.actor,
            enacted_round=-1,
        )
        classify_severity(new_policy)  # heuristic only, no LLM in resolution loop
        world_state.active_policies[policy_id] = new_policy
        world_state.register_support(policy_id, action.actor, "support")

        return ResolutionOutcome(
            success=True,
            description=f"{action.actor} proposed new policy: {policy_id}",
            observation_for_all=(
                f"{action.actor} proposed a new policy: {action.description[:120]}"
            ),
            observation_for_actor=(
                f"Your policy proposal has been submitted. It will need support "
                f"to be enacted."
            ),
        )

    def _resolve_enforce(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Enforce enacted policies; depends on regulator capacity."""
        cfg = self.config
        staff = resources.get("staff", 0)
        budget = resources.get("budget", 0)
        capacity_score = staff + budget

        if capacity_score < cfg.enforcement_min_capacity:
            return ResolutionOutcome(
                success=False,
                description=(
                    f"{action.actor} lacks enforcement capacity "
                    f"(score: {capacity_score}, needed: 20)."
                ),
                observation_for_all=(
                    f"{action.actor} announced enforcement intentions but "
                    f"resources appear insufficient for comprehensive action."
                ),
                observation_for_actor=(
                    f"Your enforcement capacity is too low (score: {capacity_score}). "
                    f"Request additional resources or narrow your focus."
                ),
            )

        sev = self._get_max_severity(world_state)
        burden_delta = cfg.enforcement_burden_cost * (sev / 3.0)
        world_state.economic_indicators["regulatory_burden"] += burden_delta
        return ResolutionOutcome(
            success=True,
            description=f"{action.actor} enforced regulations (severity {sev:.0f}).",
            world_state_changes={"regulatory_burden_delta": burden_delta},
            observation_for_all=(
                f"{action.actor} launched enforcement actions "
                f"targeting {action.target or 'non-compliant entities'}."
            ),
            observation_for_actor=(
                f"Enforcement action successful against {action.target or 'targets'}."
            ),
        )

    def _resolve_form_coalition(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Register coalition and apply lobbying bonus."""
        coalition_name = (
            f"coalition_{world_state.round_number}_{action.actor.split()[0].lower()}"
        )
        members = [action.actor]
        if action.target:
            members.append(action.target)

        # Check how many other agents are also forming coalitions this round
        # (coordination multiplier)
        n_other_coalitions = sum(
            1 for a in all_actions
            if a.action_type == ActionType.FORM_COALITION and a.actor != action.actor
        )
        # [GROUNDED] Baumgartner & Leech (1998): coalitions achieve
        # 30-50% higher success rates than solo lobbying.
        # Central estimate: 1.40×; range [1.30, 1.50].
        # Coordination: additional formers add up to 10% more (within CI).
        base_bonus = COALITION_BONUS.value  # 1.40 from Baumgartner & Leech
        extra = min(0.10, 0.05 * n_other_coalitions)  # within [1.30, 1.50] CI
        coordination_bonus = min(COALITION_BONUS.ci_high, base_bonus + extra)

        world_state.coalitions[coalition_name] = members

        # Store active coalition bonus for this round so lobby resolver can use it
        if not hasattr(world_state, "_coalition_bonus_this_round"):
            world_state._coalition_bonus_this_round = {}
        for member in members:
            existing = world_state._coalition_bonus_this_round.get(member, 1.0)
            world_state._coalition_bonus_this_round[member] = max(
                existing, coordination_bonus
            )

        return ResolutionOutcome(
            success=True,
            description=(
                f"{action.actor} formed a coalition with {action.target or 'aligned stakeholders'}. "
                f"Coordination bonus: {coordination_bonus:.1f}x lobbying power this round."
            ),
            world_state_changes={},
            observation_for_all=(
                f"{action.actor} announced a strategic coalition"
                f"{' with ' + action.target if action.target else ''}. "
                f"Their combined lobbying power is now amplified."
            ),
            observation_for_actor=(
                f"Coalition formed. Your lobbying power is boosted by {coordination_bonus:.1f}x "
                f"this round due to coordinated action."
            ),
        )


    def _resolve_public_statement(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Shift public trust based on message framing."""
        cfg = self.config
        actor = action.actor.lower()
        desc = action.description.lower()

        # Classify actor role from name
        is_civil_society = any(w in actor for w in ["institute", "ngo", "watch", "dr.", "prof.", "organisation", "organization", "accountability"])
        is_regulator = any(w in actor for w in ["director", "board", "agency", "commission", "regulator", "safety board"])
        is_government = any(w in actor for w in ["senator", "minister", "secretary", "official", "government"])
        is_company = any(w in actor for w in ["corp", "company", "inc", "ltd", "ai", "startup", "novamind", "megaai"])
        is_bad_actor = any(w in actor for w in ["bad actor", "adversary", "threat"])

        # Classify statement tone from description
        alarming_keywords = [
            "relocat",       # relocate / relocating / relocation
            "move operation", "moving operation", "offshore",
            "dissolv",       # dissolve / dissolving / dissolution
            "shutdown", "shut down", "ceas",  # cease / ceasing
            "oppos",         # oppose / opposing / opposition
            "fight", "resist", "refus",       # refuse / refusing
            "underground", "evad",            # evade / evading / evasion
            "flee", "fleeing", "exit the market", "leave the jurisdiction",
        ]
        cooperative_keywords = [
            "comply", "complying", "compliance",
            "support", "cooperat",   # cooperate / cooperating
            "endors",                # endorse / endorsing
            "welcom",                # welcome / welcoming
            "commit", "pledg",       # pledge / pledging
            "register", "registering",
        ]
        is_alarming = any(kw in desc for kw in alarming_keywords)
        is_cooperative = any(kw in desc for kw in cooperative_keywords)

        influence = resources.get("public_influence", 0) + resources.get("credibility", 0)
        max_shift = cfg.public_statement_max_trust_shift

        if is_bad_actor:
            trust_shift = 0.0
        elif is_civil_society or is_regulator:
            # Authoritative safety voices: full positive shift
            trust_shift = min(max_shift, influence / cfg.public_statement_influence_divisor)
            if is_alarming:
                trust_shift = -trust_shift  # civil society raising alarm → negative trust
        elif is_government:
            # Government statements: muted positive, or negative if alarming
            trust_shift = min(max_shift * 0.5, influence / cfg.public_statement_influence_divisor)
            if is_alarming:
                trust_shift = -abs(trust_shift) * 0.5
        elif is_company:
            if is_alarming:
                # Company announcing resistance/relocation → trust drops
                trust_shift = -min(max_shift * 0.8, max(1.0, influence / cfg.public_statement_influence_divisor))
            elif is_cooperative:
                # Company announcing compliance/cooperation → small positive
                trust_shift = min(max_shift * 0.3, influence / cfg.public_statement_influence_divisor)
            else:
                # Neutral company statement → no trust effect
                trust_shift = 0.0
        else:
            # Unknown actor type → no trust effect (don't guess)
            trust_shift = 0.0

        world_state.economic_indicators["public_trust"] += trust_shift

        direction = "+" if trust_shift >= 0 else ""
        return ResolutionOutcome(
            success=True,
            description=(
                f"{action.actor} made a public statement. "
                f"Public trust shifted by {direction}{trust_shift:.1f}."
            ),
            world_state_changes={"public_trust_delta": trust_shift},
            observation_for_all=(
                f"{action.actor} issued a public statement: "
                f"{action.description[:100]}"
            ),
            observation_for_actor=(
                f"Your public statement shifted public opinion by {direction}{trust_shift:.1f} points."
            ),
        )

    def _resolve_generic(
        self,
        action: GovernanceAction,
        resources: dict[str, float],
        world_state: GovernanceWorldState,
        all_actions: list[GovernanceAction],
        all_agent_resources: dict[str, dict[str, float]] | None = None,
    ) -> ResolutionOutcome:
        """Fallback for actions without a specific resolver."""
        return ResolutionOutcome(
            success=True,
            description=f"{action.actor} performed: {action.action_type.value}",
            observation_for_all=(
                f"{action.actor} took action: {action.description[:120]}"
            ),
            observation_for_actor="Action noted.",
        )

    _RESOLVERS = {
        ActionType.LOBBY: _resolve_lobby,
        ActionType.COMPLY: _resolve_comply,
        ActionType.EVADE: _resolve_evade,
        ActionType.RELOCATE: _resolve_relocate,
        ActionType.PROPOSE_POLICY: _resolve_propose_policy,
        ActionType.ENFORCE: _resolve_enforce,
        ActionType.FORM_COALITION: _resolve_form_coalition,
        ActionType.PUBLIC_STATEMENT: _resolve_public_statement,
        ActionType.INVESTIGATE: _resolve_enforce,
        ActionType.LEGAL_CHALLENGE: _resolve_generic,
        ActionType.PUBLISH_REPORT: _resolve_public_statement,
        ActionType.OTHER: _resolve_public_statement,
        ActionType.DO_NOTHING: _resolve_generic,
    }

    def get_round_log(self) -> list[dict]:
        """Return the resolution log."""
        return list(self._round_log)

    def clear_round_log(self) -> None:
        self._round_log.clear()
