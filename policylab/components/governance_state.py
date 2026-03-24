"""World-state tracking for governance simulations."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


@dataclasses.dataclass
class Policy:
    """Represents a single regulation with its requirements, penalties, compliance rate, and lifecycle status."""
    id: str
    name: str
    description: str
    regulated_entities: list[str]
    requirements: list[str]
    penalties: list[str]
    status: str = "proposed"  # proposed, enacted, enforced, repealed
    compliance_rate: float = 0.0
    proposed_by: str = ""
    enacted_round: int = -1
    severity: float = 0.0  # 0 = not yet classified

    def effective_severity(self) -> float:
        """Return severity, defaulting to 3.0 (moderate) if not classified."""
        return self.severity if self.severity > 0 else 3.0


@dataclasses.dataclass
class GovernanceWorldState:
    """Tracks all shared simulation state including active policies, compliance records, economic indicators, and events."""
    active_policies: dict[str, Policy] = dataclasses.field(default_factory=dict)
    compliance_tracker: dict[str, dict[str, str]] = dataclasses.field(
        default_factory=dict
    )  # {company: {policy_id: "compliant"|"evading"|"relocating"}}
    economic_indicators: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "ai_investment_index": 100.0,
            "innovation_rate": 100.0,
            "public_trust": 50.0,
            "regulatory_burden": 0.0,
            "market_concentration": 30.0,
        }
    )
    events_log: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    round_number: int = 0
    coalitions: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    # {policy_id: {agent_name: "support"|"oppose"|"neutral"}}
    policy_support: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    # {agent_name: {policy_id: rounds_of_non_compliance}}
    non_compliance_rounds: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)

    def clamp_indicators(self) -> None:
        """Clamp all economic indicators to [0.0, 100.0]."""
        for key in self.economic_indicators:
            self.economic_indicators[key] = max(0.0, min(100.0, self.economic_indicators[key]))

    def check_policy_enactment(self, policy_id: str, threshold: float = 0.5) -> bool:
        """Check whether a policy has crossed the support threshold and enact it if so. Requires ≥2 agents to have registered a stance before enactment can occur."""
        policy = self.active_policies.get(policy_id)
        if not policy or policy.status != "proposed":
            return False

        support = self.policy_support.get(policy_id, {})
        if not support:
            return False

        n_support = sum(1 for v in support.values() if v == "support")
        n_oppose = sum(1 for v in support.values() if v == "oppose")
        n_total = len(support)

        # Require at least two agents to have weighed in.
        if n_total < 2:
            return False

        support_ratio = n_support / n_total
        if support_ratio >= threshold:
            policy.status = "enacted"
            policy.enacted_round = self.round_number
            return True

        if n_oppose / n_total > (1 - threshold):
            policy.status = "rejected"
            return False

        return False

    def register_support(self, policy_id: str, agent_name: str, position: str) -> None:
        """Register an agent's position on a policy.

        ``position`` must be one of "support", "oppose", "neutral".
        This is the only legitimate path for updating policy_support.
        """
        if policy_id not in self.policy_support:
            self.policy_support[policy_id] = {}
        self.policy_support[policy_id][agent_name] = position

    # Preferred alias — use this in new code.
    def register_stance(self, policy_id: str, agent_name: str, stance: str) -> None:
        """Record an agent's stance on a policy (support/oppose/neutral)."""
        self.register_support(policy_id, agent_name, stance)

    # ------------------------------------------------------------------
    # Compliance state helpers — always use these, never write directly
    # ------------------------------------------------------------------

    def set_compliance(self, actor: str, policy_id: str, status: str) -> None:
        """Record compliance status for one actor under one policy."""
        if actor not in self.compliance_tracker:
            self.compliance_tracker[actor] = {}
        self.compliance_tracker[actor][policy_id] = status

    def set_relocated(self, actor: str) -> None:
        """Mark actor as relocated under every active policy."""
        if actor not in self.compliance_tracker:
            self.compliance_tracker[actor] = {}
        for policy_id in self.active_policies:
            self.compliance_tracker[actor][policy_id] = "relocated"

    def to_summary(self) -> str:
        """Return a human-readable world state summary for agent context."""
        lines = [f"=== WORLD STATE (Round {self.round_number}) ==="]

        if self.active_policies:
            lines.append("\nACTIVE POLICIES:")
            for pid, policy in self.active_policies.items():
                lines.append(
                    f"  [{policy.status.upper()}] {policy.name}: "
                    f"{policy.description[:100]}"
                )
                if policy.status == "enforced":
                    lines.append(
                        f"    Compliance rate: {policy.compliance_rate:.0%}"
                    )
        else:
            lines.append("\nNo active policies.")

        if self.compliance_tracker:
            lines.append("\nCOMPLIANCE STATUS:")
            for company, statuses in self.compliance_tracker.items():
                for pid, status in statuses.items():
                    policy_name = self.active_policies.get(pid, Policy(pid, pid, "", [], [], [])).name
                    lines.append(f"  {company} → {policy_name}: {status}")

        lines.append("\nECONOMIC INDICATORS:")
        for indicator, value in self.economic_indicators.items():
            lines.append(f"  {indicator}: {value:.1f}")

        if self.coalitions:
            lines.append("\nACTIVE COALITIONS:")
            for name, members in self.coalitions.items():
                lines.append(f"  {name}: {', '.join(members)}")

        if self.events_log:
            recent = self.events_log[-5:]
            lines.append(f"\nRECENT EVENTS (last {len(recent)}):")
            for event in recent:
                summary = event.get("message") or event.get("summary", "?")
                visibility = event.get("visibility", "public")
                prefix = f"[{visibility[:3].upper()}] " if visibility != "public" else ""
                lines.append(f"  Round {event.get('round', '?')}: {prefix}{summary[:100]}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize world state to a JSON-compatible dict."""
        return {
            "active_policies": {
                pid: dataclasses.asdict(p)
                for pid, p in self.active_policies.items()
            },
            "compliance_tracker": self.compliance_tracker,
            "economic_indicators": self.economic_indicators,
            "events_log": self.events_log,
            "round_number": self.round_number,
            "coalitions": self.coalitions,
            "non_compliance_rounds": self.non_compliance_rounds,
        }


class WorldStateComponent(action_spec_ignored.ActionSpecIgnored):
    """Injects role-filtered world state into agent reasoning.

    Inherits from ActionSpecIgnored so it can be referenced in
    QuestionOfRecentMemories(components=(...)) without raising AttributeError
    on get_pre_act_label().
    """

    VISIBILITY = {
        "government": {"policies", "compliance", "economics", "coalitions", "events"},
        "regulator": {"policies", "compliance", "economics", "events"},
        "company": {"policies", "economics", "own_compliance", "events"},
        "civil_society": {"policies", "economics", "events"},
        "bad_actor": {"policies", "economics", "events"},
        "full": {"policies", "compliance", "economics", "coalitions", "events"},
    }

    def __init__(
        self,
        world_state: GovernanceWorldState,
        agent_name: str = "",
        agent_role: str = "full",
        pre_act_label: str = "Current world state",
    ):
        """Initialize with a shared GovernanceWorldState, the owning agent's name and role, and a context label."""
        super().__init__(pre_act_label)
        self._world_state = world_state
        self._agent_name = agent_name
        self._agent_role = agent_role

    @property
    def state(self) -> GovernanceWorldState:
        """Expose the underlying GovernanceWorldState for direct read access by the simulation runner."""
        return self._world_state

    def _make_pre_act_value(self) -> str:
        return self._filtered_summary()

    def _filtered_summary(self) -> str:
        """Generate role-filtered world state text for agent context."""
        ws = self._world_state
        visible = self.VISIBILITY.get(self._agent_role, self.VISIBILITY["full"])
        lines = [f"=== WORLD STATE (Round {ws.round_number}) ==="]

        if "policies" in visible:
            if ws.active_policies:
                lines.append("\nACTIVE POLICIES:")
                for pid, policy in ws.active_policies.items():
                    sev = policy.effective_severity()
                    lines.append(
                        f"  [{policy.status.upper()}] {policy.name} "
                        f"(severity {sev:.0f}/5): {policy.description[:150]}"
                    )
                    # Show penalties for enacted policies
                    if policy.status == "enacted" and policy.penalties:
                        real_pens = [p for p in policy.penalties if "as described" not in p.lower()]
                        if real_pens:
                            lines.append(f"    Penalties: {'; '.join(str(p) for p in real_pens[:3])}")
            else:
                lines.append("\nNo active policies.")

        if "compliance" in visible and ws.compliance_tracker:
            lines.append("\nCOMPLIANCE STATUS:")
            for company, statuses in ws.compliance_tracker.items():
                for pid, status in statuses.items():
                    policy_name = ws.active_policies.get(
                        pid, Policy(pid, pid, "", [], [], [])
                    ).name
                    lines.append(f"  {company} → {policy_name}: {status}")
        elif "own_compliance" in visible:
            own_comp = ws.compliance_tracker.get(self._agent_name, {})
            own_nc = ws.non_compliance_rounds.get(self._agent_name, {})
            has_any = own_comp or own_nc

            if has_any or any(p.status == "enacted" for p in ws.active_policies.values()):
                lines.append("\nYOUR COMPLIANCE STATUS:")
                for pid, policy in ws.active_policies.items():
                    if policy.status != "enacted":
                        continue
                    status = own_comp.get(pid, "NOT COMPLIANT")
                    nc_rounds = own_nc.get(pid, 0)
                    sev = policy.effective_severity()

                    if status == "compliant":
                        lines.append(f"  {policy.name}: COMPLIANT")
                    elif status == "relocated":
                        lines.append(f"  {policy.name}: RELOCATED (no longer subject)")
                    else:
                        pen_str = ""
                        real_pens = [p for p in policy.penalties if "as described" not in p.lower()]
                        if real_pens:
                            pen_str = f" Penalties you face: {'; '.join(str(p) for p in real_pens[:2])}"
                        grace = max(1, 3 - int(sev / 2))
                        if nc_rounds > grace:
                            lines.append(
                                f"  {policy.name}: NON-COMPLIANT ({nc_rounds} rounds). "
                                f"PAST GRACE PERIOD — active investigation risk.{pen_str}"
                            )
                        elif nc_rounds > 0:
                            lines.append(
                                f"  {policy.name}: NON-COMPLIANT ({nc_rounds}/{grace} grace rounds remaining).{pen_str}"
                            )
                        else:
                            lines.append(f"  {policy.name}: STATUS UNKNOWN — you have not declared compliance.{pen_str}")

            total_companies = len(ws.compliance_tracker)
            if total_companies > 0:
                lines.append(f"  ({total_companies} companies tracked, details hidden)")

        if "economics" in visible:
            lines.append("\nECONOMIC INDICATORS:")
            for indicator, value in ws.economic_indicators.items():
                lines.append(f"  {indicator}: {value:.1f}")

        if "coalitions" in visible and ws.coalitions:
            lines.append("\nACTIVE COALITIONS:")
            for name, members in ws.coalitions.items():
                lines.append(f"  {name}: {', '.join(members)}")

        if "events" in visible and ws.events_log:
            # Private events are only shown to the agent they concern.
            # Public events are shown to all agents with event visibility.
            visible_events = [
                e for e in ws.events_log
                if e.get("visibility", "public") == "public"
                or e.get("agent") == self._agent_name
            ]
            recent = visible_events[-3:]
            if recent:
                lines.append(f"\nRECENT EVENTS:")
                for event in recent:
                    summary = event.get("message") or event.get("summary", "?")
                    lines.append(f"  Round {event.get('round', '?')}: {summary}")

        return "\n".join(lines)

    def get_state(self) -> entity_component.ComponentState:
        """Serialize the world state to a JSON-compatible dict via GovernanceWorldState.to_dict()."""
        return self._world_state.to_dict()

    def set_state(self, state: entity_component.ComponentState) -> None:
        """Reconstruct the GovernanceWorldState from a previously serialized state dict."""
        self._world_state = GovernanceWorldState(
            active_policies={
                pid: Policy(**pdata)
                for pid, pdata in state.get("active_policies", {}).items()
            },
            compliance_tracker=state.get("compliance_tracker", {}),
            economic_indicators=state.get("economic_indicators", {}),
            events_log=state.get("events_log", []),
            round_number=state.get("round_number", 0),
            coalitions=state.get("coalitions", {}),
        )
