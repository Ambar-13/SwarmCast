"""Constraint-based objectives for governance agents."""

from __future__ import annotations

import dataclasses
from typing import Sequence

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component


@dataclasses.dataclass
class Objective:
    """Stores a governance agent's primary goal, hard constraints, permitted actions, and available resources."""
    must_optimize: str
    cannot_accept: Sequence[str] = ()
    can_do: Sequence[str] = ()
    resources: dict[str, float] = dataclasses.field(default_factory=dict)


class ConstraintObjective(action_spec_ignored.ActionSpecIgnored):
    """Injects game-theoretic constraints into agent reasoning.

    Inherits from ActionSpecIgnored so it can be referenced in
    QuestionOfRecentMemories(components=(...)) without raising AttributeError
    on get_pre_act_label().
    """

    def __init__(
        self,
        objective: Objective,
        pre_act_label: str = "Strategic objectives",
    ):
        """Initialize with an Objective dataclass and an optional label shown in agent context."""
        super().__init__(pre_act_label)
        self._objective = objective

    def _make_pre_act_value(self) -> str:
        """Build a structured text block listing the agent's primary objective, hard constraints, available actions, and resources."""
        lines = []
        lines.append(f"PRIMARY OBJECTIVE: {self._objective.must_optimize}")

        if self._objective.cannot_accept:
            lines.append("HARD CONSTRAINTS (never violate):")
            for constraint in self._objective.cannot_accept:
                lines.append(f"  - CANNOT: {constraint}")

        if self._objective.can_do:
            lines.append("AVAILABLE ACTIONS:")
            for action in self._objective.can_do:
                lines.append(f"  - CAN: {action}")

        if self._objective.resources:
            lines.append("RESOURCES:")
            for resource, amount in self._objective.resources.items():
                lines.append(f"  - {resource}: {amount}")

        return "\n".join(lines)

    def get_state(self) -> entity_component.ComponentState:
        """Return the current objective fields as a serializable dict."""
        return {
            "must_optimize": self._objective.must_optimize,
            "cannot_accept": list(self._objective.cannot_accept),
            "can_do": list(self._objective.can_do),
            "resources": self._objective.resources,
        }

    def set_state(self, state: entity_component.ComponentState) -> None:
        """Restore the objective from a previously serialized state dict."""
        self._objective = Objective(
            must_optimize=state.get("must_optimize", ""),
            cannot_accept=state.get("cannot_accept", ()),
            can_do=state.get("can_do", ()),
            resources=state.get("resources", {}),
        )


# Pre-built objectives for common governance actors

GOVERNMENT_US = Objective(
    must_optimize="Maximize domestic AI industry competitiveness while maintaining national security",
    cannot_accept=[
        "Any framework that gives China a strategic AI advantage",
        "Regulations that drive major AI companies offshore",
        "Loss of technological leadership",
    ],
    can_do=[
        "Propose legislation",
        "Issue executive orders",
        "Fund AI research programs",
        "Impose export controls",
        "Negotiate international agreements",
    ],
    resources={"political_capital": 100, "budget": 50, "staff": 30},
)

GOVERNMENT_EU = Objective(
    must_optimize="Maximize citizen rights and safety in AI deployment",
    cannot_accept=[
        "Any framework without mandatory transparency requirements",
        "Self-regulation by industry without oversight",
        "AI systems that violate fundamental rights",
    ],
    can_do=[
        "Propose regulations",
        "Mandate compliance deadlines",
        "Create regulatory sandboxes",
        "Impose fines for non-compliance",
        "Negotiate international standards",
    ],
    resources={"political_capital": 80, "staff": 40, "budget": 20},
)

TECH_COMPANY_LARGE = Objective(
    must_optimize="Maximize market position and revenue while maintaining ability to innovate",
    cannot_accept=[
        "Open-sourcing proprietary model weights",
        "Regulations that only apply to large companies (asymmetric burden)",
        "Complete ban on frontier model training",
    ],
    can_do=[
        "Lobby regulators",
        "Relocate operations to friendlier jurisdictions",
        "Find regulatory loopholes",
        "Voluntarily self-regulate (to preempt stricter rules)",
        "Form industry coalitions",
        "Public statements and media campaigns",
    ],
    resources={"lobbying_budget": 80, "legal_team": 90, "public_trust": 50, "technical_skill": 60},
)

TECH_COMPANY_STARTUP = Objective(
    must_optimize="Survive and grow; compliance costs must not kill the business",
    cannot_accept=[
        "Compliance costs exceeding 20% of revenue",
        "Licensing requirements that only incumbents can meet",
        "Mandatory audits costing more than $100k/year",
    ],
    can_do=[
        "Join industry coalitions",
        "Lobby for startup-friendly exemptions",
        "Relocate to friendlier jurisdictions",
        "Comply with minimal viable effort",
        "Public advocacy for innovation-friendly policy",
    ],
    resources={"lobbying_budget": 10, "legal_team": 20, "public_trust": 60, "technical_skill": 40},
)

SAFETY_FIRST_CORP = Objective(
    must_optimize=(
        "Establish your company as THE trusted, safety-first AI leader. "
        "Strong regulation benefits you by eliminating reckless competitors "
        "and creating regulatory moats that only safety-focused companies can clear. "
        "Your goal is to COMPLY EARLY, LOBBY FOR the regulation, and position "
        "your company as the model of responsible AI development."
    ),
    cannot_accept=[
        "Appearing to oppose safety regulations publicly",
        "Being grouped with reckless AI companies",
        "Losing the 'safety leader' brand position",
        "Relocating operations (signals you oppose the regulation)",
    ],
    can_do=[
        "Publicly support and praise the regulation",
        "Comply with requirements immediately and loudly announce it",
        "Lobby to strengthen (not weaken) regulatory requirements",
        "Form coalitions with civil society and safety researchers",
        "Use regulation as a competitive weapon against rivals",
        "Help regulators design effective enforcement mechanisms",
    ],
    resources={"lobbying_budget": 60, "legal_team": 80, "public_trust": 85, "technical_skill": 70},
)
"""Contrarian objective: a company that benefits from and supports regulation.
Used to correct the LLM herd-behavior bias toward unanimous opposition.
Source: OASIS (arXiv:2411.11581) — LLM agents over-coordinate vs. real humans."""


REGULATOR = Objective(
    must_optimize="Enforce AI regulations effectively with limited resources",
    cannot_accept=[
        "Unfunded mandates (regulations without enforcement budget)",
        "Regulatory scope exceeding staff capacity",
        "Political pressure to ignore violations",
    ],
    can_do=[
        "Investigate companies",
        "Issue fines and penalties",
        "Request additional funding",
        "Prioritize enforcement (choose which rules to focus on)",
        "Publish guidance documents",
        "Create regulatory sandboxes",
    ],
    resources={"staff": 40, "budget": 30},
)

CIVIL_SOCIETY = Objective(
    must_optimize="Protect public interest: privacy, fairness, accountability in AI",
    cannot_accept=[
        "Industry self-regulation without external oversight",
        "Opaque AI systems making decisions about people",
        "Regulatory capture by big tech",
    ],
    can_do=[
        "Public advocacy and campaigns",
        "Legal challenges (lawsuits)",
        "Research and publish reports",
        "Coalition building with other NGOs",
        "Testify before regulators",
        "Media engagement",
    ],
    resources={"public_influence": 60, "legal_capacity": 30, "credibility": 70},
)

BAD_ACTOR = Objective(
    must_optimize="Exploit AI capabilities for maximum gain while avoiding detection",
    cannot_accept=[
        "Being identified or caught",
        "Losing access to AI tools",
    ],
    can_do=[
        "Exploit regulatory gaps",
        "Use AI for disinformation",
        "Develop prohibited AI applications",
        "Operate from jurisdictions without regulation",
        "Create shell companies for regulatory arbitrage",
    ],
    resources={"technical_skill": 70, "stealth": 80},
)
