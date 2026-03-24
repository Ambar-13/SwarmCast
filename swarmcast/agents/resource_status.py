"""Resource status Concordia component.

Injects the agent's current resource levels into the pre-act context so that
QuestionOfRecentMemories components (situation, strategic_thinking) can reason
about strategy with knowledge of actual game state — not just past observations.
"""

from __future__ import annotations

from typing import Callable

from concordia.typing import entity_component


class ResourceStatusComponent(entity_component.ContextComponent):
    """Shows the agent its current resource levels before strategic reasoning.

    This component is placed BEFORE strategic_thinking in the ConcatActComponent
    order so that the LLM sees resource constraints when deciding what to do.
    Without this, an agent might plan to lobby when it has 0 lobbying_budget,
    because its memory only contains observations, not balance-sheet data.

    Implements the full Concordia ContextComponent interface so it can be
    registered as a named component without raising AttributeError on set_entity.
    """

    def __init__(
        self,
        resource_getter: Callable[[], dict[str, float]],
        pre_act_label: str = "Current resources",
    ):
        self._get = resource_getter
        self._pre_act_label = pre_act_label

    def pre_act(self, action_spec) -> str:  # noqa: ARG002
        resources = self._get()
        if not resources:
            return ""
        lines = [f"{self._pre_act_label}:"]
        for k, v in resources.items():
            lines.append(f"  {k}: {v:.0f}")
        return "\n".join(lines)

    def get_state(self) -> entity_component.ComponentState:
        return {}

    def set_state(self, state: entity_component.ComponentState) -> None:
        pass  # No persistent state to restore
