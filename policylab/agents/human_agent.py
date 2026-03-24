"""Human-in-the-loop agent for interactive play."""

from __future__ import annotations

from concordia.agents import entity_agent_with_logging
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class HumanActComponent(entity_component.ActingComponent):
    """Acting component that replaces LLM sampling with an interactive terminal prompt for a human player."""

    def __init__(self, agent_name: str):
        """Store the agent name used to label terminal prompts and action strings."""
        self._agent_name = agent_name

    def get_action_attempt(
        self,
        contexts: dict[str, str],
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Prompt the human player for input."""
        print(f"\n{'=' * 50}")
        print(f"YOUR TURN: {self._agent_name}")
        print(f"{'=' * 50}")

        for key, value in contexts.items():
            if value and value.strip():
                print(f"\n[{key}]")
                if len(value) > 300:
                    print(value[:300] + "...")
                else:
                    print(value)

        print(f"\n{action_spec.call_to_action}")
        print()

        if action_spec.output_type == entity_lib.OutputType.CHOICE:
            print("OPTIONS:")
            for i, opt in enumerate(action_spec.options):
                print(f"  {i + 1}. {opt}")
            while True:
                try:
                    choice = int(input(f"Choose (1-{len(action_spec.options)}): ")) - 1
                    if 0 <= choice < len(action_spec.options):
                        return action_spec.options[choice]
                except (ValueError, EOFError):
                    pass
                print("Invalid choice. Try again.")
        else:
            try:
                response = input(f"{self._agent_name}'s action: ")
                return f"{self._agent_name}: {response}" if response else f"{self._agent_name}: (no action)"
            except EOFError:
                return f"{self._agent_name}: (no action)"

    def get_state(self) -> entity_component.ComponentState:
        return {"agent_name": self._agent_name}

    def set_state(self, state: entity_component.ComponentState) -> None:
        if "agent_name" in state:
            self._agent_name = state["agent_name"]


class HumanObserverComponent(entity_component.ContextComponent):
    """Context component that prints incoming observations to the terminal and surfaces recent ones during the act phase."""

    def __init__(self):
        """Initialize with an empty observations buffer."""
        super().__init__()
        self._observations: list[str] = []

    def pre_observe(self, observation: str) -> str:
        """Append the observation to the buffer and echo a truncated preview to the terminal."""
        self._observations.append(observation)
        print(f"  >> {observation[:150]}")
        return ""

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Return the five most recent observations as a formatted context string for the human player."""
        if self._observations:
            recent = self._observations[-5:]
            return "Recent events:\n" + "\n".join(f"  - {o[:100]}" for o in recent)
        return ""

    def get_state(self) -> entity_component.ComponentState:
        return {"observations": self._observations[-20:]}

    def set_state(self, state: entity_component.ComponentState) -> None:
        self._observations = state.get("observations", [])


def build_human_agent(
    name: str,
    world_state=None,
    **kwargs,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a human-controlled agent."""
    observer = HumanObserverComponent()
    act = HumanActComponent(agent_name=name)

    components = {
        "observer": observer,
    }

    if world_state is not None:
        from policylab.components.governance_state import WorldStateComponent
        components["world_state"] = WorldStateComponent(
            world_state=world_state,
            agent_name=name,
            agent_role="full",
            pre_act_label="World state",
        )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act,
        context_components=components,
    )
