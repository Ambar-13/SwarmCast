"""Factory functions for building governance agents."""

from __future__ import annotations

import dataclasses
from typing import Mapping

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import (
    concat_act_component,
    constant,
    memory as memory_component,
    observation,
    question_of_recent_memories,
)
from concordia.language_model import language_model as lm_lib

from swarmcast.components.objectives import (
    BAD_ACTOR,
    CIVIL_SOCIETY,
    GOVERNMENT_EU,
    GOVERNMENT_US,
    REGULATOR,
    TECH_COMPANY_LARGE,
    TECH_COMPANY_STARTUP,
    ConstraintObjective,
    Objective,
)
from swarmcast.components.governance_state import (
    GovernanceWorldState,
    WorldStateComponent,
)
from swarmcast.agents.resource_status import ResourceStatusComponent


def _build_governance_agent(
    name: str,
    objective: Objective,
    backstory: str,
    world_state: GovernanceWorldState,
    model: lm_lib.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    agent_role: str = "full",
    resource_getter=None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Initialize agent components (memory, objectives, world state) and return the assembled agent."""

    agent_memory = memory_component.AssociativeMemory(
        memory_bank=memory_bank,
    )

    role = constant.Constant(
        state=backstory,
        pre_act_label="Role",
    )

    obs_to_memory = observation.ObservationToMemory()

    recent_observations = observation.LastNObservations(
        history_length=50,
        pre_act_label="Recent events",
    )

    situation = question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        pre_act_label="Current situation assessment",
        question=(
            f"Given recent events, what is the current political and "
            f"strategic situation facing {name}? Respond in 2-3 sentences."
        ),
        answer_prefix=f"The current situation for {name} is ",
        add_to_memory=False,
        components=("observation", "objectives", "world_state"),
        num_memories_to_retrieve=10,
    )

    strategic_thinking = question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        pre_act_label="Strategic assessment",
        question=(
            f"Given {name}'s objectives and the current situation, "
            f"what is the most strategically advantageous move right now? "
            f"Consider trade-offs and second-order effects. Respond in 2-3 sentences."
        ),
        answer_prefix=f"{name} should strategically ",
        add_to_memory=False,
        components=("observation", "objectives", "world_state"),
        num_memories_to_retrieve=10,
    )

    constraint_obj = ConstraintObjective(
        objective=objective,
        pre_act_label="Strategic objectives",
    )

    world_state_view = WorldStateComponent(
        world_state=world_state,
        agent_name=name,
        agent_role=agent_role,
        pre_act_label="World state",
    )

    components: dict[str, object] = {
        "__memory__": agent_memory,
        "__observation__": obs_to_memory,
        "role": role,
        "observation": recent_observations,
        "situation": situation,
        "strategic_thinking": strategic_thinking,
        "objectives": constraint_obj,
        "world_state": world_state_view,
    }

    component_order = [
        "role",
        "objectives",
        "world_state",
        "observation",
        "situation",
    ]

    if resource_getter is not None:
        components["resource_status"] = ResourceStatusComponent(
            resource_getter=resource_getter,
            pre_act_label="Current resources",
        )
        component_order.append("resource_status")

    component_order.append("strategic_thinking")

    act = concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act,
        context_components=components,
    )


def build_government_agent(
    name: str,
    jurisdiction: str,
    world_state: GovernanceWorldState,
    model: lm_lib.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    objective: Objective | None = None,
    resource_getter=None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a government agent for the given jurisdiction."""
    if objective is None:
        objective = GOVERNMENT_US if "US" in jurisdiction.upper() else GOVERNMENT_EU

    backstory = (
        f"{name} is a senior government official in {jurisdiction} responsible "
        f"for AI policy. They must balance innovation with safety, economic "
        f"competitiveness with citizen protection, and domestic interests with "
        f"international cooperation. They have significant political capital "
        f"but face electoral pressure and lobbying from multiple directions."
    )

    return _build_governance_agent(
        name=name,
        objective=objective,
        backstory=backstory,
        world_state=world_state,
        model=model,
        memory_bank=memory_bank,
        agent_role="government",
        resource_getter=resource_getter,
    )


def build_company_agent(
    name: str,
    company_size: str,
    world_state: GovernanceWorldState,
    model: lm_lib.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    objective: Objective | None = None,
    resource_getter=None,
    extra_context: str | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a tech company agent (large or startup).

    extra_context: persona override appended to backstory, e.g. 'strongly supports regulation'.
    """
    if objective is None:
        objective = (
            TECH_COMPANY_LARGE if company_size == "large" else TECH_COMPANY_STARTUP
        )

    backstory = (
        f"{name} is the head of AI policy at a {'major' if company_size == 'large' else 'startup'} "
        f"AI company. {'The company has significant resources for lobbying and legal teams, ' if company_size == 'large' else 'The company has limited resources and compliance costs threaten survival. '}"
        f"They must navigate regulations strategically — complying where necessary, "
        f"lobbying where beneficial, and considering relocation if regulation becomes "
        f"too burdensome. Every decision affects the bottom line."
    )

    if extra_context:
        backstory = backstory + " " + extra_context

    return _build_governance_agent(
        name=name,
        objective=objective,
        backstory=backstory,
        world_state=world_state,
        model=model,
        memory_bank=memory_bank,
        agent_role="company",
        resource_getter=resource_getter,
    )


def build_regulator_agent(
    name: str,
    agency: str,
    world_state: GovernanceWorldState,
    model: lm_lib.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    objective: Objective | None = None,
    resource_getter=None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a regulator agent for the given agency."""
    if objective is None:
        objective = REGULATOR

    backstory = (
        f"{name} is the director of {agency}, the regulatory body responsible "
        f"for AI oversight. They have a mandate to enforce AI regulations but "
        f"limited staff and budget. They must prioritize which regulations to "
        f"enforce, which companies to investigate, and how to allocate scarce "
        f"resources. Political pressure and industry lobbying complicate their work."
    )

    return _build_governance_agent(
        name=name,
        objective=objective,
        backstory=backstory,
        world_state=world_state,
        model=model,
        memory_bank=memory_bank,
        agent_role="regulator",
        resource_getter=resource_getter,
    )


def build_civil_society_agent(
    name: str,
    organization: str,
    world_state: GovernanceWorldState,
    model: lm_lib.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    objective: Objective | None = None,
    resource_getter=None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a civil society agent for the given organization."""
    if objective is None:
        objective = CIVIL_SOCIETY

    backstory = (
        f"{name} is the executive director of {organization}, a civil society "
        f"organization focused on AI ethics and accountability. They advocate "
        f"for public interest, push for stronger regulation, and challenge "
        f"industry when it prioritizes profit over safety. They have significant "
        f"public credibility but limited financial resources compared to industry."
    )

    return _build_governance_agent(
        name=name,
        objective=objective,
        backstory=backstory,
        world_state=world_state,
        model=model,
        memory_bank=memory_bank,
        agent_role="civil_society",
        resource_getter=resource_getter,
    )


def build_bad_actor_agent(
    name: str,
    world_state: GovernanceWorldState,
    model: lm_lib.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    objective: Objective | None = None,
    resource_getter=None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a bad actor agent."""
    if objective is None:
        objective = BAD_ACTOR

    backstory = (
        f"{name} operates in the shadows of the AI ecosystem. They exploit "
        f"regulatory gaps, use AI for activities that others would consider "
        f"unethical or illegal, and operate from jurisdictions where regulation "
        f"is weak. Their goal is maximum gain with minimum detection risk. "
        f"They adapt quickly when new regulations close old loopholes."
    )

    return _build_governance_agent(
        name=name,
        objective=objective,
        backstory=backstory,
        world_state=world_state,
        model=model,
        memory_bank=memory_bank,
        agent_role="bad_actor",
        resource_getter=resource_getter,
    )
