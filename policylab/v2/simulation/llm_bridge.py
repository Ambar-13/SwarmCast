"""Connects 5 Concordia-based LLM strategic agents to the v2 population engine.

The five agents represent the key institutional actors in an AI governance scenario:
  - Government Official: proposes counter-policies, sets enforcement priorities
  - Regulator: executes enforcement, allocates inspection capacity
  - Industry Association: coordinates lobbying, negotiates exemptions
  - Civil Society Leader: public campaigns, coalition formation
  - Safety-First Corp: early compliance, regulatory moat strategy (contrarian)

Each round the bridge:
  1. Injects current population statistics and stock state into the shared world state
  2. Each LLM agent observes that context, reasons, and chooses a structured action
  3. The action is parsed (keyword matching on the raw text) and broadcast back to world_state
  4. Population agents observe the broadcast in the next round and update beliefs via DeGroot

Actions that modify stocks close the LLM→population feedback loop: an
"enforce" action adds enforcement burden immediately; a "reform_policy" action
fires an LLMProposalEffect that discharges BurdenStock, which the population
observes as policy softening in the following round.
"""

from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class RandomEmbedder:
    """Fallback random embedder when no real embedding model is available."""
    def embed_text(self, text: str):
        import numpy as np, hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        return rng.normal(0, 1, 128)


@dataclasses.dataclass
class LLMRoundResult:
    agent_name: str
    action: str
    reasoning: str
    affects_population: bool  # True if this action will update population beliefs
    observation_text: str    # What population agents will see


def build_llm_strategic_agents(
    policy_name: str,
    policy_description: str,
    policy_severity: float,
    model,
    embedder,
    seed: int = 42,
):
    """Construct the five Concordia-based strategic agents and their shared world state.

    The five agents are Government Official, Regulator, Industry Association CEO,
    Civil Society Leader, and Safety-First Corp CEO. Each gets an associative memory
    bank seeded with the policy premise and population-dynamics context.
    Returns (agents_dict, resources_dict, objectives_dict, world_state).
    Requires concordia to be installed; raises ImportError otherwise.
    """
    from policylab.components.governance_state import (
        GovernanceWorldState, Policy
    )
    from policylab.agents.governance_agents import (
        build_government_agent, build_regulator_agent,
        build_company_agent, build_civil_society_agent,
    )
    from policylab.components.objectives import (
        GOVERNMENT_US, REGULATOR, TECH_COMPANY_LARGE,
        CIVIL_SOCIETY, SAFETY_FIRST_CORP,
    )
    from concordia.associative_memory import basic_associative_memory
    import random

    rng = random.Random(seed)

    # Build world state with policy already enacted
    world_state = GovernanceWorldState()
    policy = Policy(
        id=policy_name.lower().replace(" ", "_"),
        name=policy_name,
        description=policy_description,
        regulated_entities=[],
        requirements=[],
        penalties=[],
        status="enacted",
        enacted_round=0,
    )
    world_state.active_policies[policy.id] = policy

    # V2-enhanced premise: includes population dynamics context
    premise = (
        f'This regulation IS NOW LAW: "{policy_name}"\n\n'
        f"Description: {policy_description}\n\n"
        f"The regulation has been enacted. You are operating in a hybrid environment:\n"
        f"  - 5 key institutional actors (you and your peers)\n"
        f"  - ~100 firms, researchers, and investors responding according to their own logic\n"
        f"You will receive population-level statistics each round showing aggregate behavior.\n"
        f"Your strategic decisions can shift the population's beliefs and actions.\n"
        f"Each round, take ONE concrete strategic action with real consequences."
    )
    world_state.events_log.append({
        "round": 0, "type": "premise", "visibility": "public",
        "message": premise,
    })

    agents = {}
    resources = {}
    objectives = {}

    configs = [
        {
            "name": "Government Official",
            "builder_fn": build_government_agent,
            "kwargs": {"jurisdiction": "United States"},
            "resources": dict(GOVERNMENT_US.resources),
            "objective": GOVERNMENT_US,
        },
        {
            "name": "Regulator",
            "builder_fn": build_regulator_agent,
            "kwargs": {"agency": "Federal AI Prohibition Bureau"},
            "resources": {"staff": 150, "budget": 60, "expertise": 80},
            "objective": REGULATOR,
        },
        {
            "name": "Industry Association CEO",
            "builder_fn": build_company_agent,
            "kwargs": {
                "company_size": "large",
                "extra_context": (
                    "You represent the entire AI industry as head of the "
                    "National AI Industry Association. Your job is to coordinate "
                    "industry response: lobby for amendments, negotiate exemptions, "
                    "and manage the public narrative."
                ),
            },
            "resources": dict(TECH_COMPANY_LARGE.resources),
            "objective": TECH_COMPANY_LARGE,
        },
        {
            "name": "Civil Society Leader",
            "builder_fn": build_civil_society_agent,
            "kwargs": {"organization": "AI Safety Coalition"},
            "resources": dict(CIVIL_SOCIETY.resources),
            "objective": CIVIL_SOCIETY,
        },
        {
            "name": "Safety-First Corp CEO",
            "builder_fn": build_company_agent,
            "kwargs": {
                "company_size": "large",
                "extra_context": (
                    "CRITICAL IDENTITY: You support this regulation as competitive "
                    "strategy. Comply early, lobby FOR stronger enforcement, "
                    "help design implementation. Do NOT relocate or oppose."
                ),
            },
            "resources": dict(SAFETY_FIRST_CORP.resources),
            "objective": SAFETY_FIRST_CORP,
        },
    ]

    for cfg in configs:
        name = cfg["name"]
        res = dict(cfg["resources"])
        resources[name] = res

        mem = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder
        )
        agent = cfg["builder_fn"](
            name=name,
            world_state=world_state,
            model=model,
            memory_bank=mem,
            resource_getter=lambda n=name: resources[n],
            **cfg.get("kwargs", {}),
        )
        agents[name] = agent
        objectives[name] = cfg["objective"]

    return agents, resources, objectives, world_state


def run_llm_round(
    llm_agents: dict,
    llm_agent_resources: dict,
    llm_agent_objectives: dict,
    llm_world_state,
    population_summary: dict,
    stocks,
    round_num: int,
    model,
) -> list[dict]:
    """Run one round of LLM strategic decisions and return a list of action dicts.

    Flow: population stats and stock state are injected as public world-state
    observations → each agent calls act() → raw text is keyword-parsed to a
    structured action type → action is broadcast back to world_state so population
    agents observe it as a belief signal in the next round.
    Blocked actions are silently replaced with do_nothing (see objective constraint
    check below). Returns one dict per agent with keys: agent, action, reasoning, round.
    """
    from policylab.game_master.simulation_loop import (
        _make_action_spec,
        _parse_agent_action,
    )
    from policylab.components.actions import ActionType
    from policylab.components.constraint_enforcer import enforce_constraints

    # Inject population statistics as a world-state observation
    pop_obs = (
        f"[Round {round_num} Population Report] "
        f"Compliance: {population_summary.get('compliance_rate', 0):.0%} | "
        f"Relocation: {population_summary.get('relocation_rate', 0):.0%} | "
        f"Evasion: {population_summary.get('evasion_rate', 0):.0%} | "
        f"Domestic companies remaining: {population_summary.get('domestic_companies', 100):.0f}/100 | "
        f"Mean belief policy is harmful: {population_summary.get('mean_belief_harmful', 0.5):.2f}"
    )
    llm_world_state.events_log.append({
        "round": round_num, "type": "population_report",
        "visibility": "public", "message": pop_obs,
    })

    # Also inject stock state
    stock_obs = (
        f"[Round {round_num} Economic State] "
        f"Investment: {stocks.ai_investment_rate:.0f}/100 | "
        f"Burden: {stocks.burden.level:.0f}/100 | "
        f"Innovation: {stocks.innovation.level:.0f}/100"
    )
    llm_world_state.events_log.append({
        "round": round_num, "type": "stock_state",
        "visibility": "public", "message": stock_obs,
    })

    results = []
    for agent_name, agent in llm_agents.items():
        try:
            # Get agent action via Concordia
            action_spec = agent.act(action_spec=None)
            if hasattr(action_spec, "output"):
                raw_action = action_spec.output
            else:
                raw_action = str(action_spec)

            # Keyword-parse the raw LLM text to a structured action type.
            # "reform_policy" maps to LLMProposalEffect, which modifies BurdenStock
            # directly — this is the primary LLM→population feedback path.
            from policylab.game_master.resolution_engine import ResolutionEngine
            engine = ResolutionEngine()

            action_lower = raw_action.lower()
            if "enforce" in action_lower:
                action_type = "enforce"
            elif "reform" in action_lower or "repeal" in action_lower or "amend" in action_lower:
                action_type = "reform_policy"
            elif "lobby" in action_lower:
                action_type = "lobby"
            elif "comply" in action_lower:
                action_type = "comply"
            elif "relocate" in action_lower:
                action_type = "relocate"
            elif "coalition" in action_lower:
                action_type = "form_coalition"
            elif "statement" in action_lower or "public" in action_lower:
                action_type = "public_statement"
            else:
                action_type = "other"

            # SAFETY_FIRST_CORP cannot_accept includes relocation — this is the
            # contrarian agent that corrects LLM herd-behavior bias. When all
            # other agents pile onto a "relocate" signal, Safety-First Corp is
            # structurally prevented from joining, ensuring at least one domestic
            # institutional voice remains in the simulation.
            objective = llm_agent_objectives.get(agent_name)
            if objective and action_type == "relocate":
                cannot = " ".join(objective.cannot_accept).lower()
                if "relocat" in cannot:
                    action_type = "do_nothing"  # blocked by objective

            results.append({
                "agent": agent_name,
                "action": action_type,
                "reasoning": raw_action[:200],
                "round": round_num,
            })

            # Broadcast to world state
            llm_world_state.events_log.append({
                "round": round_num,
                "type": "llm_action",
                "visibility": "public",
                "message": f"{agent_name} chose {action_type}: {raw_action[:100]}",
            })
        except Exception as e:
            results.append({
                "agent": agent_name, "action": "do_nothing",
                "reasoning": f"error: {e}", "round": round_num,
            })

    return results
