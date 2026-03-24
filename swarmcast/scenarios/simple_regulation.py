"""Five-agent scenario for the US AI Model Registration Act.

Agents: Senator Williams (government), Diana Chen/MegaAI Corp (large company),
Alex Rivera/NovaMind (startup), Director Park/AI Safety Board (regulator),
and Dr. Okonkwo/AI Accountability Institute (civil society).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model as lm_lib

from swarmcast.agents.governance_agents import (
    build_civil_society_agent,
    build_company_agent,
    build_government_agent,
    build_regulator_agent,
)
from swarmcast.components.actions import (
    ActionType,
    GovernanceAction,
    get_available_actions,
    parse_action_from_text,
)
from swarmcast.components.governance_state import (
    GovernanceWorldState,
    Policy,
)
from swarmcast.components.objectives import (
    CIVIL_SOCIETY,
    GOVERNMENT_US,
    REGULATOR,
    TECH_COMPANY_LARGE,
    TECH_COMPANY_STARTUP,
)
from swarmcast.game_master.resolution_engine import (
    ResolutionEngine,
)


def _get_model() -> lm_lib.LanguageModel:
    """Return an OpenAI-backed language model if OPENAI_API_KEY is set, otherwise a no-op mock."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    model_name = os.environ.get("POLICYLAB_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from swarmcast.llm_backend import OpenAIModel
            label = f"{model_name}"
            if base_url:
                label += f" @ {base_url}"
            print(f"[LLM] {label}")
            return OpenAIModel(
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
            )
        except Exception as e:
            print(f"[LLM] Failed: {e}")

    from concordia.language_model import no_language_model
    print("[LLM] No API key — using mock model (set OPENAI_API_KEY for real LLM)")
    return no_language_model.NoLanguageModel()


def _get_embedder():
    """Return a SentenceTransformer embedder if available, otherwise a deterministic random fallback."""
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Embedder] Using all-MiniLM-L6-v2")
        return lambda text: _model.encode(text)
    except ImportError:
        print("[Embedder] Using random embedder (install sentence-transformers for real)")
        def embedder(text: str) -> np.ndarray:
            rng = np.random.RandomState(hash(text) % 2**31)
            return rng.randn(384).astype(np.float32)
        return embedder


PREMISE = """\
It is 2026. The US government has proposed the "AI Model Registration Act" — \
mandatory registration for any AI model trained with more than 10^25 FLOPS. \
Requirements: pre-training notification, post-training safety evaluation, \
quarterly reporting. Penalties up to $10M per violation.

All stakeholders must now decide how to respond. Each round, each actor takes \
ONE concrete strategic action. Actions have real consequences — lobbying costs \
political capital, compliance costs innovation, evasion risks detection, and \
relocation damages the ecosystem.

Think strategically. Your resources are limited. Choose wisely.
"""

AGENT_CONFIGS = [
    {
        "name": "Senator Williams",
        "builder": "government",
        "kwargs": {"jurisdiction": "United States"},
        "resources": dict(GOVERNMENT_US.resources),
    },
    {
        "name": "Diana Chen (MegaAI Corp)",
        "builder": "company_large",
        "kwargs": {"company_size": "large"},
        "resources": dict(TECH_COMPANY_LARGE.resources),
    },
    {
        "name": "Alex Rivera (NovaMind)",
        "builder": "company_startup",
        "kwargs": {"company_size": "startup"},
        "resources": dict(TECH_COMPANY_STARTUP.resources),
    },
    {
        "name": "Director Park (AI Safety Board)",
        "builder": "regulator",
        "kwargs": {"agency": "the AI Safety Board"},
        "resources": dict(REGULATOR.resources),
    },
    {
        "name": "Dr. Okonkwo (AI Accountability Institute)",
        "builder": "civil_society",
        "kwargs": {"organization": "the AI Accountability Institute"},
        "resources": dict(CIVIL_SOCIETY.resources),
    },
]


def build_scenario():
    """Instantiate the model, embedder, world state, five agents, and resolution engine for the AI Model Registration Act scenario.

    Returns a tuple of (model, world_state, agents, agent_resources, resolution_engine) ready to pass to run_simulation.
    """
    print("=" * 70)
    print("AI GOVERNANCE SIMULATOR")
    print("Scenario: US AI Model Registration Act")
    print("=" * 70)

    model = _get_model()
    embedder = _get_embedder()

    world_state = GovernanceWorldState()
    world_state.active_policies["ai_registration_act"] = Policy(
        id="ai_registration_act",
        name="AI Model Registration Act",
        description=(
            "Mandatory registration for AI models trained with >10^25 FLOPS. "
            "Pre-training notification, post-training safety eval, quarterly reports. "
            "Penalties up to $10M."
        ),
        regulated_entities=["AI companies training frontier models"],
        requirements=[
            "Pre-training notification to AI Safety Board",
            "Post-training safety evaluation before deployment",
            "Quarterly capability and incident reports",
        ],
        penalties=["Up to $10M per violation", "Injunctive relief"],
        status="proposed",
        proposed_by="Senator Williams",
    )

    resolution_engine = ResolutionEngine(seed=42)

    agents = {}
    agent_resources = {}

    builders = {
        "government": build_government_agent,
        "company_large": build_company_agent,
        "company_startup": build_company_agent,
        "regulator": build_regulator_agent,
        "civil_society": build_civil_society_agent,
    }

    print("\nBuilding agents...")
    for config in AGENT_CONFIGS:
        name = config["name"]
        mem = basic_associative_memory.AssociativeMemoryBank(sentence_embedder=embedder)
        builder_fn = builders[config["builder"]]
        agents[name] = builder_fn(
            name=name.split(" (")[0],
            world_state=world_state,
            model=model,
            memory_bank=mem,
            **config["kwargs"],
        )
        agent_resources[name] = dict(config["resources"])

    print(f"  Built {len(agents)} agents")
    return model, world_state, agents, agent_resources, resolution_engine


def run_simulation(
    model: lm_lib.LanguageModel,
    world_state: GovernanceWorldState,
    agents: dict,
    agent_resources: dict[str, dict[str, float]],
    resolution_engine: ResolutionEngine,
    num_rounds: int = 8,
) -> dict:
    """Run the scenario for the given number of rounds."""
    from concordia.typing import entity as entity_lib

    print(f"\n{'=' * 70}")
    print(f"SIMULATION — {num_rounds} rounds")
    print(f"{'=' * 70}")

    for agent in agents.values():
        agent.observe(PREMISE)

    all_results = []

    for round_num in range(1, num_rounds + 1):
        world_state.round_number = round_num
        print(f"\n{'─' * 70}")
        print(f"ROUND {round_num}")
        print(f"{'─' * 70}")

        round_actions: list[GovernanceAction] = []
        round_results = []

        for agent_name, agent in agents.items():
            resources = agent_resources[agent_name]
            available = get_available_actions(resources)
            available_str = ", ".join(a.value for a in available)

            action_spec = entity_lib.free_action_spec(
                call_to_action=(
                    f"Round {round_num}/{num_rounds}. "
                    f"Your available resources: {json.dumps(resources)}. "
                    f"Actions you can afford: [{available_str}]. "
                    f"Choose ONE action. Be concrete and strategic. "
                    f"State the action type and target."
                ),
                tag="governance_action",
            )

            raw_action = agent.act(action_spec)
            parsed = parse_action_from_text(raw_action, agent_name)

            print(f"\n  [{agent_name}]")
            print(f"  Resources: { {k: f'{v:.0f}' for k, v in resources.items()} }")
            print(f"  Raw: {raw_action[:120]}...")
            print(f"  Parsed: {parsed.action_type.value}")

            round_actions.append(parsed)

        print(f"\n  --- RESOLUTION ---")
        for action in round_actions:
            outcome = resolution_engine.resolve(
                action=action,
                agent_resources=agent_resources[action.actor],
                world_state=world_state,
                all_actions_this_round=round_actions,
            )

            if action.actor in outcome.resource_changes:
                agent_resources[action.actor] = outcome.resource_changes[action.actor]

            status = "SUCCESS" if outcome.success else "BLOCKED" if outcome.blocked_reason else "FAILED"
            print(f"  [{action.actor}] {action.action_type.value} → {status}")
            if outcome.blocked_reason:
                print(f"    Reason: {outcome.blocked_reason}")
            print(f"    Effect: {outcome.description[:100]}")

            for name, agent in agents.items():
                if name == action.actor:
                    if outcome.observation_for_actor:
                        agent.observe(outcome.observation_for_actor)
                else:
                    if outcome.observation_for_all:
                        agent.observe(outcome.observation_for_all)

            round_results.append({
                "action": action.to_dict(),
                "outcome": outcome.to_dict(),
            })

        world_state.events_log.append({
            "round": round_num,
            "summary": f"Round {round_num}: {len(round_actions)} actions resolved",
            "results": round_results,
        })
        all_results.extend(round_results)

        print(f"\n  --- WORLD STATE ---")
        indicators = world_state.economic_indicators
        print(f"  Investment: {indicators['ai_investment_index']:.0f} | "
              f"Innovation: {indicators['innovation_rate']:.0f} | "
              f"Trust: {indicators['public_trust']:.0f} | "
              f"Reg burden: {indicators['regulatory_burden']:.0f} | "
              f"Concentration: {indicators['market_concentration']:.0f}")

    print(f"\n{'=' * 70}")
    print("SIMULATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nFinal World State:")
    print(world_state.to_summary())
    print(f"\nFinal Agent Resources:")
    for name, resources in agent_resources.items():
        print(f"  {name}: { {k: f'{v:.0f}' for k, v in resources.items()} }")

    return {
        "scenario": "AI Model Registration Act",
        "timestamp": datetime.now().isoformat(),
        "num_rounds": num_rounds,
        "total_actions": len(all_results),
        "results": all_results,
        "final_world_state": world_state.to_dict(),
        "final_resources": agent_resources,
    }


def save_results(data: dict, output_dir: str) -> str:
    """Write simulation results as JSON to output_dir/simulation_results.json and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "simulation_results.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {path}")
    return path


def main():
    """Build and run the AI Model Registration Act scenario for 8 rounds, then save results."""
    model, world_state, agents, resources, engine = build_scenario()
    data = run_simulation(
        model=model,
        world_state=world_state,
        agents=agents,
        agent_resources=resources,
        resolution_engine=engine,
        num_rounds=8,
    )
    save_results(data, "./results/simple_regulation")


if __name__ == "__main__":
    main()
