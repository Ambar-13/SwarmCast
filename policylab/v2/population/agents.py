"""Heterogeneous population agents for PolicyLab v2 — legacy per-agent path.

This module implements the per-agent (non-vectorized) simulation path.
It is used when use_vectorized=False in HybridSimConfig. For large populations
(n > 500), the vectorized path in population_vectorized.py is faster; this
path is retained for interpretability, debugging, and scenarios where per-agent
memory/history tracking is needed.

KEY ADVANCE OVER V1
────────────────────
V1 used 6 LLM agents to represent the entire stakeholder population. This meant
every "population" decision was actually the output of a language model call on
a single narrative — not a distribution of heterogeneous actors. V2 replaces
this with 50-200 rule-based population agents using empirically calibrated
response functions (see response_functions.py). This enables:
  1. Population-level statistics (compliance rates, relocation rates) that
     are not dominated by a single agent's idiosyncratic response.
  2. Fast simulation — microseconds per agent vs ~30 seconds per LLM call.
  3. Genuine heterogeneity — risk tolerance, size, and resources vary across
     agents drawn from realistic statistical distributions.
  4. Social learning via DeGroot belief updating on an explicit network topology.

This architecture follows the IIASA macroeconomic ABM and the Bank of England
housing market model, both of which use rule-based heterogeneous agents for
population dynamics while reserving richer deliberation for a small number of
strategic actors.

AGENT TYPES
───────────
  large_company  — Top 20 AI firms (OpenAI, Google, Anthropic tier)
  mid_company    — Mid-tier AI firms ($100M-$10B valuation)
  startup        — Pre-revenue or seed-stage AI startups
  researcher     — Academic and independent researchers
  investor       — VC funds and institutional AI investors
  civil_society  — NGOs, advocacy orgs, public interest groups

POPULATION SYNTHESIS DISTRIBUTIONS
────────────────────────────────────
  Size:           Pareto(α=1.1) — power-law firm size distribution
                  (Axtell 2001, Science: US firm sizes follow Zipf/Pareto law)
  Risk tolerance: Beta(2, 5) — mean=0.29, mode=0.2; most agents are risk-averse.
                  A small right tail of higher-risk agents models the startups
                  and contrarian investors who relocate or evade sooner.
  Resources:      Proportional to size with uniform noise ×[0.8, 1.2].
  Network:        Barabasi-Albert (scale-free), assigned by the calling simulation
                  layer via populate agent.connections lists.
"""

from __future__ import annotations

import dataclasses
import random
import uuid
from collections import deque
from typing import Literal

import numpy as np

from policylab.v2.population.response_functions import (
    compliance_probability,
    relocation_probability,
    evasion_probability,
    lobbying_probability,
    update_belief,
    STUBBORNNESS,
)

AgentType = Literal[
    "large_company", "mid_company", "startup",
    "researcher", "investor", "civil_society"
]


@dataclasses.dataclass
class PopulationAgent:
    """A rule-based population agent with calibrated behavioral responses.

    Unlike LLM agents, these agents make decisions via calibrated probability
    functions, not language model calls. They are fast, numerous, and
    collectively produce realistic population-level dynamics.
    """
    id: str
    name: str
    agent_type: AgentType
    size: float               # 0-1 (relative size; Pareto distributed)
    risk_tolerance: float     # 0-1 (Beta(2,5) distributed — mostly risk-averse)
    resources: dict[str, float]
    belief_policy_harmful: float   # 0-1: belief that policy is harmful to them
    memory: deque             # last N observations (actions by self and neighbors)
    connections: list[str]    # neighbor agent IDs (from social graph)
    rounds_since_enactment: int = 0
    is_compliant: bool = False
    has_relocated: bool = False
    has_evaded: bool = False
    rounds_evading: int = 0
    domestic_innovation_share: float = 1.0  # 1.0 = fully domestic; 0 = relocated

    def decide_action(
        self,
        burden: float,
        severity: float,
        compliance_cost: float,
        detection_prob: float,
        fine_amount: float,
        neighbor_beliefs: list[float] | None = None,
        neighbor_weights: list[float] | None = None,
    ) -> dict:
        """Decide this round's action using calibrated response functions.

        Returns a dict with:
          action: str
          p_action: float (probability that drove the decision)
          updated_belief: float
        """
        # 1. Update belief via DeGroot social learning
        if neighbor_beliefs:
            stubbornness = STUBBORNNESS.get(self.agent_type, 0.55)
            self.belief_policy_harmful = update_belief(
                self.belief_policy_harmful,
                neighbor_beliefs,
                neighbor_weights,
                stubbornness=stubbornness,
            )

        # 2. Compute action probabilities
        p_comply = compliance_probability(
            self.rounds_since_enactment, self.agent_type, burden, severity
        )
        p_relocate = relocation_probability(
            burden, self.agent_type, self.risk_tolerance, self.has_relocated,
            policy_severity=severity,
        )
        p_evade = evasion_probability(
            compliance_cost, detection_prob, fine_amount, self.risk_tolerance
        )
        p_lobby = lobbying_probability(
            self.agent_type,
            perceived_policy_threat=burden * self.belief_policy_harmful,
            resources=self.resources.get("lobbying_budget", 30.0),
        )

        # 3. Action selection (priority ordering with stochastic sampling)
        r = random.random()

        # Already relocated — can't do much else
        if self.has_relocated:
            return {"action": "do_nothing", "p_action": 1.0,
                    "updated_belief": self.belief_policy_harmful}

        # Relocation is a terminal, high-stakes decision — check first
        if r < p_relocate * self.belief_policy_harmful:
            self.has_relocated = True
            self.domestic_innovation_share = 0.0
            return {"action": "relocate", "p_action": p_relocate,
                    "updated_belief": self.belief_policy_harmful}

        # Compliance (once compliant, stays compliant)
        if not self.is_compliant and r < p_comply:
            self.is_compliant = True
            return {"action": "comply", "p_action": p_comply,
                    "updated_belief": self.belief_policy_harmful}

        # Evasion (if not compliant and not relocated)
        if not self.is_compliant and r < p_evade:
            self.has_evaded = True
            self.rounds_evading += 1
            return {"action": "evade", "p_action": p_evade,
                    "updated_belief": self.belief_policy_harmful}

        # Lobbying
        if r < p_lobby:
            return {"action": "lobby", "p_action": p_lobby,
                    "updated_belief": self.belief_policy_harmful}

        # Default: do nothing this round
        return {"action": "do_nothing", "p_action": 1.0 - p_lobby,
                "updated_belief": self.belief_policy_harmful}


def _pareto_size(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate firm sizes from a Pareto distribution.

    Power-law firm sizes are empirically documented (Axtell 2001, Science).
    Pareto shape parameter α ≈ 1.1 for US firm size distribution.
    Normalized to [0, 1] range.
    """
    raw = rng.pareto(1.1, n) + 1.0
    return raw / raw.max()


def _beta_risk_tolerance(rng: np.random.Generator, n: int) -> np.ndarray:
    """Risk tolerance from Beta(2, 5) — most firms are risk-averse.

    Beta(2,5) has mean 0.29, mode 0.2. Most firms will avoid risk;
    a small tail of high-risk-tolerance actors (startups, etc.) will
    be more willing to evade or relocate quickly.
    """
    return rng.beta(2.0, 5.0, n)


def generate_population(
    n_total: int = 100,
    type_distribution: dict[str, float] | None = None,
    seed: int = 42,
    policy_severity: float = 3.0,
) -> list[PopulationAgent]:
    """Generate a heterogeneous population of n_total rule-based agents.

    The default type distribution is:
      40% startups, 25% mid companies, 15% large companies,
      10% researchers, 5% investors, 5% civil society.

    This is calibrated against the EU AI Act stakeholder consultation composition
    reported in EC Impact Assessment SWD(2021)84 final, Annex II — which found
    that SMEs and startups constituted the largest share of AI Act-affected firms
    by count (even though large companies dominated by revenue and lobbying spend).

    Note: this is the composition of the affected stakeholder population, not
    the distribution of lobbying influence or economic weight. Large companies
    and investors are over-represented in policy influence relative to their
    population share; that asymmetry is captured separately through the resource
    endowment and lobbying budget draws, not through population counts.
    """
    if type_distribution is None:
        type_distribution = {
            "startup": 0.40,
            "mid_company": 0.25,
            "large_company": 0.15,
            "researcher": 0.10,
            "investor": 0.05,
            "civil_society": 0.05,
        }

    rng = np.random.default_rng(seed)
    agents = []

    # Draw sizes and risk tolerances for entire population
    sizes = _pareto_size(rng, n_total)
    risk_tolerances = _beta_risk_tolerance(rng, n_total)

    # Assign types according to distribution
    types = []
    for t, frac in type_distribution.items():
        count = max(1, round(frac * n_total))
        types.extend([t] * count)
    # Trim or pad to exactly n_total
    types = types[:n_total]
    while len(types) < n_total:
        types.append("startup")
    rng.shuffle(types)  # type: ignore

    for i in range(n_total):
        atype: AgentType = types[i]  # type: ignore
        size = float(sizes[i])
        risk_tol = float(risk_tolerances[i])

        # Resource endowment scales with size (with noise)
        base = size * 100
        noise = float(rng.uniform(0.8, 1.2))
        resources = {
            "lobbying_budget": base * noise * (1.5 if atype == "large_company" else 0.8),
            "legal_team":      base * noise * (1.2 if atype in ("large_company", "mid_company") else 0.5),
            "public_trust":    float(rng.uniform(30, 80)),
            "technical_skill": base * noise,
        }

        # Initial belief that policy is harmful: large companies believe it's
        # more harmful (they have more to lose); civil society believes less so
        belief_map = {
            "large_company": float(rng.uniform(0.6, 0.95)),
            "mid_company":   float(rng.uniform(0.5, 0.85)),
            "startup":       float(rng.uniform(0.6, 0.95)),  # existential threat
            "researcher":    float(rng.uniform(0.3, 0.70)),  # mixed views
            "investor":      float(rng.uniform(0.5, 0.80)),
            "civil_society": float(rng.uniform(0.05, 0.40)), # generally supportive of regulation
        }
        belief = belief_map.get(atype, 0.5)

        agent = PopulationAgent(
            id=str(uuid.uuid4())[:8],
            name=f"{atype.replace('_',' ').title()} #{i+1}",
            agent_type=atype,
            size=size,
            risk_tolerance=risk_tol,
            resources=resources,
            belief_policy_harmful=belief,
            memory=deque(maxlen=5),  # remember last 5 rounds
            connections=[],          # filled by social graph
            rounds_since_enactment=0,
            is_compliant=False,
            has_relocated=False,
            domestic_innovation_share=1.0,
        )
        agents.append(agent)

    return agents


def summarize_population(
    agents: list[PopulationAgent],
) -> dict:
    """Compute population-level statistics at current state."""
    n = len(agents)
    if n == 0:
        return {}

    n_compliant = sum(1 for a in agents if a.is_compliant)
    n_relocated = sum(1 for a in agents if a.has_relocated)
    n_evading = sum(1 for a in agents if a.has_evaded and not a.is_compliant)

    type_counts = {}
    for a in agents:
        type_counts[a.agent_type] = type_counts.get(a.agent_type, 0) + 1

    mean_belief = sum(a.belief_policy_harmful for a in agents) / n

    return {
        "n_total": n,
        "n_compliant": n_compliant,
        "compliance_rate": n_compliant / n,
        "n_relocated": n_relocated,
        "relocation_rate": n_relocated / n,
        "n_evading": n_evading,
        "evasion_rate": n_evading / n,
        "mean_belief_harmful": mean_belief,
        "type_distribution": type_counts,
        "domestic_innovation_share": sum(
            a.domestic_innovation_share for a in agents
        ) / n,
    }
