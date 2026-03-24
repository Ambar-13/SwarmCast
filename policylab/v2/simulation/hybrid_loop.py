"""Main simulation entry point for PolicyLab v2.

Orchestrates the per-round hybrid loop:
  1. Population generation (vectorized PopulationArray or legacy per-agent)
  2. Network construction (sparse Barabasi-Albert influence matrix)
  3. Stock initialization (CompanyStock, BurdenStock, InnovationCapacity, RelocationPipeline)
  4. Multi-jurisdiction routing (EU source → US/UK/Singapore/UAE via softmax)
  5. Event queue processing (PolicyAmendmentEffect, TrustShockEffect, etc.)
  6. Per round: population decisions → relocation pipeline → stock updates
     → optional LLM strategic agents → memory update
  7. SMM moment computation and distance to GDPR/EU AI Act calibration targets

EventQueue deep_copy: events carry fired=True after they fire. If the same
EventQueue instance were shared across ensemble runs, run 2 would silently skip
every event that fired in run 1. deep_copy() resets that state so each run
sees a fresh queue.

LLM mode (run_llm_strategic) is off by default: it requires an OPENAI_API_KEY
and costs ~$5-15 per run. Population-only mode is free and covers most use cases.
"""

from __future__ import annotations

import dataclasses
import random
import time
from typing import Any

import numpy as np

from policylab.v2.population.agents import (
    PopulationAgent,
    generate_population,
    summarize_population,
)
from policylab.v2.network.social_graph import (
    build_governance_network,
    get_neighbor_beliefs,
    compute_network_statistics,
    identify_hubs,
)
from policylab.v2.stocks.governance_stocks import (
    GovernanceStocks,
    CompanyStock,
    BurdenStock,
    InnovationCapacity,
    RelocationPipeline,
    DimensionalAnchors,
)
from policylab.v2.calibration.smm_framework import (
    SimulatedMoments,
    compute_simulated_moments,
    TargetMoments,
    smm_objective,
)
from policylab.game_master.indicator_dynamics import (
    apply_indicator_feedback, RIGOROUS_BASELINE,
)
from policylab.game_master.resolution_config import DEFAULT_CONFIG
from policylab.v2.population.vectorized import (
    PopulationArray, build_influence_matrix, vectorized_round, update_memory,
)
from policylab.v2.international.jurisdictions import (
    MultiJurisdictionState, make_eu, make_us, make_uk, make_singapore, make_uae,
)
from policylab.v2.simulation.events import EventQueue


@dataclasses.dataclass
class HybridSimConfig:
    """Configuration for the v2 hybrid simulation."""
    n_population: int = 100
    """Number of rule-based population agents.
    Tradeoff: larger n reduces Monte Carlo variance but increases runtime roughly linearly.
    Below 50 the compliance S-curve becomes noisy; above 500 gains are marginal."""

    num_rounds: int = 16
    """Simulation horizon. 16 rounds = 4 years, matching the EU AI Act implementation cycle.
    Do not go below 8: Weibull compliance S-curves and relocation sigmoid both require
    ~6-8 rounds to reach their inflection points — shorter runs cut off before dynamics complete."""

    spillover_factor: float = 0.5
    """Fraction of domestic innovation preserved globally when a company relocates.
    A relocating firm takes its R&D capacity with it rather than destroying it.
    [ASSUMED] 0.5 is the midpoint; sweep 0.1–0.9 for sensitivity."""

    type_distribution: dict | None = None
    seed: int = 42
    use_network: bool = True
    verbose: bool = True

    run_llm_strategic: bool = False
    """Run 5 strategic LLM agents alongside population agents.

    When True, the simulation becomes a TRUE HYBRID:
      - 5 LLM strategic agents: government, regulator, industry association,
        civil society leader, safety-first company (from v1 StressTester)
      - n_population rule-based agents: the rest of the ecosystem

    LLM agents handle complex institutional reasoning:
      - Drafting counter-policies and amendments
      - Deciding whether to escalate enforcement
      - Strategic coalition building
      - Novel regulatory arbitrage strategies

    Population agents handle the realistic behavioral population:
      - Calibrated compliance trajectories (Weibull)
      - Sigmoid relocation decisions
      - Social belief updating via DeGroot learning
      - Market dynamics from sheer numbers

    This extends the MiroFish design to larger populations, but with:
      - Empirically calibrated behavioral responses (MiroFish has none)
      - Governance-domain action space (lobbying, enforcement, compliance)
      - Stock-flow accounting (proper conservation laws)
      - SMM calibration framework

    Off by default: requires concordia + OPENAI_API_KEY, costs ~$5-15 per run.
    Population-only mode (False) is free and sufficient for most analyses.
    """

    llm_model: object | None = None
    """Language model for LLM strategic agents. If None and run_llm_strategic=True,
    will attempt to auto-configure from environment."""

    llm_embedder: object | None = None

    # ── New in v2.1: vectorized engine ──────────────────────────────────────
    use_vectorized: bool = True
    """Use vectorized NumPy engine instead of per-agent Python loop.
    Enables 10,000+ agents at near-zero cost. Set False only when debugging
    per-agent decision logic — the per-agent path is ~100× slower."""

    # ── New in v2.1: multi-jurisdiction ─────────────────────────────────────
    source_jurisdiction: str = "EU"
    """Source jurisdiction (where policy is enacted). Options: EU, US, UK"""

    destination_jurisdictions: list[str] = dataclasses.field(
        default_factory=lambda: ["US", "UK", "Singapore", "UAE"]
    )
    """Destination jurisdictions for relocating companies."""

    relocation_temperature: float = 0.1
    """Softmax temperature for destination selection.
    [ASSUMED] 0.1 = mostly rational (cheapest wins); 0.5 = more random.
    Sweep [0.05, 0.1, 0.2, 0.5] for sensitivity."""

    # ── New in v2.1: event injection ─────────────────────────────────────────
    event_queue: object | None = None
    """Optional EventQueue for policy amendments, incidents, and shocks.
    None = static policy (no events). Pass an EventQueue for dynamic scenarios."""

    # ── Adversarial belief injection ─────────────────────────────────────────
    adversarial_injection_rate: float = 0.0
    """Fraction of high-degree network hubs that receive external belief
    manipulation each round. 0.0 disables injection (default).

    Models: industry association narrative campaigns, state-actor influence
    operations, well-funded lobbying targeting influential companies.

    [ASSUMED] all parameters here. Run sweep [0.01, 0.05, 0.10] before citing.
    """

    adversarial_injection_direction: float = 1.0
    """Direction of injected belief shift.
    +1.0 = push belief_policy_harmful upward (anti-regulation narrative)
    -1.0 = push belief_policy_harmful downward (pro-compliance narrative)
    """

    adversarial_injection_magnitude: float = 0.08
    """Per-round belief shift applied to each targeted hub.
    [ASSUMED] 0.08 ≈ a noticeable but not overwhelming influence per round.
    Sweep [0.03, 0.08, 0.15].
    """

    adversarial_injection_start_round: int = 1
    """Round when injection begins. Set > 1 to model delayed campaigns."""

    # ── Compliance cost scaling (AI vs GDPR proxy) ────────────────────────────
    compute_cost_factor: float = 1.0
    """Multiplier on Weibull compliance λ to account for the higher capital
    intensity of AI regulation compared to GDPR, from which λ was calibrated.

    GDPR compliance was primarily administrative (DPO appointment, privacy
    notices, DPIAs). AI regulation at the compute threshold requires novel
    technical capabilities: red-teaming pipelines, compute audit infrastructure,
    third-party safety evaluations — none of which have established commercial
    markets. These take longer to build and cost significantly more per dollar
    of R&D than GDPR compliance.

    compute_cost_factor = 1.0 → GDPR-equivalent compliance speed (the default;
      preserves all SMM calibration exactly).
    compute_cost_factor = 2.0 → AI compliance takes twice as long as GDPR.
    compute_cost_factor = 4.0 → AI compliance takes four times as long.

    [ASSUMED] No empirical compliance timeline data exists for AI-specific
    regulations at the compute threshold level. This parameter bridges that gap
    transparently. Always show results at [1.0, 2.0, 4.0] in any evidence pack
    citing compliance trajectories for compute-threshold legislation.

    When using parse_bill(), pass spec.compute_cost_factor if set, otherwise
    derive from compute_threshold_flops:
      1e27+ FLOPS → factor ~ 1.5  (few affected models, modest overhead)
      1e26  FLOPS → factor ~ 2.0  (SB-53 / EU AI Act GPAI level)
      1e25  FLOPS → factor ~ 3.0  (broad frontier coverage)
      1e24- FLOPS → factor ~ 4.0  (wide coverage, many models affected)
    """

    # ── Bounded confidence (Hegselmann-Krause) ────────────────────────────────
    hk_epsilon: float = 1.0
    """Confidence bound for belief updating. Agents only update their belief
    from neighbours whose belief is within hk_epsilon of their own.

    hk_epsilon = 1.0 (default): reduces exactly to pure DeGroot — all agents
      influence all neighbours regardless of belief distance. Calibration
      unchanged.
    hk_epsilon = 0.5: agents only update from neighbours within 0.5 belief
      units. Allows persistent belief clusters (pro- and anti-regulation camps)
      rather than forcing mathematical consensus.
    hk_epsilon = 0.3: tighter clustering; more polarisation.

    [ASSUMED] The HK mechanism is theoretically grounded (Hegselmann & Krause
    2002); the specific epsilon is not calibrated against empirical data. Run
    sensitivity across [0.3, 0.5, 1.0] before citing lobbying trajectories —
    lobbying rate is most sensitive to belief distribution.
    """


@dataclasses.dataclass
class HybridSimResult:
    config: HybridSimConfig
    policy_name: str
    policy_severity: float
    round_summaries: list[dict]
    stock_history: list[dict]
    final_stocks: dict
    final_population_summary: dict
    network_statistics: dict
    network_hubs: list[dict]
    simulated_moments: SimulatedMoments
    smm_distance_to_gdpr: float | None
    elapsed_seconds: float
    seed: int
    event_log: list  # descriptions of fired events
    jurisdiction_summary: dict  # final state of all jurisdictions

    def compliance_trajectory(self) -> list[float]:
        return [r.get("compliance_rate", 0.0) for r in self.round_summaries]

    def relocation_trajectory(self) -> list[float]:
        return [r.get("relocation_rate", 0.0) for r in self.round_summaries]

    def summary(self) -> str:
        sep = "─" * 68
        lines = [
            f"\n{sep}",
            f"HYBRID SIMULATION REPORT — {self.policy_name}",
            sep,
            f"Population: {self.config.n_population} agents | "
            f"Rounds: {self.config.num_rounds} | Seed: {self.seed}",
            f"Elapsed: {self.elapsed_seconds:.1f}s",
            "",
            "FINAL STOCK STATE:",
            f"  domestic_companies: {self.final_stocks.get("domestic_companies", 0):.0f}/100 remaining",
            f"  regulatory_burden:  {self.final_stocks.get("regulatory_burden", 0):.1f}/100"
            f" = {self.final_stocks.get("regulatory_burden", 0):.0f}% R&D overhead",
            f"  innovation_rate:    {self.final_stocks.get("innovation_rate", 0):.1f}/100"
            f" ≈ {DimensionalAnchors.innovation_to_tfp(self.final_stocks.get("innovation_rate", 0)):.2f}% TFP/yr",
            f"  ai_investment:      {self.final_stocks.get("ai_investment_index", 0):.1f}/100"
            f" ≈ ${DimensionalAnchors.investment_to_billions(self.final_stocks.get("ai_investment_index", 0)):.0f}B/yr",
            f"  public_trust:       {self.final_stocks.get("public_trust", 0):.1f}/100",
            "",
            f"POPULATION (n={self.config.n_population}, calibrated Weibull/sigmoid):",
            f"  compliance_rate:  {self.final_population_summary.get("compliance_rate", 0):.1%}",
            f"  relocation_rate:  {self.final_population_summary.get("relocation_rate", 0):.1%}",
            f"  evasion_rate:     {self.final_population_summary.get("evasion_rate", 0):.1%}",
            "",
        ]

        if self.smm_distance_to_gdpr is not None:
            lines += [
                "CALIBRATION (SMM distance to GDPR/EU AI Act moments):",
                f"  SMM objective: {self.smm_distance_to_gdpr:.4f}",
                f"  lobbying:   {self.simulated_moments.lobbying_rate:.2f} (target 0.85)",
                f"  relocation: {self.simulated_moments.relocation_rate:.2f} (target 0.12)",
                f"  compliance: {self.simulated_moments.compliance_rate_y1:.2f} (target 0.23 at 1yr)",
                "",
            ]

        if self.network_statistics:
            lines += [
                "NETWORK (Barabasi-Albert scale-free):",
                f"  {self.network_statistics.get("n_nodes", 0)} agents | "
                f"{self.network_statistics.get("n_edges", 0)} edges | "
                f"density={self.network_statistics.get("density", 0):.3f}",
            ]
            for hub in self.network_hubs[:3]:
                lines.append(f"  hub: {hub["name"]} ({hub["type"]}) cent={hub["centrality"]:.3f}")
            lines.append("")

        lines += [
            sep,
            "EPISTEMIC STATUS (v2):",
            "  Compliance trajectory: GROUNDED (DLA Piper GDPR Weibull fit)",
            "  Relocation sigmoid:    DIRECTIONAL (calibrated to EU AI Act data)",
            "  Stock-flow:            STRUCTURALLY VALID (conservation laws hold)",
            "  Spillover factor:      ASSUMED (0.5; sweep [0.3, 0.5, 0.7])",
            "  v2 improvements:       vectorized population, dimensional anchors, relocation pipeline",
            "  staff_scaling:         [ASSUMED] 0.3; sweep [0.1, 0.3, 0.5, 1.0]",
            "  severity scoring:      [ASSUMED] req_weight=0.5, pen_weight=0.7",
            "  periodization:         91.25 days/round (quarterly, ROUNDS_PER_YEAR=4)",
            sep,
        ]
        return "\n".join(lines)


def run_hybrid_simulation(
    policy_name: str,
    policy_description: str,
    policy_severity: float,
    config: HybridSimConfig | None = None,
    initial_stocks: GovernanceStocks | None = None,
) -> HybridSimResult:
    """Run the v2 hybrid simulation and return a fully populated HybridSimResult.

    If calling in an ensemble loop, pass a fresh config each time or rely on the
    internal deep_copy of event_queue — the same EventQueue must not be reused
    across calls without deep_copy or events fired in run 1 will be silently
    skipped in run 2. config.seed controls all stochastic elements; fix it for
    reproducibility or vary it across ensemble members.
    """
    if config is None:
        config = HybridSimConfig()

    rng = random.Random(config.seed)
    t0 = time.time()

    # --- Population generation ---
    if config.verbose:
        print(f"  [v2] Generating {config.n_population} population agents "
              f"({'vectorized' if config.use_vectorized else 'per-agent'})...")

    if config.use_vectorized:
        pop_arr = PopulationArray.generate(
            n=config.n_population,
            type_distribution=config.type_distribution,
            policy_severity=policy_severity,
            seed=config.seed,
        )
        agents = None
        agents_by_id = {}
    else:
        agents = generate_population(
            n_total=config.n_population,
            type_distribution=config.type_distribution,
            seed=config.seed,
            policy_severity=policy_severity,
        )
        agents_by_id = {a.id: a for a in agents}
        pop_arr = None

    # --- Network ---
    graph = None
    W = None
    W_raw = None  # raw (unnormalized) adjacency for degree statistics
    if config.use_network:
        if config.verbose:
            print(f"  [v2] Building scale-free influence network (n={config.n_population})...")
        if config.use_vectorized and pop_arr is not None:
            W, W_raw = build_influence_matrix(pop_arr, m=3, seed=config.seed,
                                              return_raw=True)
            # Degrees from RAW adjacency (W is row-normalized → row sums = 1.0,
            # not actual degree. Hub detection needs actual edge counts.)
            raw_degrees = np.asarray(W_raw.sum(axis=1)).ravel()
            net_stats = {
                "n_nodes": config.n_population,
                "n_edges": W_raw.nnz // 2,  # undirected: each edge counted twice
                "density": (W_raw.nnz // 2) / max(1, config.n_population * (config.n_population-1) / 2),
                "mean_degree": float(raw_degrees.mean()),
                "max_degree": float(raw_degrees.max()),
                "is_connected": True,
            }
            # Top hubs by raw degree (high degree = influential hub in BA network)
            top_idx = np.argsort(raw_degrees)[-5:][::-1]
            net_hubs = [
                {
                    "id": str(int(i)),
                    "name": f"agent_{i}",
                    "type": ["large_company","mid_company","startup",
                              "researcher","investor","civil_society",
                              "frontier_lab"][pop_arr.type_idx[i]],
                    # Degree centrality = degree / (n-1) using RAW adjacency counts
                    "centrality": round(
                        float(raw_degrees[i]) / max(1, config.n_population - 1), 4
                    ),
                    "raw_degree": int(raw_degrees[i]),
                }
                for i in top_idx
            ]
        else:
            graph = build_governance_network(agents, m=3, seed=config.seed)
            net_stats = compute_network_statistics(graph)
            net_hubs = identify_hubs(graph)
    else:
        net_stats = {}
        net_hubs = []

    # --- Stock initialization ---
    stocks = initial_stocks if initial_stocks is not None else GovernanceStocks()
    stocks.burden.add_policy_burden(policy_severity)

    # 3b. Multi-jurisdiction state
    _JUR_MAP = {"EU": make_eu, "US": make_us, "UK": make_uk,
                "Singapore": make_singapore, "UAE": make_uae}
    source_jur = _JUR_MAP.get(config.source_jurisdiction, make_eu)()
    source_jur.company_count = config.n_population
    dest_jurs = [_JUR_MAP[j]() for j in config.destination_jurisdictions
                 if j in _JUR_MAP and j != config.source_jurisdiction]
    mj_state = MultiJurisdictionState(source=source_jur, destinations=dest_jurs)
    np_rng = np.random.default_rng(config.seed)

    # --- Event queue ---
    # Events carry a fired=True flag after they execute. Without deep_copy, a
    # shared EventQueue passed across ensemble runs would silently skip every
    # event that already fired in run 1, producing incorrect later runs.
    event_queue: EventQueue = (
        config.event_queue.deep_copy()
        if config.event_queue is not None
        else EventQueue()
    )

    # --- LLM strategic agents (setup) ---
    llm_agents = {}
    llm_agent_resources = {}
    llm_agent_objectives = {}
    llm_world_state = None

    if config.run_llm_strategic:
        if config.verbose:
            print("  [v2] Initialising 5 LLM strategic agents...")
        try:
            from policylab.v2.simulation.llm_bridge import (
                build_llm_strategic_agents, LLMRoundResult
            )
            model = config.llm_model
            embedder = config.llm_embedder
            if model is None:
                from policylab.llm_backend import OpenAIModel
                import os
                model = OpenAIModel(
                    model_name=os.environ.get("POLICYLAB_MODEL", "gpt-4o-mini"),
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
            if embedder is None:
                from policylab.v2.simulation.llm_bridge import RandomEmbedder
                embedder = RandomEmbedder()
            llm_agents, llm_agent_resources, llm_agent_objectives, llm_world_state = \
                build_llm_strategic_agents(
                    policy_name=policy_name,
                    policy_description=policy_description,
                    policy_severity=policy_severity,
                    model=model, embedder=embedder, seed=config.seed,
                )
            if config.verbose:
                print(f"    Strategic agents: {list(llm_agents.keys())}")
        except Exception as e:
            if config.verbose:
                print(f"  [v2] LLM agents failed: {e} — population-only mode")
            llm_agents = {}

    # Track history
    round_summaries = []
    stock_history = []
    action_log: list[dict] = []
    event_log: list[str] = []

    # --- Main loop ---
    if config.verbose:
        print(f"  [v2] Running {config.num_rounds} rounds "
              f"(n={config.n_population}, vectorized={config.use_vectorized})...")

    for round_num in range(1, config.num_rounds + 1):
        round_actions = {"lobby": 0, "comply": 0, "evade": 0,
                         "relocate": 0, "do_nothing": 0}

        burden = stocks.burden.level
        compliance_cost = burden * 0.5 + policy_severity * 5.0
        detection_prob = DEFAULT_CONFIG.enforcement_base_prob_per_severity * policy_severity
        fine_amount = policy_severity * DEFAULT_CONFIG.enforcement_penalty_innovation_per_severity * 10

        # ── Process events (amendments, incidents, shocks) ──────────────────
        fired = event_queue.process(round_num, stocks, {}, None, mj_state, rng=rng)
        for desc in fired:
            event_log.append(f"R{round_num:02d}: {desc}")
            if config.verbose:
                print(f"    [EVENT] {desc}")

        # --- Round N: population decisions ---
        if config.use_vectorized and pop_arr is not None:
            # Vectorized path: all agents in one NumPy pass
            vec_result = vectorized_round(
                pop=pop_arr, W=W,
                burden=burden, policy_severity=policy_severity,
                compliance_cost=compliance_cost, detection_prob=detection_prob,
                fine_amount=fine_amount, round_num=round_num,
                run_seed=config.seed,
                compute_cost_factor=config.compute_cost_factor,
                hk_epsilon=config.hk_epsilon,
            )
            n_relocating_this_round = vec_result["n_relocating"]
            n_complying_this_round  = vec_result["n_complying"]
            round_actions["relocate"]   = n_relocating_this_round
            round_actions["comply"]     = n_complying_this_round
            round_actions["evade"]      = vec_result["n_evading"]
            round_actions["lobby"]      = vec_result["n_lobbying"]
            round_actions["do_nothing"] = vec_result["n_nothing"]

            # Mark relocating agents as having departed (via pipeline)
            new_reloc_mask = vec_result["new_relocators_mask"]
            if new_reloc_mask.any():
                for i in np.where(new_reloc_mask)[0]:
                    stocks.relocation_pipeline.add(f"agent_{i}", round_num)
                pop_arr.has_relocated |= new_reloc_mask

            compliance_rate = pop_arr.compliance_rate()
        else:
            # Legacy per-agent path
            n_relocating_this_round = 0
            n_complying_this_round = 0
            for agent in (agents or []):
                if agent.has_relocated:
                    continue
                agent.rounds_since_enactment += 1
                nb, nw = [], []
                if graph is not None:
                    nb, nw = get_neighbor_beliefs(agent.id, graph, agents_by_id, k=5)
                decision = agent.decide_action(
                    burden=burden, severity=policy_severity,
                    compliance_cost=compliance_cost, detection_prob=detection_prob,
                    fine_amount=fine_amount, neighbor_beliefs=nb, neighbor_weights=nw,
                )
                action = decision["action"]
                round_actions[action] = round_actions.get(action, 0) + 1
                if action == "relocate":
                    n_relocating_this_round += 1
                    stocks.relocation_pipeline.add(agent.id, round_num)
                elif action == "comply":
                    n_complying_this_round += 1
            compliance_rate = summarize_population(agents).get("compliance_rate", 0.0)

        # --- Relocation pipeline ---
        # Destination selection uses softmax discrete choice over jurisdictional
        # attractiveness (burden×0.7 + tax×0.3 − subsidy). The temperature
        # parameter (relocation_temperature) controls rationality: lower values
        # concentrate flows toward the cheapest destination; higher values spread
        # them more evenly. Burden enters in log scale to reflect diminishing
        # marginal sensitivity to extreme regulatory costs.
        departing, cancelled = stocks.relocation_pipeline.process(
            round_num, policy_softened=(stocks.burden.level < 30.0),
        )

        if departing:
            domestic_loss, global_preserve = stocks.innovation.apply_relocation_effect(
                n_relocated_this_round=len(departing),
                spillover_factor=config.spillover_factor,
            )
            # NOTE: stocks.companies.relocated is incremented in update() below
            # via n_arriving_from_pipeline — do NOT increment it here too.
            # Route to destination jurisdictions (not a black hole)
            flows = mj_state.process_relocations(
                n_relocating=len(departing), round_num=round_num, rng=np_rng,
                temperature=config.relocation_temperature,
            )
            if config.verbose and departing:
                dest_str = ", ".join(f"{k}:{v}" for k,v in flows.items()) or "unknown"
                print(f"    Round {round_num:02d}: {len(departing)} departed "
                      f"→ {dest_str} "
                      f"(dom −{domestic_loss:.1f}, global +{global_preserve:.1f})")

        # --- Stock updates ---
        stocks.companies.update(
            n_relocating_this_round=n_relocating_this_round,
            n_arriving_from_pipeline=len(departing),
            burden=burden,
            innovation_expectation=stocks.innovation.expected_future_level,
        )

        # Ongoing enforcement overhead: wired to ResolutionConfig.ongoing_burden_per_severity
        # [DIRECTIONAL] magnitude — see resolution_config.py for sweep range and calibration note
        ongoing_burden = policy_severity * DEFAULT_CONFIG.ongoing_burden_per_severity
        stocks.burden.level = min(100.0, stocks.burden.level + ongoing_burden)
        stocks.burden.cumulative_inflow += ongoing_burden
        stocks.burden.discharge_compliance(compliance_rate)

        # Probabilistic enforcement (population-only mode — enables SMM m6 computation)
        # Without LLM agents, enforcement must be stochastic to produce observable m6.
        # P(enforcement this round) = enf_prob per round = 0.015 × severity.
        # At severity=3: P=0.045/round = 0.36/year → fraction of rounds with enforcement ≈ 0.45
        # DLA Piper m6 target is 0.06/entity/year; this is the system-level enforcement rate.
        if not llm_agents:
            # Enforcement: use independent uniform draw
            # seed = run_seed * prime1 + round * prime2 (avoids correlation with action RNG)
            _ep = DEFAULT_CONFIG.enforcement_base_prob_per_severity * policy_severity
            _ev = float(np.random.default_rng(config.seed * 99991 + round_num * 39989).random())
            if _ev < _ep:
                stocks.burden.add_enforcement_burden(1)
                round_actions["enforce"] = round_actions.get("enforce", 0) + 1

        # R&D investment effect on innovation stock. Formula is symmetric around
        # investment=50: below 50 the stock declines, at 50 it is stable, above 50
        # it grows. Source: Ugur 2016 meta-analysis of R&D–innovation elasticities.
        stocks.innovation.apply_rd_investment(stocks.ai_investment_rate)
        stocks.innovation.apply_depreciation()
        stocks.innovation.update_expectation(
            current_level=stocks.innovation.level,
            investment=stocks.ai_investment_rate,
            burden=stocks.burden.level,
            domestic_company_fraction=stocks.companies.domestic_fraction(),
        )
        stocks.compute_investment_rate()

        if n_relocating_this_round > 0:
            stocks.public_trust = max(
                0.0, stocks.public_trust - n_relocating_this_round * 0.5
            )

        # --- LLM strategic agents (if enabled) ---
        # Each LLM call costs real money (~$5-15/run total). This block is
        # skipped entirely when run_llm_strategic=False (the default).
        llm_round_actions = []
        if llm_agents and llm_world_state is not None:
            # Build rich population context for LLM agents (closes feedback loop)
            if config.use_vectorized and pop_arr is not None:
                pop_ctx = pop_arr.to_summary()
            else:
                pop_ctx = summarize_population(agents) if agents else {}
            pop_ctx["action_frequencies"] = dict(round_actions)
            pop_ctx["destination_flows"] = mj_state.destination_summary()
            pop_ctx["fired_events"] = fired  # LLM agents know what just happened

            try:
                from policylab.v2.simulation.llm_bridge import run_llm_round
                llm_round_actions = run_llm_round(
                    llm_agents=llm_agents,
                    llm_agent_resources=llm_agent_resources,
                    llm_agent_objectives=llm_agent_objectives,
                    llm_world_state=llm_world_state,
                    population_summary=pop_ctx,
                    stocks=stocks,
                    round_num=round_num,
                    model=config.llm_model,
                )
                for llm_action in llm_round_actions:
                    act = llm_action.get("action", "other")
                    if act == "enforce":
                        stocks.burden.add_enforcement_burden(1)
                    elif act == "reform_policy":
                        # LLM reform proposal fires as an event (closes LLM→population loop)
                        from policylab.v2.simulation.events import (
                            PolicyEvent, RoundTrigger, LLMProposalEffect
                        )
                        effect = LLMProposalEffect(
                            proposer=llm_action.get("agent", "unknown"),
                            amendment_text=llm_action.get("reasoning", "Amendment proposed")[:80],
                            burden_delta=-8.0,
                        )
                        effect.apply(stocks, None, mj_state)
                        event_log.append(f"R{round_num:02d}: [LLM REFORM] {effect.amendment_text}")
                    round_actions[act] = round_actions.get(act, 0) + 1
            except Exception as e:
                if config.verbose and round_num == 1:
                    print(f"    LLM round failed: {e}")

        # --- Adversarial belief injection ---
        # Runs after LLM agents so it can't be neutralised by an LLM reform this round.
        # Skipped when injection_rate is 0 (the default) — zero overhead.
        if (
            config.adversarial_injection_rate > 0
            and config.use_vectorized
            and pop_arr is not None
            and W_raw is not None
            and round_num >= config.adversarial_injection_start_round
        ):
            try:
                from policylab.v2.influence.adversarial import inject_beliefs
                n_injected = inject_beliefs(
                    pop=pop_arr,
                    W_raw=W_raw,
                    injection_rate=config.adversarial_injection_rate,
                    injection_direction=config.adversarial_injection_direction,
                    injection_magnitude=config.adversarial_injection_magnitude,
                    round_num=round_num,
                    rng=np.random.default_rng(config.seed * 77777 + round_num),
                )
                if config.verbose:
                    print(f"    [injection] Round {round_num}: {n_injected} hubs injected "
                          f"(direction={config.adversarial_injection_direction:+.0f})")
            except Exception as e:
                if config.verbose:
                    print(f"    [injection] Injection failed: {e}")

        # --- Memory update ---
        if config.use_vectorized and pop_arr is not None:
            # policy_softened: True if any amendment event reduced burden this round
            policy_softened = any(
                "Δ-" in e or "burden Δ-" in e or "burden_delta=-" in e
                for e in fired
            )
            update_memory(
                pop=pop_arr,
                observed_reloc_frac=n_relocating_this_round / max(1, config.n_population),
                # observed_enforcement_frac: fraction of agents contacted by enforcement.
                # In population-only mode (llm_agents=None), enforce count comes from
                # vectorized_round() action counts; denominator is n_population.
                # OLD BUG: len(llm_agents or [1]) = 1 when llm_agents=None → sends
                # binary 0/1 into memory instead of a fraction.
                observed_enforcement_frac=round_actions.get("enforce", 0) / max(1, config.n_population),
                burden_norm=burden,
                round_num=round_num,
                policy_softened=policy_softened,
            )

        mj_state.record_round()

        # ── Population summary ───────────────────────────────────────────────
        if config.use_vectorized and pop_arr is not None:
            pop_summary = pop_arr.to_summary()
        else:
            pop_summary = summarize_population(agents) if agents else {}
        pop_summary["round"] = round_num
        pop_summary["action_frequencies"] = dict(round_actions)
        pop_summary["n_departing"] = len(departing)
        pop_summary["n_cancelled"] = len(cancelled)
        pop_summary["fired_events"] = len(fired)
        round_summaries.append(pop_summary)

        stock_snap = stocks.to_indicators_dict()
        stock_snap["round"] = round_num
        # Flatten jurisdiction data (avoid nested dicts breaking mean_final)
        for jname, jinfo in mj_state.destination_summary().items():
            stock_snap[f"dest_{jname}_companies"] = jinfo.get("company_count", 0)
            stock_snap[f"dest_{jname}_burden"] = jinfo.get("burden", 0)
        stock_history.append(stock_snap)
        action_log.append({"round": round_num, **round_actions})

    # --- SMM moments ---
    simulated_moments = compute_simulated_moments(round_summaries, n_rounds=config.num_rounds)
    target = TargetMoments()
    smm_dist = smm_objective(simulated_moments, target)
    # NOTE: SMM targets (m1-m6) are fitted to moderate policies (GDPR severity~3,
    # EU AI Act severity~3-4). High SMM distance for severity-5 policies is
    # EXPECTED — the model is extrapolating beyond its calibration regime.
    # The compliance trajectory SHAPE (S-curve) remains valid via Weibull.
    # Low SMM distance validates the model is well-calibrated for moderate policies.

    elapsed = time.time() - t0

    return HybridSimResult(
        config=config,
        policy_name=policy_name,
        policy_severity=policy_severity,
        round_summaries=round_summaries,
        stock_history=stock_history,
        final_stocks=stock_history[-1] if stock_history else {},
        final_population_summary=round_summaries[-1] if round_summaries else {},
        network_statistics=net_stats,
        network_hubs=net_hubs,
        simulated_moments=simulated_moments,
        smm_distance_to_gdpr=smm_dist,
        elapsed_seconds=elapsed,
        seed=config.seed,
        event_log=event_log,
        jurisdiction_summary=mj_state.destination_summary(),
    )
