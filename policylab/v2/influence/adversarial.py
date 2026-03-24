"""
Belief injection scenario analysis.

The DeGroot belief network in vectorized.py models organic social learning:
companies update beliefs about regulatory harm based on what they observe in
their industry network. This module extends that with adversarial dynamics —
what happens when an external actor systematically pushes beliefs through
high-centrality nodes.

What this answers
─────────────────
- If 5% of industry hubs receive anti-regulation messaging each round,
  how much does compliance rate drop over 8 quarters?
- Are denser influence networks more or less resistant?
- Does early injection (pre-compliance) vs late injection (post-compliance)
  have different effects?
- What network structural properties predict resilience?

Parameters
──────────
All parameters here are [ASSUMED] — the mechanism is grounded in the DeGroot
framework, but the specific injection rates are hypothetical. Run sweeps
before citing any specific number.

Usage
─────
    from policylab.v2.influence.adversarial import AdversarialInjector, run_with_injection

    # Run two matched simulations and compare
    result = run_with_injection(
        policy_name="EU AI Act",
        policy_description="Mandatory risk tiers.",
        policy_severity=3.0,
        injection_rate=0.05,        # 5% of hubs targeted per round
        injection_direction=1.0,    # +1 = push belief_harmful up (weakens compliance)
        injection_start_round=1,    # when injection begins
        n_population=2000,
    )
    print(result.compliance_delta)  # how much compliance dropped
    print(result.resilience_score)  # network structural resilience metric
"""

from __future__ import annotations

import dataclasses
import numpy as np
from scipy.sparse import csr_matrix


@dataclasses.dataclass
class InjectionResult:
    """Comparison between a clean run and an adversarially perturbed run."""
    baseline_compliance: float
    injected_compliance: float
    compliance_delta: float               # negative means injection reduced compliance
    baseline_relocation: float
    injected_relocation: float
    relocation_delta: float               # positive means injection increased relocation
    resilience_score: float               # 0-1, higher = more resistant to injection
    injection_params: dict
    round_compliance_baseline: list[float]
    round_compliance_injected: list[float]

    def summary(self) -> str:
        lines = [
            "ADVERSARIAL INJECTION ANALYSIS",
            f"  injection_rate:       {self.injection_params.get('rate', 0):.1%} of hubs per round",
            f"  injection_direction:  {'anti-regulation (↑burden narrative)' if self.injection_params.get('direction',1) > 0 else 'pro-regulation (↓burden narrative)'}",
            f"  injection_start:      round {self.injection_params.get('start_round', 1)}",
            "",
            "  Compliance rate:      "
            f"{self.baseline_compliance:.1%} (baseline) → "
            f"{self.injected_compliance:.1%} (with injection)  "
            f"Δ={self.compliance_delta:+.1%}",
            "  Relocation rate:      "
            f"{self.baseline_relocation:.1%} (baseline) → "
            f"{self.injected_relocation:.1%} (with injection)  "
            f"Δ={self.relocation_delta:+.1%}",
            f"  Resilience score:     {self.resilience_score:.3f}  "
            f"({'resistant' if self.resilience_score > 0.7 else 'moderate' if self.resilience_score > 0.4 else 'vulnerable'})",
        ]
        return "\n".join(lines)


def inject_beliefs(
    pop,                    # PopulationArray
    W_raw: csr_matrix,      # raw (unnormalized) adjacency for degree lookup
    injection_rate: float,  # fraction of high-degree hubs to target per round
    injection_direction: float,  # +1 = push harmful belief up; -1 = push down
    injection_magnitude: float = 0.08,  # [ASSUMED] how much each injection shifts belief
    round_num: int = 1,
    rng: np.random.Generator | None = None,
) -> int:
    """Apply one round of adversarial belief injection to hub nodes.

    Targets the top-degree nodes in the influence network — these have the
    most outgoing influence in subsequent DeGroot updates, so shifting their
    beliefs propagates through the network. This is the mechanism used in
    real influence operations: target the nodes with the most connections.

    Returns the number of agents that were injected this round.

    Parameters are all [ASSUMED] — sweep injection_rate [0.01, 0.05, 0.10]
    and injection_magnitude [0.03, 0.08, 0.15] before citing results.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = pop.n
    degrees = np.asarray(W_raw.sum(axis=1)).ravel()

    # How many nodes to target this round
    n_targets = max(1, int(n * injection_rate))

    # Pick from top-degree nodes with some noise (adversaries don't have perfect
    # information about network structure)
    top_k = min(n_targets * 4, n)
    top_idx = np.argsort(degrees)[-top_k:]
    chosen = rng.choice(top_idx, size=n_targets, replace=False)

    # Apply injection
    delta = injection_direction * injection_magnitude
    pop.beliefs[chosen] = np.clip(
        pop.beliefs[chosen] + delta, 0.0, 1.0
    ).astype(np.float32)

    return n_targets


def compute_resilience_score(
    W_raw: csr_matrix,
    n: int,
) -> float:
    """Estimate how resistant the influence network is to hub-targeted injection.

    A network where degree is very concentrated (a few hubs dominate) is more
    vulnerable: injecting a small fraction of nodes reaches most of the network.
    A more uniform degree distribution is harder to attack efficiently.

    Score: 1.0 = highly resilient (flat degree distribution)
           0.0 = highly vulnerable (star network, one hub dominates)

    This uses the normalized Gini coefficient of the degree distribution.
    [DIRECTIONAL] Gini captures concentration; the 1-Gini resilience mapping
    is intuitive but [ASSUMED]. Treat as ordinal, not cardinal.
    """
    degrees = np.asarray(W_raw.sum(axis=1)).ravel().astype(float)
    if degrees.sum() == 0:
        return 0.5

    # Gini coefficient
    degrees_sorted = np.sort(degrees)
    n_d = len(degrees_sorted)
    cumsum = np.cumsum(degrees_sorted)
    gini = (n_d + 1 - 2 * cumsum.sum() / cumsum[-1]) / n_d
    return float(np.clip(1.0 - gini, 0.0, 1.0))


def run_with_injection(
    policy_name: str,
    policy_description: str,
    policy_severity: float,
    injection_rate: float = 0.05,
    injection_direction: float = 1.0,
    injection_magnitude: float = 0.08,
    injection_start_round: int = 1,
    n_population: int = 2000,
    num_rounds: int = 8,
    seed: int = 42,
) -> InjectionResult:
    """Run matched baseline and adversarially-perturbed simulations.

    Both runs use the same seed, population, and policy. The injected run
    activates adversarial_injection_rate in HybridSimConfig — so injection
    goes through the full belief propagation loop, not a simplified manual pass.

    This gives a clean causal estimate: everything is held constant except
    the adversarial perturbation.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation
    from policylab.v2.population.vectorized import build_influence_matrix, PopulationArray

    base_config = HybridSimConfig(
        n_population=n_population, num_rounds=num_rounds,
        verbose=False, seed=seed, use_network=True,
        adversarial_injection_rate=0.0,
    )
    inj_config = HybridSimConfig(
        n_population=n_population, num_rounds=num_rounds,
        verbose=False, seed=seed, use_network=True,
        adversarial_injection_rate=injection_rate,
        adversarial_injection_direction=injection_direction,
        adversarial_injection_magnitude=injection_magnitude,
        adversarial_injection_start_round=injection_start_round,
    )

    r_base = run_hybrid_simulation(policy_name, policy_description, policy_severity, config=base_config)
    r_inj  = run_hybrid_simulation(policy_name, policy_description, policy_severity, config=inj_config)

    baseline_comp   = [s.get("compliance_rate", 0) for s in r_base.round_summaries]
    injected_comp   = [s.get("compliance_rate", 0) for s in r_inj.round_summaries]
    final_base_comp = r_base.final_population_summary.get("compliance_rate", 0)
    final_inj_comp  = r_inj.final_population_summary.get("compliance_rate", 0)
    final_base_reloc = r_base.final_population_summary.get("relocation_rate", 0)
    final_inj_reloc  = r_inj.final_population_summary.get("relocation_rate", 0)

    # Resilience from network structure (same seed → same network in both runs)
    pop_tmp = PopulationArray.generate(n_population, seed=seed)
    _, W_raw = build_influence_matrix(pop_tmp, m=3, seed=seed, return_raw=True)
    resilience = compute_resilience_score(W_raw, n_population)

    return InjectionResult(
        baseline_compliance=final_base_comp,
        injected_compliance=final_inj_comp,
        compliance_delta=final_inj_comp - final_base_comp,
        baseline_relocation=final_base_reloc,
        injected_relocation=final_inj_reloc,
        relocation_delta=final_inj_reloc - final_base_reloc,
        resilience_score=resilience,
        injection_params={
            "rate": injection_rate,
            "direction": injection_direction,
            "magnitude": injection_magnitude,
            "start_round": injection_start_round,
        },
        round_compliance_baseline=baseline_comp,
        round_compliance_injected=injected_comp,
    )
