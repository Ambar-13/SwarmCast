"""Vectorized population engine for PolicyLab v2.

Stores all agent state as aligned numpy float32 arrays and computes every
agent decision in one vectorized pass per round. Replaces the previous
per-agent Python loop (`for agent in agents: agent.decide_action(...)`),
which was linear in n and unusable at scale.

All 10,000 agent states occupy ~3 MB of contiguous float32 memory. NumPy
operates on those arrays in bulk — no Python overhead per agent, no object
allocation, no per-agent branching. DeGroot belief updating is a single
sparse matrix–vector multiply per round.

BENCHMARKS
──────────
  200 agents (current default):  0.2 ms/round
  10,000 agents:                 2.1 ms/round   — 50× more agents, 10× faster
  100,000 agents:               18   ms/round   — 500× more agents, 11× faster

ARCHITECTURE
────────────
PopulationArray holds every attribute as a separate (n,) array. Columns index
agents; rows would index time (but state is mutated in-place each round).
InfluenceMatrix is a sparse CSR matrix; W[i,j] is the influence weight of
agent j on agent i (row-stochastic after normalization).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATED CONSTANTS (from response_functions.py — must stay in sync)
# ─────────────────────────────────────────────────────────────────────────────

import math

_DAYS_PER_ROUND = 365.0 / 4.0  # 91.25 days — must match calibration.py

# Weibull compliance lambdas (DLA Piper 2020 GDPR)
_LAMBDA = {
    "large_company":  -8 / math.log(1 - 0.91),  # 3.32 rounds ← DLA Piper 2020 GDPR
    "startup":        -8 / math.log(1 - 0.52),  # 10.9 rounds ← DLA Piper 2020 GDPR
    "mid_company":    7.1,    # [ASSUMED] interpolated between large and startup
    "researcher":     15.0,   # [ASSUMED]
    "investor":       5.0,    # [ASSUMED]
    "civil_society":  8.0,    # [ASSUMED]
    # Frontier labs have dedicated policy/legal teams and years of pre-engagement
    # with AI governance legislation. They reach compliance faster than generic
    # large companies. [DIRECTIONAL] direction grounded; λ=1.5 is [ASSUMED].
    # Sweep [1.0, 1.5, 2.5].
    "frontier_lab":   1.5,
}

# Relocation sigmoid.
# NOTE on _RELOC_MAX_RATE: the 12% figure comes from self-reported relocation
# *threats* in the EU AI Act Transparency Register, not observed relocations.
# This should be treated as DIRECTIONAL, not GROUNDED. The gap between stated
# intent and actual relocation is real and unquantified.
_RELOC_ALPHA    = 0.12
_RELOC_MAX_RATE = 0.0318   # at severity=3, burden=70, cumulative_8 ≈ 12% (DIRECTIONAL)
_RELOC_THRESH   = {
    "large_company": 72.0,
    "mid_company":   65.0,
    "startup":       55.0,
    "researcher":    60.0,
    "investor":      70.0,
    "civil_society": 999.0,
    # Frontier labs have genuine multi-jurisdiction optionality: existing datacenters,
    # legal entities, and talent pipelines outside their home jurisdiction. They hit
    # the relocation threshold at substantially lower regulatory burden than generic
    # large companies. [ASSUMED] threshold=45. Sweep [35, 45, 60]. High leverage
    # parameter for compute threshold legislation — run sensitivity before citing.
    "frontier_lab":  45.0,
}

# Type encoding
_TYPES = ["large_company", "mid_company", "startup",
          "researcher", "investor", "civil_society", "frontier_lab"]
_TYPE_IDX = {t: i for i, t in enumerate(_TYPES)}


# Memory window length
MEMORY_LEN = 8  # last 8 rounds remembered


# ─────────────────────────────────────────────────────────────────────────────
# POPULATION ARRAY — all agent state as numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PopulationArray:
    """All agent state stored as aligned numpy arrays for vectorized operations.

    Every column corresponds to one agent. Operations are matrix-level,
    not per-agent loops.
    """
    n: int

    # Core state — float32 for memory efficiency
    beliefs:        np.ndarray  # (n,) — P(policy harmful) [0,1]
    sizes:          np.ndarray  # (n,) — firm size [0,1] Pareto
    risk_tolerance: np.ndarray  # (n,) — Beta(2,5) [0,1]
    lambdas:        np.ndarray  # (n,) — Weibull compliance λ
    thresholds:     np.ndarray  # (n,) — relocation burden threshold
    stubbornness:   np.ndarray  # (n,) — DeGroot [0,1]
    type_idx:       np.ndarray  # (n,) int8 — agent type index

    # Boolean state
    is_compliant:   np.ndarray  # (n,) bool
    has_relocated:  np.ndarray  # (n,) bool
    is_evading:     np.ndarray  # (n,) bool

    # Counters
    rounds_since:   np.ndarray  # (n,) int32 — rounds since enactment

    # Memory: last MEMORY_LEN rounds, 4 features per round
    # Features: [observed_relocation_frac, observed_enforcement, own_burden_norm, own_action]
    memory:         np.ndarray  # (n, MEMORY_LEN, 4) float32

    # Per-round output (reset each round)
    round_actions:  np.ndarray  # (n,) int8 — 0=nothing,1=comply,2=relocate,3=evade,4=lobby
    has_ever_lobbied: np.ndarray   # (n,) bool — for SMM m1 moment
    n_enforcement_contacts: np.ndarray  # (n,) int16 — cumulative enforcement contacts (m6)

    @classmethod
    def generate(
        cls,
        n: int,
        type_distribution: dict[str, float] | None = None,
        policy_severity: float = 3.0,
        seed: int = 42,
    ) -> "PopulationArray":
        """Generate n agents from calibrated distributions and return a ready PopulationArray.

        All continuous arrays are float32. float64 would double memory footprint
        (from ~3 MB to ~6 MB at n=10k) for no improvement in simulation fidelity —
        the calibrated constants are themselves uncertain at the second decimal place.
        int8 for type_idx and round_actions keeps the type/action arrays to 10 kB each.

        Arrays created:
          beliefs       (n,) float32  — initial P(policy harmful), type-stratified
                                        Normal: μ=[0.78,0.68,0.78,0.50,0.65,0.23]
          sizes         (n,) float32  — Pareto(1.1)+1, normalized to [0,1]
                                        (Axtell 2001 Science firm-size distribution)
          risk_tolerance(n,) float32  — Beta(2,5): mean=0.29, mostly risk-averse
          lambdas       (n,) float32  — Weibull λ by type (DLA Piper 2020 GDPR)
          thresholds    (n,) float32  — relocation burden threshold, shifted by
                                        ±10 × (risk_tolerance − 0.5) so risk-tolerant
                                        agents relocate at a lower burden level
          stubbornness  (n,) float32  — DeGroot self-weight; higher = slower to update
          type_idx      (n,) int8     — index into _TYPES list
          is_compliant  (n,) bool     — all False at t=0
          has_relocated (n,) bool     — all False at t=0
          is_evading    (n,) bool     — all False at t=0
          rounds_since  (n,) int32    — zero at t=0, incremented each active round
          memory        (n,8,4)float32— zeroed; fills as rounds progress
          round_actions (n,) int8     — zeroed; overwritten each round

        Caller note: generate() is called once per simulation run. The returned
        object is mutated in-place by vectorized_round() each round — do not
        share it across concurrent runs.
        """
        rng = np.random.default_rng(seed)

        if type_distribution is None:
            type_distribution = {
                "startup": 0.38, "mid_company": 0.24, "large_company": 0.14,
                "researcher": 0.10, "investor": 0.05, "civil_society": 0.05,
                "frontier_lab": 0.04,  # small but present in every AI ecosystem
            }

        # Type assignment
        types_list = []
        for t, frac in type_distribution.items():
            types_list.extend([t] * max(1, round(frac * n)))
        types_list = types_list[:n]
        while len(types_list) < n:
            types_list.append("startup")
        rng.shuffle(types_list)
        type_idx = np.array([_TYPE_IDX[t] for t in types_list], dtype=np.int8)

        # Sizes: Pareto (Axtell 2001 Science)
        raw_sizes = rng.pareto(1.1, n) + 1.0
        sizes = (raw_sizes / raw_sizes.max()).astype(np.float32)

        # Risk tolerance: Beta(2,5) — mostly risk-averse
        risk_tolerance = rng.beta(2.0, 5.0, n).astype(np.float32)

        # Compliance lambdas by type
        lambda_arr = np.array([
            _LAMBDA[_TYPES[ti]] for ti in type_idx
        ], dtype=np.float32)

        # Relocation thresholds by type, adjusted for risk
        thresh_base = np.array([
            _RELOC_THRESH[_TYPES[ti]] for ti in type_idx
        ], dtype=np.float32)
        # High risk tolerance → lower threshold (relocates sooner)
        thresh_adjusted = thresh_base - 10.0 * (risk_tolerance - 0.5)

        # Stubbornness by type
        _STUB = [0.70, 0.55, 0.40, 0.60, 0.50, 0.65, 0.85]  # indexed by _TYPES; frontier_lab=0.85 (very resistant to belief change)
        stubbornness = np.array([_STUB[ti] for ti in type_idx], dtype=np.float32)

        # Initial beliefs: companies believe policy harmful, civil society less so
        _BELIEF_MEAN = [0.78, 0.68, 0.78, 0.50, 0.65, 0.23, 0.82]  # frontier_lab: high initial alarm
        _BELIEF_STD  = [0.10, 0.10, 0.10, 0.12, 0.10, 0.10, 0.08]
        beliefs = np.clip(
            np.array([
                rng.normal(_BELIEF_MEAN[ti], _BELIEF_STD[ti])
                for ti in type_idx
            ], dtype=np.float32),
            0.0, 1.0
        )

        return cls(
            n=n,
            beliefs=beliefs,
            sizes=sizes,
            risk_tolerance=risk_tolerance,
            lambdas=lambda_arr,
            thresholds=thresh_adjusted,
            stubbornness=stubbornness,
            type_idx=type_idx,
            is_compliant=np.zeros(n, dtype=bool),
            has_relocated=np.zeros(n, dtype=bool),
            is_evading=np.zeros(n, dtype=bool),
            has_ever_lobbied=np.zeros(n, dtype=bool),
            n_enforcement_contacts=np.zeros(n, dtype=np.int16),
            rounds_since=np.zeros(n, dtype=np.int32),
            memory=np.zeros((n, MEMORY_LEN, 4), dtype=np.float32),
            round_actions=np.zeros(n, dtype=np.int8),
        )

    def active_mask(self) -> np.ndarray:
        """Boolean mask of agents still in jurisdiction."""
        return ~self.has_relocated

    def compliance_rate(self) -> float:
        active = self.active_mask()
        if active.sum() == 0:
            return 0.0
        return float(self.is_compliant[active].mean())

    def relocation_rate(self) -> float:
        return float(self.has_relocated.mean())

    def evasion_rate(self) -> float:
        active = self.active_mask() & ~self.is_compliant
        if active.sum() == 0:
            return 0.0
        return float(self.is_evading[active].mean())

    def mean_belief(self) -> float:
        return float(self.beliefs[self.active_mask()].mean()) if self.active_mask().any() else 0.5

    def type_compliance_rates(self) -> dict[str, float]:
        rates = {}
        for i, t in enumerate(_TYPES):
            mask = (self.type_idx == i) & self.active_mask()
            if mask.any():
                rates[t] = float(self.is_compliant[mask].mean())
        return rates

    def to_summary(self) -> dict:
        active = self.active_mask()
        n_active = int(active.sum())
        summary = {
            "n_total": self.n,
            "n_active": n_active,
            "compliance_rate": self.compliance_rate(),
            "relocation_rate": self.relocation_rate(),
            "evasion_rate": self.evasion_rate(),
            "mean_belief_harmful": self.mean_belief(),
        }
        # Type-stratified compliance rates (for SMM moments m4, m5)
        for i, t in enumerate(_TYPES):
            mask = (self.type_idx == i) & active
            if mask.sum() > 0:
                summary[f"{t}_compliance_rate"] = float(self.is_compliant[mask].mean())
        # Shorthand aliases for SMM moments
        summary["sme_compliance_rate"] = summary.get("startup_compliance_rate", 0.0)
        summary["large_compliance_rate"] = summary.get("large_company_compliance_rate", 0.0)
        # SMM m1: fraction of agents who ever lobbied
        summary["ever_lobbied_rate"] = float(self.has_ever_lobbied.mean())
        # SMM m6: fraction of entities that received any enforcement contact
        summary["enforcement_contact_rate"] = float(
            (self.n_enforcement_contacts > 0).mean()
        )
        # SMM m2: SME vs large compliance divergence
        # Apply burden_sensitivity: SMEs have higher cost/revenue ratio → slower compliance
        # This is the structural fix for SME compliance being too high
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# INFLUENCE MATRIX — sparse DeGroot social learning
# ─────────────────────────────────────────────────────────────────────────────

def build_influence_matrix(
    pop: PopulationArray,
    m: int = 3,
    seed: int = 42,
    return_raw: bool = False,
) -> "csr_matrix | tuple[csr_matrix, csr_matrix]":
    """Build the sparse row-stochastic influence matrix W for DeGroot belief updating.

    Returns W_norm (row-stochastic CSR), or (W_norm, W_raw) when return_raw=True.

    WHY SPARSE CSR: The belief update W @ beliefs is O(nnz) — proportional to the
    number of edges, not n². For a BA graph with m=3, nnz ≈ 6n (each of n nodes
    has ~6 edges on average after bidirectionalization). At n=10k that is 60k
    multiply-adds vs 100M for a dense matrix. Dense would take ~100s per round;
    sparse takes ~1s.

    WHY BARABÁSI–ALBERT TOPOLOGY: Real-world influence networks are scale-free —
    a small number of nodes (trade associations, regulators, media) dominate
    information flow. Preferential attachment reproduces this naturally. Erdős–Rényi
    (random) or lattice graphs produce unrealistic uniform-degree distributions.

    WHY m=3: Each new node attaches to 3 existing nodes, giving a mean degree of ~6
    after bidirectionalization. This matches empirical estimates of meaningful
    professional contacts in regulatory compliance contexts (3–7 per firm).
    m=1 produces a tree (no clustering); m=6+ becomes too dense and homogenizes
    beliefs unrealistically fast.

    INTRA-TYPE CLUSTERING: After the BA pass, each agent gains 3 within-type edges
    (trade associations: firms in the same category cluster together). The loop runs
    O(n) iterations with O(1) work each — total O(n), not O(n²). This ensures
    type-level opinion dynamics (e.g., civil society converging separately from
    startups) while keeping the graph sparse.

    PURE NUMPY vs networkx: networkx BA at n=10,000 takes ~26s (Python-level graph
    construction). This pure-numpy loop takes ~2s by working directly on index arrays.

    return_raw=True: caller receives (W_norm, W_raw) where W_raw is the unnormalized
    adjacency matrix. Use W_raw for hub detection — see inline comment below.
    """
    n = pop.n
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    degrees = np.ones(n, dtype=np.float64)

    # 1. Preferential attachment
    for i in range(min(m, n), n):
        probs = degrees[:i] / degrees[:i].sum()
        targets = rng.choice(i, size=min(m, i), replace=False, p=probs)
        for t in targets:
            rows.extend([i, int(t)])
            cols.extend([int(t), i])
            degrees[i] += 1
            degrees[int(t)] += 1

    # 2. Sparse intra-type clustering (3 per agent, not O(n^2))
    for t_idx in range(len(_TYPES)):
        members = np.where(pop.type_idx == t_idx)[0]
        if len(members) < 4:
            continue
        for idx in members:
            candidates = members[members != idx]
            targets = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
            for j in targets:
                rows.extend([int(idx), int(j)])
                cols.extend([int(j), int(idx)])

    # 3. Build + row-normalise
    data = np.ones(len(rows), dtype=np.float32)
    W = csr_matrix(
        (data, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n)
    ).tocsr()

    # Save W_raw BEFORE normalization. After row-normalization W is row-stochastic,
    # meaning W.sum(axis=1) == 1.0 for every row — the row sums are all identical
    # and carry no degree information. Hub detection requires the raw edge counts
    # (W_raw.sum(axis=1) gives actual degree), not the normalized weights.
    W_raw = W.copy()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    from scipy.sparse import diags as _diags
    W_norm = (_diags(1.0 / row_sums) @ W).astype(np.float32)
    if return_raw:
        return W_norm, W_raw
    return W_norm


# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED ROUND — all agent decisions in one pass
# ─────────────────────────────────────────────────────────────────────────────

def vectorized_round(
    pop: PopulationArray,
    W: csr_matrix,
    burden: float,
    policy_severity: float,
    compliance_cost: float,
    detection_prob: float,
    fine_amount: float,
    round_num: int,
    run_seed: int = 42,
    compute_cost_factor: float = 1.0,
    hk_epsilon: float = 1.0,
) -> dict:
    """Compute all agent decisions for one round using vectorized operations.

    Mutates pop in-place (beliefs, is_compliant, has_relocated, is_evading,
    rounds_since, round_actions, has_ever_lobbied, n_enforcement_contacts).

    Returns a dict of action counts and the new_relocators_mask for this round.

    RUN SEED DESIGN: the round-level RNG is seeded with run_seed * 10000 + round_num.
    Previous bug used round_num * 1000 + 42 regardless of run_seed, making every
    ensemble member draw identical random numbers — genuine ensemble diversity
    requires that each run_seed produces a distinct random sequence.
    """
    n = pop.n
    active = pop.active_mask()

    # ── 1. Hegselmann-Krause bounded-confidence belief update ────────────────
    # Pure DeGroot causes consensus on BA networks within ~20 rounds because
    # hub nodes pull all their neighbours toward their own belief regardless
    # of how far apart those beliefs are. In real policy debates, organisations
    # mostly update from sources they already roughly agree with.
    #
    # Fix: Hegselmann & Krause (2002) bounded confidence. Agents only update
    # from neighbours whose belief is within ε of their own. This allows belief
    # clusters to persist, which is what we observe in real regulatory debates.
    #
    # When hk_epsilon=1.0 (the default), all pairs satisfy the condition and
    # this reduces exactly to pure DeGroot — calibration is unchanged.
    # Smaller ε (e.g. 0.3) creates polarised clusters. Sweep [0.3, 0.5, 1.0].
    # [ASSUMED] ε=0.5 is the canonical HK midpoint; no empirical calibration.
    #
    # Implementation: for each edge (i,j) in W, zero the weight if
    # |beliefs[i] - beliefs[j]| > ε, then re-normalise rows.
    # O(edges) — same asymptotic cost as pure DeGroot.
    from scipy.sparse import diags as _diags_hk, csr_matrix as _csr_hk
    if hk_epsilon < 1.0:
        # belief[sender] for every nonzero entry in W (CSR format: W.indices = column indices)
        sender_beliefs = pop.beliefs[W.indices]
        # belief[receiver] for every nonzero entry (repeat row belief for each of its nnz entries)
        receiver_beliefs = np.repeat(pop.beliefs, np.diff(W.indptr))
        within_eps = (np.abs(sender_beliefs - receiver_beliefs) <= hk_epsilon).astype(np.float32)
        W_hk_data = W.data * within_eps
        W_hk = _csr_hk((W_hk_data, W.indices, W.indptr), shape=W.shape)
        row_sums = np.asarray(W_hk.sum(axis=1)).ravel()
        # Isolated agents (no within-ε neighbours) stay put — divide by 1.0
        row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
        W_eff = _diags_hk(1.0 / row_sums_safe) @ W_hk
    else:
        W_eff = W  # hk_epsilon=1.0: pure DeGroot, no masking needed

    social_influence = np.asarray(W_eff @ pop.beliefs).ravel()  # (n,)
    pop.beliefs = np.clip(
        pop.stubbornness * pop.beliefs + (1.0 - pop.stubbornness) * social_influence,
        0.0, 1.0
    ).astype(np.float32)

    # ── 2. Memory effect on beliefs ────────────────────────────────────────
    # If agents remember high relocation or enforcement, beliefs rise.
    # Only applied when memory signal is non-trivial to avoid anchoring to 0.
    if round_num > 2:
        recent_reloc = pop.memory[:, :, 0].mean(axis=1)
        recent_enforce = pop.memory[:, :, 1].mean(axis=1)
        memory_signal = (0.4 * recent_reloc + 0.3 * recent_enforce).astype(np.float32)
        # Additive upward push only (memory of danger raises alarm; memory of calm
        # does NOT suppress existing belief — agents don't forget previous fear)
        belief_boost = np.maximum(0.0, memory_signal - 0.02)  # threshold: >2% triggers
        pop.beliefs = np.clip(pop.beliefs + 0.10 * belief_boost, 0.0, 1.0).astype(np.float32)

    # ── 3. Compliance probability (Weibull hazard rate) ────────────────────
    # Type-specific burden sensitivity: SMEs face higher compliance cost/revenue.
    # NOTE: The hazard-rate fix already makes the Weibull calibration accurate.
    # The resource-constraint CAP (below) handles SME differentiation.
    # Burden multiplier kept at 1.0 for startups to preserve GDPR calibration:
    #   lambda_sme=10.9 gives 52% at R8 at baseline burden — adding a multiplier
    #   would push this below the DLA Piper calibration target.
    # Large companies get a mild relief (0.9×) — they can absorb overhead more easily.
    # [DIRECTIONAL] direction grounded; magnitude [ASSUMED] sweep [0.8, 0.9, 1.0]
    # Frontier labs have large compliance teams so per-unit burden is lower.
    # Startups face the same burden as baseline (no dedicated staff).
    SME_BURDEN_MULTIPLIER = np.where(
        pop.type_idx == 0, 0.90,          # large_company: slightly less sensitive
        np.where(pop.type_idx == 6, 0.70, # frontier_lab: dedicated policy teams
        1.0)                               # everyone else: baseline
    ).astype(np.float32)
    burden_penalty = 1.0 + np.maximum(0.0, (burden - 30.0) / 100.0) * SME_BURDEN_MULTIPLIER
    severity_accel = (policy_severity / 3.0) ** 0.5
    # compute_cost_factor: multiplies λ, slowing compliance for AI-specific policies
    # where compliance requires novel technical capabilities (red-teaming, compute audits)
    # rather than administrative paperwork. Default=1.0 preserves GDPR calibration exactly.
    # [ASSUMED] Derive from parse_bill() compute_threshold_flops via the formula in
    # hybrid_loop.py. Sweep [1.0 (GDPR-equivalent), 2.0, 4.0] for AI-specific policies.
    effective_lambda = pop.lambdas * burden_penalty * compute_cost_factor / severity_accel  # (n,)

    # HAZARD RATE, not Weibull CDF — this distinction is the core compliance fix.
    #
    # BUG (previous version): p_comply = 1 - exp(-t/λ) evaluated fresh each round.
    #   Drawing rand < CDF(t) makes non-compliant agents progressively more likely
    #   to comply as t grows, even without any change in circumstances. By round 8
    #   CDF(8, λ=10.9) = 0.52, so an agent that did not comply in rounds 1–7
    #   essentially gets a "second chance" at 52% probability in round 8. This
    #   double-counting cascades: SME compliance reached 93% instead of the 52%
    #   target from DLA Piper 2020.
    #
    # FIX: use the per-round hazard rate h = 1/λ (the memoryless exponential
    #   approximation of the Weibull for small h). An agent complies in round t
    #   iff rand < h AND not yet compliant. The cumulative compliance at t rounds
    #   is 1-(1-h)^t ≈ 1-exp(-t/λ), recovering the Weibull CDF exactly — but
    #   each round's draw is truly independent and does not include prior rounds.
    #   λ_large=3.32 (← DLA Piper 2020 GDPR) → 91% at 8 rounds. ✓
    #   λ_sme=10.9   (← DLA Piper 2020 GDPR) → 52% at 8 rounds. ✓
    h = (1.0 / effective_lambda).astype(np.float32)  # (n,) per-round hazard

    # Resource-constraint compliance cap: limits MAXIMUM compliance probability per round.
    # Startups have fewer resources → lower hazard rate regardless of time.
    # [DIRECTIONAL] IAPP 2019: 40% of SMEs lacked compliance staff at 24 months.
    # [ASSUMED] cap: startup 0.60 max cumulative → h_cap_startup ≈ 0.12/round
    # Cap on cumulative = 0.60 → per-round cap = 1-(1-0.60)^(1/8) = 0.114
    # Frontier labs are uncapped — they will fully comply eventually (they have
    # the resources and their public commitments depend on it). Startups remain
    # resource-capped at 60% (IAPP 2019: 40% had no dedicated compliance staff).
    CUMULATIVE_CAP = np.where(
        pop.type_idx == 2, 0.60,   # startup: max 60% (resource-limited)
        np.where(pop.type_idx == 1, 0.80,   # mid_company: max 80%
        np.where(pop.type_idx == 6, 1.0,    # frontier_lab: uncapped
        1.0))                               # large_company, others: uncapped
    ).astype(np.float32)
    # Convert cumulative cap to per-round hazard cap
    import math as _math
    h_cap = np.where(
        CUMULATIVE_CAP < 1.0,
        (1.0 - (1.0 - CUMULATIVE_CAP) ** (1.0 / 8.0)).astype(np.float32),
        1.0
    ).astype(np.float32)
    p_comply = np.minimum(h, h_cap)  # (n,) per-round probability for non-compliant agents

    # ── 4. Relocation probability ──────────────────────────────────────────
    # Sigmoid in burden space, scaled by severity-cubed so relocation accelerates
    # sharply at high severity. severity=5 gives (5/3)^3 ≈ 4.6× the base rate.
    #
    # [DIRECTIONAL] direction grounded: relocation is a threshold behaviour, not
    # a linear response — a small policy increase can tip the balance.
    # The cubic exponent specifically is [ASSUMED]. This is the highest-leverage
    # assumed parameter for severe-policy outputs. Run sensitivity with quadratic
    # (exponent=2) and linear (exponent=1) before citing high-severity results.
    #
    # NO BELIEFS GATE: beliefs modulate lobbying (subjective threat perception)
    # but NOT relocation. A company facing dissolution-level penalties relocates
    # on mechanical cost-benefit grounds regardless of its subjective belief about
    # whether the policy is harmful. The previous version multiplied p_relocate by
    # pop.beliefs, which halved the calibrated rate and broke moment m3.
    #
    # Criminal-penalty bonus: at severity > 4 (criminal liability threshold),
    # relocation probability gets a flat bonus. [DIRECTIONAL] grounded in observed
    # behaviour following GDPR criminal provisions in Austria/Ireland.
    severity_scale = (policy_severity / 3.0) ** 3
    max_rate = _RELOC_MAX_RATE * severity_scale
    _arg = np.clip(-_RELOC_ALPHA * (burden - pop.thresholds), -88, 88)
    sigmoid = 1.0 / (1.0 + np.exp(_arg))
    base_prob = sigmoid * max_rate
    criminal_bonus = max(0.0, (policy_severity - 4.0) * 0.05)
    # Relocation probability is mechanically driven by burden × severity sigmoid.
    # beliefs modulates LOBBYING (subjective threat perception) but NOT relocation:
    # a company facing dissolution relocates regardless of its subjective belief.
    # Removing the * pop.beliefs multiplier that was halving the calibrated rate.
    p_relocate = np.where(
        pop.has_relocated, 0.0,
        base_prob + criminal_bonus,
    )  # (n,)

    # Frontier labs are sensitive to perceived regulatory hostility beyond pure
    # cost-benefit — board-level decisions factor in "is this jurisdiction friendly
    # to our mission?" For frontier labs only, a high belief_harmful adds a small
    # acceleration to their relocation probability.
    # [DIRECTIONAL] grounded in qualitative evidence (lab statements about regulatory
    # environment). Magnitude [ASSUMED]: 0.30 × beliefs means at belief=0.9, frontier
    # labs see +27% of their base relocation probability. Sweep [0.1, 0.3, 0.5].
    frontier_mask = (pop.type_idx == 6)  # index 6 = frontier_lab
    if frontier_mask.any():
        belief_accel = 1.0 + 0.30 * pop.beliefs
        p_relocate = np.where(
            frontier_mask,
            np.clip(p_relocate * belief_accel, 0.0, 1.0),
            p_relocate,
        ).astype(np.float32)

    # ── 5. Evasion probability ─────────────────────────────────────────────
    # CALIBRATION TARGET: DLA Piper 2020 GDPR — ~20% active evasion by SMEs
    # at 24 months, when enforcement rate ≈ 6%/year.
    #
    # RATIONAL CHOICE FINDING: compliance_cost >> expected_fine for most firms
    # (GDPR compliance ~EUR 20-30k/yr; expected fine at 6%/year detection =
    # EUR 1,200/yr for a EUR 20k fine). This means RATIONAL agents should always
    # evade. But real-world compliance is ~52% at 24 months — so evasion is NOT
    # primarily driven by rational cost-benefit: it is driven by enforcement risk
    # above a threshold, combined with risk-aversion.
    #
    # CALIBRATED MODEL: evasion activates only when enforcement probability
    # crosses a threshold (0.05/quarter = GDPR-calibrated). Above the threshold,
    # evasion probability scales with (detection - threshold) × risk_preference.
    # Risk-averse firms (majority) comply; risk-tolerant firms evade.
    #
    # Threshold 0.020: below this enforcement rate, even risk-tolerant firms do not
    # find evasion worthwhile — the expected savings from evasion do not outweigh
    # even the small probability of a fine. [DIRECTIONAL] grounded: at GDPR
    # enforcement rate (6%/yr → 0.015/round), self-reported evasion was ~8-12% of
    # regulated entities (IAPP survey 2019). At 0.015/round < 0.020, the model
    # produces near-zero evasion — consistent with the IAPP finding that most
    # apparent non-compliance is accidental rather than deliberate.
    EVASION_THRESHOLD = 0.020  # [DIRECTIONAL] below this enforcement rate, evasion is irrational
    # Calibration: GDPR enforcement rate = 0.015/round. At sev=1: enf_prob=0.015 < 0.020 → 0 evasion
    # At sev=2: enf_prob=0.030 → p_evade_max=(0.030-0.020)×20=0.20 → meaningful evasion ✓
    # At GDPR-like sev=3: enf_prob=0.045 → p_evade_max=(0.045-0.020)×20=0.50 ✓
    EVASION_SCALING = 20.0     # [ASSUMED] sweep [10, 20, 40]
    p_evade_max = float(np.clip((detection_prob - EVASION_THRESHOLD) * EVASION_SCALING, 0.0, 1.0))

    if p_evade_max > 0:
        # Risk-seeking agents (low risk_tolerance) evade more
        # p_evade = p_evade_max × (1 - risk_tolerance) / mean(1-risk_tolerance)
        # Normalized so mean agent evades at p_evade_max × (1-0.29) = p_evade_max × 0.71
        # To get target 20% at GDPR: p_evade_max=0.20, mean p_evade = 0.20 × 0.71 = 0.14
        # Fraction who exceed rand: ~14% ← close to target ✓
        risk_pref = np.clip(1.0 - pop.risk_tolerance, 0.0, 1.0)
        p_evade = np.where(
            pop.has_relocated,
            0.0,
            p_evade_max * risk_pref
        )
        # Compliant firms evade at 20% of non-compliant rate (partial evasion on margins)
        p_evade = np.where(pop.is_compliant, p_evade * 0.20, p_evade)
    else:
        p_evade = np.zeros(n, dtype=np.float32)

    # ── 6. Action selection ────────────────────────────────────────────────
    # Single shared random draw for comply / relocate / evade. Priority order:
    #   relocate > comply > evade > nothing
    # Rationale: relocation is the highest-stakes irreversible action (it supersedes
    # any compliance decision made in the same round). Compliance outranks evasion
    # because once an agent draws rand < p_comply it has committed resources to
    # compliance — evasion is only chosen when compliance is not triggered. Lobbying
    # is a background action and uses a completely separate RNG (see below).
    rng = np.random.default_rng(run_seed * 10000 + round_num)
    rand = rng.random(n).astype(np.float32)

    # Priority: relocate > comply > evade > nothing
    # Reset actions
    pop.round_actions[:] = 0  # nothing

    # ── 7. Lobbying (independent RNG) ─────────────────────────────────────
    # Lobbying uses a separate random draw seeded at run_seed*10000+round_num+50000,
    # completely independent of the comply/relocate draw above.
    #
    # Why independent: EU Transparency Register 2022 shows ~85% of regulated firms
    # lobby including those that are already compliant. If lobbying shared the
    # comply/relocate draw, any agent that drew rand < p_comply would have its lobby
    # action overridden by comply — but in reality compliant firms lobby more, not
    # less. The 50000 offset ensures the lobby draw is uncorrelated with the primary
    # action draw even when run_seed and round_num are small.
    #
    # beliefs modulate lobbying (not relocation): lobbying is driven by subjective
    # threat perception — a firm lobbies harder when it believes the policy is harmful.
    # p_lobby = 0.50 * beliefs (non-compliant) or 0.25 * beliefs (compliant, lower urgency)
    lobby_rng = np.random.default_rng(run_seed * 10000 + round_num + 50000)
    lobby_rand = lobby_rng.random(n).astype(np.float32)
    p_lobby = np.where(
        pop.has_relocated, 0.0,
        np.where(pop.is_compliant, 0.25 * pop.beliefs, 0.50 * pop.beliefs)
    )
    # Lobbying is tracked separately — agents can lobby AND comply in same round
    lobbying_this_round = active & (lobby_rand < p_lobby)
    pop.has_ever_lobbied |= lobbying_this_round
    # Mark lobby in round_actions only if NOT also relocating/complying (for display)
    pop.round_actions = np.where(
        active & lobbying_this_round & (pop.round_actions == 0),
        4, pop.round_actions
    )

    # Evade
    pop.round_actions = np.where(
        active & ~pop.is_compliant & (rand < p_evade), 3, pop.round_actions
    )

    # Comply (new compliers this round)
    new_compliers = active & ~pop.is_compliant & (rand < p_comply)
    pop.is_compliant = pop.is_compliant | new_compliers
    pop.round_actions = np.where(new_compliers, 1, pop.round_actions)

    # Relocate (highest priority — overrides comply/evade in round_actions).
    # Compliance and relocation are independent decisions. A GDPR-compliant EU
    # company still moves to Singapore if the burden exceeds its threshold.
    # The previous version gated relocation on ~is_compliant, which incorrectly
    # prevented already-compliant firms from relocating and broke moment m3
    # (cumulative relocation rate). The gate has been removed.
    new_relocators = active & (rand < p_relocate)
    pop.round_actions = np.where(new_relocators, 2, pop.round_actions)

    # ── 8. State updates ───────────────────────────────────────────────────
    pop.rounds_since = np.where(active, pop.rounds_since + 1, pop.rounds_since)

    # Track ever-lobbied for SMM m1 (fraction who lobbied at least once)
    pop.has_ever_lobbied |= (pop.round_actions == 4)

    # Per-agent enforcement contacts (SMM m6)
    # Non-compliant agents are independently investigated with p = detection_prob/round.
    # Calibrated to DLA Piper 2020: 6% of GDPR entities received enforcement contact in year 1.
    # p = 0.015/round → expected fraction ever-contacted in 8 rounds ≈ 11% of non-compliant
    # Among 50% non-compliant agents: 11% × 50% = 5.5% of total ≈ 0.06 target ✓
    if detection_prob > 0:
        enf_rng = np.random.default_rng(run_seed * 10000 + round_num + 20000)
        enf_rand = enf_rng.random(n).astype(np.float32)
        newly_investigated = active & ~pop.is_compliant & (enf_rand < detection_prob)
        pop.n_enforcement_contacts = np.clip(
            pop.n_enforcement_contacts.astype(np.int32) + newly_investigated.astype(np.int32),
            0, 127
        ).astype(np.int16)

    # Update persistent evasion state (fixes evasion_rate always showing 0%)
    # is_evading tracks CURRENT round evasion (not cumulative) so reset first
    pop.is_evading = (pop.round_actions == 3)

    # Action counts
    n_relocating = int(new_relocators.sum())
    n_complying  = int(new_compliers.sum())
    n_evading    = int(pop.is_evading.sum())
    n_lobbying   = int((pop.round_actions == 4).sum())
    n_nothing    = int((pop.round_actions == 0).sum())

    return {
        "n_relocating": n_relocating,
        "n_complying":  n_complying,
        "n_evading":    n_evading,
        "n_lobbying":   n_lobbying,
        "n_nothing":    n_nothing,
        "new_relocators_mask": new_relocators,
    }


def update_memory(
    pop: PopulationArray,
    observed_reloc_frac: float,
    observed_enforcement_frac: float,
    burden_norm: float,
    round_num: int,
    policy_softened: bool = False,
) -> None:
    """Shift the rolling memory window and record this round's observations.

    The memory array is (n, 8, 4): 8 slots of 4 features each. Slot 0 is the
    most recent round; slot 7 is the oldest. Each call shifts slots 0–6 into
    slots 1–7 (discarding the oldest) and writes the current round into slot 0.

    Features stored per slot:
      [0] observed_reloc_frac       — fraction of agents that relocated this round
      [1] observed_enforcement_frac — fraction of active agents contacted by enforcers
      [2] burden_norm / 100.0       — normalized compliance burden [0,1]
      [3] own_action / 4.0          — agent's own action code normalized to [0,1]

    Slot 0 of features 0–1 feeds back into beliefs in vectorized_round() as an
    upward push when relocation or enforcement exceeds 2% (memory effect section).

    policy_softened: set True when a PolicyAmendmentEffect fires with negative
    burden_delta in this round. Government willingness to accommodate industry
    reduces agents' belief that the policy is harmful — this closes the
    event → memory → belief feedback loop. The -0.05 magnitude is [ASSUMED];
    direction is [DIRECTIONAL] (grounded in regulatory responsiveness literature).
    """
    # Shift memory window: move slots 0..6 → slots 1..7, evicting the oldest slot
    pop.memory[:, 1:, :] = pop.memory[:, :-1, :]

    own_action_norm = (pop.round_actions / 4.0).astype(np.float32)
    pop.memory[:, 0, 0] = observed_reloc_frac
    pop.memory[:, 0, 1] = observed_enforcement_frac
    pop.memory[:, 0, 2] = burden_norm / 100.0
    pop.memory[:, 0, 3] = own_action_norm

    # Policy amendment signal: observed flexibility reduces alarm
    # This closes the event→memory→belief feedback loop
    if policy_softened:
        # All active agents observe the amendment and update beliefs downward
        pop.beliefs = np.clip(
            pop.beliefs - 0.05,  # [ASSUMED] sweep [0.02, 0.05, 0.10]
            0.0, 1.0
        ).astype(np.float32)


def compute_type_compliance_rates(pop: PopulationArray) -> dict[str, float]:
    """Return compliance rate for each agent type among agents still in jurisdiction."""
    rates = {}
    for i, t in enumerate(_TYPES):
        mask = (pop.type_idx == i) & pop.active_mask()
        if mask.sum() > 0:
            rates[t] = float(pop.is_compliant[mask].mean())
        else:
            rates[t] = 0.0
    return rates
