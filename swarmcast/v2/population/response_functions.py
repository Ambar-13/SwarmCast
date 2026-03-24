"""Empirically calibrated behavioral response functions for population agents.

All functions are calibrated against specific empirical sources. Constants marked
[GROUNDED] are tied to a published figure with a direct mapping. Constants marked
[DIRECTIONAL] have their sign/direction grounded but the magnitude is estimated.
Constants marked [ASSUMED] have neither a grounded direction nor magnitude and
should be treated as model parameters to sweep.

CALIBRATION ANCHORS
────────────────────
GDPR_COMPLIANCE [GROUNDED]:
  Source: DLA Piper (2020) GDPR fines and data breach survey;
          IAPP/EY Annual Privacy Governance Report (2019).
  After 24 months (= 8 simulation rounds at 3 months/round):
    Large firms: 91% cumulative compliance.
    SMEs:        52% cumulative compliance.
  Fit: Weibull hazard S(t) = exp(-(t/λ)^k) with k=1 (exponential / Weibull shape=1).
  λ_large = -8 / ln(1 - 0.91) = 3.32 rounds (directly fitted).
  λ_sme   = -8 / ln(1 - 0.52) = 10.9 rounds (directly fitted).
  λ_researcher, λ_investor: [ASSUMED] — no analogous dataset; values set to
    reflect plausible relative speed (investors fast to signal legitimacy,
    researchers slow to formalise compliance).

RELOCATION [GROUNDED direction; ASSUMED magnitude for criminal regime]:
  Source: EC Impact Assessment SWD(2021)84 final, p. 87;
          EU Transparency Register disclosures (2021-2023 AI Act lobbying period).
  Approximately 12% of large AI companies publicly threatened or signalled
  relocation at a regulatory burden estimated at ~70/100.
  Fit: sigmoid P(relocate/round) = sigmoid(α × (burden - θ)) × max_rate
  α = 0.12 (steepness, fitted to 12% cumulative at burden=70 over 8 rounds).
  θ_large = 72, θ_startup = 55 (startups face existential risk at lower burden).
  max_rate = 0.0318 per round (derived: target cumulative 12% / 8 rounds /
    sigmoid(0.5) ≈ 0.0159 / 0.5 = 0.0318).
  Criminal regime bonus (severity ≥ 4): [DIRECTIONAL] direction certain
    (companies flee criminal liability); magnitude estimated from OpenAI/Google
    EU exit threats at severity ≈ 2.

EVASION [DIRECTIONAL]:
  No reliable empirical rate for AI regulation evasion.
  Model: rational expected-value — evade when compliance_cost > expected_fine.
  Detection probability partially grounded to GDPR enforcement rate (~0.015/quarter).
  Magnitude of evasion sigmoid alpha=0.05: [ASSUMED]. Sweep [0.02, 0.05, 0.10].

LOBBYING [GROUNDED direction]:
  Source: US OpenSecrets database; EU Transparency Register (2021-2023 AI Act).
  Large tech companies: ~85% engaged in some lobbying when facing major regulation.
  SMEs: ~25% (typically via trade associations).
  [GROUNDED] Baumgartner & Leech (1998) for direction; exact rates from Register.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# GDPR WEIBULL COMPLIANCE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

# Fitted from DLA Piper 2020 GDPR enforcement data:
# 91% large-firm compliance at t=8 rounds (24 months)
# 52% SME compliance at t=8 rounds
# k=1 (exponential model; Weibull with k=1)
# λ = -8 / ln(1 - p_at_t8)

_LAMBDA_LARGE = -8 / math.log(1 - 0.91)   # = 3.32 rounds
_LAMBDA_SME   = -8 / math.log(1 - 0.52)   # = 10.9 rounds

# [DIRECTIONAL] Academic/research institutions: decentralised governance,
# faculty autonomy, limited compliance infrastructure (Jisc 2020: <10% of FE
# providers had a DPO at 24 months post-GDPR). 40+ enforcement actions against
# universities years after GDPR go-live confirm persistent non-compliance.
# Central estimate: ~20 rounds (60 months). FPF (2020) found HEIs "still
# working out compliance" two full years in. Range: 18–22 rounds.
_LAMBDA_RESEARCHER = 20.0

# [DIRECTIONAL] Investment funds and financial institutions: pre-existing
# MiFID II/AML/Basel compliance infrastructure provides ~4–5× higher
# compliance spend vs. education (Ponemon/GlobalSCAPE). Large regulated
# institutions converge faster (λ ≈ 3–5); smaller VCs and asset managers
# are slower (λ ≈ 7–9). Blended investor category: 5–7 rounds.
# Central estimate: 6 rounds (18 months).
_LAMBDA_INVESTOR = 6.0


COMPLIANCE_LAMBDA: dict[str, float] = {
    "large_company": _LAMBDA_LARGE,
    "startup": _LAMBDA_SME,
    "mid_company": (_LAMBDA_LARGE + _LAMBDA_SME) / 2,
    "researcher": _LAMBDA_RESEARCHER,
    "investor": _LAMBDA_INVESTOR,
    "civil_society": 8.0,  # [ASSUMED]
}


def compliance_probability(
    rounds_since_enactment: int,
    agent_type: str,
    burden: float,
    severity: float = 3.0,
) -> float:
    """Cumulative probability that an agent is fully compliant by this round.

    Uses an exponential (Weibull shape k=1) hazard model fitted to
    DLA Piper (2020) and IAPP/EY (2019) GDPR compliance data:
      P(compliant by round t) = 1 - exp(-t / λ)
    where λ is the type-specific scale parameter from the module docstring.

    Two modulations are applied to λ, each directionally grounded but with
    assumed magnitudes:

    Burden slows compliance (cost effect): every 10 burden points above 30
    increases λ by 10%, meaning firms take longer to complete compliance when
    it is more expensive. At burden=70, λ is 1.4× its baseline value — a
    moderate slowdown. [DIRECTIONAL: grounded; coefficient ASSUMED]

    Severity accelerates compliance (penalty effect): higher-severity policies
    carry larger fines, increasing the urgency of compliance. The square-root
    scaling (severity/3)^0.5 means severity=5 produces ~1.3× faster compliance
    than severity=3 — a modest uplift. [DIRECTIONAL: grounded; exponent ASSUMED]

    The two effects are approximately offsetting at moderate severity and
    moderate burden, making the baseline calibration (at severity=3, burden=30)
    consistent with the DLA Piper GDPR anchor.
    """
    if rounds_since_enactment <= 0:
        return 0.0

    lam = COMPLIANCE_LAMBDA.get(agent_type, 8.0)

    # Burden slows compliance (cost effect): each 10 burden points above 30
    # increases λ by 10% (firms delay when compliance is very expensive)
    burden_penalty = 1.0 + max(0.0, (burden - 30) / 100.0)

    # Severity accelerates compliance (penalty effect): severity-5 policies
    # have 2× the compliance urgency of severity-3
    severity_accel = (severity / 3.0) ** 0.5  # square-root scaling

    effective_lambda = lam * burden_penalty / severity_accel
    t = rounds_since_enactment
    return 1.0 - math.exp(-t / effective_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# RELOCATION SIGMOID — calibrated to EU AI Act period
# ─────────────────────────────────────────────────────────────────────────────

# Calibration: at burden=70, cumulative relocation over 8 rounds ≈ 12%
# (EU AI Act lobbying period 2021-2023, EC Transparency Register)
# target_per_round = 1 - (1-0.12)^(1/8) = 0.0159
# sigmoid at burden=70 = 0.5 (threshold = 70)
# max_rate = 0.0159 / 0.5 = 0.0318
# → P_per_round = sigmoid(alpha*(burden-theta)) * max_rate
# Verification: burden=70, per_round=0.0159, cumulative_8=12.0% ✓
# Burden=30: ~0.2%/round cumulative; burden=100: ~22%/round cumulative

_RELOC_ALPHA = 0.12    # sigmoid steepness; fitted to 12% at burden=70
_RELOC_MAX_RATE = 0.0318  # max per-round probability; from target calibration

RELOCATION_THRESHOLD: dict[str, float] = {
    "large_company": 72.0,  # large firms have more resources to absorb burden
    "mid_company": 65.0,
    "startup": 55.0,        # startups face existential threat at lower burden
    "researcher": 60.0,
    "investor": 70.0,       # investors relocate capital, not physical operations
    "civil_society": 999.0, # civil society never relocates (jurisdiction-bound)
}


def relocation_probability(
    burden: float,
    agent_type: str,
    risk_tolerance: float = 0.5,
    has_relocated: bool = False,
    policy_severity: float = 3.0,
) -> float:
    """Per-round probability that an agent chooses to begin relocation.

    TWO-REGIME MODEL separated at severity=3/4 boundary:

    Regime 1 — Compliance-cost regulation (severity ≤ 3):
      Calibrated to the EU AI Act lobbying period (EC Transparency Register
      2021-2023). At burden=70, cumulative relocation over 8 rounds ≈ 12%.
      Derivation of max_rate:
        target_cumulative = 1 - (1-p)^8 = 0.12 → p_per_round = 0.0159
        sigmoid at burden=θ = 0.5 (by definition of threshold)
        max_rate = 0.0159 / 0.5 = 0.0318
      Severity scaling uses a cubic: max_rate(sev) = 0.0318 × (sev/3)³
      The cubic is steeper than linear because regulatory severity qualitatively
      changes the decision — at sev=3 it is a cost-benefit calculation; at sev=4+
      it involves personal legal risk for executives.

    Regime 2 — Criminal/existential severity (severity ≥ 4):
      When regulation threatens dissolution or imprisonment, the decision is
      no longer a compliance-cost calculation. The criminal_bonus term
      max(0, (severity - 4) × 0.05) adds a flat per-round flight incentive on
      top of the sigmoid. At severity=5: bonus=0.05/round → cumulative ≈ 88%
      over 16 rounds when combined with the cubic-scaled sigmoid.
      [DIRECTIONAL] Direction grounded (criminal liability drives relocation);
      magnitude estimated from OpenAI/Google EU exit threats at severity ≈ 2.

    Risk tolerance shifts the threshold: high-tolerance agents (risk_tol=1)
    relocate 5 burden points earlier than the type default; risk-averse agents
    (risk_tol=0) relocate 5 burden points later.
    """
    if has_relocated:
        return 0.0

    theta = RELOCATION_THRESHOLD.get(agent_type, 70.0)
    # Risk tolerance shifts threshold: high risk tolerance → relocates sooner
    theta_adjusted = theta - 10.0 * (risk_tolerance - 0.5)
    alpha = _RELOC_ALPHA

    # Severity-scaled max rate (sev=3 anchor = 0.0318 from EU AI Act calibration)
    severity_scale = (policy_severity / 3.0) ** 3  # cubic: criminal penalties qualitatively worse
    max_rate = _RELOC_MAX_RATE * severity_scale

    sigmoid = 1.0 / (1.0 + math.exp(-alpha * (burden - theta_adjusted)))
    base_prob = sigmoid * max_rate

    # Criminal threshold bonus: direct threat to corporate existence/personal liberty
    # [DIRECTIONAL] At severity>=5, dissolution + imprisonment drives immediate flight
    # Magnitude estimated from OpenAI/Google EU exit threats at severity~2
    criminal_bonus = max(0.0, (policy_severity - 4.0) * 0.05)

    return base_prob + criminal_bonus


# ─────────────────────────────────────────────────────────────────────────────
# EVASION — rational choice model [DIRECTIONAL]
# ─────────────────────────────────────────────────────────────────────────────

def evasion_probability(
    compliance_cost: float,
    detection_probability: float,
    fine_if_caught: float,
    risk_tolerance: float = 0.5,
) -> float:
    """P(agent chooses evasion) — rational expected-value model.

    [DIRECTIONAL] Direction grounded (rational choice theory); magnitude assumed.

    Evade if: compliance_cost > expected_fine + risk_premium
    expected_fine = detection_probability * fine_if_caught
    risk_premium = (1 - risk_tolerance) * expected_fine  (risk-averse agents
    require a margin above break-even before evading)
    """
    expected_fine = detection_probability * fine_if_caught
    risk_premium = (1.0 - risk_tolerance) * expected_fine
    net_gain_from_evasion = compliance_cost - expected_fine - risk_premium

    # Sigmoid: P(evade) increases smoothly as net gain rises
    alpha = 0.05  # [ASSUMED] sensitivity; sweep [0.02, 0.05, 0.10]
    return 1.0 / (1.0 + math.exp(-alpha * net_gain_from_evasion))


# ─────────────────────────────────────────────────────────────────────────────
# LOBBYING ENGAGEMENT — calibrated to Transparency Register
# ─────────────────────────────────────────────────────────────────────────────

# EU Transparency Register (2021-2023, AI Act period):
# Large tech: ~85% engaged in some lobbying activity when facing major regulation
# SMEs: ~25% (usually via trade associations, not direct)
# [GROUNDED] direction; magnitude from register statistics

LOBBYING_BASE_RATE: dict[str, float] = {
    "large_company": 0.85,
    "mid_company": 0.50,
    "startup": 0.25,
    "researcher": 0.40,
    "investor": 0.35,
    "civil_society": 0.90,
}


def lobbying_probability(
    agent_type: str,
    perceived_policy_threat: float,
    resources: float,
) -> float:
    """P(agent engages in lobbying this round).

    Base rate from EU Transparency Register 2021-2023.
    Scales with perceived threat and available resources.
    """
    base = LOBBYING_BASE_RATE.get(agent_type, 0.30)

    # Scale by threat perception and resources
    threat_multiplier = 0.5 + 0.5 * min(1.0, perceived_policy_threat / 80.0)
    resource_multiplier = 0.5 + 0.5 * min(1.0, resources / 100.0)

    return min(1.0, base * threat_multiplier * resource_multiplier)


# ─────────────────────────────────────────────────────────────────────────────
# DEGROOT BELIEF UPDATING — social learning
# ─────────────────────────────────────────────────────────────────────────────

def update_belief(
    own_belief: float,
    neighbor_beliefs: list[float],
    neighbor_weights: list[float] | None = None,
    stubbornness: float = 0.5,
) -> float:
    """Update an agent's belief via DeGroot (1974) social learning.

    The DeGroot model is the canonical social-influence model for belief
    convergence in networks (DeGroot 1974; Golub & Jackson 2010). Each round,
    an agent updates by blending its own belief with a weighted average of its
    neighbours' beliefs:

      belief_new = stubbornness × own_belief + (1 - stubbornness) × neighbour_avg

    stubbornness ∈ [0, 1] parameterises resistance to social influence:
      1.0 = fully stubborn (never updates from neighbours)
      0.0 = pure conformist (adopts the neighbour average entirely)

    The STUBBORNNESS dict sets type defaults:
      Large companies (0.7): committed corporate strategies, long planning cycles.
      Startups (0.4): more reactive to peer signals, less institutional inertia.
    [DIRECTIONAL] Direction (large firms more stubborn than startups) grounded;
    exact values from STUBBORNNESS dict are [ASSUMED].

    Source: DeGroot, M.H. (1974) "Reaching a Consensus", JASA 69(345).
    Empirical validation: Golub & Jackson (2010) "Naive Learning in Social Networks".
    """
    if not neighbor_beliefs:
        return own_belief

    if neighbor_weights is None:
        neighbor_weights = [1.0 / len(neighbor_beliefs)] * len(neighbor_beliefs)

    # Normalize weights
    total_w = sum(neighbor_weights)
    if total_w == 0:
        return own_belief
    norm_weights = [w / total_w for w in neighbor_weights]

    neighbor_avg = sum(b * w for b, w in zip(neighbor_beliefs, norm_weights))
    return stubbornness * own_belief + (1.0 - stubbornness) * neighbor_avg


STUBBORNNESS: dict[str, float] = {
    "large_company": 0.70,
    "mid_company": 0.55,
    "startup": 0.40,
    "researcher": 0.60,
    "investor": 0.50,
    "civil_society": 0.65,
}
