"""Coupled indicator dynamics — three-layer architecture.

LAYER 1 — GROUNDED (empirically calibrated)
─────────────────────────────────────────────
Only four effects have specific empirical estimates. These run by default.
Every coefficient has a derivation in calibration.py with source and CI.

  [GROUNDED] regulatory_burden → ai_investment_index
      Source: OECD PMR FDI gravity (Springler et al. 2023)
      Coefficient: −0.00295 per round per burden index point [CI: −0.00389, −0.00200]

  [GROUNDED] ai_investment_index → innovation_rate (R&D proxy)
      Source: R&D-to-productivity meta-analysis (Ugur et al. 2016)
      Coefficient: +0.000345 per round per investment index point [CI: +0.000286, +0.000404]
      NOTE: approximately 1/44th of the previously used 0.015 coefficient (actual ratio: 0.015/0.000345 = 43.5×) — that was not grounded.

  [GROUNDED] market_concentration → innovation_rate (Schumpeterian inverted-U)
      Source: Aghion, Bloom et al. (2005) QJE
      Peak at Lerner index 0.4 → concentration = 40 on 0-100 scale [CI: 25-60]

  [GROUNDED] coalition_lobbying_multiplier (applied in resolution_engine.py)
      Source: Baumgartner & Leech (1998)
      Range: 1.30-1.50×

LAYER 2 — DIRECTIONAL (direction known, magnitude = 0 in rigorous baseline)
─────────────────────────────────────────────────────────────────────────────
These effects are real but have no empirically calibrated magnitude.
In the RIGOROUS_BASELINE config all Layer-2 coefficients are 0.0.
They appear only in SENSITIVITY configs.

  trust ↔ regulatory_burden   (legitimacy theory; Suchman 1995)
  innovation → public_trust   (TAM; Davis 1989)
  passive burden accumulation (regulatory compliance, conceptual)
  resource regeneration scaling (conceptually obvious)

LAYER 3 — ASSUMED (invented; swept as sensitivity parameters)
──────────────────────────────────────────────────────────────
These exist only to demonstrate model-dependence.
Results that ONLY appear with Layer-3 active are flagged as NON-ROBUST.

  tipping-point thresholds    — no empirical basis
  cascade multiplier          — no empirical basis
  ongoing relocation drain    — calibration to desired output (worst bias)
  severity exponents          — ad hoc mathematical choices

REFERENCES
──────────
See calibration.py for full derivations and citations.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from swarmcast.game_master.calibration import (
    BURDEN_TO_INVESTMENT,
    RD_TO_INNOVATION,
    SCHUMPETERIAN_PEAK,
)

if TYPE_CHECKING:
    from swarmcast.components.governance_state import GovernanceWorldState


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: GROUNDED CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class GroundedDynamicsConfig:
    """Only empirically derived coefficients. No invented numbers.

    Every field here maps directly to a GroundedCoefficient in calibration.py.
    Defaults are the central estimates; CI bounds available from calibration.py.
    """

    # [GROUNDED] OECD PMR FDI gravity → per-round suppression of investment
    # Source: Springler et al. 2023 Table 4 col 4; ε = −0.197
    # Derived: −0.197 / (100/6) / 4 = −0.00295
    burden_to_investment: float = BURDEN_TO_INVESTMENT.value  # −0.00295
    burden_to_investment_ci_low: float = BURDEN_TO_INVESTMENT.ci_low
    burden_to_investment_ci_high: float = BURDEN_TO_INVESTMENT.ci_high

    # Threshold above which burden suppresses investment
    # (mild regulation = baseline; the elasticity applies to excess burden)
    burden_threshold: float = 30.0
    """[DIRECTIONAL] Threshold below which burden elasticity is near-zero.
    30/100 corresponds to a moderate regulatory environment.
    Exact threshold value is assumed — only direction is grounded."""

    # [GROUNDED] Ugur et al. 2016 meta-analysis; ε = 0.138
    # Derived: 0.138 / 4 / 100 = 0.000345
    investment_to_innovation: float = RD_TO_INNOVATION.value  # 0.000345
    investment_to_innovation_ci_low: float = RD_TO_INNOVATION.ci_low
    investment_to_innovation_ci_high: float = RD_TO_INNOVATION.ci_high

    # Above-median threshold for investment → innovation coupling
    investment_threshold: float = 50.0
    """[DIRECTIONAL] Investment must be above this to drive innovation.
    Conceptually: the economy needs a minimum investment base.
    50/100 = midpoint; exact value is assumed."""

    # [GROUNDED] Aghion et al. (2005) QJE; Lerner peak ≈ 0.4 → 40 on 0-100
    schumpeterian_peak: float = SCHUMPETERIAN_PEAK.value  # 40.0
    schumpeterian_peak_ci_low: float = SCHUMPETERIAN_PEAK.ci_low   # 25.0
    schumpeterian_peak_ci_high: float = SCHUMPETERIAN_PEAK.ci_high  # 60.0

    # Strength of inverted-U effect per unit deviation from peak
    # [DIRECTIONAL] Direction grounded (Aghion 2005); magnitude assumed.
    # Set very small to avoid overclaiming. Swept in sensitivity.
    schumpeterian_strength: float = 0.0005
    """[DIRECTIONAL] Strength of the competition-innovation effect.
    0.0005 per index point deviation produces ≤ 3 points/round at extreme
    concentration. Larger values exaggerate the effect."""


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: DIRECTIONAL CONFIG (all magnitudes zero in rigorous baseline)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DirectionalDynamicsConfig:
    """Layer-2 coefficients: direction confirmed, magnitude unknown.

    ALL values are 0.0 in the RIGOROUS_BASELINE configuration.
    Non-zero values are SENSITIVITY parameters, not predictions.
    """

    # trust ↔ burden (legitimacy theory, Suchman 1995)
    trust_to_burden: float = 0.0
    """[DIRECTIONAL] Low trust → regulatory demand increase.
    Set 0 in rigorous baseline. Sweep [-0.001, -0.003, -0.006] in sensitivity."""

    burden_to_trust: float = 0.0
    """[DIRECTIONAL] High burden → erodes institutional trust.
    Set 0 in rigorous baseline."""

    # innovation → trust (TAM, Davis 1989)
    innovation_to_trust: float = 0.0
    """[DIRECTIONAL] Sustained high innovation → public trust in AI.
    Set 0 in rigorous baseline."""

    # passive burden accumulation
    passive_burden_per_severity: float = 0.0
    """[DIRECTIONAL] Enacted policies accumulate compliance burden each round.
    Set 0 in rigorous baseline. Sweep [0.1, 0.5, 1.0] in sensitivity."""


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: ASSUMED / SENSITIVITY CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class AssumedDynamicsConfig:
    """Layer-3 parameters: no empirical basis. For sensitivity analysis only.

    Results that appear ONLY when these are non-zero are labeled NON-ROBUST.
    cascade_multiplier was designed to show model-dependence; it amplifies dynamics at threshold crossings independent of policy content — a purely computational artifact. See sensitivity_layer.py for robustness classification.
    """

    # Tipping points — entirely invented
    investment_cascade_threshold: float = 0.0
    """[ASSUMED] Below this, investment enters cascade. Set to 0 = disabled.
    Previously 20 — had no empirical basis. Sweep [0, 10, 20, 30]."""

    trust_collapse_threshold: float = 0.0
    """[ASSUMED] Below this, trust enters collapse. Set to 0 = disabled."""

    innovation_death_threshold: float = 0.0
    """[ASSUMED] Below this, innovation is considered dead. Set to 0 = disabled."""

    cascade_multiplier: float = 1.0
    """[ASSUMED] Amplifier when any threshold is crossed.
    1.0 = no effect (disabled). Previously 1.5 — had no empirical basis.
    Sweep [1.0, 1.2, 1.5, 2.0] in sensitivity."""


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED CONFIG AND PRESETS
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DynamicsConfig:
    """Full dynamics configuration: all three layers.

    Prefer RIGOROUS_BASELINE for defensible results.
    Use SENSITIVITY_* variants to probe model-dependence.
    """
    grounded: GroundedDynamicsConfig = dataclasses.field(
        default_factory=GroundedDynamicsConfig
    )
    directional: DirectionalDynamicsConfig = dataclasses.field(
        default_factory=DirectionalDynamicsConfig
    )
    assumed: AssumedDynamicsConfig = dataclasses.field(
        default_factory=AssumedDynamicsConfig
    )

    def active_layers(self) -> list[str]:
        """Report which layers have non-zero parameters."""
        layers = ["GROUNDED"]  # always active
        d = self.directional
        if any([d.trust_to_burden, d.burden_to_trust,
                d.innovation_to_trust, d.passive_burden_per_severity]):
            layers.append("DIRECTIONAL")
        a = self.assumed
        if any([a.investment_cascade_threshold, a.trust_collapse_threshold,
                a.innovation_death_threshold, a.cascade_multiplier != 1.0]):
            layers.append("ASSUMED")
        return layers


# ─── PRESET CONFIGURATIONS ────────────────────────────────────────────────────

RIGOROUS_BASELINE = DynamicsConfig(
    grounded=GroundedDynamicsConfig(),
    directional=DirectionalDynamicsConfig(),   # all zeros
    assumed=AssumedDynamicsConfig(),           # all disabled
)
"""Only empirically grounded coefficients. Most defensible configuration.
Results from this preset can be cited with confidence in direction.
Magnitudes remain approximate pending proper SMM/Bayesian calibration."""

SENSITIVITY_DIRECTIONAL = DynamicsConfig(
    grounded=GroundedDynamicsConfig(),
    directional=DirectionalDynamicsConfig(
        trust_to_burden=-0.003,
        burden_to_trust=-0.002,
        innovation_to_trust=0.001,
        passive_burden_per_severity=0.5,
    ),
    assumed=AssumedDynamicsConfig(),
)
"""Adds directional effects at CONSERVATIVE magnitudes.
Results that change materially vs. RIGOROUS_BASELINE are flagged as
DIRECTIONAL-DEPENDENT in the sensitivity report."""

SENSITIVITY_ASSUMED = DynamicsConfig(
    grounded=GroundedDynamicsConfig(),
    directional=DirectionalDynamicsConfig(
        trust_to_burden=-0.003,
        burden_to_trust=-0.002,
        innovation_to_trust=0.001,
        passive_burden_per_severity=0.5,
    ),
    assumed=AssumedDynamicsConfig(
        investment_cascade_threshold=20.0,
        trust_collapse_threshold=20.0,
        innovation_death_threshold=15.0,
        cascade_multiplier=1.5,
    ),
)
"""Full model including all assumed parameters. Previously the DEFAULT.
Use ONLY for illustration that tipping-point dynamics are possible —
not for policy recommendations. Label all outputs NON-ROBUST."""

DEFAULT_DYNAMICS = RIGOROUS_BASELINE


# ─────────────────────────────────────────────────────────────────────────────
# TIPPING POINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class TippingPointReport:
    investment_cascade: bool = False
    trust_collapse: bool = False
    innovation_death: bool = False
    cascade_active: bool = False
    config_layer: str = "GROUNDED"

    def any_active(self) -> bool:
        return self.investment_cascade or self.trust_collapse or self.innovation_death

    def describe(self) -> str:
        msgs = []
        if self.investment_cascade:
            msgs.append(
                "[ASSUMED-LAYER] CAPITAL FLIGHT CASCADE: investment below assumed "
                "threshold — this is a sensitivity result, not a grounded prediction"
            )
        if self.trust_collapse:
            msgs.append(
                "[ASSUMED-LAYER] TRUST COLLAPSE: below assumed threshold"
            )
        if self.innovation_death:
            msgs.append(
                "[ASSUMED-LAYER] INNOVATION DEATH: below assumed threshold"
            )
        return " | ".join(msgs) if msgs else ""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DYNAMICS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def apply_indicator_feedback(
    world_state: "GovernanceWorldState",
    config: DynamicsConfig = DEFAULT_DYNAMICS,
) -> TippingPointReport:
    """Apply one round of coupled indicator dynamics.

    Returns TippingPointReport (only non-empty if assumed-layer thresholds
    are configured and crossed — tagged [ASSUMED-LAYER] in output).

    Modifies world_state.economic_indicators in-place.
    Call AFTER action resolution, BEFORE clamp_indicators().
    """
    ind = world_state.economic_indicators
    inv = ind.get("ai_investment_index", 50.0)
    inn = ind.get("innovation_rate", 50.0)
    tru = ind.get("public_trust", 50.0)
    bur = ind.get("regulatory_burden", 30.0)
    con = ind.get("market_concentration", 40.0)

    g = config.grounded
    d = config.directional
    a = config.assumed

    # ── LAYER 1: GROUNDED ────────────────────────────────────────────────

    d_investment = 0.0
    d_innovation = 0.0
    d_trust = 0.0
    d_burden = 0.0

    # [GROUNDED] Regulatory burden → investment suppression
    # ε = −0.00295 per round per burden-index-point above threshold
    bur_excess = max(0.0, bur - g.burden_threshold)
    d_investment += g.burden_to_investment * bur_excess

    # [GROUNDED] Investment → innovation (R&D proxy)
    # ε = +0.000345 per round per investment-index-point above threshold
    inv_excess = max(0.0, inv - g.investment_threshold)
    d_innovation += g.investment_to_innovation * inv_excess

    # [GROUNDED] Market concentration → innovation (Schumpeterian inverted-U)
    # Peak at concentration = 40 (Lerner 0.4, Aghion 2005)
    # Below peak: slight positive (escape-competition effect)
    # Above peak: negative (Schumpeterian incumbents block disruptors)
    con_deviation = con - g.schumpeterian_peak
    if con_deviation > 0:
        # Above peak: incumbents suppress innovation
        d_innovation -= g.schumpeterian_strength * con_deviation
    else:
        # Below peak: mild positive (escape-competition, weaker effect)
        d_innovation += g.schumpeterian_strength * abs(con_deviation) * 0.3

    # ── LAYER 2: DIRECTIONAL (zero in RIGOROUS_BASELINE) ────────────────

    # trust → burden
    if d.trust_to_burden != 0.0:
        trust_deficit = max(0.0, 50.0 - tru)
        d_burden += abs(d.trust_to_burden) * trust_deficit  # low trust → more burden

    # burden → trust
    if d.burden_to_trust != 0.0:
        bur_excess2 = max(0.0, bur - 60.0)
        d_trust += d.burden_to_trust * bur_excess2

    # innovation → trust
    if d.innovation_to_trust != 0.0:
        inn_benefit = max(0.0, inn - 70.0)
        d_trust += d.innovation_to_trust * inn_benefit

    # passive burden (enacted policies)
    if d.passive_burden_per_severity != 0.0:
        passive = 0.0
        for policy in world_state.active_policies.values():
            if policy.status == "enacted":
                sev = policy.effective_severity()
                # [DIRECTIONAL] Higher severity = more burden per round
                # Exponent 1.8 is assumed; using linear (exponent=1) here
                passive += d.passive_burden_per_severity * (sev / 3.0)
        d_burden += passive

    # ── LAYER 3: ASSUMED (disabled in RIGOROUS_BASELINE) ─────────────────

    report = TippingPointReport(config_layer=", ".join(config.active_layers()))

    if a.investment_cascade_threshold > 0.0 and inv < a.investment_cascade_threshold:
        report.investment_cascade = True
    if a.trust_collapse_threshold > 0.0 and tru < a.trust_collapse_threshold:
        report.trust_collapse = True
    if a.innovation_death_threshold > 0.0 and inn < a.innovation_death_threshold:
        report.innovation_death = True
    report.cascade_active = report.any_active()

    # Cascade multiplier amplifies adverse dynamics (assumed effect)
    if report.cascade_active and a.cascade_multiplier > 1.0:
        m = a.cascade_multiplier
        if d_investment < 0: d_investment *= m
        if d_innovation < 0: d_innovation *= m
        if d_trust < 0:      d_trust *= m
        if d_burden > 0:     d_burden *= m

    # ── APPLY ─────────────────────────────────────────────────────────────

    ind["innovation_rate"] = inn + d_innovation
    ind["ai_investment_index"] = inv + d_investment
    ind["public_trust"] = tru + d_trust
    ind["regulatory_burden"] = bur + d_burden

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def convergence_check(
    indicator_history: list[dict[str, float]],
    window: int = 3,
    threshold: float = 2.0,
) -> dict[str, bool]:
    """Check whether each indicator has converged (variance < threshold)."""
    if len(indicator_history) < window:
        return {}
    recent = indicator_history[-window:]
    result = {}
    all_keys = set().union(*[d.keys() for d in recent])
    for key in all_keys:
        vals = [d.get(key, 0.0) for d in recent]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        result[key] = variance < threshold
    return result


def is_system_converged(convergence: dict[str, bool]) -> bool:
    return bool(convergence) and all(convergence.values())
