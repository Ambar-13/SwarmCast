"""Configurable resolution parameters for PolicyLab's resolution engine.

Every parameter carries an epistemic tag:
  [GROUNDED]    Value derived from a published empirical estimate.
  [DIRECTIONAL] Direction confirmed; magnitude is a sensitivity parameter.
  [ASSUMED]     No empirical basis; ad hoc design choice.

DEFAULT_CONFIG uses conservative defaults. Use sensitivity_layer.py to
identify which results are robust vs model-dependent.
"""

from __future__ import annotations
import dataclasses
from typing import Any


@dataclasses.dataclass
class ResolutionConfig:

    # LOBBYING
    lobbying_base_resistance: float = 30.0
    """[ASSUMED] Government inertia. Sweep [15,30,50,75].
    Coalition bonus IS grounded [1.30-1.50x, Baumgartner & Leech 1998]."""

    lobbying_opposition_weight: float = 20.0
    """[ASSUMED] Resistance per opposing lobby. Sweep [10,20,35]."""

    lobbying_public_opposition_weight: float = 10.0
    """[ASSUMED] Resistance per opposing public statement. Sweep [5,10,20]."""

    lobbying_trust_cost: float = 2.0
    """[ASSUMED] Trust cost of successful lobbying. Sweep [1,2,4]."""

    # COMPLIANCE
    compliance_regulatory_burden: float = 5.0
    """[ASSUMED] Burden increase per compliance event. Sweep [2,5,10]."""

    compliance_innovation_cost: float = 3.0
    """[ASSUMED] Innovation decrease per compliance event. Sweep [1,3,6].
    Direction grounded (compliance diverts R&D); magnitude not calibrated."""

    # EVASION
    evasion_enforcement_strength_per_action: float = 30.0
    """[ASSUMED] Detection increase per enforcement action. Sweep [15,30,50]."""

    evasion_max_detection_probability: float = 0.9
    """[ASSUMED] Detection probability cap. Sweep [0.7,0.9,0.95]."""

    evasion_trust_cost_if_detected: float = 5.0
    """[ASSUMED] Trust cost when evasion detected. Sweep [3,5,10]."""

    evasion_innovation_gain_if_undetected: float = 2.0
    """[ASSUMED] Innovation gain from undetected evasion. Sweep [1,2,5]."""

    # RELOCATION
    relocation_investment_cost: float = 15.0
    """[ASSUMED] Investment drop per relocating company. Sweep [5,10,15,25].
    Previously 25 to force near-zero investment for moratorium scenario.
    That was calibration-to-desired-output. Reset to 15 (conservative)."""

    relocation_innovation_cost: float = 10.0
    """[ASSUMED] Innovation drop per relocating company. Sweep [5,10,20].
    Previously 15; reset to 10 (conservative)."""

    relocation_concentration_increase: float = 5.0
    """[ASSUMED] Concentration increase on relocation. Sweep [2,5,10]."""

    relocation_ongoing_innovation_drain: float = 0.0
    """[ASSUMED - PERMANENTLY DISABLED] Previously 3.0/round, chosen to make
    moratorium show near-zero innovation (calibration-to-desired-output,
    the most dangerous bias). Set to 0.0 permanently."""

    # ENFORCEMENT
    enforcement_min_capacity: float = 20.0
    """[ASSUMED] Min capacity for enforcement to succeed. Sweep [10,20,35]."""

    enforcement_burden_cost: float = 3.0
    """[ASSUMED] Burden increase from enforcement. Sweep [1,3,5]."""

    # PUBLIC STATEMENTS
    public_statement_max_trust_shift: float = 5.0
    """[ASSUMED] Max trust shift per public statement. Sweep [2,5,10]."""

    public_statement_influence_divisor: float = 20.0
    """[ASSUMED] Influence-to-trust conversion divisor. Sweep [10,20,40]."""

    # SEVERITY SCALING EXPONENTS  (ALL ASSUMED)
    # Formula: multiplier = (severity / 3.0) ** exponent
    severity_compliance_exponent: float = 1.5
    """[ASSUMED] (5/3)^1.5 = 2.15x at sev-5. Sweep [1.0,1.5,2.0]."""

    severity_relocation_exponent: float = 1.3
    """[ASSUMED] (5/3)^1.3 = 1.94x at sev-5. Sweep [1.0,1.3,1.8]."""

    severity_innovation_exponent: float = 1.5
    """[ASSUMED] Sweep [1.0,1.5,2.0]."""

    severity_trust_exponent: float = 0.8
    """[ASSUMED] Sublinear. Sweep [0.5,0.8,1.0]."""

    severity_lobbying_resistance_exponent: float = 1.0
    """[ASSUMED] Linear. Sweep [0.8,1.0,1.3]."""

    severity_evasion_detection_bonus: float = 0.10
    """[ASSUMED] Detection bonus per sev point above 3. Sweep [0.05,0.10,0.20]."""

    # AUTOMATIC ENFORCEMENT LOOP
    enforcement_grace_rounds_base: int = 3
    """[ASSUMED] Grace rounds before auto-investigation. Sweep [1,3,5]."""

    enforcement_base_prob_per_severity: float = 0.005
    """[DIRECTIONAL] Per-entity enforcement contact probability per round per severity unit.

    CORRECTED DERIVATION (v2.1):
      DLA Piper (2020) GDPR: 6% of entities received enforcement contact in year 1.
      6%/year = 0.015/quarter = 0.015/round (1 round ≈ 1 quarter, ROUNDS_PER_YEAR=4)
      But detection_prob = base × policy_severity.
      GDPR ≈ severity 3. To get 0.015/round at sev=3:
        base = 0.015 / 3 = 0.005  ✓

    Verification at sev=3 (n=2000, 8 rounds, 50% non-compliant):
      detection_prob = 0.005 × 3 = 0.015/round
      fraction ever contacted = 1-(1-0.015)^8 = 11.4% of non-compliant agents
      11.4% × 50% non-compliant = 5.7% of all entities ≈ 6% ✓

    At sev=5 (criminal ban):
      detection_prob = 0.005 × 5 = 0.025/round
      Among ~20% non-compliant: 1-(1-0.025)^8 = 18.3% × 20% = 3.7%
      (fewer non-compliant agents at sev=5 → absolute enforcement rate lower)

    PREVIOUS BUG (0.015): multiplied by severity=3 gave 0.045/round → 18.5% contact
    rate vs DLA Piper target of 6%. Factor of 3x too high.

    Sweep [0.002, 0.005, 0.010, 0.015] in sensitivity analysis.
    Note: v1 used this for auto-enforcement trigger, not per-agent contact.
    In v2 vectorized mode, this is used as per-agent detection probability."""

    enforcement_escalation_per_round: float = 0.03
    """[ASSUMED] Prob increase per non-compliant round. Sweep [0.01,0.03,0.07]."""

    enforcement_penalty_innovation_per_severity: float = 3.0
    """[ASSUMED] Innovation penalty when caught = sev x this. Sweep [1,3,5]."""

    # Severity extraction heuristics
    severity_req_weight: float = 0.5
    """[ASSUMED] Severity score per policy requirement.
    No empirical calibration target. Hand-fitted to GDPR/EU AI Act/US EO 14110.
    Sweep [0.3, 0.5, 0.8]. Higher = more requirements → higher severity.
    Consequence: propagates into relocation_exponent, compliance_lambda, enforcement_prob."""

    severity_penalty_weight: float = 0.7
    """[ASSUMED] Severity score per penalty clause.
    No empirical calibration target. Hand-fitted.
    Sweep [0.4, 0.7, 1.0]. Criminal penalty clauses get an additional +1.0 bonus."""

    # Enforcement capacity scaling
    enforcement_staff_scaling: float = 0.3
    """[ASSUMED] Real-world bureau headcount → in-game staff units.
    '500-person Bureau' × 0.3 = 150 in-game staff units.
    No empirical calibration. Sweep [0.1, 0.3, 0.5, 1.0]."""

    # V2 ONGOING BURDEN — dominates dynamics, must be explicitly configurable
    ongoing_burden_per_severity: float = 1.5
    """[DIRECTIONAL] Burden points added per round per severity unit.
    Models ongoing compliance monitoring costs: audits, reporting, legal review.
    At severity=5: 7.5 burden points/round before compliance discharge.
    [DIRECTIONAL] Direction grounded (active enforcement creates overhead).
    Magnitude [ASSUMED]. This parameter dominates v2 dynamics — it drives burden
    accumulation faster than the grounded OECD coefficient.
    Sweep [0.5, 1.0, 1.5, 2.5, 4.0].
    Calibration target: EU AI Act administrative cost ~3-7% of R&D budget/year
    at severity=3 → ~0.5 burden points/round (suggests 1.5 is 3x too high for EU AI Act,
    but may be appropriate for severity=5 criminal enforcement).
    TODO: derive per-severity from DLA Piper enforcement cost data."""

    enforcement_penalty_trust_per_severity: float = 2.0
    """[ASSUMED] Trust penalty when caught = sev x this. Sweep [1,2,4]."""

    # FAILURE DETECTION THRESHOLDS
    failure_trust_threshold: float = 30.0
    """[ASSUMED] Trust below this = trust_collapse failure. Sweep [20,30,40]."""

    failure_investment_threshold: float = 60.0
    """[ASSUMED] Investment below this = investment_flight. Sweep [40,60,75]."""

    failure_innovation_threshold: float = 50.0
    """[ASSUMED] Innovation below this = innovation_death. Sweep [30,50,65]."""

    failure_relocation_rate_threshold: float = 0.5
    """[ASSUMED] Relocation in >50% runs = mass_relocation. Sweep [0.25,0.50,0.75]."""

    def severity_multiplier(self, severity: float, exponent: float) -> float:
        """(severity / 3.0) ** exponent. All exponents are ASSUMED."""
        ratio = max(0.1, severity) / 3.0
        return ratio ** exponent

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolutionConfig":
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def epistemic_summary(self) -> str:
        return (
            "ResolutionConfig: all 33 parameters are [ASSUMED] or [DIRECTIONAL]. "
            "No resolution-engine parameter has a published empirical estimate. "
            "Use sensitivity_layer.py to identify which results are robust."
        )

    def describe(self) -> str:
        lines = ["ResolutionConfig (all [ASSUMED]):"]
        for field in dataclasses.fields(self):
            lines.append(f"  {field.name}: {getattr(self, field.name)}")
        return "\n".join(lines)


DEFAULT_CONFIG = ResolutionConfig()

PERMISSIVE_CONFIG = ResolutionConfig(
    lobbying_base_resistance=15.0,
    compliance_innovation_cost=1.0,
    enforcement_min_capacity=30.0,
    evasion_max_detection_probability=0.7,
)

STRICT_CONFIG = ResolutionConfig(
    lobbying_base_resistance=50.0,
    compliance_innovation_cost=5.0,
    enforcement_min_capacity=10.0,
    evasion_max_detection_probability=0.95,
    evasion_trust_cost_if_detected=10.0,
)

CONSERVATIVE_CONFIG = ResolutionConfig(
    relocation_investment_cost=10.0,
    relocation_innovation_cost=8.0,
    compliance_innovation_cost=1.5,
    enforcement_base_prob_per_severity=0.04,
    severity_compliance_exponent=1.0,
    severity_relocation_exponent=1.0,
    severity_innovation_exponent=1.0,
)
"""Linear severity scaling + reduced relocation costs.
Results that differ between DEFAULT_CONFIG and CONSERVATIVE_CONFIG
depend on assumed superlinear exponents -- treat as NON-ROBUST."""
