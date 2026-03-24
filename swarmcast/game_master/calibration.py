"""Empirical calibration of PolicyLab's quantitative coefficients.

Every number in this file has a specific derivation from a published
empirical source. Numbers WITHOUT a derivation here do not belong in
the model's quantitative layer.

CLASSIFICATION SYSTEM
─────────────────────
  [GROUNDED]    — Specific empirical estimate exists; coefficient derived
                  from it with documented unit conversion.
  [DIRECTIONAL] — Direction confirmed by multiple studies; magnitude unknown.
                  Coefficient must be treated as a sensitivity parameter.
  [ASSUMED]     — No empirical basis. Must be DELETED from the quantitative
                  layer or clearly labeled as a model design choice swept
                  in sensitivity analysis.

DERIVATION METHODOLOGY
──────────────────────
Each grounded coefficient is derived as follows:

  1. Identify the empirical elasticity ε from the source paper
     (units: Δoutput / Δinput, measured over the paper's time horizon T_paper).

  2. Establish the simulation unit conversion:
       rounds_per_year = 4  (1 round ≈ 3 months, a governance review cycle)
       index_scale     = 100 (all indicators run 0-100)

  3. Convert:
       per_round_coefficient = ε × (1 / rounds_per_year) × (1 / index_scale)

  4. Report the 95% confidence interval from the source SE.

This gives coefficients in units of "index points per round per index point"
which is what simulation_loop.py's indicator arithmetic expects.

CALIBRATION TARGETS (SMM target moments for future calibration)
───────────────────────────────────────────────────────────────
These are the historical moments that a properly calibrated model should
reproduce. They come from public records of AI governance episodes.

  EU AI Act 2021-2024 (source: EU Transparency Register, EC Impact Assessment):
    m1 = large-company lobbying rate          ≈ 0.85
    m2 = first-year compliance announcement   ≈ 0.23
    m3 = relocation-threat rate               ≈ 0.12
    m4 = EU AI investment change 2021-2023    ≈ +0.12 (Dealroom data)
         [TEMPORAL CAVEAT: EU AI Act not enacted until December 2023.
          The 2021-2023 window measures pre-regulation dynamics, not policy effects.
          Suitable as a COUNTERFACTUAL BASELINE (what happened without full regulation)
          but NOT as a post-enactment target. Units unclear: +0.12 on 0-100 index?
          That is +0.12 points = negligible. Likely means +12% absolute investment.]
    m5 = EU public AI trust change 2021-2023  ≈ −0.07
         [REMOVED: Eurobarometer 87.1 was fielded autumn 2016, not 2021-2023.
          Correct surveys are Eurobarometer 98 (autumn 2022) and 99 (autumn 2023).
          This moment target requires verification against EB-98/99 before use.]

  GDPR 2018-2020 (source: IAPP survey, DLA Piper annual reports):
    m6 = SME compliance rate after 24 months  ≈ 0.52
    m7 = large-firm compliance rate            ≈ 0.91
         [MEASUREMENT GAP: Model has no 'current compliance %' output variable.
          Model tracks discrete compliance EVENTS, not a continuous rate.
          To use m7 in SMM: must implement compliance_rate = (agents_complied /
          total_regulated_agents) tracked each round. Currently unmeasurable.]
    m8 = DPA enforcement actions year 1        ≈ 0.06 of regulated entities

  California SB 1047 2023 (source: CA legislature, OpenSecrets):
    m9 = industry coalition opposition rate    ≈ 0.95 (OpenAI, Google, Meta, Anthropic)
    m10 = academic-civil-society support rate  ≈ 0.60 (Hinton, Bengio letters)

REFERENCES
──────────
  [OECD_FDI_2023]
    Springler, E. et al. (2023). "The trade effects of product market
    regulation in global value chains." Empirica. Springer.
    Key finding: PMR Barriers-to-Trade-and-Investment coefficient = −0.197
    (exporting country), −0.109 (importing country). 60 countries, 1997-2016.
    URL: https://doi.org/10.1007/s10663-023-09574-z

  [UGUR_2016]
    Ugur, M. et al. (2016). "R&D and productivity: an updated meta-analysis."
    R&D Management. 1,000+ empirical estimates across 29 studies.
    Key finding: R&D-to-productivity elasticity = 0.138 (SE = 0.012).

  [AGHION_2005]
    Aghion, P., Bloom, N., Blundell, R., Griffith, R., Howitt, P. (2005).
    "Competition and Innovation: An Inverted-U Relationship."
    QJE 120(2): 701-728.
    Key finding: inverted-U peak at Lerner index ≈ 0.4; escape-competition
    effect below peak, Schumpeterian effect above. UK panel 1968-1997.

  [BAUMGARTNER_LEECH_1998]
    Baumgartner, F.R. & Leech, B.L. (1998). "Basic Interests: The
    Importance of Groups in Politics and in Political Science."
    Princeton University Press.
    Key finding: organized coalitions achieve 30-50% higher success
    rates than solo lobbying in US federal rulemaking.

  [AGHION_HOWITT_1992]
    Aghion, P. & Howitt, P. (1992). "A Model of Growth Through Creative
    Destruction." Econometrica 60(2): 323-351. [Aghion & Howitt, 2025 Nobel Prize in Economic Sciences]
    Key finding: R&D intensity ∝ market investment (structural relationship,
    not a per-period linear coupling). Innovation is forward-looking,
    dependent on expected rents, not current investment level.
    NOTE: Does NOT provide a per-period linear coupling coefficient.
"""

from __future__ import annotations

import dataclasses
import math


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION UNIT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ROUNDS_PER_YEAR: float = 4.0
"""One simulation round ≈ one governance review cycle ≈ 3 months."""

INDEX_SCALE: float = 100.0
"""All indicators (investment_index, innovation_rate, etc.) run 0-100."""


# ─────────────────────────────────────────────────────────────────────────────
# GROUNDED COEFFICIENTS — with full derivations
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class GroundedCoefficient:
    """A single empirically-derived coefficient with provenance."""
    name: str
    value: float
    ci_low: float
    ci_high: float
    source: str
    derivation: str
    status: str  # GROUNDED | DIRECTIONAL | ASSUMED


def derive_burden_to_investment() -> GroundedCoefficient:
    """Derive the regulatory burden → AI investment suppression coefficient.

    Source: [OECD_FDI_2023]
    Empirical estimate: ε = −0.197 (elasticity of bilateral FDI to PMR
    barriers-to-trade-and-investment, exporting country).
    SE = 0.032 (from Table 4, column 4 of Springler et al. 2023).
    95% CI: [−0.197 − 1.96×0.032, −0.197 + 1.96×0.032] = [−0.260, −0.134]

    Unit conversion:
      The OECD PMR index runs 0-6.
      Our regulatory_burden indicator runs 0-100.
      Mapping: 1 PMR unit ≈ 100/6 ≈ 16.7 burden index points.
      So ε_per_burden_point = −0.197 / 16.7 ≈ −0.0118 per PMR unit.

      The OECD study measures annual FDI elasticity.
      Per-round (quarterly): divide by ROUNDS_PER_YEAR = 4.
      Per-round-per-index-point: −0.0118 / 4 ≈ −0.00295

    We apply this only above a threshold of 30 (mild regulation baseline),
    so the effective per-round coefficient is:
      −0.00295  [CI: −0.00389, −0.00200]

    NOTE: The OECD study uses bilateral FDI across 60 countries, not
    AI-sector investment. AI sector may have higher sensitivity given
    its mobility (compute can move). This is a CONSERVATIVE estimate.
    """
    eps_annual = -0.197       # from OECD_FDI_2023 Table 4
    eps_se = 0.032
    pmr_units_per_100 = 100.0 / 6.0  # PMR scale is 0-6

    # Per PMR unit → per burden index point
    eps_per_burden = eps_annual / pmr_units_per_100

    # Annual → per round
    per_round = eps_per_burden / ROUNDS_PER_YEAR

    # CI bounds
    ci_low = (eps_annual - 1.96 * eps_se) / pmr_units_per_100 / ROUNDS_PER_YEAR
    ci_high = (eps_annual + 1.96 * eps_se) / pmr_units_per_100 / ROUNDS_PER_YEAR

    return GroundedCoefficient(
        name="burden_to_investment",
        value=round(per_round, 6),
        ci_low=round(ci_low, 6),
        ci_high=round(ci_high, 6),
        source="OECD_FDI_2023 (Springler et al. 2023, Table 4 col 4)",
        derivation=(
            f"ε_annual={eps_annual} / (PMR_scale={pmr_units_per_100:.1f}) "
            f"/ (rounds_per_year={ROUNDS_PER_YEAR}) = {per_round:.6f}"
        ),
        status="GROUNDED",
    )


def derive_rd_to_innovation() -> GroundedCoefficient:
    """Derive the R&D/investment → innovation coupling coefficient.

    Source: [UGUR_2016]
    Empirical estimate: ε = 0.138, SE = 0.012 (meta-analysis of 1000+ estimates).
    This is a R&D-expenditure-to-productivity elasticity.

    Unit conversion:
      Ugur et al. measure elasticity of TFP to R&D expenditure.
      We proxy: AI investment index ≈ R&D expenditure intensity.
      Innovation rate ≈ TFP growth rate.

      Per-round (quarterly): 0.138 / ROUNDS_PER_YEAR = 0.0345
      Per-round-per-index-point: 0.0345 / INDEX_SCALE = 0.000345

    95% CI:
      per_round_ci_low  = (0.138 - 1.96×0.012) / 4 / 100 = 0.000286
      per_round_ci_high = (0.138 + 1.96×0.012) / 4 / 100 = 0.000404

    IMPORTANT CAVEAT: Aghion-Howitt (1992) [Nobel 2025] shows the structural
    relationship is forward-looking (innovation depends on EXPECTED future
    rents, not current investment). Our linear time-step approximation is a
    LOCAL linearisation that is valid only near equilibrium. It systematically
    underestimates innovation in boom phases and overestimates it in bust.
    The actual coefficient is approximately 44× smaller than our previous 0.015 value.
    """
    eps = 0.138
    eps_se = 0.012

    per_round = eps / ROUNDS_PER_YEAR / INDEX_SCALE
    ci_low = (eps - 1.96 * eps_se) / ROUNDS_PER_YEAR / INDEX_SCALE
    ci_high = (eps + 1.96 * eps_se) / ROUNDS_PER_YEAR / INDEX_SCALE

    return GroundedCoefficient(
        name="rd_to_innovation",
        value=round(per_round, 7),
        ci_low=round(ci_low, 7),
        ci_high=round(ci_high, 7),
        source="UGUR_2016 (Ugur et al. 2016, R&D-to-productivity meta-analysis)",
        derivation=(
            f"ε={eps} / rounds_per_year={ROUNDS_PER_YEAR} "
            f"/ index_scale={INDEX_SCALE} = {per_round:.7f}"
        ),
        status="GROUNDED",
    )


def derive_schumpeterian_peak() -> GroundedCoefficient:
    """Derive the Schumpeterian inverted-U peak market concentration.

    Source: [AGHION_2005]
    Empirical finding: inverted-U peak at Lerner index approximately 0.4.
    UK firm panel data, 1968-1997. (Aghion, Bloom, Blundell, Griffith, Howitt 2005)

    Unit conversion:
      Lerner index L = (P - MC) / P, runs 0-1.
      Our market_concentration runs 0-100.
      Mapping: L = 0.4 to concentration = 40 on our scale.

    Note: Aghion et al. (2005) do not report a CI around the peak.
    The ci_low=25.0, ci_high=60.0 are SENSITIVITY RANGE estimates from
    cross-industry heterogeneity in the paper's figures -- NOT a reported CI.
    The peak varies by industry neck-and-neckness (Proposition 5) and a
    single system-wide peak of 40 is a strong simplification.
    Treat ci_low/ci_high as sweep bounds, not statistical uncertainty.
    """
    lerner_peak = 0.4  # Aghion et al. 2005 Figure II
    peak_on_scale = lerner_peak * INDEX_SCALE

    # Sensitivity range from cross-industry variation -- NOT a reported CI
    industry_range_low = 25.0   # neck-and-neck industries: lower peak
    industry_range_high = 60.0  # leveled industries: higher peak

    return GroundedCoefficient(
        name="schumpeterian_concentration_peak",
        value=peak_on_scale,
        ci_low=industry_range_low,
        ci_high=industry_range_high,
        source="AGHION_2005 (Aghion, Bloom et al. 2005 QJE Figure II)",
        derivation=(
            f"Lerner peak={lerner_peak} x index_scale={INDEX_SCALE} = {peak_on_scale}. "
            f"ci_low/ci_high: sensitivity range [25,60], NOT a reported CI (see A5 note)."
        ),
        status="GROUNDED",
    )


def derive_coalition_bonus() -> GroundedCoefficient:
    """Derive the coalition lobbying effectiveness multiplier.

    Source: [BAUMGARTNER_LEECH_1998] — NOTE: SYNTHESIS BOOK, NOT PRIMARY STUDY
    Baumgartner & Leech (1998) is a book-length survey of the lobbying literature,
    not a primary empirical study reporting a specific elasticity. The 30-50%
    advantage figure synthesizes multiple studies but does not have a standard error.

    CRITICAL ISSUE: The conversion from success-rate advantage to budget multiplier
    is unvalidated. If P_coalition = 1.4 × P_solo, the required budget multiplier
    depends on the slope of the probability function at the operating point, which
    itself depends on assumed parameters (lobbying_base_resistance, etc.).
    This introduces a potential 2-3× error in the implied multiplier.

    STATUS: DIRECTIONAL. Direction confirmed (coalitions help). Magnitude unknown.
    Treat the [1.30, 1.50] range as a sensitivity sweep, NOT an empirical CI.
    A primary study with a reported elasticity is needed for GROUNDED status.
    """
    lower = 1.30
    upper = 1.50
    central = (lower + upper) / 2  # = 1.40

    return GroundedCoefficient(
        name="coalition_lobbying_multiplier",
        value=central,
        ci_low=lower,
        ci_high=upper,
        source="BAUMGARTNER_LEECH_1998 (Basic Interests, Princeton UP) — SYNTHESIS BOOK",
        derivation=(
            f"30-50% success-rate advantage from literature synthesis (not primary study). "
            f"Success-rate to budget-multiplier conversion unvalidated (potential 2-3x error). "
            f"Range [{lower}, {upper}] treated as sensitivity sweep, not empirical CI."
        ),
        status="DIRECTIONAL",  # demoted from GROUNDED: source is synthesis not primary study
    )


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTIONAL COEFFICIENTS — direction known, magnitude unknown
# ─────────────────────────────────────────────────────────────────────────────

DIRECTIONAL_PARAMETERS: dict[str, str] = {
    "trust_to_burden": (
        "Direction: low public trust → higher regulatory demand. "
        "Source: Legitimacy theory (Suchman 1995, Ayres & Braithwaite 1992). "
        "Magnitude: NO empirical estimate. Set to 0 in rigorous baseline; "
        "sweep [0, -0.003, -0.006, -0.010] in sensitivity analysis."
    ),
    "burden_to_trust": (
        "Direction: high regulatory burden → reduced public trust in institutions. "
        "Source: Regulatory overreach backlash literature (general). "
        "Magnitude: NO empirical estimate. Set to 0 in rigorous baseline."
    ),
    "innovation_to_trust": (
        "Direction: sustained innovation → gradual trust in AI technology. "
        "Source: Technology Acceptance Model (Davis 1989, TAM). "
        "Magnitude: NO empirical estimate. Set to 0 in rigorous baseline."
    ),
    "passive_regulatory_burden": (
        "Direction: enacted policies accumulate burden each round. "
        "Source: Regulatory compliance literature (conceptual). "
        "Magnitude: NO empirical estimate. Set to 0 in rigorous baseline; "
        "sweep [0, 0.1, 0.5, 1.0, 2.0] in sensitivity analysis."
    ),
    "resource_regen_economy_scaling": (
        "Direction: company resources deplete when the investment ecosystem collapses. "
        "Source: Conceptually obvious. "
        "Magnitude: NO empirical estimate. Linear scale [0.5, 1.5] is assumed."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# ASSUMED / MUST-DELETE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

ASSUMED_PARAMETERS: dict[str, str] = {
    "tipping_investment_cascade_threshold": (
        "ASSUMED. No empirical basis for threshold of 20/100. "
        "Neither Lempert (2003) nor Axelrod (2006) provide AI-governance thresholds. "
        "Action: DELETE from quantitative layer. Use as sensitivity parameter only."
    ),
    "tipping_trust_collapse_threshold": (
        "ASSUMED. No empirical basis for threshold of 20/100. "
        "Action: DELETE from quantitative layer."
    ),
    "tipping_innovation_death_threshold": (
        "ASSUMED. No empirical basis for threshold of 15/100. "
        "Action: DELETE from quantitative layer."
    ),
    "tipping_cascade_multiplier": (
        "ASSUMED. 1.5× has no empirical backing. "
        "Systematically produces catastrophism when any threshold is grazed. "
        "Action: DELETE from quantitative layer."
    ),
    "relocation_ongoing_innovation_drain": (
        "ASSUMED. 3 points/round/company chosen to produce near-zero innovation "
        "for the moratorium scenario. This is calibration-to-desired-output, "
        "the most dangerous form of bias. "
        "Action: Set to 0.0 in rigorous baseline."
    ),
    "severity_compliance_exponent": (
        "ASSUMED. Exponent of 1.5 chosen for mathematical consistency. "
        "No empirical study provides exponents for AI governance severity scaling."
    ),
    "severity_relocation_exponent": "ASSUMED. Exponent of 1.3, no empirical basis.",
    "severity_innovation_exponent": "ASSUMED. Exponent of 1.5, no empirical basis.",
    "lobbying_base_resistance": (
        "ASSUMED. Value of 30 chosen ad hoc. No empirical calibration. "
        "DIRECTIONAL only: higher values make lobbying harder."
    ),
    "enforcement_base_prob_per_severity": (
        "ASSUMED. 0.10 per severity point chosen ad hoc. "
        "EU DPA enforcement rates (0.06 per regulated entity per year from "
        "DLA Piper 2020 GDPR report) could ground this after proper unit conversion."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTED CALIBRATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

# Compute all grounded coefficients at module load time
BURDEN_TO_INVESTMENT = derive_burden_to_investment()
RD_TO_INNOVATION = derive_rd_to_innovation()
SCHUMPETERIAN_PEAK = derive_schumpeterian_peak()
COALITION_BONUS = derive_coalition_bonus()

GROUNDED_COEFFICIENTS: tuple[GroundedCoefficient, ...] = (
    BURDEN_TO_INVESTMENT,
    RD_TO_INNOVATION,
    SCHUMPETERIAN_PEAK,
    # COALITION_BONUS is DIRECTIONAL (Baumgartner & Leech 1998
    # is a synthesis book, not a primary empirical study; success-rate →
    # budget-multiplier conversion is unvalidated). It lives in
    # DIRECTIONAL_COEFFICIENTS below.
)

DIRECTIONAL_COEFFICIENTS: tuple[GroundedCoefficient, ...] = (
    COALITION_BONUS,
    # Add future directional coefficients here as data becomes available.
)


def calibration_report() -> str:
    """Print a full calibration audit — every number with its status."""
    sep = "─" * 72
    lines = [
        sep,
        "POLICYLAB CALIBRATION AUDIT",
        sep,
        "",
        "GROUNDED COEFFICIENTS (empirically derived):",
        "",
    ]
    for c in GROUNDED_COEFFICIENTS:
        lines += [
            f"  [{c.status}] {c.name}",
            f"    value  : {c.value}",
            f"    95% CI : [{c.ci_low}, {c.ci_high}]",
            f"    source : {c.source}",
            f"    derived: {c.derivation}",
            "",
        ]

    lines += ["", "DIRECTIONAL COEFFICIENTS (direction confirmed; magnitude from synthesis/indirect):", ""]
    for c in DIRECTIONAL_COEFFICIENTS:
        lines += [
            f"  [{c.status}] {c.name}",
            f"    value  : {c.value}",
            f"    range  : [{c.ci_low}, {c.ci_high}]",
            f"    source : {c.source}",
            f"    derived: {c.derivation}",
            "",
        ]

    lines += ["", "DIRECTIONAL PARAMETERS (direction known, magnitude unknown):", ""]
    for name, desc in DIRECTIONAL_PARAMETERS.items():
        lines.append(f"  [DIRECTIONAL] {name}")
        lines.append(f"    {desc[:100]}")
        lines.append("")

    lines += ["", "ASSUMED / MUST BE SWEPT AS SENSITIVITY (no empirical basis):", ""]
    for name, desc in ASSUMED_PARAMETERS.items():
        lines.append(f"  [ASSUMED] {name}")
        lines.append(f"    {desc[:100]}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


if __name__ == "__main__":
    print(calibration_report())
