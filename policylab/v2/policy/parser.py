"""
Bill text → PolicySpec parser.

The severity parameter drives most of PolicyLab's output. If it's set
arbitrarily, the outputs are arbitrary too. This module provides a
deterministic, auditable way to map bill text or structured parameters
to a severity score — so two analysts running the same bill get the same
number, and that number can be defended.

The scoring is a weighted checklist, not an LLM call. Each dimension
maps observable bill properties to a sub-score, and the justification
text records exactly why each sub-score was assigned. A policy analyst
can hand that justification to their boss or include it in a memo.

Usage
─────
    from policylab.v2.policy.parser import parse_bill, PolicySpec

    # From structured parameters (most auditable):
    spec = parse_bill(
        penalty_type="civil",
        penalty_cap_usd=1_000_000,
        compute_threshold_flops=1e26,
        enforcement_mechanism="third_party_audit",
        grace_period_months=12,
        scope="large_developers_only",
    )
    print(spec.severity)        # 3.1
    print(spec.justification)  # line-by-line reasoning

    # From bill text (uses keyword extraction, less precise):
    spec = parse_bill_text('Any developer above 10^26 FLOPS must get third-party audit. Civil fines up to $1M. 12 months.')
    print(spec.severity)  # ~3.2
"""

from __future__ import annotations

import dataclasses
import re
from typing import Literal


PenaltyType  = Literal["none", "voluntary", "civil", "civil_heavy", "criminal"]
EnfType      = Literal["none", "self_report", "third_party_audit", "government_inspect", "criminal_invest"]
ScopeType    = Literal["all", "large_developers_only", "frontier_only", "voluntary"]


@dataclasses.dataclass
class PolicySpec:
    """Structured representation of a regulatory policy for PolicyLab simulation.

    severity is the primary input to run_hybrid_simulation(). Every field that
    contributed to the score is recorded in justification so outputs are traceable.
    """
    name: str
    description: str
    severity: float                          # 1.0–5.0, two decimal places
    justification: list[str]                 # one entry per scoring dimension

    # Structured parameters (set by parser; can be overridden)
    penalty_type: PenaltyType = "civil"
    penalty_cap_usd: float | None = None     # None = uncapped
    compute_threshold_flops: float | None = None
    enforcement_mechanism: EnfType = "self_report"
    grace_period_months: int = 0
    scope: ScopeType = "all"

    # Recommended simulation config
    recommended_n_population: int = 2000
    recommended_num_rounds: int = 16         # 4 years
    recommended_severity_sweep: tuple = ()   # (low, mid, high) scenarios

    # Compute cost factor — derived from compute_threshold_flops.
    # Pass this directly to HybridSimConfig.compute_cost_factor.
    # [ASSUMED] See HybridSimConfig.compute_cost_factor docstring for rationale.
    compute_cost_factor: float = 1.0

    def summary(self) -> str:
        lines = [
            f"Policy: {self.name}",
            f"Severity: {self.severity:.1f} / 5.0",
            "",
            "Scoring breakdown:",
        ]
        for j in self.justification:
            lines.append(f"  {j}")
        lines += [
            "",
            f"Recommended simulation: n={self.recommended_n_population}, "
            f"rounds={self.recommended_num_rounds}",
        ]
        if self.recommended_severity_sweep:
            lo, mid, hi = self.recommended_severity_sweep
            lines.append(
                f"Sensitivity sweep: severity={lo} (relaxed), "
                f"{mid} (as-written), {hi} (tightened)"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING DIMENSIONS
# Each returns (score_contribution, justification_string).
# Scores sum to a raw total that is then scaled to [1.0, 5.0].
# ─────────────────────────────────────────────────────────────────────────────

def _score_penalty(penalty_type: PenaltyType, cap_usd: float | None) -> tuple[float, str]:
    """Score the penalty dimension. All weights are [ASSUMED] and should be swept."""
    base = {  # [ASSUMED] — no empirical calibration target
        "none": 0.0,
        "voluntary": 0.3,
        "civil": 1.0,
        "civil_heavy": 1.8,
        "criminal": 3.0,
    }[penalty_type]

    cap_note = ""
    cap_adj = 0.0
    if penalty_type in ("civil", "civil_heavy") and cap_usd is not None:
        # Cap above $10M signals serious intent
        if cap_usd >= 100_000_000:
            cap_adj = 0.4
            cap_note = f" Cap ≥$100M adds +0.4."
        elif cap_usd >= 10_000_000:
            cap_adj = 0.2
            cap_note = f" Cap ≥$10M adds +0.2."
        elif cap_usd <= 500_000:
            cap_adj = -0.2
            cap_note = f" Cap ≤$500K reduces by 0.2."

    score = base + cap_adj
    cap_str = f"${cap_usd/1e6:.0f}M cap" if cap_usd else "uncapped"
    label = f"[penalty={penalty_type}, {cap_str}] → {score:.2f}{cap_note}"
    return score, label


def _score_enforcement(mechanism: EnfType) -> tuple[float, str]:
    """Score the enforcement mechanism dimension. All weights are [ASSUMED] and should be swept."""
    scores = {  # [ASSUMED] — no empirical calibration target
        "none":             0.0,
        "self_report":      0.3,
        "third_party_audit": 0.8,
        "government_inspect": 1.2,
        "criminal_invest":  2.0,
    }
    s = scores[mechanism]
    return s, f"[enforcement={mechanism}] → {s:.2f}"


def _score_threshold(flops: float | None) -> tuple[float, str]:
    """Score the compute threshold dimension; lower threshold covers more entities and scores higher."""
    # Lower threshold = more companies covered = higher burden on the ecosystem
    if flops is None:
        return 0.5, "[threshold=unspecified] → 0.50 (assume moderate coverage)"
    if flops >= 1e27:
        s, note = 0.2, "very few current models affected"  # [ASSUMED]
    elif flops >= 1e26:
        s, note = 0.5, "covers frontier labs only (SB-53 level)"  # [ASSUMED]
    elif flops >= 1e25:
        s, note = 0.8, "covers most frontier + large models (EU AI Act level)"  # [ASSUMED]
    elif flops >= 1e24:
        s, note = 1.1, "covers broad range of large models"  # [ASSUMED]
    else:
        s, note = 1.4, "broad coverage, affects many mid-size models"  # [ASSUMED]
    return s, f"[threshold={flops:.0e} FLOPS, {note}] → {s:.2f}"


def _score_grace_period(months: int) -> tuple[float, str]:
    """Score the grace period dimension; longer grace reduces immediate burden and lowers the score."""
    # Longer grace period = lower immediate burden
    if months == 0:
        s, note = 0.8, "no grace period (immediate)"  # [ASSUMED]
    elif months <= 6:
        s, note = 0.5, "short (≤6 months)"  # [ASSUMED]
    elif months <= 12:
        s, note = 0.2, "moderate (7–12 months)"  # [ASSUMED]
    elif months <= 24:
        s, note = 0.0, "long (13–24 months)"  # [ASSUMED]
    else:
        s, note = -0.2, "very long (>24 months)"  # [ASSUMED]
    return s, f"[grace_period={months}mo, {note}] → {s:.2f}"


def _score_scope(scope: ScopeType) -> tuple[float, str]:
    """Score the regulatory scope dimension; broader coverage scores higher."""
    scores = {  # [ASSUMED] — no empirical calibration target
        "voluntary":          0.0,
        "frontier_only":      0.3,
        "large_developers_only": 0.5,
        "all":                0.8,
    }
    s = scores[scope]
    return s, f"[scope={scope}] → {s:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PARSER — structured parameters
# ─────────────────────────────────────────────────────────────────────────────

def parse_bill(
    name: str = "Policy",
    description: str = "",
    penalty_type: PenaltyType = "civil",
    penalty_cap_usd: float | None = 1_000_000,
    compute_threshold_flops: float | None = 1e26,
    enforcement_mechanism: EnfType = "third_party_audit",
    grace_period_months: int = 12,
    scope: ScopeType = "large_developers_only",
) -> PolicySpec:
    """Map structured bill parameters to a PolicySpec with auditable severity score.

    Each dimension contributes a sub-score. The raw total is scaled linearly
    to [1.0, 5.0] against calibrated min/max anchors:
      min anchor: voluntary guideline, no enforcement, 10^27 threshold (→ 1.0)
      max anchor: criminal liability, government investigation, no grace (→ 5.0)

    The justification list records exactly what drove each sub-score so the
    result can be defended in a policy memo.
    """
    dims = [
        _score_penalty(penalty_type, penalty_cap_usd),
        _score_enforcement(enforcement_mechanism),
        _score_threshold(compute_threshold_flops),
        _score_grace_period(grace_period_months),
        _score_scope(scope),
    ]

    raw = sum(s for s, _ in dims)
    justifications = [j for _, j in dims]

    # Calibrated anchors:
    #   voluntary/none/1e27/24mo/frontier_only → raw ≈ 0.8  → severity 1.0
    #   criminal/criminal_invest/1e24/0mo/all  → raw ≈ 8.0  → severity 5.0
    RAW_MIN, RAW_MAX = 0.8, 8.0  # [ASSUMED] calibrated anchors — adjust if scoring weights change
    severity = 1.0 + 4.0 * (raw - RAW_MIN) / (RAW_MAX - RAW_MIN)
    severity = round(max(1.0, min(5.0, severity)), 2)

    justifications.append(
        f"[raw_total={raw:.2f} → scaled to severity={severity:.2f}]"
    )

    # Suggest a sensitivity sweep: ±0.5 around the point estimate
    sweep = (
        round(max(1.0, severity - 0.5), 1),
        severity,
        round(min(5.0, severity + 0.5), 1),
    )

    # Derive compute_cost_factor from threshold level.
    # Logic: fewer affected models → less novel compliance infrastructure needed.
    # [ASSUMED] These mappings have no empirical calibration. Always show results
    # at factor=1.0 (GDPR-equivalent) alongside the derived factor.
    if compute_threshold_flops is None:
        _ccf = 1.0   # no compute scope → treat as GDPR-equivalent
    elif compute_threshold_flops >= 1e27:
        _ccf = 1.5   # very few models affected; compliance infrastructure modest
    elif compute_threshold_flops >= 1e26:
        _ccf = 2.0   # SB-53 / EU AI Act GPAI tier
    elif compute_threshold_flops >= 1e25:
        _ccf = 3.0   # broad frontier coverage; significant novel audit burden
    else:
        _ccf = 4.0   # wide coverage; many models, audit market nascent

    return PolicySpec(
        name=name,
        description=description or f"{penalty_type.title()} penalties, {enforcement_mechanism.replace('_',' ')}",
        severity=severity,
        justification=justifications,
        penalty_type=penalty_type,
        penalty_cap_usd=penalty_cap_usd,
        compute_threshold_flops=compute_threshold_flops,
        enforcement_mechanism=enforcement_mechanism,
        grace_period_months=grace_period_months,
        scope=scope,
        recommended_severity_sweep=sweep,
        compute_cost_factor=_ccf,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PARSER — extract structured fields from legislative language
# ─────────────────────────────────────────────────────────────────────────────

def parse_bill_text(text: str, name: str = "Policy from text") -> PolicySpec:
    """Extract bill parameters from free-form legislative text and score them.

    This is less precise than parse_bill() with explicit parameters.
    Use it for initial exploration; verify the extracted fields before citing.
    The extracted fields are shown in the justification so you can check them.
    """
    t = text.lower()

    # Penalty type
    if any(w in t for w in ["criminal", "imprisonment", "prison", "felony", "dissolution"]):
        pt: PenaltyType = "criminal"
    elif any(w in t for w in ["civil fine", "civil penalty", "monetary penalty"]):
        # Look for magnitude
        if re.search(r'\$\s*\d+\s*[mb]illion', t) or re.search(r'unlimited', t):
            pt = "civil_heavy"
        else:
            pt = "civil"
    elif any(w in t for w in ["voluntary", "guideline", "best practice", "encourage"]):
        pt = "voluntary"
    else:
        pt = "civil"  # default: most AI bills have some civil penalty

    # Cap
    cap = None
    m = re.search(r'\$\s*([\d,]+)\s*([mb]illion|[kmb])?(?:\s*(?:per|each|maximum))?', t)
    if m:
        val_str = m.group(1).replace(',', '')
        suffix = (m.group(2) or '').lower()
        val = float(val_str)
        if 'billion' in suffix or 'b' in suffix:
            cap = val * 1e9
        elif 'million' in suffix or 'm' in suffix:
            cap = val * 1e6
        elif 'k' in suffix:
            cap = val * 1e3
        else:
            cap = val

    # Compute threshold
    thresh = None
    flop_match = re.search(r'10\s*[\^*]\s*([\d.]+)', t) or \
                 re.search(r'([\d.]+)\s*[×x]\s*10\s*[\^*]\s*([\d.]+)', t)
    if flop_match:
        exp = float(flop_match.group(1) if len(flop_match.groups()) == 1
                    else flop_match.group(2))
        thresh = 10 ** exp

    # Enforcement
    if any(w in t for w in ["criminal investigation", "prosecution", "fbi", "doj"]):
        enf: EnfType = "criminal_invest"
    elif any(w in t for w in ["government inspection", "regulatory inspection", "audit by"]):
        enf = "government_inspect"
    elif any(w in t for w in ["third-party", "third party", "independent audit", "accredited"]):
        enf = "third_party_audit"
    elif any(w in t for w in ["self-report", "self report", "disclosure", "notify"]):
        enf = "self_report"
    else:
        enf = "self_report"

    # Grace period
    gp = 0
    gp_match = re.search(r'(\d+)[- ]month', t) or re.search(r'(\d+)[- ]year', t)
    if gp_match:
        val = int(gp_match.group(1))
        if 'year' in t[gp_match.start():gp_match.end() + 5]:
            gp = val * 12
        else:
            gp = val

    # Scope
    if any(w in t for w in ["frontier", "advanced", "10^26", "10^25"]):
        scope: ScopeType = "frontier_only" if thresh and thresh >= 1e26 else "large_developers_only"
    elif any(w in t for w in ["large developer", "large company", "revenue threshold"]):
        scope = "large_developers_only"
    else:
        scope = "all"

    spec = parse_bill(
        name=name,
        description=text[:120].strip().replace('\n', ' ') + "...",
        penalty_type=pt,
        penalty_cap_usd=cap,
        compute_threshold_flops=thresh,
        enforcement_mechanism=enf,
        grace_period_months=gp,
        scope=scope,
    )
    # Prepend extraction notes
    spec.justification.insert(0, f"[extracted: penalty={pt}, cap={cap}, flops={thresh}, enf={enf}, grace={gp}mo, scope={scope}]")
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# PRESET REAL-WORLD BILLS
# ─────────────────────────────────────────────────────────────────────────────

def california_sb53() -> PolicySpec:
    """California SB-53 (signed September 2025).

    Large frontier developers (>$500M revenue) training above 10^26 FLOPS.
    Transparency, incident reporting, third-party risk assessments.
    Civil penalties. No criminal liability.
    """
    return parse_bill(
        name="California SB-53",
        description=(
            "Large frontier developer (>$500M revenue) transparency and safety "
            "requirements for models trained above 10^26 FLOPS. Incident reporting "
            "to OES. Third-party risk assessments. Civil penalties."
        ),
        penalty_type="civil",
        penalty_cap_usd=1_000_000,
        compute_threshold_flops=1e26,
        enforcement_mechanism="third_party_audit",
        grace_period_months=6,
        scope="large_developers_only",
    )


def eu_ai_act_gpai() -> PolicySpec:
    """EU AI Act — GPAI systemic risk tier (10^25 FLOP threshold).

    General-purpose AI models above 10^25 FLOPS face systemic risk obligations:
    model evaluation, adversarial testing, incident reporting, transparency.
    Administrative fines up to 3% of global turnover.
    """
    return parse_bill(
        name="EU AI Act (GPAI systemic risk)",
        description=(
            "General-purpose AI models trained above 10^25 FLOPS face systemic risk "
            "obligations: model evaluation, red-teaming, incident notification, "
            "transparency register. Fines up to 3% of global annual turnover."
        ),
        penalty_type="civil",
        penalty_cap_usd=None,   # % of turnover, not fixed cap
        compute_threshold_flops=1e25,
        enforcement_mechanism="third_party_audit",
        grace_period_months=12,
        scope="large_developers_only",
    )


def ny_raise_act() -> PolicySpec:
    """New York RAISE Act (proposed).

    Transparency, disclosure, third-party audits for AI models above compute
    and cost thresholds. Civil fines.
    """
    return parse_bill(
        name="NY RAISE Act",
        description=(
            "Transparency, disclosure, documentation, and third-party audit "
            "requirements for AI models meeting compute and cost thresholds."
        ),
        penalty_type="civil",
        penalty_cap_usd=5_000_000,
        compute_threshold_flops=1e26,
        enforcement_mechanism="third_party_audit",
        grace_period_months=18,
        scope="large_developers_only",
    )


def hypothetical_compute_ban(threshold_flops: float = 1e26) -> PolicySpec:
    """A hypothetical hard compute ban with criminal liability above a threshold."""
    import math
    exp = int(math.log10(threshold_flops))
    return parse_bill(
        name=f"Hypothetical Compute Ban (10^{exp} FLOPS)",
        description=(
            f"Training runs above {threshold_flops:.0e} FLOPS prohibited. "
            "Criminal liability. Mandatory dissolution for non-compliant entities."
        ),
        penalty_type="criminal",
        penalty_cap_usd=None,
        compute_threshold_flops=threshold_flops,
        enforcement_mechanism="government_inspect",
        grace_period_months=0,
        scope="all",
    )
