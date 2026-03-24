"""Extract regulatory severity scores (1-5) from policy description text.

Severity 1 = voluntary guidance; 5 = total ban or moratorium.
Keyword scan and structural scan run in parallel; an optional LLM call
supplements both at scenario setup time only, not per-round.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

# PERIODIZATION ANCHOR — must match calibration.py ROUNDS_PER_YEAR = 4
# 1 simulation round = 1 governance review cycle ≈ 3 months = 91.25 days
# All day→round conversions in this file must use this constant.
# See calibration.py lines 116-117 for source of ROUNDS_PER_YEAR.
_DAYS_PER_ROUND: float = 365.0 / 4.0  # = 91.25

if TYPE_CHECKING:
    from concordia.language_model import language_model as lm_lib
    from swarmcast.components.governance_state import Policy


# Each tuple: (keywords_list, severity_score)
# Ordered from highest to lowest so we can short-circuit.
_SEVERITY_KEYWORDS: list[tuple[list[str], float]] = [
    # Severity 5: total bans, existential penalties
    (["moratorium", "total ban", "prohibit all", "ban all",
      "dissolution", "delete all weights", "delete all models",
      "block all", "block at the network level",
      "10 years imprisonment", "years imprisonment"], 5.0),

    # Severity 4: criminal liability, shutdowns, license revocation
    (["criminal liability", "criminal penalty", "imprisonment",
      "mandatory shutdown", "immediate shutdown", "cease operations",
      "revoke license", "suspend operations", "face dissolution"], 4.0),

    # Severity 3: significant financial penalties, mandatory audits
    (["million", "percent of revenue", "percent of global revenue",
      "mandatory audit", "third-party audit", "certification required",
      "annual review", "annual re-certification", "stress test",
      "capital requirements", "reserves equal to"], 3.0),

    # Severity 2: reporting and registration
    (["registration", "register", "notification", "disclosure",
      "reporting requirement", "transparency obligation",
      "self-certify", "self-report"], 2.0),

    # Severity 1: voluntary, non-binding
    (["guidance", "recommendation", "best practice", "voluntary",
      "encouraged", "non-binding", "no penalties", "no enforcement"], 1.0),
]

_NEGATION_WORDS = {"not", "no", "without", "exempt", "except", "excluding"}


def extract_severity_keywords(description: str) -> float:
    """Scan policy description for severity-indicating keywords with negation handling.

    Return a severity score 1–5, or 0.0 if no keywords matched.
    """
    text = description.lower()
    words = text.split()
    max_score = 0.0

    for keywords, score in _SEVERITY_KEYWORDS:
        for kw in keywords:
            pos = text.find(kw)
            if pos < 0:
                continue

            # Check for negation within 5 words before the keyword
            word_idx = len(text[:pos].split()) - 1
            start = max(0, word_idx - 5)
            preceding = words[start:max(0, word_idx + 1)]
            negated = any(w in _NEGATION_WORDS for w in preceding)

            if not negated:
                max_score = max(max_score, score)

    return max_score


def extract_severity_structural(policy: "Policy") -> float:
    """Derive severity from the count and content of the policy's requirements and penalties."""
    reqs = policy.requirements if policy.requirements else []
    pens = policy.penalties if policy.penalties else []

    # Filter out placeholder strings
    reqs = [r for r in reqs if r and "as described" not in r.lower()]
    pens = [p for p in pens if p and "as described" not in p.lower()]

    # [ASSUMED] coefficients — no calibration target (see docstring)
    REQ_WEIGHT = 0.5       # [ASSUMED] per requirement; sweep [0.3, 0.5, 0.8]
    PEN_WEIGHT = 0.7       # [ASSUMED] per penalty; sweep [0.4, 0.7, 1.0]
    base = 1.0 + len(reqs) * REQ_WEIGHT + len(pens) * PEN_WEIGHT

    # Bonus for money amounts in penalties
    pen_text = " ".join(pens).lower()
    if re.search(r'\$?\d+\s*million|\d+\s*percent', pen_text):
        base += 1.0  # [ASSUMED] sweep [0.5, 1.0, 1.5]
    if any(w in pen_text for w in ["criminal", "imprisonment", "jail", "dissolution"]):
        base += 1.0  # [ASSUMED] sweep [0.5, 1.0, 2.0]

    return min(5.0, max(1.0, base))


_LLM_SEVERITY_PROMPT = """\
Rate this policy's regulatory severity from 1 to 5.

1 = voluntary guidance, no enforcement
2 = registration or reporting requirements
3 = mandatory compliance with moderate financial penalties
4 = strict enforcement with criminal penalties or mandatory shutdowns
5 = total ban, moratorium, or dissolution of non-compliant entities

Policy: {description}

Penalties: {penalties}

Reply with ONLY a single number (1, 2, 3, 4, or 5). Nothing else."""


def _extract_severity_llm(policy: "Policy", model: "lm_lib.LanguageModel") -> float:
    """Rate policy severity via one LLM call at setup time; return 0.0 on failure."""
    pen_text = "; ".join(str(p) for p in policy.penalties[:5]) if policy.penalties else "None specified"
    desc = policy.description[:500] if policy.description else "No description"

    prompt = _LLM_SEVERITY_PROMPT.format(description=desc, penalties=pen_text)

    try:
        result = model.sample_text(prompt, max_tokens=5, temperature=0.0)
        result = result.strip()
        for char in result:
            if char in "12345":
                return float(char)
    except Exception as e:
        import warnings
        warnings.warn(
            f"_extract_severity_llm: LLM call failed ({type(e).__name__}: {e}). "
            f"Falling back to keyword/structural severity score.",
            RuntimeWarning,
            stacklevel=2,
        )

    return 0.0


def classify_severity(
    policy: "Policy",
    model: "lm_lib.LanguageModel | None" = None,
) -> float:
    """Classify policy severity by combining keyword and structural scores, with optional LLM fallback.

    Set policy.severity as a side effect and return the final score (default 3.0 if all signals are zero).
    """
    kw_score = extract_severity_keywords(policy.description)
    struct_score = extract_severity_structural(policy)

    scores = [kw_score, struct_score]

    # Fall back to LLM only when the heuristics find nothing.
    # The keyword heuristic is reliable for clear cases (total ban → 5.0, guidance → 1.0).
    if model is not None and max(scores) == 0:
        llm_score = _extract_severity_llm(policy, model)
        scores.append(llm_score)

    final = max(scores) if any(s > 0 for s in scores) else 3.0
    policy.severity = final
    return final


# ---------------------------------------------------------------------------
# Enforcement capacity extraction
# ---------------------------------------------------------------------------

# Patterns for bureau/agency staff sizes embedded in policy descriptions.
# Matches: "500-person Bureau", "bureau of 500 agents", "1,000 staff", etc.
_BUREAU_PATTERNS = [
    # "500-person [optional words] Bureau/Agency/Force"  (up to 5 words between)
    re.compile(r'(\d[\d,]*)-person(?:\s+\w+){0,5}?\s+(?:bureau|agency|office|force|unit)', re.I),
    # "Bureau/Agency of 500 staff/agents"
    re.compile(r'(?:bureau|agency|office|force|unit)\s+of\s+(\d[\d,]*)\s+(?:staff|agents|officers|inspectors)', re.I),
    # "500 staff/agents enforcing/monitoring"
    re.compile(r'(\d[\d,]*)\s+(?:staff|agents|officers|inspectors)\s+(?:enforc|monitor|oversee)', re.I),
    # "enforcing bureau/agency of 500"
    re.compile(r'(?:enforc|monitor)\w*\s+(?:bureau|agency|force)\s+of\s+(\d[\d,]*)', re.I),
]

# Patterns for enforcement deadlines — shorter = more severe.
_DEADLINE_PATTERNS = [
    re.compile(r'(\d+)\s*-?\s*day\s+(?:deadline|window|period|compliance)', re.I),
    re.compile(r'within\s+(\d+)\s+days', re.I),
    re.compile(r'(\d+)\s+days?\s+to\s+(?:comply|register|deregister|delete)', re.I),
]


def extract_enforcement_capacity(
    policy_description: str,
    staff_scaling: float = 0.3,  # [ASSUMED] sweep [0.1, 0.3, 0.5, 1.0]
) -> dict[str, float]:
    """Parse bureau headcount and compliance deadlines from policy text to produce concrete game parameters.

    Return a dict with ``staff_override`` (regulator staff units, 0 if not found) and
    ``grace_rounds_override`` (enforcement grace period in rounds, 0 if not found).
    """
    result: dict[str, float] = {"staff_override": 0.0, "grace_rounds_override": 0.0}

    # Extract bureau / agency size
    for pat in _BUREAU_PATTERNS:
        m = pat.search(policy_description)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                staff = float(raw)
                # [ASSUMED] Staff scaling factor: real-world bureau size → in-game staff units
                # Rationale: regulator resources are capped at 500 for numerical stability;
                # a 500-person real bureau maps to ~150 in-game staff units at 0.3×.
                # NO empirical basis for 0.3×. Alternative interpretation: the in-game
                # "staff" unit represents capacity to run one enforcement action, and
                # a 500-person bureau can sustain ~150 concurrent enforcement tracks.
                # Sensitivity range: [ASSUMED] sweep [0.1, 0.3, 0.5, 1.0]
                # At 0.1: 500 staff → 50 units (under-powered regulator)
                # At 1.0: 500 staff → 500 units (regulator resource = stated headcount)
                STAFF_SCALING = staff_scaling  # [ASSUMED] no calibration target; sweep [0.1, 0.3, 0.5, 1.0]
                result["staff_override"] = min(500.0, staff) * STAFF_SCALING
            except ValueError:
                pass
            break

    # Extract compliance deadline → grace rounds
    # Periodization: 1 round = 91.25 days (quarterly, matching ROUNDS_PER_YEAR=4)
    #   1 round = 91 days (≈ 3 months, one governance review cycle)
    #   This is CONSISTENT with calibration.py ROUNDS_PER_YEAR=4 and the
    #   OECD/Ugur elasticity conversions (annual → quarterly).
    #
    #   Previous code used 14 days/round — a 6.5x error:
    #     90-day window → 5 grace rounds (at 14 days/round)  [WRONG]
    #     90-day window → 0 grace rounds (at 91 days/round)  [CORRECT]
    #
    #   Real-world calibration (at 91 days/round):
    #     90-day window  = 0.99 rounds → 0 grace rounds (immediate compliance)
    #     180-day window = 1.98 rounds → 1 grace round  (~3 months)
    #     365-day window = 4.00 rounds → 3 grace rounds (~9 months)
    DAYS_PER_ROUND = _DAYS_PER_ROUND  # 91.25 days (must match calibration.py)
    for pat in _DEADLINE_PATTERNS:
        m = pat.search(policy_description)
        if m:
            try:
                days = float(m.group(1))
                rounds = max(0.0, round(days / DAYS_PER_ROUND) - 1)
                result["grace_rounds_override"] = rounds
            except ValueError:
                pass
            break

    return result
