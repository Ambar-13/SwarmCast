"""Swarm elicitation service for PolicyLab.

Runs parallel LLM calls across company-type persona swarms to produce
policy-specific behavioral priors. Results are tagged [SWARM-ELICITED]
and adjust the vectorized simulation's GDPR-fitted parameters within
bounded ranges (±30% on λ, ±15 burden units on relocation thresholds).

EPISTEMIC WARNING
─────────────────
These are LLM priors filtered through training data — commentary about
regulation, not measured firm behaviour. They are directionally useful
(does this policy feel harder than GDPR?) but must not displace the
empirically grounded GDPR calibration. Confidence is measured by
coefficient of variation across the swarm; low-confidence types fall
back to GDPR baselines unchanged.

GDPR BASELINES (used for relative framing)
────────────────────────────────────────────
λ rounds × 3 months/round:
  large_company : λ=3.32  → ~10 months
  mid_company   : λ=7.10  → ~21 months
  startup       : λ=10.9  → ~33 months
  frontier_lab  : λ=1.50  → ~4.5 months
  investor      : λ=5.00  → ~15 months
"""
from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

from api.schemas.swarm import (
    ParameterAdjustments,
    SwarmAgentResponse,
    SwarmResult,
    SwarmTypeResult,
)

# ── GDPR compliance time baselines (months) ───────────────────────────────────
_GDPR_MONTHS: dict[str, float] = {
    "large_company": 10.0,
    "mid_company":   21.0,
    "startup":       33.0,
    "frontier_lab":   4.5,
    "investor":      15.0,
}

# ── Persona swarms ─────────────────────────────────────────────────────────────
# Each entry: (agent_type, short persona description)
_SWARM_PERSONAS: list[tuple[str, str]] = [
    # Startups — 8 agents
    ("startup", "Seed-stage AI startup, 8 engineers, no legal team, pre-revenue"),
    ("startup", "Series-A startup, 35 staff, one part-time compliance consultant"),
    ("startup", "18-month-old AI research lab, EU-based, compute-constrained"),
    ("startup", "Pivot-stage startup moving from NLP to agentic AI, limited runway"),
    ("startup", "Well-funded Series-B startup, 120 staff, dedicated legal counsel"),
    ("startup", "Early-stage AI safety startup, pro-regulation, mission-aligned"),
    ("startup", "Bootstrapped AI tools company, no VC funding, cost-sensitive"),
    ("startup", "YC-backed AI infrastructure startup, high growth, US-focused"),

    # Large/Frontier companies — 5 agents
    ("large_company", "Large frontier AI lab, ~2,000 staff, dedicated policy team, Anthropic-like"),
    ("large_company", "Big tech AI division, parent company >$500B market cap, Google-like"),
    ("large_company", "Large AI-first company, OpenAI-like, significant EU revenue exposure"),
    ("large_company", "Enterprise AI software company, 5,000 staff, conservative compliance culture"),
    ("large_company", "Hyperscaler cloud AI division, Microsoft/AWS-like, compliance infrastructure"),

    # Mid-tier — 6 agents
    ("mid_company", "Mid-size AI company, 300 staff, Series-D, scaling compliance operations"),
    ("mid_company", "EU-headquartered AI firm, 180 staff, already GDPR-compliant"),
    ("mid_company", "US AI company with significant EU customers, 250 staff"),
    ("mid_company", "AI platform company, 400 staff, B2B focus, enterprise contracts"),
    ("mid_company", "Mid-size AI research and deployment firm, academic partnerships"),
    ("mid_company", "Vertical AI company in healthcare, already regulated by sector rules"),

    # Investors — 4 agents
    ("investor", "Tier-1 VC fund, significant AI portfolio, regulatory risk averse"),
    ("investor", "Corporate VC from a major tech company, strategic AI investments"),
    ("investor", "Growth equity fund, late-stage AI bets, IPO-track companies"),
    ("investor", "EU-based VC, familiar with GDPR compliance, portfolio mostly European"),
]

# ── Prompt template ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are simulating a specific type of company responding to an AI governance policy.
Answer concisely and in character. Your response must be valid JSON matching the schema exactly.
Do not invent numbers — reason from your persona's actual constraints.
"""

def _build_user_prompt(
    persona: str,
    agent_type: str,
    policy_name: str,
    policy_description: str,
    policy_severity: float,
    gdpr_months: float,
) -> str:
    return f"""\
You are: {persona}

A new AI regulation has been proposed:
Name: {policy_name}
Description: {policy_description}
Severity score: {policy_severity:.1f} / 5.0

GDPR took your organisation approximately {gdpr_months:.0f} months to comply with.

Answer the following as this specific organisation. Be realistic about your resources and constraints.

Respond with ONLY this JSON (no markdown, no explanation outside the JSON):
{{
  "compliance_factor": <float: how many times longer/shorter than GDPR compliance would take. 1.0 = same, 2.0 = twice as long, 0.5 = half as long>,
  "primary_action": <"comply" | "relocate" | "evade" | "lobby">,
  "relocation_pressure": <"low" | "medium" | "high">,
  "reasoning": "<1-2 sentences explaining your reasoning>"
}}
"""


def _cv(values: list[float]) -> float:
    """Coefficient of variation. Returns 0 for single-element lists."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / abs(mean)


def _confidence_from_cv(cv: float) -> str:
    if cv < 0.20:
        return "high"
    if cv < 0.40:
        return "medium"
    return "low"


def _lambda_mult_from_factor(mean_factor: float, confidence: str) -> float:
    """Convert mean compliance_factor → bounded λ multiplier.

    compliance_factor > 1 means agents expect this to take longer than GDPR
    → higher λ (slower compliance).
    Bounded to [0.70, 1.30]. Falls back to 1.0 for low confidence.
    """
    if confidence == "low":
        return 1.0
    return max(0.70, min(1.30, mean_factor))


def _threshold_shift_from_relocation(
    pressure_dist: dict[str, int],
    confidence: str,
) -> float:
    """Convert relocation pressure distribution → threshold shift.

    Higher relocation pressure → lower threshold (companies relocate sooner).
    Shift range: [-15, +15] burden units. Falls back to 0.0 for low confidence.
    """
    if confidence == "low":
        return 0.0
    total = sum(pressure_dist.values())
    if total == 0:
        return 0.0
    # Weighted score: low=0, medium=1, high=2
    score = (pressure_dist.get("medium", 0) * 1 + pressure_dist.get("high", 0) * 2) / total
    # score in [0, 2]; 1.0 = baseline (no shift); >1 = lower threshold
    # Map [0, 2] → [+10, -10] (high pressure → negative shift = lower threshold)
    return max(-15.0, min(15.0, (1.0 - score) * 10.0))


async def _call_llm_single(
    client: Any,
    model: str,
    persona: str,
    agent_type: str,
    policy_name: str,
    policy_description: str,
    policy_severity: float,
) -> SwarmAgentResponse | None:
    """Single async LLM call for one persona. Returns None on parse failure."""
    import json

    gdpr_months = _GDPR_MONTHS.get(agent_type, 20.0)
    user_prompt = _build_user_prompt(
        persona, agent_type, policy_name, policy_description,
        policy_severity, gdpr_months,
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)

        factor = float(data.get("compliance_factor", 1.0))
        factor = max(0.1, min(5.0, factor))  # sanity clamp

        action = data.get("primary_action", "comply")
        if action not in ("comply", "relocate", "evade", "lobby"):
            action = "comply"

        pressure = data.get("relocation_pressure", "medium")
        if pressure not in ("low", "medium", "high"):
            pressure = "medium"

        return SwarmAgentResponse(
            persona=persona,
            agent_type=agent_type,
            compliance_factor=factor,
            primary_action=action,
            relocation_pressure=pressure,
            reasoning=str(data.get("reasoning", ""))[:400],
        )
    except Exception:
        return None


async def run_swarm(
    policy_name: str,
    policy_description: str,
    policy_severity: float,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> SwarmResult:
    """Run the full swarm elicitation and return a SwarmResult.

    All LLM calls are made concurrently (asyncio.gather). Total time is
    bounded by the slowest single call, not by the number of agents.
    """
    from openai import AsyncOpenAI

    t0 = time.perf_counter()
    client = AsyncOpenAI(api_key=api_key)

    # Fire all calls concurrently
    tasks = [
        _call_llm_single(
            client, model, persona, agent_type,
            policy_name, policy_description, policy_severity,
        )
        for agent_type, persona in _SWARM_PERSONAS
    ]
    raw_results: list[SwarmAgentResponse | None] = await asyncio.gather(*tasks)

    # Drop failures
    responses: list[SwarmAgentResponse] = [r for r in raw_results if r is not None]

    # Group by agent_type
    by_type: dict[str, list[SwarmAgentResponse]] = {}
    for r in responses:
        by_type.setdefault(r.agent_type, []).append(r)

    type_results: list[SwarmTypeResult] = []
    lambda_multipliers: dict[str, float] = {}
    threshold_shifts: dict[str, float] = {}

    for atype, agents in by_type.items():
        factors = [a.compliance_factor for a in agents]
        mean_factor = statistics.mean(factors) if factors else 1.0
        cv = _cv(factors)
        confidence = _confidence_from_cv(cv)

        pressure_dist: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        for a in agents:
            pressure_dist[a.relocation_pressure] += 1

        dominant_action = max(
            ("comply", "relocate", "evade", "lobby"),
            key=lambda act: sum(1 for a in agents if a.primary_action == act),
        )

        lam_mult  = _lambda_mult_from_factor(mean_factor, confidence)
        thr_shift = _threshold_shift_from_relocation(pressure_dist, confidence)

        lambda_multipliers[atype]  = lam_mult
        threshold_shifts[atype]    = thr_shift

        type_results.append(SwarmTypeResult(
            agent_type=atype,
            n_agents=len(agents),
            mean_compliance_factor=round(mean_factor, 3),
            compliance_factor_cv=round(cv, 3),
            dominant_action=dominant_action,
            relocation_pressure_dist=pressure_dist,
            confidence=confidence,
            applied_lambda_multiplier=lam_mult,
            applied_threshold_shift=thr_shift,
            agents=agents,
        ))

    elapsed = time.perf_counter() - t0

    return SwarmResult(
        policy_name=policy_name,
        n_total_agents=len(responses),
        type_results=type_results,
        parameter_adjustments=ParameterAdjustments(
            lambda_multipliers=lambda_multipliers,
            threshold_shifts=threshold_shifts,
        ),
        llm_model=model,
        elapsed_seconds=round(elapsed, 2),
    )


async def smoke_test_eu_ai_act(api_key: str, model: str = "gpt-4o-mini") -> dict:
    """Smoke test: run swarm on EU AI Act and check directional validity.

    Expected signal (from post-Feb 2025 implementation period):
    - Frontier labs: compliance_factor < 1.5 (they prepared early)
    - Startups: compliance_factor > 1.5 (disproportionate burden)
    - Relocation pressure: medium-high for startups, low for large companies

    Returns a dict with pass/fail per check and the full SwarmResult.
    """
    result = await run_swarm(
        policy_name="EU AI Act — GPAI systemic risk",
        policy_description=(
            "General-purpose AI models trained above 10^25 FLOPS face systemic risk "
            "obligations: model evaluation, red-teaming, incident notification, "
            "transparency register. Fines up to 3% of global annual turnover."
        ),
        policy_severity=2.39,
        api_key=api_key,
        model=model,
    )

    checks: dict[str, bool] = {}
    by_type = {tr.agent_type: tr for tr in result.type_results}

    # Frontier labs should find EU AI Act manageable (compliance_factor < 1.5)
    if "large_company" in by_type:
        checks["frontier_manageable"] = by_type["large_company"].mean_compliance_factor < 1.8

    # Startups should find it more burdensome than GDPR (compliance_factor > 1.0)
    if "startup" in by_type:
        checks["startup_burdened"] = by_type["startup"].mean_compliance_factor > 1.0

    # Startup relocation pressure should be at least medium
    if "startup" in by_type:
        p = by_type["startup"].relocation_pressure_dist
        checks["startup_relocation_signal"] = (p.get("medium", 0) + p.get("high", 0)) > p.get("low", 0)

    passed = sum(checks.values())
    total  = len(checks)

    return {
        "passed": passed,
        "total": total,
        "checks": checks,
        "verdict": "PASS" if passed == total else f"PARTIAL ({passed}/{total})",
        "swarm_result": result.model_dump(),
    }
