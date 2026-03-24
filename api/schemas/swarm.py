"""Pydantic schemas for swarm elicitation results.

Swarm elicitation is an [SWARM-ELICITED] epistemic tier — distinct from both
[GROUNDED] empirical calibration (DLA Piper / GDPR data) and [ASSUMED] model
parameters. LLM-elicited responses represent model priors filtered through
training data, not measured behavioural observations. Treat accordingly.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SwarmAgentResponse(BaseModel):
    """Raw response from a single swarm agent."""
    persona: str                          # e.g. "Seed-stage startup, 12 staff"
    agent_type: str                       # "startup" | "large_company" | ...
    compliance_factor: float              # multiplier vs GDPR baseline (1.0 = same)
    primary_action: Literal["comply", "relocate", "evade", "lobby"]
    relocation_pressure: Literal["low", "medium", "high"]
    reasoning: str                        # raw LLM reasoning (1-2 sentences)


class SwarmTypeResult(BaseModel):
    """Aggregated result for one agent type."""
    agent_type: str
    n_agents: int
    mean_compliance_factor: float         # mean of compliance_factor across agents
    compliance_factor_cv: float           # coefficient of variation (std/mean)
    dominant_action: str                  # most common primary_action
    relocation_pressure_dist: dict[str, int]   # {"low": 3, "medium": 3, "high": 2}
    confidence: Literal["high", "medium", "low"]
    # high: CV < 0.2 | medium: 0.2–0.4 | low: CV > 0.4 → fall back to baseline
    applied_lambda_multiplier: float      # actual multiplier applied (1.0 if low confidence)
    applied_threshold_shift: float        # actual shift applied (0.0 if low confidence)
    agents: list[SwarmAgentResponse]


class ParameterAdjustments(BaseModel):
    """The actual adjustments passed to PopulationArray.generate().

    All values are bounded to ±30% / ±15 units of empirical baseline.
    Low-confidence types fall back to 1.0 / 0.0 (no adjustment).
    """
    lambda_multipliers: dict[str, float] = Field(default_factory=dict)
    threshold_shifts: dict[str, float]   = Field(default_factory=dict)


class SwarmResult(BaseModel):
    """Full swarm elicitation result attached to a SimulateResponse."""
    epistemic_tag: Literal["SWARM-ELICITED"] = "SWARM-ELICITED"
    policy_name: str
    n_total_agents: int
    type_results: list[SwarmTypeResult]
    parameter_adjustments: ParameterAdjustments
    llm_model: str
    elapsed_seconds: float
    warning: str = (
        "Swarm-elicited adjustments are LLM priors, not empirical observations. "
        "Do not treat as equivalent to GDPR-grounded calibration data."
    )
