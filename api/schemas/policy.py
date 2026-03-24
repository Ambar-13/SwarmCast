"""Pydantic schemas for PolicySpec and preset policies.

Derived from policylab/v2/policy/parser.py — do not invent fields.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PenaltyType = Literal["none", "voluntary", "civil", "civil_heavy", "criminal"]
EnforcementType = Literal[
    "none", "self_report", "third_party_audit", "government_inspect", "criminal_invest"
]
ScopeType = Literal["all", "large_developers_only", "frontier_only", "voluntary"]


class PolicySpec(BaseModel):
    name: str
    description: str
    severity: float = Field(..., ge=1.0, le=5.0)
    justification: list[str] = []
    penalty_type: PenaltyType = "none"
    penalty_cap_usd: float | None = None
    compute_threshold_flops: float | None = None
    enforcement_mechanism: EnforcementType = "none"
    grace_period_months: int = 0
    scope: ScopeType = "all"
    recommended_n_population: int = 1000
    recommended_num_rounds: int = 16
    recommended_severity_sweep: list[float] = []
    compute_cost_factor: float = 1.0


class PresetPolicy(BaseModel):
    slug: str
    label: str
    spec: PolicySpec
