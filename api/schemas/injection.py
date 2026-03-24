"""Pydantic schemas for adversarial injection (influence scenario).

Derived from policylab/v2/influence/adversarial.py.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class InjectionConfig(BaseModel):
    injection_rate: float = Field(0.05, ge=0.0, le=1.0)
    injection_direction: float = Field(1.0, ge=-1.0, le=1.0)
    injection_magnitude: float = Field(0.08, ge=0.0, le=1.0)
    injection_start_round: int = Field(1, ge=1)


class InjectRequest(BaseModel):
    policy_name: str
    policy_description: str
    policy_severity: float = Field(..., ge=1.0, le=5.0)
    n_population: int = Field(1000, ge=1, le=20000)
    num_rounds: int = Field(8, ge=1, le=64)
    seed: int = 42
    injection: InjectionConfig = Field(default_factory=InjectionConfig)


class InjectionResult(BaseModel):
    baseline_compliance: float
    injected_compliance: float
    compliance_delta: float
    baseline_relocation: float
    injected_relocation: float
    relocation_delta: float
    resilience_score: float
    injection_params: dict
    round_compliance_baseline: list[float]
    round_compliance_injected: list[float]
