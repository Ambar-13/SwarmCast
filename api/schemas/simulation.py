"""Pydantic schemas for simulation requests and responses.

Derived from policylab/v2/simulation/hybrid_loop.py — do not invent fields.
Non-serializable HybridSimConfig fields (llm_model, llm_embedder, event_queue)
are excluded from both request and response.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from api.schemas.swarm import SwarmResult


class SimConfigRequest(BaseModel):
    """Serializable subset of HybridSimConfig accepted over the API."""
    n_population: int = Field(1000, ge=1, le=20000)
    num_rounds: int = Field(16, ge=1, le=64)
    spillover_factor: float = Field(0.5, ge=0.0, le=2.0)
    seed: int = 42
    use_network: bool = True
    use_vectorized: bool = True
    source_jurisdiction: str = "EU"
    destination_jurisdictions: list[str] = ["US", "UK", "Singapore", "UAE"]
    relocation_temperature: float = 0.1
    adversarial_injection_rate: float = 0.0
    adversarial_injection_direction: float = 1.0
    adversarial_injection_magnitude: float = 0.08
    adversarial_injection_start_round: int = 1
    compute_cost_factor: float = 1.0
    hk_epsilon: float = 1.0
    type_distribution: dict[str, float] | None = None


class SimulateRequest(BaseModel):
    policy_name: str
    policy_description: str
    policy_severity: float = Field(..., ge=1.0, le=5.0)
    config: SimConfigRequest = Field(default_factory=SimConfigRequest)
    use_swarm_elicitation: bool = False  # runs swarm before simulation if True
    openai_api_key: str | None = None  # overrides env var for this request


class RoundSummary(BaseModel):
    round: int
    compliance_rate: float
    relocation_rate: float
    ai_investment_index: float
    enforcement_contact_rate: float


class SimulatedMoments(BaseModel):
    lobbying_rate: float = 0.0
    compliance_rate_y1: float = 0.0
    relocation_rate: float = 0.0
    sme_compliance_24mo: float = 0.0
    large_compliance_24mo: float = 0.0
    enforcement_rate: float = 0.0
    n_runs: int = 1


class RunMetadata(BaseModel):
    duration_ms: float
    seed: int
    n_population: int


class SimulateResponse(BaseModel):
    policy_name: str
    policy_severity: float
    round_summaries: list[dict[str, Any]]
    stock_history: list[dict[str, Any]]
    final_stocks: dict[str, Any]
    final_population_summary: dict[str, Any]
    network_statistics: dict[str, Any]
    network_hubs: list[dict[str, Any]]
    simulated_moments: SimulatedMoments
    smm_distance_to_gdpr: float | None
    jurisdiction_summary: dict[str, Any]
    run_metadata: RunMetadata
    event_log: list[Any]
    swarm_result: SwarmResult | None = None
