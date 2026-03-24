"""Pydantic schemas for async jobs (evidence pack background task)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ConfidenceBand(BaseModel):
    round: int
    mean: float
    p10: float
    p90: float


class EvidencePackResult(BaseModel):
    severity_levels: list[float]
    bands: dict[str, list[ConfidenceBand]]  # keyed by metric name


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "error"]
    progress: float = 0.0  # 0.0–1.0
    result: EvidencePackResult | None = None
    error: str | None = None


class EvidencePackRequest(BaseModel):
    policy_name: str
    policy_description: str
    base_severity: float
    sim_config: dict[str, Any] = {}
    ensemble_size: int = 3


class CompareRequest(BaseModel):
    policies: list[dict[str, Any]]  # list of SimulateRequest-compatible dicts
