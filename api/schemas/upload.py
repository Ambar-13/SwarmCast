"""Pydantic schemas for document upload and extraction results.

Derived from policylab/v2/ingest/provision_extractor.py and pipeline.py.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from api.schemas.policy import PolicySpec
from api.schemas.simulation import SimulateResponse


class ExtractedField(BaseModel):
    value: Any
    confidence: float
    source_passage: str
    extraction_method: str
    epistemic_tag: str  # GROUNDED | DIRECTIONAL | ASSUMED


class ExtractionResult(BaseModel):
    policy_name: ExtractedField
    policy_description: ExtractedField
    penalty_type: ExtractedField
    penalty_cap_usd: ExtractedField
    compute_threshold_flops: ExtractedField
    enforcement_mechanism: ExtractedField
    grace_period_months: ExtractedField
    scope: ExtractedField
    source_jurisdiction: ExtractedField
    has_sme_provisions: ExtractedField
    has_frontier_lab_focus: ExtractedField
    has_research_exemptions: ExtractedField
    has_investor_provisions: ExtractedField
    estimated_n_regulated: ExtractedField
    key_provisions: list[list[str]]  # list of [provision_text, source_id] pairs
    extraction_method_used: str
    model_used: str | None
    unresolved_provisions: list[str]


class UploadResponse(BaseModel):
    document_name: str
    spec: PolicySpec
    extraction: ExtractionResult
    warnings: list[str]
    elapsed_seconds: float
    result: SimulateResponse
    confidence_summary: str
