"""Adapter between FastAPI and the policylab ingestion pipeline.

Wraps ingest() / ingest_text() and converts IngestResult → response dicts.
Never modifies policylab/.
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.schemas.policy import PolicySpec
from api.schemas.simulation import SimulateRequest, SimConfigRequest
from api.schemas.upload import ExtractionResult, ExtractedField

_executor = ThreadPoolExecutor(max_workers=2)


def _extracted_field(ef) -> ExtractedField:
    return ExtractedField(
        value=ef.value,
        confidence=ef.confidence,
        source_passage=ef.source_passage,
        extraction_method=ef.extraction_method,
        epistemic_tag=ef.epistemic_tag,
    )


def _convert_extraction(er) -> ExtractionResult:
    return ExtractionResult(
        policy_name=_extracted_field(er.policy_name),
        policy_description=_extracted_field(er.policy_description),
        penalty_type=_extracted_field(er.penalty_type),
        penalty_cap_usd=_extracted_field(er.penalty_cap_usd),
        compute_threshold_flops=_extracted_field(er.compute_threshold_flops),
        enforcement_mechanism=_extracted_field(er.enforcement_mechanism),
        grace_period_months=_extracted_field(er.grace_period_months),
        scope=_extracted_field(er.scope),
        source_jurisdiction=_extracted_field(er.source_jurisdiction),
        has_sme_provisions=_extracted_field(er.has_sme_provisions),
        has_frontier_lab_focus=_extracted_field(er.has_frontier_lab_focus),
        has_research_exemptions=_extracted_field(er.has_research_exemptions),
        has_investor_provisions=_extracted_field(er.has_investor_provisions),
        estimated_n_regulated=_extracted_field(er.estimated_n_regulated),
        key_provisions=[list(p) for p in er.key_provisions],
        extraction_method_used=er.extraction_method_used,
        model_used=er.model_used,
        unresolved_provisions=er.unresolved_provisions,
    )


def _convert_spec(ps) -> PolicySpec:
    d = dataclasses.asdict(ps)
    # recommended_severity_sweep is a tuple — convert to list
    if "recommended_severity_sweep" in d and isinstance(d["recommended_severity_sweep"], (tuple, list)):
        d["recommended_severity_sweep"] = list(d["recommended_severity_sweep"])
    return PolicySpec(**d)


def _ingest_file_sync(file_path: str, api_key: str | None, model: str):
    from policylab.v2.ingest.pipeline import ingest
    return ingest(file_path, api_key=api_key, model=model, verbose=False)


def _ingest_text_sync(text: str, name: str, api_key: str | None, model: str):
    from policylab.v2.ingest.pipeline import ingest_text
    return ingest_text(text, name=name, api_key=api_key, model=model, verbose=False)


def ingest_result_to_simulate_request(ingest_result) -> SimulateRequest:
    """Convert IngestResult.config (plain dict) to a SimulateRequest."""
    config_dict = ingest_result.config.copy()
    # Remove keys not in SimConfigRequest
    allowed = set(SimConfigRequest.model_fields.keys())
    filtered = {k: v for k, v in config_dict.items() if k in allowed}
    return SimulateRequest(
        policy_name=ingest_result.spec.name,
        policy_description=ingest_result.spec.description,
        policy_severity=ingest_result.spec.severity,
        config=SimConfigRequest(**filtered),
    )


async def ingest_file(
    file_path: str,
    api_key: str | None = None,
    model: str = "gpt-4o",
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, _ingest_file_sync, file_path, api_key, model
    )


async def ingest_text(
    text: str,
    name: str = "Uploaded document",
    api_key: str | None = None,
    model: str = "gpt-4o",
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, _ingest_text_sync, text, name, api_key, model
    )
