"""Adapter between FastAPI schemas and the policylab simulation engine.

Never modifies policylab/. Converts Pydantic request → policylab kwargs,
calls run_hybrid_simulation(), converts HybridSimResult → response dict.
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Ensure policylab is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from api.schemas.simulation import SimulateRequest, SimulateResponse, SimulatedMoments, RunMetadata


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

_executor = ThreadPoolExecutor(max_workers=4)


def _build_hybrid_config(req: SimulateRequest):
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig

    kwargs: dict[str, Any] = {
        "n_population": req.config.n_population,
        "num_rounds": req.config.num_rounds,
        "spillover_factor": req.config.spillover_factor,
        "seed": req.config.seed,
        "use_network": req.config.use_network,
        "use_vectorized": req.config.use_vectorized,
        "source_jurisdiction": req.config.source_jurisdiction,
        "destination_jurisdictions": req.config.destination_jurisdictions,
        "relocation_temperature": req.config.relocation_temperature,
        "adversarial_injection_rate": req.config.adversarial_injection_rate,
        "adversarial_injection_direction": req.config.adversarial_injection_direction,
        "adversarial_injection_magnitude": req.config.adversarial_injection_magnitude,
        "adversarial_injection_start_round": req.config.adversarial_injection_start_round,
        "compute_cost_factor": req.config.compute_cost_factor,
        "hk_epsilon": req.config.hk_epsilon,
        "verbose": False,
    }
    if req.config.type_distribution is not None:
        kwargs["type_distribution"] = req.config.type_distribution

    return HybridSimConfig(**kwargs)


def _run_simulation_sync(req: SimulateRequest) -> SimulateResponse:
    from policylab.v2.simulation.hybrid_loop import run_hybrid_simulation

    t0 = time.perf_counter()
    config = _build_hybrid_config(req)
    result = run_hybrid_simulation(
        req.policy_name,
        req.policy_description,
        req.policy_severity,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    moments_dict = dataclasses.asdict(result.simulated_moments)

    return SimulateResponse(
        policy_name=result.policy_name,
        policy_severity=float(result.policy_severity),
        round_summaries=_sanitize(result.round_summaries),
        stock_history=_sanitize(result.stock_history),
        final_stocks=_sanitize(result.final_stocks),
        final_population_summary=_sanitize(result.final_population_summary),
        network_statistics=_sanitize(result.network_statistics),
        network_hubs=_sanitize(result.network_hubs),
        simulated_moments=SimulatedMoments(**moments_dict),
        smm_distance_to_gdpr=result.smm_distance_to_gdpr,
        jurisdiction_summary=_sanitize(result.jurisdiction_summary),
        run_metadata=RunMetadata(
            duration_ms=elapsed_ms,
            seed=result.seed,
            n_population=req.config.n_population,
        ),
        event_log=result.event_log,
    )


async def run_simulation(req: SimulateRequest) -> SimulateResponse:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run_simulation_sync, req)
