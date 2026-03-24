"""Adapter between the FastAPI layer and the simulation engine.

Converts Pydantic request schemas → run_hybrid_simulation() kwargs,
then converts HybridSimResult back to a serialisable response dict.
The policylab/ package is never modified here — this is a pure wrapper.
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


def _build_hybrid_config(req: SimulateRequest, parameter_overrides: dict | None = None):
    from swarmcast.v2.simulation.hybrid_loop import HybridSimConfig

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
        "parameter_overrides": parameter_overrides,
    }
    if req.config.type_distribution is not None:
        kwargs["type_distribution"] = req.config.type_distribution

    return HybridSimConfig(**kwargs)


def _run_simulation_sync(req: SimulateRequest, swarm_result=None) -> SimulateResponse:
    from swarmcast.v2.simulation.hybrid_loop import run_hybrid_simulation

    t0 = time.perf_counter()
    parameter_overrides = (
        swarm_result.parameter_adjustments.model_dump() if swarm_result else None
    )
    config = _build_hybrid_config(req, parameter_overrides=parameter_overrides)
    result = run_hybrid_simulation(
        req.policy_name,
        req.policy_description,
        req.policy_severity,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    moments_dict = dataclasses.asdict(result.simulated_moments)

    # Merge ai_investment_index (0–100 scale) from stock_history into each
    # round_summary so the frontend trajectory chart can render it.
    # Normalise to 0–1 to match compliance_rate / relocation_rate scale.
    stock_by_round: dict[int, dict] = {
        s.get("round", -1): s for s in (result.stock_history or [])
    }
    merged_round_summaries = []
    for rs in result.round_summaries:
        rnum = rs.get("round", -1)
        stock = stock_by_round.get(rnum, {})
        raw_invest = stock.get("ai_investment_index", 50.0)
        merged_round_summaries.append({
            **rs,
            "ai_investment_index": round(float(raw_invest) / 100.0, 4),
        })

    return SimulateResponse(
        policy_name=result.policy_name,
        policy_severity=float(result.policy_severity),
        round_summaries=_sanitize(merged_round_summaries),
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
        swarm_result=swarm_result,
    )


async def run_simulation(req: SimulateRequest) -> SimulateResponse:
    from api.config import settings
    from api.services.swarm_calibration_service import run_swarm

    swarm_result = None
    if req.use_swarm_elicitation:
        api_key = req.openai_api_key or settings.get_openai_key()
        if not api_key:
            raise ValueError(
                "Swarm elicitation requires an OpenAI API key. "
                "Provide it in the API key field or set OPENAI_API_KEY in the environment."
            )
        swarm_result = await run_swarm(
            policy_name=req.policy_name,
            policy_description=req.policy_description,
            policy_severity=req.policy_severity,
            api_key=api_key,
        )

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, _run_simulation_sync, req, swarm_result
    )
