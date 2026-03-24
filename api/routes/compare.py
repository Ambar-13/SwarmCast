"""POST /compare — run up to 4 simulations in parallel and return all results."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from api.schemas.simulation import SimulateRequest, SimulateResponse
from api.schemas.jobs import CompareRequest
from api.services import simulation_service

router = APIRouter()


@router.post("/compare")
async def compare_policies(req: CompareRequest) -> dict:
    if not (2 <= len(req.policies) <= 4):
        raise HTTPException(
            status_code=400,
            detail="Provide 2–4 policies to compare.",
        )

    # Parse each policy dict into SimulateRequest
    try:
        sim_requests = [SimulateRequest(**p) for p in req.policies]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Run in parallel
    try:
        results: list[SimulateResponse] = await asyncio.gather(
            *[simulation_service.run_simulation(r) for r in sim_requests]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"results": [r.model_dump() for r in results]}
