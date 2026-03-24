"""POST /inject — adversarial influence scenario."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.injection import InjectRequest, InjectionResult
from api.services import injection_service

router = APIRouter()


@router.post("/inject", response_model=InjectionResult)
async def inject(req: InjectRequest) -> InjectionResult:
    try:
        return await injection_service.run_injection(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
