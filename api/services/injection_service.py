"""Adapter for adversarial injection scenario.

Wraps policylab/v2/influence/adversarial.py run_with_injection().
"""
from __future__ import annotations

import asyncio
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.schemas.injection import InjectRequest, InjectionResult

_executor = ThreadPoolExecutor(max_workers=2)


def _run_injection_sync(req: InjectRequest) -> InjectionResult:
    from policylab.v2.influence.adversarial import run_with_injection

    result = run_with_injection(
        policy_name=req.policy_name,
        policy_description=req.policy_description,
        policy_severity=req.policy_severity,
        injection_rate=req.injection.injection_rate,
        injection_direction=req.injection.injection_direction,
        injection_magnitude=req.injection.injection_magnitude,
        injection_start_round=req.injection.injection_start_round,
        n_population=req.n_population,
        num_rounds=req.num_rounds,
        seed=req.seed,
    )

    return InjectionResult(
        baseline_compliance=result.baseline_compliance,
        injected_compliance=result.injected_compliance,
        compliance_delta=result.compliance_delta,
        baseline_relocation=result.baseline_relocation,
        injected_relocation=result.injected_relocation,
        relocation_delta=result.relocation_delta,
        resilience_score=result.resilience_score,
        injection_params=result.injection_params,
        round_compliance_baseline=result.round_compliance_baseline,
        round_compliance_injected=result.round_compliance_injected,
    )


async def run_injection(req: InjectRequest) -> InjectionResult:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run_injection_sync, req)
