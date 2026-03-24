"""Background task: run ensemble simulations for evidence pack (confidence bands).

For each severity level in [base-0.5, base, base+0.5]:
  Run ensemble_size simulations with different seeds.
  Aggregate compliance_rate per round into mean/p10/p90.

Result is stored in the job_registry and polled by the client.
"""
from __future__ import annotations

import asyncio
import sys
import os
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.services.job_registry import registry
from api.schemas.jobs import ConfidenceBand, EvidencePackResult


def _run_one_sync(
    policy_name: str,
    policy_description: str,
    severity: float,
    config_kwargs: dict[str, Any],
    seed: int,
) -> list[dict]:
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation

    allowed_keys = {
        "n_population", "num_rounds", "spillover_factor", "use_network",
        "use_vectorized", "source_jurisdiction", "destination_jurisdictions",
        "relocation_temperature", "compute_cost_factor", "hk_epsilon",
    }
    filtered = {k: v for k, v in config_kwargs.items() if k in allowed_keys}
    config = HybridSimConfig(verbose=False, seed=seed, **filtered)
    result = run_hybrid_simulation(policy_name, policy_description, severity, config=config)
    return result.round_summaries


def _compute_bands(
    all_round_summaries: list[list[dict]],
    metric: str,
) -> list[ConfidenceBand]:
    if not all_round_summaries:
        return []
    n_rounds = max(len(rs) for rs in all_round_summaries)
    bands = []
    for r in range(n_rounds):
        vals = [
            rs[r].get(metric, 0.0)
            for rs in all_round_summaries
            if r < len(rs)
        ]
        if vals:
            arr = np.array(vals, dtype=float)
            bands.append(ConfidenceBand(
                round=r + 1,
                mean=float(np.mean(arr)),
                p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)),
            ))
    return bands


async def run_evidence_pack(
    job_id: str,
    policy_name: str,
    policy_description: str,
    base_severity: float,
    sim_config: dict[str, Any],
    ensemble_size: int = 3,
) -> None:
    await registry.update(job_id, status="running", progress=0.0)

    severity_levels = sorted({
        max(1.0, base_severity - 0.5),
        base_severity,
        min(5.0, base_severity + 0.5),
    })

    loop = asyncio.get_running_loop()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=4)

    metrics = ["compliance_rate", "relocation_rate", "ai_investment_index", "enforcement_contact_rate"]
    all_results: dict[str, list[list[dict]]] = {str(sv): [] for sv in severity_levels}

    total_tasks = len(severity_levels) * ensemble_size
    completed = 0

    for sv in severity_levels:
        for i in range(ensemble_size):
            seed = 42 + i * 100
            try:
                summaries = await loop.run_in_executor(
                    executor, _run_one_sync, policy_name, policy_description,
                    sv, sim_config, seed,
                )
                all_results[str(sv)].append(summaries)
            except Exception as exc:
                await registry.update(job_id, status="error", error=str(exc))
                executor.shutdown(wait=False)
                return

            completed += 1
            await registry.update(job_id, progress=completed / total_tasks)

    # Build confidence bands per metric
    bands_by_metric: dict[str, list[ConfidenceBand]] = {}
    for metric in metrics:
        combined: list[list[dict]] = []
        for sv in severity_levels:
            combined.extend(all_results[str(sv)])
        bands_by_metric[metric] = _compute_bands(combined, metric)

    result = EvidencePackResult(
        severity_levels=severity_levels,
        bands=bands_by_metric,
    )
    await registry.update(job_id, status="complete", progress=1.0, result=result)
    executor.shutdown(wait=False)
