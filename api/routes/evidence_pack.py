"""POST /evidence-pack (start job) and GET /evidence-pack/{job_id} (poll)."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas.jobs import EvidencePackRequest, JobStatus
from api.services.job_registry import registry
from api.services import evidence_service

router = APIRouter()


@router.post("/evidence-pack")
async def start_evidence_pack(
    req: EvidencePackRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    job_id = registry.create_job()
    background_tasks.add_task(
        evidence_service.run_evidence_pack,
        job_id=job_id,
        policy_name=req.policy_name,
        policy_description=req.policy_description,
        base_severity=req.base_severity,
        sim_config=req.sim_config,
        ensemble_size=req.ensemble_size,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "estimated_seconds": req.ensemble_size * 3 * 3,  # rough: 3 severity × ensemble × ~3s each
    }


@router.get("/evidence-pack/{job_id}", response_model=JobStatus)
def poll_evidence_pack(job_id: str) -> JobStatus:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return JobStatus(
        job_id=job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
    )
