"""POST /simulate and POST /simulate/upload endpoints."""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from api.schemas.simulation import SimulateRequest, SimulateResponse
from api.schemas.upload import UploadResponse
from api.services import simulation_service, extraction_service
from api.config import settings

router = APIRouter()


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest) -> SimulateResponse:
    try:
        return await simulation_service.run_simulation(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate/upload", response_model=UploadResponse)
async def simulate_upload(
    file: UploadFile = File(...),
    api_key: str | None = Form(default=None),
    model: str = Form(default="gpt-4o"),
) -> UploadResponse:
    # Validate file type
    allowed_suffixes = {".pdf", ".txt", ".md", ".docx"}
    filename = file.filename or "upload"
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use PDF, txt, md, or docx.",
        )

    # Check size
    contents = await file.read()
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents):,} bytes). Max is {settings.max_upload_bytes:,}.",
        )

    # Write to temp file and ingest
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        ingest_result = await extraction_service.ingest_file(
            file_path=tmp_path,
            api_key=api_key,
            model=model,
        )
    finally:
        os.unlink(tmp_path)

    # Run simulation with extracted config
    sim_req = extraction_service.ingest_result_to_simulate_request(ingest_result)
    try:
        sim_result = await simulation_service.run_simulation(sim_req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    spec = extraction_service._convert_spec(ingest_result.spec)
    extraction = extraction_service._convert_extraction(ingest_result.extraction)

    return UploadResponse(
        document_name=filename,
        spec=spec,
        extraction=extraction,
        warnings=ingest_result.warnings,
        elapsed_seconds=ingest_result.elapsed_seconds,
        result=sim_result,
        confidence_summary=ingest_result.confidence_summary(),
    )
