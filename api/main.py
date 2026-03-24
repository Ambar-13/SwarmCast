"""Swarmcast FastAPI application.

Start with:
    cd api && uvicorn main:app --reload --port 8000

Endpoints:
    GET  /presets
    POST /simulate
    POST /simulate/upload
    POST /compare
    POST /inject
    POST /evidence-pack
    GET  /evidence-pack/{job_id}
"""
from __future__ import annotations

import sys
import os

# Make policylab importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager

import json
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class NumpyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(content, cls=_NumpyEncoder).encode("utf-8")

from api.config import settings
from api.routes import presets, simulate, compare, inject, evidence_pack


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: load preset policies at startup so first request is fast
    try:
        from policylab.v2.policy.parser import (
            california_sb53, eu_ai_act_gpai, ny_raise_act, hypothetical_compute_ban
        )
        _ = california_sb53(), eu_ai_act_gpai(), ny_raise_act(), hypothetical_compute_ban(1e26)
    except Exception:
        pass  # Non-fatal — will load on first request
    yield


app = FastAPI(
    title="Swarmcast API",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=NumpyJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(presets.router)
app.include_router(simulate.router)
app.include_router(compare.router)
app.include_router(inject.router)
app.include_router(evidence_pack.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
