"""GET /presets — return the 4 built-in preset policies."""
from __future__ import annotations

import dataclasses
import sys
import os

from fastapi import APIRouter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.schemas.policy import PolicySpec, PresetPolicy

router = APIRouter()


def _load_presets() -> list[PresetPolicy]:
    from swarmcast.v2.policy.parser import (
        california_sb53,
        eu_ai_act_gpai,
        ny_raise_act,
        hypothetical_compute_ban,
    )

    raw = [
        ("ca-sb-53",            "California SB-53 (2025)",             california_sb53()),
        ("eu-ai-act-gpai",      "EU AI Act — GPAI systemic risk",      eu_ai_act_gpai()),
        ("ny-raise-act",        "NY RAISE Act (proposed)",             ny_raise_act()),
        ("compute-ban-1e26",    "Hypothetical compute ban (10²⁶)",     hypothetical_compute_ban(1e26)),
    ]

    presets = []
    for slug, label, ps in raw:
        d = dataclasses.asdict(ps)
        if "recommended_severity_sweep" in d:
            d["recommended_severity_sweep"] = list(d["recommended_severity_sweep"])
        presets.append(PresetPolicy(slug=slug, label=label, spec=PolicySpec(**d)))
    return presets


@router.get("/presets")
def get_presets() -> dict:
    presets = _load_presets()
    return {"presets": [p.model_dump() for p in presets]}
