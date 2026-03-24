"""
PolicyLab document ingestion pipeline.

Extracts regulatory provisions from documents (PDF, txt, md, docx) and
automatically populates PolicySpec + HybridSimConfig overrides with full
epistemic traceability (GROUNDED/DIRECTIONAL/ASSUMED per field).

Quick start:
    from swarmcast.v2.ingest import ingest, ingest_text

    result = ingest("eu_ai_act_impact_assessment.pdf")
    # result.spec → PolicySpec
    # result.config → dict of HybridSimConfig overrides
    # result.traceability_report() → full derivation chain
"""

from swarmcast.v2.ingest.pipeline import ingest, ingest_text, ingest_and_simulate
from swarmcast.v2.ingest.pipeline import IngestResult
from swarmcast.v2.ingest.document_loader import load_document, load_text_string
from swarmcast.v2.ingest.provision_extractor import extract_provisions, ExtractionResult
from swarmcast.v2.ingest.entity_graph import EntityGraph, build_entity_graph
from swarmcast.v2.ingest.spec_builder import build_spec, BuiltSpec

__all__ = [
    "ingest", "ingest_text", "ingest_and_simulate", "IngestResult",
    "load_document", "load_text_string",
    "extract_provisions", "ExtractionResult",
    "EntityGraph", "build_entity_graph",
    "build_spec", "BuiltSpec",
]
