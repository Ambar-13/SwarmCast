"""
Swarmcast document ingestion pipeline.

End-to-end: document file (PDF/txt/md/docx) or raw text string
→ PolicySpec + HybridSimConfig overrides ready for simulation.

Every derived parameter carries its confidence score and source passage,
mapping automatically to GROUNDED/DIRECTIONAL/ASSUMED in the epistemic framework.

Usage
─────
    from policylab.v2.ingest.pipeline import ingest

    # From a file (PDF, txt, md, docx)
    result = ingest("eu_ai_act_impact_assessment.pdf", api_key="sk-...")

    # From a string (pasted text, e.g. from a UI)
    result = ingest_text(bill_text, name="Custom bill", api_key="sk-...")

    # Use the result directly
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation

    r = run_hybrid_simulation(
        result.spec.name,
        result.spec.description,
        result.spec.severity,
        config=HybridSimConfig(**result.config),
    )

    # Print the full traceability report
    print(result.traceability_report())

    # Run with compute_cost_factor sweep
    for ccf in result.spec.recommended_severity_sweep:
        ...

No external services required unless you pass an api_key.
Without an API key, regex extraction runs as fallback (lower confidence, no hallucination).
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any

from policylab.v2.ingest.document_loader import LoadedDocument, load_document, load_text_string
from policylab.v2.ingest.provision_extractor import ExtractionResult, extract_provisions
from policylab.v2.ingest.entity_graph import EntityGraph, build_entity_graph
from policylab.v2.ingest.spec_builder import BuiltSpec, build_spec


@dataclasses.dataclass
class IngestResult:
    """Complete result of the ingestion pipeline.

    All fields are populated and ready to use. Nothing is lazy-evaluated.
    """
    # The primary outputs — pass these to the simulation
    spec: Any           # PolicySpec
    config: dict        # kwargs for HybridSimConfig(**result.config)

    # Full traceability chain
    document: LoadedDocument
    extraction: ExtractionResult
    graph: EntityGraph
    built_spec: BuiltSpec

    # Pipeline metadata
    elapsed_seconds: float
    warnings: list[str]

    def traceability_report(self) -> str:
        """Full human-readable traceability from document to simulation parameters."""
        return self.built_spec.traceability_report()

    def extraction_summary(self) -> str:
        """Concise extraction summary table."""
        return self.extraction.summary_table()

    def ready_to_simulate(self) -> bool:
        """True if extraction confidence is high enough for reliable simulation.

        Low confidence means many parameters are ASSUMED — simulation results will
        be highly sensitive to those assumptions. Still runnable, but flag to the
        analyst that they should verify the extracted parameters before citing results.
        """
        from policylab.v2.ingest.provision_extractor import ExtractedField
        core_fields = [
            self.extraction.penalty_type,
            self.extraction.enforcement_mechanism,
            self.extraction.scope,
        ]
        avg_conf = sum(f.confidence for f in core_fields) / len(core_fields)
        return avg_conf >= 0.50

    def confidence_summary(self) -> str:
        """One-line summary of extraction confidence."""
        fields = [
            ("penalty",      self.extraction.penalty_type.confidence),
            ("enforcement",  self.extraction.enforcement_mechanism.confidence),
            ("threshold",    self.extraction.compute_threshold_flops.confidence),
            ("grace",        self.extraction.grace_period_months.confidence),
            ("scope",        self.extraction.scope.confidence),
        ]
        tags = []
        for name, conf in fields:
            if conf >= 0.80:
                tags.append(f"{name}=GROUNDED")
            elif conf >= 0.50:
                tags.append(f"{name}=DIRECTIONAL")
            else:
                tags.append(f"{name}=ASSUMED")
        return "  ".join(tags)

    def __repr__(self) -> str:
        return (
            f"IngestResult(\n"
            f"  spec.name={self.spec.name!r}\n"
            f"  spec.severity={self.spec.severity:.2f}\n"
            f"  compute_cost_factor={self.spec.compute_cost_factor:.2f}\n"
            f"  config.num_rounds={self.config.get('num_rounds')}\n"
            f"  config.n_population={self.config.get('n_population')}\n"
            f"  extraction_method={self.extraction.extraction_method_used}\n"
            f"  {self.confidence_summary()}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def ingest(
    file_path: str,
    api_key: str | None = None,
    model: str = "gpt-4o",
    base_url: str | None = None,
    verbose: bool = True,
) -> IngestResult:
    """Run the full ingestion pipeline on a document file.

    Parameters
    ──────────
    file_path  : Path to PDF, txt, md, or docx file.
    api_key    : OpenAI-compatible API key. Falls back to OPENAI_API_KEY env var,
                 then ANTHROPIC_API_KEY, then regex extraction.
    model      : LLM model name. "gpt-4o" recommended for best extraction quality.
    base_url   : Optional non-OpenAI endpoint (e.g. local Ollama, Azure OpenAI).
    verbose    : Print progress to stdout.

    Returns
    ───────
    IngestResult with spec, config, and full traceability chain.

    Raises
    ──────
    FileNotFoundError : file_path does not exist.
    ValueError        : unsupported file format.
    """
    t0 = time.perf_counter()
    warnings: list[str] = []

    # ── Step 1: Load document ─────────────────────────────────────────────────
    if verbose:
        print(f"[ingest] Loading {file_path}...")
    doc = load_document(file_path)
    if doc.encoding_notes:
        warnings.extend(doc.encoding_notes)
        if verbose:
            for note in doc.encoding_notes:
                print(f"[ingest] Warning: {note}")
    if verbose:
        print(f"[ingest] Loaded {len(doc.full_text):,} chars, "
              f"{doc.n_pages} {'pages' if doc.file_type == 'pdf' else 'sections'}")

    return _run_pipeline(doc, api_key, model, base_url, verbose, warnings, t0)


def ingest_text(
    text: str,
    name: str = "Uploaded document",
    api_key: str | None = None,
    model: str = "gpt-4o",
    base_url: str | None = None,
    verbose: bool = True,
) -> IngestResult:
    """Run the ingestion pipeline on a text string (e.g. pasted bill text).

    Use this when the document is already in memory rather than on disk.

    Parameters
    ──────────
    text       : Raw document text (any length; will be truncated for LLM if needed).
    name       : Human-readable name for the document (used in traceability reports).
    api_key    : See ingest() above.
    model      : See ingest() above.
    base_url   : See ingest() above.
    verbose    : See ingest() above.

    Returns
    ───────
    IngestResult — same as ingest().
    """
    t0 = time.perf_counter()
    warnings: list[str] = []

    if verbose:
        print(f"[ingest] Processing {len(text):,} character string '{name}'...")

    doc = load_text_string(text, name=name)
    return _run_pipeline(doc, api_key, model, base_url, verbose, warnings, t0)


def _run_pipeline(
    doc: LoadedDocument,
    api_key: str | None,
    model: str,
    base_url: str | None,
    verbose: bool,
    warnings: list[str],
    t0: float,
) -> IngestResult:
    """Shared internal pipeline execution."""

    # ── Step 2: Extract provisions ────────────────────────────────────────────
    if verbose:
        key_available = bool(
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        method = f"LLM ({model})" if key_available else "regex fallback (no API key)"
        print(f"[ingest] Extracting provisions using {method}...")

    extraction = extract_provisions(
        doc=doc,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=0.0,
    )

    if extraction.unresolved_provisions:
        warnings.extend(extraction.unresolved_provisions[:3])

    if verbose:
        print(f"[ingest] Extracted: "
              f"penalty={extraction.penalty_type.value} "
              f"(conf={extraction.penalty_type.confidence:.2f}), "
              f"scope={extraction.scope.value} "
              f"(conf={extraction.scope.confidence:.2f})")
        if extraction.compute_threshold_flops.value:
            print(f"[ingest] Compute threshold: "
                  f"{extraction.compute_threshold_flops.value:.0e} FLOPS "
                  f"(conf={extraction.compute_threshold_flops.confidence:.2f})")

    # ── Step 3: Build entity graph ────────────────────────────────────────────
    if verbose:
        print(f"[ingest] Building entity graph...")

    graph = build_entity_graph(extraction)

    if verbose:
        print(f"[ingest] Graph: {len(graph)} nodes, "
              f"regulated types: {graph.regulated_entity_types()}")

    # ── Step 4: Build PolicySpec + config ─────────────────────────────────────
    if verbose:
        print(f"[ingest] Deriving PolicySpec and HybridSimConfig overrides...")

    built = build_spec(extraction, graph)

    if verbose:
        print(f"[ingest] Done in {time.perf_counter()-t0:.1f}s")
        print(f"[ingest] PolicySpec: severity={built.policy_spec.severity:.2f}, "
              f"compute_cost_factor={built.policy_spec.compute_cost_factor:.2f}")
        print(f"[ingest] Config: n_pop={built.config_overrides['n_population']}, "
              f"rounds={built.config_overrides['num_rounds']}, "
              f"type_dist={built.config_overrides['type_distribution']}")

    # Low confidence warning
    core_conf = (
        extraction.penalty_type.confidence
        + extraction.enforcement_mechanism.confidence
        + extraction.scope.confidence
    ) / 3
    if core_conf < 0.50:
        msg = (
            f"Core extraction confidence is low ({core_conf:.2f}). "
            f"Many parameters are ASSUMED. Verify the extracted values before simulation."
        )
        warnings.append(msg)
        if verbose:
            print(f"[ingest] WARNING: {msg}")

    return IngestResult(
        spec=built.policy_spec,
        config=built.config_overrides,
        document=doc,
        extraction=extraction,
        graph=graph,
        built_spec=built,
        elapsed_seconds=time.perf_counter() - t0,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: ingest → simulate in one call
# ─────────────────────────────────────────────────────────────────────────────

def ingest_and_simulate(
    file_path: str,
    api_key: str | None = None,
    model: str = "gpt-4o",
    base_url: str | None = None,
    verbose: bool = True,
    extra_config: dict | None = None,
) -> tuple["IngestResult", Any]:
    """Ingest a document and immediately run a simulation.

    Returns (IngestResult, HybridSimResult) so you have both the traceability
    chain and the simulation output.

    base_url    : Non-OpenAI endpoint (e.g. local Ollama). Forwarded to ingest().
    extra_config: Additional kwargs for HybridSimConfig (override ingest defaults).
    """
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation

    import warnings
    warnings.warn(
        "ingest_and_simulate() is a convenience wrapper not used by the API or CLI. "
        "Call ingest() and run_hybrid_simulation() directly for full control. "
        "This function will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = ingest(file_path, api_key=api_key, model=model,
                    base_url=base_url, verbose=verbose)

    config_kwargs = result.config.copy()
    if extra_config:
        config_kwargs.update(extra_config)

    if verbose:
        print(f"\n[simulate] Running simulation for '{result.spec.name}'...")

    sim_result = run_hybrid_simulation(
        policy_name=result.spec.name,
        policy_description=result.spec.description,
        policy_severity=result.spec.severity,
        config=HybridSimConfig(**config_kwargs),
    )

    return result, sim_result
