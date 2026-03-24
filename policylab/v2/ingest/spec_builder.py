"""
Spec builder: ExtractionResult + EntityGraph → PolicySpec + HybridSimConfig overrides.

This is where the extracted regulatory intelligence becomes simulation parameters.
Every field in the output carries:
  - the derived value
  - which extraction field(s) it came from
  - the confidence of those source fields
  - the resulting epistemic tag

The type_distribution derivation is the most important function here. It translates
the graph question "who does this regulation cover?" into a concrete population mix
that drives the simulation's agent composition. This is what MiroFish does with
GraphRAG but for social media personas — we do it for regulatory compliance actors.

Design:
  1. Parse entity graph for regulated/exempted entity types
  2. Assign base fractions by entity type presence and confidence
  3. Adjust fractions for scope (frontier_only vs all)
  4. Normalise to sum=1.0
  5. Record justification for every fraction so it can be shown in evidence pack
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from policylab.v2.ingest.provision_extractor import ExtractionResult
from policylab.v2.ingest.entity_graph import EntityGraph


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED PARAMETER — carries value + traceability
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DerivedParameter:
    """A simulation parameter derived from the extracted regulatory text."""
    value: Any
    confidence: float
    epistemic_tag: str         # GROUNDED / DIRECTIONAL / ASSUMED
    justification: str         # human-readable reasoning chain
    source_fields: list[str]   # which ExtractionResult fields contributed

    @classmethod
    def from_extraction(
        cls,
        value: Any,
        conf: float,
        justification: str,
        source_fields: list[str],
    ) -> "DerivedParameter":
        if conf >= 0.80:
            tag = "GROUNDED"
        elif conf >= 0.50:
            tag = "DIRECTIONAL"
        else:
            tag = "ASSUMED"
        return cls(value=value, confidence=conf, epistemic_tag=tag,
                   justification=justification, source_fields=source_fields)

    def __repr__(self) -> str:
        return f"DerivedParameter({self.value!r}, [{self.epistemic_tag}] conf={self.confidence:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# BUILT SPEC — what the pipeline returns
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BuiltSpec:
    """Result of running the full spec builder pipeline on a document.

    Contains a ready-to-use PolicySpec and HybridSimConfig override dict,
    plus full traceability of every derived parameter.
    """
    # Ready to use
    policy_spec: Any           # PolicySpec (imported lazily to avoid circular)
    config_overrides: dict     # kwargs for HybridSimConfig

    # Full traceability
    derived_params: dict[str, DerivedParameter]

    # Source data (for evidence pack)
    extraction: ExtractionResult
    entity_graph: EntityGraph

    def traceability_report(self) -> str:
        """Full human-readable traceability report."""
        lines = [
            "PARAMETER DERIVATION REPORT",
            "=" * 72,
            f"Policy: {self.policy_spec.name}",
            f"Severity: {self.policy_spec.severity:.2f} / 5.0",
            "",
            "DERIVED PARAMETERS:",
            "─" * 72,
        ]
        for name, dp in self.derived_params.items():
            lines.append(f"\n  {name}")
            lines.append(f"    Value:         {dp.value!r}")
            lines.append(f"    Confidence:    {dp.confidence:.2f}  [{dp.epistemic_tag}]")
            lines.append(f"    Source fields: {', '.join(dp.source_fields)}")
            lines.append(f"    Reasoning:     {dp.justification}")

        lines += [
            "",
            "ENTITY GRAPH:",
            self.entity_graph.summary(),
            "",
            "EXTRACTION SUMMARY:",
            self.extraction.summary_table(),
        ]
        return "\n".join(lines)

    def config_with_overrides(self) -> dict:
        """Merge config_overrides with PolicySpec's compute_cost_factor.

        Returns a dict of kwargs that can be passed to HybridSimConfig(**kwargs).
        """
        import warnings
        warnings.warn(
            "config_with_overrides() is redundant: config_overrides already contains "
            "compute_cost_factor. Use result.config directly. "
            "This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = dict(self.config_overrides)
        result["compute_cost_factor"] = self.policy_spec.compute_cost_factor
        return result


# ─────────────────────────────────────────────────────────────────────────────
# TYPE DISTRIBUTION DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

# Base distribution: what the ecosystem looks like when a policy doesn't
# specifically target or exempt any particular type.
#
# [ASSUMED] These fractions are not directly calibrated to the Dealroom 2022
# figure (~3000 AI companies EU) because that dataset does not decompose by
# the PolicyLab agent type schema. The fractions are set to approximate a
# typical AI startup ecosystem (startup-heavy, few frontier labs) and then
# adjusted by the document-specific signals from the extraction.
# They represent the PRIOR before evidence from the document is applied.
# Sweep with [startup ±0.10, frontier_lab ±0.05] before citing type-composition
# results.
_BASE_DISTRIBUTION = {
    "startup":       0.40,
    "mid_company":   0.25,
    "large_company": 0.15,
    "researcher":    0.10,
    "investor":      0.05,
    "civil_society": 0.03,
    "frontier_lab":  0.02,
}


def _derive_type_distribution(
    extraction: ExtractionResult,
    graph: EntityGraph,
) -> tuple[dict[str, float], DerivedParameter]:
    """Derive agent type_distribution from extraction and entity graph.

    Returns (distribution_dict, traceability_DerivedParameter).

    The distribution reflects who this regulation primarily applies to.
    A compute-threshold regulation focused on frontier labs needs more
    frontier_lab and large_company agents; a broad "all AI systems" regulation
    needs more startups.
    """
    dist = dict(_BASE_DISTRIBUTION)
    justification_parts = []
    max_conf = 0.0

    scope = extraction.scope.value
    has_frontier = extraction.has_frontier_lab_focus.value
    has_sme = extraction.has_sme_provisions.value
    has_research_exempt = extraction.has_research_exemptions.value
    thresh = extraction.compute_threshold_flops.value

    # ── Adjust based on scope ────────────────────────────────────────────────
    if scope == "frontier_only":
        # Regulation only covers a handful of companies above a high compute threshold.
        # Simulate with more frontier labs and large companies; fewer startups.
        dist["frontier_lab"]  = 0.15
        dist["large_company"] = 0.25
        dist["mid_company"]   = 0.20
        dist["startup"]       = 0.25
        dist["researcher"]    = 0.07
        dist["investor"]      = 0.05
        dist["civil_society"] = 0.03
        justification_parts.append(
            f"scope=frontier_only: elevated frontier_lab ({dist['frontier_lab']:.0%}) "
            f"and large_company ({dist['large_company']:.0%}) fractions."
        )
        max_conf = max(max_conf, extraction.scope.confidence)

    elif scope == "large_developers_only":
        # Covers large but not necessarily frontier developers.
        dist["frontier_lab"]  = 0.08
        dist["large_company"] = 0.28
        dist["mid_company"]   = 0.28
        dist["startup"]       = 0.22
        dist["researcher"]    = 0.07
        dist["investor"]      = 0.04
        dist["civil_society"] = 0.03
        justification_parts.append(
            "scope=large_developers_only: elevated large_company fraction."
        )
        max_conf = max(max_conf, extraction.scope.confidence)

    elif scope == "all":
        # All AI developers — startup-heavy because most AI companies are startups.
        dist["startup"]       = 0.42
        dist["mid_company"]   = 0.24
        dist["large_company"] = 0.14
        dist["frontier_lab"]  = 0.04
        dist["researcher"]    = 0.08
        dist["investor"]      = 0.05
        dist["civil_society"] = 0.03
        justification_parts.append(
            "scope=all: high startup fraction (AI ecosystem is predominantly startups)."
        )
        max_conf = max(max_conf, extraction.scope.confidence)

    # ── Boost frontier_lab if compute threshold is specific ──────────────────
    if thresh is not None:
        try:
            exp = math.log10(float(thresh))
            if exp >= 26:
                # Very high threshold → very few entities → more frontier_lab weight
                boost = min(0.10, (exp - 25) * 0.05)
                dist["frontier_lab"] = min(0.20, dist.get("frontier_lab", 0.02) + boost)
                dist["startup"] = max(0.15, dist.get("startup", 0.40) - boost / 2)
                dist["mid_company"] = max(0.15, dist.get("mid_company", 0.25) - boost / 2)
                justification_parts.append(
                    f"compute_threshold=10^{exp:.0f} FLOPS: "
                    f"frontier_lab boosted by {boost:.0%} (fewer but larger entities affected)."
                )
                max_conf = max(max_conf, extraction.compute_threshold_flops.confidence)
        except (ValueError, TypeError, OverflowError):
            pass

    # ── Reduce researcher if explicitly exempted ─────────────────────────────
    if has_research_exempt:
        dist["researcher"] = max(0.03, dist.get("researcher", 0.10) * 0.5)
        justification_parts.append(
            "research_exemption=True: researcher fraction halved "
            "(they don't face compliance costs)."
        )
        max_conf = max(max_conf, extraction.has_research_exemptions.confidence)

    # ── Add investor if investor provisions present ──────────────────────────
    if extraction.has_investor_provisions.value:
        dist["investor"] = max(0.06, dist.get("investor", 0.05))
        justification_parts.append("investor_provisions=True: investor fraction elevated.")
        max_conf = max(max_conf, extraction.has_investor_provisions.confidence)

    # ── Use entity graph to cross-check regulated types ─────────────────────
    # The graph tells us which entity types actually appear in the document.
    # If the graph has NO frontier_lab node (low confidence extraction), don't
    # assume they're regulated just because we chose scope=frontier_only by default.
    graph_regulated = set(graph.regulated_entity_types(min_confidence=0.40))
    if "frontier_lab" not in graph_regulated and extraction.has_frontier_lab_focus.confidence < 0.50:
        # Downgrade frontier_lab if neither the graph nor the extraction is confident
        dist["frontier_lab"] = min(0.04, dist.get("frontier_lab", 0.04))
        dist["large_company"] = dist.get("large_company", 0.15) + 0.04
        justification_parts.append(
            "graph: no high-confidence frontier_lab node; frontier_lab fraction capped at 4%."
        )

    if "mid_company" not in graph_regulated and scope != "all":
        # Mid companies not explicitly mentioned → reduce their fraction slightly
        reduction = dist.get("mid_company", 0.20) * 0.15
        dist["mid_company"] = dist.get("mid_company", 0.20) - reduction
        dist["large_company"] = dist.get("large_company", 0.15) + reduction
        justification_parts.append(
            f"graph: mid_company not explicitly regulated; fraction reduced by {reduction:.0%}."
        )

    # ── Apply explicit exemptions from graph ─────────────────────────────────
    exempted = graph.exempted_entity_types(min_confidence=0.50)
    for ex_type in exempted:
        if ex_type in dist and ex_type not in ("researcher",):
            dist[ex_type] = max(0.01, dist[ex_type] * 0.3)
            justification_parts.append(
                f"{ex_type} exempted in entity graph: fraction reduced to {dist[ex_type]:.0%}."
            )

    # ── Normalise to sum=1.0 ────────────────────────────────────────────────
    total = sum(dist.values())
    if total > 0:
        dist = {k: round(v / total, 4) for k, v in dist.items()}

    # Confidence of the distribution is the average of the fields that drove it
    conf = max_conf if max_conf > 0 else 0.35
    justification = (
        "Type distribution derived from: "
        + "; ".join(justification_parts)
        if justification_parts
        else "Type distribution: base ecosystem fractions (no document-specific signals)."
    )

    dp = DerivedParameter.from_extraction(
        value=dist,
        conf=conf,
        justification=justification,
        source_fields=["scope", "compute_threshold_flops", "has_frontier_lab_focus",
                        "has_research_exemptions", "has_investor_provisions"],
    )
    return dist, dp


# ─────────────────────────────────────────────────────────────────────────────
# ROUNDS / HORIZON DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def _derive_num_rounds(extraction: ExtractionResult) -> tuple[int, DerivedParameter]:
    """Derive recommended simulation horizon from grace period and typical cycle.

    The simulation should cover at least:
      - The grace period (when compliance ramps up)
      - 4–8 quarters beyond to see the post-compliance equilibrium

    1 round = 1 quarter = 3 months.
    """
    grace_mo = int(extraction.grace_period_months.value or 0)
    grace_q = math.ceil(grace_mo / 3)

    # Add 4 quarters post-grace to see equilibrium, minimum 8 quarters (2 years)
    recommended = max(8, grace_q + 4)
    # Cap at 20 quarters (5 years) — beyond that extrapolation is unreliable
    recommended = min(20, recommended)

    grace_conf = extraction.grace_period_months.confidence
    if grace_mo == 0:
        justification = (
            f"No grace period specified (grace_period_months=0). "
            f"Defaulting to 8 rounds (2 years) — the minimum GDPR calibration window."
        )
        conf = 0.40
    else:
        justification = (
            f"grace_period_months={grace_mo} ({grace_q} rounds) + "
            f"4 rounds post-grace equilibrium window = {recommended} rounds total."
        )
        conf = grace_conf * 0.85  # slightly lower than source — derivation adds uncertainty

    return recommended, DerivedParameter.from_extraction(
        value=recommended,
        conf=conf,
        justification=justification,
        source_fields=["grace_period_months"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE COST FACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _derive_compute_cost_factor(extraction: ExtractionResult) -> tuple[float, DerivedParameter]:
    """Derive compute_cost_factor from threshold and enforcement mechanism.

    Combines two signals:
      1. How high is the compute threshold? (higher → fewer entities → more novel)
      2. How demanding is the enforcement? (third-party audit → more technical burden)

    The GDPR baseline is factor=1.0. AI regulation at 10^26 with third-party
    audit is factor≈2.0. Criminal investigation is factor≈4.0.
    """
    thresh = extraction.compute_threshold_flops.value
    enf = extraction.enforcement_mechanism.value
    thresh_conf = extraction.compute_threshold_flops.confidence
    enf_conf = extraction.enforcement_mechanism.confidence

    # Base from threshold
    if thresh is None:
        thresh_factor = 1.0
        thresh_note = "no compute threshold → GDPR-equivalent compliance speed"
    else:
        try:
            exp = math.log10(float(thresh))
            if exp >= 27:
                thresh_factor, thresh_note = 1.5, f"10^{exp:.0f} FLOPS (very few models)"
            elif exp >= 26:
                thresh_factor, thresh_note = 2.0, f"10^{exp:.0f} FLOPS (SB-53/EU AI Act level)"
            elif exp >= 25:
                thresh_factor, thresh_note = 3.0, f"10^{exp:.0f} FLOPS (broad frontier coverage)"
            else:
                thresh_factor, thresh_note = 4.0, f"10^{exp:.0f} FLOPS (wide coverage)"
        except (ValueError, OverflowError):
            thresh_factor, thresh_note = 1.5, "unresolved threshold"

    # Multiplier from enforcement mechanism — demanding enforcement = more
    # novel technical capabilities needed to comply
    enf_mult = {
        "none":              1.0,
        "self_report":       1.0,
        "third_party_audit": 1.15,
        "government_inspect":1.30,
        "criminal_invest":   1.50,
    }.get(enf, 1.0)

    factor = round(thresh_factor * enf_mult, 2)
    # Clip to sensible range
    factor = max(1.0, min(6.0, factor))

    # Confidence: use lower of the two source fields
    conf = min(thresh_conf, enf_conf) if thresh is not None else enf_conf * 0.7
    conf = max(0.25, conf)  # CCF is always partially assumed

    justification = (
        f"compute_threshold → factor={thresh_factor:.1f} ({thresh_note}); "
        f"enforcement={enf} → multiplier={enf_mult:.2f}; "
        f"combined={factor:.2f}. "
        f"[ASSUMED] no empirical compliance timeline data for AI-specific regulations. "
        f"Sweep [1.0, {factor:.1f}, {min(6.0, factor*1.5):.1f}] before citing."
    )

    return factor, DerivedParameter.from_extraction(
        value=factor, conf=conf,
        justification=justification,
        source_fields=["compute_threshold_flops", "enforcement_mechanism"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_spec(
    extraction: ExtractionResult,
    graph: EntityGraph,
) -> BuiltSpec:
    """Build a PolicySpec + HybridSimConfig overrides from extraction and graph.

    This is the final step of the ingestion pipeline. Every derived parameter
    is fully traced to the extraction fields and source passages that grounded it.
    """
    # Import here to avoid circular dependency
    from policylab.v2.policy.parser import parse_bill, PolicySpec

    # ── Validate and coerce extracted values ─────────────────────────────────
    VALID_PENALTY  = {"none","voluntary","civil","civil_heavy","criminal"}
    VALID_ENF      = {"none","self_report","third_party_audit","government_inspect","criminal_invest"}
    VALID_SCOPE    = {"voluntary","frontier_only","large_developers_only","all"}

    penalty_type = str(extraction.penalty_type.value)
    if penalty_type not in VALID_PENALTY:
        penalty_type = "civil"

    enf = str(extraction.enforcement_mechanism.value)
    if enf not in VALID_ENF:
        enf = "self_report"

    scope = str(extraction.scope.value)
    if scope not in VALID_SCOPE:
        scope = "all"

    try:
        cap = float(extraction.penalty_cap_usd.value) if extraction.penalty_cap_usd.value else None
    except (TypeError, ValueError):
        cap = None

    try:
        thresh = float(extraction.compute_threshold_flops.value) if extraction.compute_threshold_flops.value else None
    except (TypeError, ValueError):
        thresh = None

    try:
        grace = int(extraction.grace_period_months.value or 0)
    except (TypeError, ValueError):
        grace = 0

    # ── Run parse_bill to get severity score ─────────────────────────────────
    spec = parse_bill(
        name=str(extraction.policy_name.value),
        description=str(extraction.policy_description.value or ""),
        penalty_type=penalty_type,
        penalty_cap_usd=cap,
        compute_threshold_flops=thresh,
        enforcement_mechanism=enf,
        grace_period_months=grace,
        scope=scope,
    )

    # ── Derive additional parameters ─────────────────────────────────────────
    type_dist, type_dist_dp = _derive_type_distribution(extraction, graph)
    num_rounds, rounds_dp = _derive_num_rounds(extraction)
    ccf, ccf_dp = _derive_compute_cost_factor(extraction)

    # Override spec's auto-derived CCF with our more nuanced derivation
    spec.compute_cost_factor = ccf

    # Set recommended_num_rounds on spec
    spec.recommended_num_rounds = num_rounds

    # ── Source jurisdiction ──────────────────────────────────────────────────
    jur_map = {"EU": "EU", "US": "US", "UK": "UK", "Singapore": "Singapore"}
    source_jur = jur_map.get(str(extraction.source_jurisdiction.value), "EU")
    jur_dp = DerivedParameter.from_extraction(
        value=source_jur,
        conf=extraction.source_jurisdiction.confidence,
        justification=f"source_jurisdiction={source_jur} from document language/references.",
        source_fields=["source_jurisdiction"],
    )

    # ── n_population recommendation ──────────────────────────────────────────
    # Larger n for frontier-focused regulations (fewer entity types → lower
    # variance per agent, so we need fewer agents for the same statistical power)
    n_reg = str(extraction.estimated_n_regulated.value)
    if n_reg == "handful":
        n_pop = 500
        n_pop_note = "handful of regulated entities → 500 agents sufficient"
    elif n_reg == "dozens":
        n_pop = 1000
        n_pop_note = "dozens of regulated entities → 1000 agents"
    elif n_reg == "hundreds":
        n_pop = 2000
        n_pop_note = "hundreds of regulated entities → 2000 agents"
    else:
        n_pop = 2000
        n_pop_note = "default (unknown N regulated)"

    n_pop_dp = DerivedParameter.from_extraction(
        value=n_pop,
        conf=extraction.estimated_n_regulated.confidence * 0.8,
        justification=f"estimated_n_regulated={n_reg}: {n_pop_note}.",
        source_fields=["estimated_n_regulated"],
    )

    # ── Extend PolicySpec justification with derivation notes ────────────────
    # Compute extraction confidence across core fields
    core_conf_fields = [
        extraction.penalty_type.confidence,
        extraction.enforcement_mechanism.confidence,
        extraction.scope.confidence,
    ]
    avg_core_conf = sum(core_conf_fields) / len(core_conf_fields)

    spec.justification.extend([
        f"[ingest] type_distribution derived from scope+threshold: {type_dist}",
        f"[ingest] compute_cost_factor={ccf:.2f} (threshold={thresh}, enf={enf})",
        f"[ingest] num_rounds={num_rounds} (grace={grace}mo + 4 quarter equilibration)",
        f"[ingest] source_jurisdiction={source_jur}",
        f"[ingest] n_population={n_pop} (estimated_n_regulated={n_reg})",
        f"[ingest] core_extraction_confidence={avg_core_conf:.2f} "
        f"({'GROUNDED' if avg_core_conf >= 0.80 else 'DIRECTIONAL' if avg_core_conf >= 0.50 else 'ASSUMED — verify before citing'})",
    ])

    # When confidence is low, widen the sweep range so the evidence pack
    # shows more uncertainty rather than false precision
    if avg_core_conf < 0.50:
        lo, _, hi = spec.recommended_severity_sweep if spec.recommended_severity_sweep else (
            max(1.0, spec.severity - 0.5), spec.severity, min(5.0, spec.severity + 0.5)
        )
        spec.recommended_severity_sweep = (
            max(1.0, spec.severity - 1.0),
            spec.severity,
            min(5.0, spec.severity + 1.0),
        )
        spec.justification.append(
            "[ingest] LOW CONFIDENCE: severity sweep widened to ±1.0 to reflect "
            "extraction uncertainty. Verify extracted parameters before simulation."
        )

    # ── HybridSimConfig overrides ─────────────────────────────────────────────
    config_overrides = {
        "n_population":         n_pop,
        "num_rounds":           num_rounds,
        "type_distribution":    type_dist,
        "source_jurisdiction":  source_jur,
        "compute_cost_factor":  ccf,
    }

    # ── Traceability dict ─────────────────────────────────────────────────────
    derived_params = {
        "type_distribution":    type_dist_dp,
        "num_rounds":           rounds_dp,
        "compute_cost_factor":  ccf_dp,
        "source_jurisdiction":  jur_dp,
        "n_population":         n_pop_dp,
    }

    return BuiltSpec(
        policy_spec=spec,
        config_overrides=config_overrides,
        derived_params=derived_params,
        extraction=extraction,
        entity_graph=graph,
    )
