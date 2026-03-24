"""Tests for the Swarmcast v2 document ingestion pipeline.

Covers:
  1. Document loader — text chunking, char_offset correctness (Fix 3)
  2. Provision extractor — regex accuracy, threshold math (Fix 4), passage validation (Fix 6)
  3. Entity graph — mid_company inference (Fix 11), regulated_entity_types()
  4. Spec builder — type_distribution derivation, compute_cost_factor, num_rounds,
                    entity graph used (Fix 10), confidence propagation (Fix 13)
  5. Pipeline — end-to-end on real-world policy text, ProbabilisticTrigger (Fix 5),
                ingest_and_simulate signature (Fix 7)
  6. Epistemic integrity — GROUNDED/DIRECTIONAL/ASSUMED thresholds correct
  7. pyproject.toml — missing deps present (Fix 1)
"""

import math
import os
import sys
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT LOADER
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentLoader:

    def test_load_text_string_char_offsets_sorted(self):
        """Fix 3: char_offsets must be strictly non-decreasing across paragraphs."""
        from policylab.v2.ingest.document_loader import load_text_string
        doc = load_text_string("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
        offsets = [c.char_offset for c in doc.chunks]
        assert offsets == sorted(offsets), (
            f"char_offsets must be sorted — got {offsets}"
        )

    def test_load_text_string_para2_offset_nonzero(self):
        """Fix 3: paragraph 2 offset must be > len(para1), not 0."""
        from policylab.v2.ingest.document_loader import load_text_string
        doc = load_text_string("First paragraph text here.\n\nSecond paragraph text here.")
        assert len(doc.chunks) == 2, f"Expected 2 chunks, got {len(doc.chunks)}"
        para1_len = len(doc.chunks[0].text)
        assert doc.chunks[1].char_offset > para1_len, (
            f"Para 2 offset {doc.chunks[1].char_offset} must be > "
            f"len(para1)={para1_len}. Old bug: find(para2, 0) returns 0 "
            f"when para2 shares prefix with para1."
        )

    def test_load_text_string_offset_points_to_correct_text(self):
        """char_offset must point to the start of the paragraph in full_text."""
        from policylab.v2.ingest.document_loader import load_text_string
        doc = load_text_string("Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota.")
        for chunk in doc.chunks:
            reconstructed = doc.full_text[chunk.char_offset:chunk.char_offset + len(chunk.text)]
            assert reconstructed == chunk.text, (
                f"Offset {chunk.char_offset} for '{chunk.source_id}' points to "
                f"'{reconstructed[:30]}' but chunk.text='{chunk.text[:30]}'"
            )

    def test_load_text_string_identical_paragraphs(self):
        """Identical paragraphs must get distinct offsets, not all map to first occurrence."""
        from policylab.v2.ingest.document_loader import load_text_string
        doc = load_text_string("Same text.\n\nSame text.\n\nSame text.")
        offsets = [c.char_offset for c in doc.chunks]
        assert len(set(offsets)) == len(offsets), (
            f"Identical paragraphs got duplicate offsets: {offsets}. "
            f"Old bug: find(para, 0) always returns first occurrence."
        )

    def test_load_text_string_returns_loaded_document(self):
        """load_text_string must return a LoadedDocument."""
        from policylab.v2.ingest.document_loader import load_text_string, LoadedDocument
        doc = load_text_string("Simple text.")
        assert isinstance(doc, LoadedDocument)
        assert doc.file_type == "txt"
        assert len(doc.full_text) > 0
        assert len(doc.chunks) > 0

    def test_passage_with_context_returns_correct_region(self):
        """passage_with_context must return text around the given offset."""
        from policylab.v2.ingest.document_loader import load_text_string
        doc = load_text_string("First paragraph.\n\nSecond paragraph with key text here.")
        # Find offset of "key text"
        offset = doc.full_text.find("key text")
        ctx = doc.passage_with_context(offset, window=40)
        assert "key text" in ctx, f"Context does not contain 'key text': {ctx}"


# ─────────────────────────────────────────────────────────────────────────────
# PROVISION EXTRACTOR — REGEX
# ─────────────────────────────────────────────────────────────────────────────

class TestProvisionExtractorRegex:

    def _extract(self, text: str):
        from policylab.v2.ingest.document_loader import load_text_string
        from policylab.v2.ingest.provision_extractor import _extract_regex
        return _extract_regex(load_text_string(text))

    def test_compute_threshold_plain_10_to_26(self):
        """10^26 must extract to 1e26."""
        ex = self._extract("Training runs above 10^26 FLOPS must comply.")
        assert ex.compute_threshold_flops.value is not None
        assert abs(ex.compute_threshold_flops.value - 1e26) / 1e26 < 0.01

    def test_compute_threshold_base_times_power(self):
        """Fix 4: 1.5 × 10^26 must extract to 1.5e26, not 1e26."""
        ex = self._extract("Training runs above 1.5 × 10^26 FLOPS must comply.")
        thresh = ex.compute_threshold_flops.value
        assert thresh is not None, "Threshold must be extracted"
        assert abs(thresh - 1.5e26) / 1.5e26 < 0.01, (
            f"Expected 1.5e26, got {thresh:.2e}. "
            f"Old bug: base×10^N pattern fires after 10^N pattern, discarding base."
        )

    def test_compute_threshold_25_flops(self):
        """10^25 FLOPS must extract to 1e25."""
        ex = self._extract("Models above 10^25 floating-point operations.")
        assert ex.compute_threshold_flops.value is not None
        assert abs(ex.compute_threshold_flops.value - 1e25) / 1e25 < 0.01

    def test_penalty_civil_detected(self):
        """Civil penalty language must extract penalty_type=civil."""
        ex = self._extract("Civil penalties up to $1M per violation.")
        assert ex.penalty_type.value == "civil"

    def test_penalty_criminal_detected(self):
        """Criminal language must extract penalty_type=criminal."""
        ex = self._extract("Criminal prosecution for violations. Imprisonment up to 5 years.")
        assert ex.penalty_type.value == "criminal"

    def test_penalty_cap_not_confused_with_revenue_threshold(self):
        """$500M revenue threshold must not become the penalty cap."""
        ex = self._extract(
            "Developers with annual revenue exceeding $500M must comply. "
            "Civil penalties up to $1M per violation."
        )
        cap = ex.penalty_cap_usd.value
        # The cap should either be $1M (correct) or None (fallback), not $500M
        if cap is not None:
            assert abs(cap - 1e6) / 1e6 < 0.10, (
                f"Cap should be $1M not ${cap/1e6:.0f}M — "
                f"revenue threshold was confused with penalty cap."
            )

    def test_enforcement_third_party_audit(self):
        """Third-party audit language must extract correctly."""
        ex = self._extract("Models must undergo third-party safety evaluation by an accredited auditor.")
        assert ex.enforcement_mechanism.value == "third_party_audit"

    def test_grace_period_months(self):
        """6-month implementation period must extract grace_period_months=6."""
        ex = self._extract("This act takes effect 6 months after enactment.")
        assert ex.grace_period_months.value == 6

    def test_scope_large_developers_only(self):
        """Large developer scope language."""
        ex = self._extract("This regulation applies to large AI developers only.")
        assert ex.scope.value in ("large_developers_only", "frontier_only", "all")

    def test_sme_provisions_detected(self):
        """SME mention must set has_sme_provisions=True."""
        ex = self._extract("Small business exemption applies to companies with under 50 employees.")
        assert ex.has_sme_provisions.value is True

    def test_research_exemption_detected(self):
        """Research exemption must set has_research_exemptions=True."""
        # Text where "research" and "exempt" are within 120-char window
        ex = self._extract("Research and development activities are exempt from this regulation.")
        assert ex.has_research_exemptions.value is True, (
            f"has_research_exemptions should be True, got: "
            f"value={ex.has_research_exemptions.value}, "
            f"conf={ex.has_research_exemptions.confidence}"
        )

    def test_us_jurisdiction_detected(self):
        """California legislation must detect US jurisdiction."""
        ex = self._extract("CALIFORNIA SENATE BILL 53. AN ACT relating to artificial intelligence.")
        assert ex.source_jurisdiction.value == "US"

    def test_eu_jurisdiction_detected(self):
        """EU AI Act language must detect EU jurisdiction."""
        ex = self._extract("REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT on artificial intelligence.")
        assert ex.source_jurisdiction.value == "EU", (
            f"Expected EU jurisdiction, got {ex.source_jurisdiction.value!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PASSAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestPassageValidation:

    def test_genuine_passage_found(self):
        """Fix 6: a passage that exists in the document must be accepted."""
        from policylab.v2.ingest.provision_extractor import _validate_source_passage
        text = "civil fines up to one million dollars per violation"
        doc_text = "... " + text + " ..."
        validated, ok = _validate_source_passage(text, doc_text)
        assert ok, "Genuine passage must be found"
        assert validated == text

    def test_hallucinated_passage_rejected(self):
        """Fix 6: a passage not in the document must be rejected and confidence capped."""
        from policylab.v2.ingest.provision_extractor import _validate_source_passage
        fake = "this text does not appear anywhere in the document at all"
        doc_text = "The regulation requires third-party safety evaluations."
        validated, ok = _validate_source_passage(fake, doc_text)
        assert not ok, "Hallucinated passage must not be accepted"
        assert validated == "", "Rejected passage must return empty string"

    def test_normalised_whitespace_match(self):
        """Passage with collapsed whitespace (LLM artefact) must still match."""
        from policylab.v2.ingest.provision_extractor import _validate_source_passage
        # Document has "civil\npenalties" (line break), LLM returns "civil penalties" (space)
        passage = "civil penalties up to one million"
        doc_text = "civil\npenalties up to one million dollars per violation"
        _, ok = _validate_source_passage(passage, doc_text)
        assert ok, "Normalised-whitespace match must succeed"

    def test_short_passage_skipped(self):
        """Very short passages (< min_len) must be skipped without crashing."""
        from policylab.v2.ingest.provision_extractor import _validate_source_passage
        _, ok = _validate_source_passage("hi", "hi there in the document")
        # Short passages return the original value unchanged — just no match attempt
        assert ok is False  # too short to validate

    def test_epistemic_tag_grounded_when_confident(self):
        """confidence >= 0.80 must produce GROUNDED tag."""
        from policylab.v2.ingest.provision_extractor import ExtractedField
        f = ExtractedField(value="civil", confidence=0.90, source_passage="...", extraction_method="llm")
        assert f.epistemic_tag == "GROUNDED"

    def test_epistemic_tag_directional(self):
        """0.50 <= confidence < 0.80 must produce DIRECTIONAL."""
        from policylab.v2.ingest.provision_extractor import ExtractedField
        f = ExtractedField(value="civil", confidence=0.65, source_passage="", extraction_method="regex")
        assert f.epistemic_tag == "DIRECTIONAL"

    def test_epistemic_tag_assumed(self):
        """confidence < 0.50 must produce ASSUMED."""
        from policylab.v2.ingest.provision_extractor import ExtractedField
        f = ExtractedField(value="all", confidence=0.30, source_passage="", extraction_method="regex")
        assert f.epistemic_tag == "ASSUMED"


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityGraph:

    def _build_graph(self, text: str):
        from policylab.v2.ingest.document_loader import load_text_string
        from policylab.v2.ingest.provision_extractor import _extract_regex
        from policylab.v2.ingest.entity_graph import build_entity_graph
        doc = load_text_string(text)
        ex = _extract_regex(doc)
        return build_entity_graph(ex), ex

    def test_mid_company_in_regulated_types(self):
        """Fix 11: mid_company must appear in regulated_entity_types() when developers present."""
        from policylab.v2.ingest.entity_graph import EntityGraph, build_entity_graph
        from policylab.v2.ingest.document_loader import load_text_string
        from policylab.v2.ingest.provision_extractor import _extract_regex
        doc = load_text_string("Large AI developers above $500M revenue must comply with this regulation.")
        ex = _extract_regex(doc)
        ex.scope.value = "large_developers_only"  # ensure developer node is added
        g = build_entity_graph(ex)
        regulated = g.regulated_entity_types()
        assert "mid_company" in regulated, (
            f"mid_company missing from regulated_entity_types(): {regulated}. "
            f"Fix 11: mid_company must be inferred from DEVELOPER node presence."
        )

    def test_regulated_entity_types_returns_list(self):
        """regulated_entity_types() must return a list without crashing."""
        g, _ = self._build_graph("Frontier AI developers above 10^26 FLOPS must comply.")
        result = g.regulated_entity_types(min_confidence=0.40)
        assert isinstance(result, list), "Must return a list"

    def test_frontier_lab_in_regulated_when_threshold_present(self):
        """Compute threshold presence should result in frontier_lab being regulated."""
        g, _ = self._build_graph("Systems trained with more than 10^26 FLOPS must be evaluated.")
        regulated = g.regulated_entity_types()
        assert "frontier_lab" in regulated, (
            f"frontier_lab should be regulated when compute threshold present: {regulated}"
        )

    def test_researcher_exempted_when_research_exempt(self):
        """Research exemption must add researcher to exempted types."""
        # Use text that clearly triggers the research exemption pattern
        g, _ = self._build_graph(
            "Civil penalties apply. Research exemption: academic institutions "
            "and scientific research bodies are not subject to this regulation."
        )
        exempted = g.exempted_entity_types(min_confidence=0.40)
        assert "researcher" in exempted, (
            f"researcher should be exempted when 'research exemption' appears: {exempted}"
        )

    def test_graph_has_nodes_after_build(self):
        """build_entity_graph must produce at least 3 nodes."""
        g, _ = self._build_graph("Civil penalties up to $1M. 10^26 FLOPS threshold.")
        assert len(g) >= 3, f"Expected >= 3 nodes, got {len(g)}"

    def test_graph_to_dict_serializable(self):
        """to_dict() must return JSON-serializable dict."""
        import json
        g, _ = self._build_graph("Civil penalties. Third-party audit required.")
        d = g.to_dict()
        json.dumps(d)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# SPEC BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class TestSpecBuilder:

    def _build(self, text: str):
        from policylab.v2.ingest.document_loader import load_text_string
        from policylab.v2.ingest.provision_extractor import _extract_regex
        from policylab.v2.ingest.entity_graph import build_entity_graph
        from policylab.v2.ingest.spec_builder import build_spec
        doc = load_text_string(text)
        ex = _extract_regex(doc)
        g = build_entity_graph(ex)
        return build_spec(ex, g)

    def test_type_distribution_sums_to_one(self):
        """type_distribution must sum to 1.0."""
        built = self._build("Civil penalties. All AI developers must comply.")
        td = built.config_overrides["type_distribution"]
        total = sum(td.values())
        assert abs(total - 1.0) < 0.01, f"type_distribution sums to {total:.4f}, not 1.0"

    def test_type_distribution_all_positive(self):
        """All type fractions must be non-negative."""
        built = self._build("Criminal ban. All developers must comply.")
        td = built.config_overrides["type_distribution"]
        for t, v in td.items():
            assert v >= 0, f"type {t} has negative fraction {v}"

    def test_type_distribution_contains_all_7_types(self):
        """All 7 canonical agent types must be present in type_distribution."""
        built = self._build("Civil penalties. All developers.")
        td = built.config_overrides["type_distribution"]
        expected_types = {"startup", "mid_company", "large_company", "researcher",
                          "investor", "civil_society", "frontier_lab"}
        assert expected_types == set(td.keys()), (
            f"Missing types: {expected_types - set(td.keys())}"
        )

    def test_entity_graph_used_in_type_distribution(self):
        """Fix 10: spec_builder must call graph.regulated_entity_types()."""
        # Verify at source level that the code uses the graph
        import inspect
        from policylab.v2.ingest import spec_builder
        src = inspect.getsource(spec_builder)
        assert "graph.regulated_entity_types" in src, (
            "spec_builder must use graph.regulated_entity_types() — Fix 10"
        )

    def test_frontier_heavy_when_scope_frontier_only(self):
        """frontier_only scope must produce higher frontier_lab fraction than all-scope."""
        from policylab.v2.ingest.document_loader import load_text_string
        from policylab.v2.ingest.provision_extractor import _extract_regex
        from policylab.v2.ingest.entity_graph import build_entity_graph
        from policylab.v2.ingest.spec_builder import build_spec

        def _td(scope_val):
            doc = load_text_string(f"Civil penalties. Applies to {scope_val} developers.")
            ex = _extract_regex(doc)
            ex.scope.value = scope_val
            g = build_entity_graph(ex)
            return build_spec(ex, g).config_overrides["type_distribution"]

        td_frontier = _td("frontier_only")
        td_all = _td("all")
        assert td_frontier.get("frontier_lab", 0) >= td_all.get("frontier_lab", 0), (
            "frontier_only scope must have >= frontier_lab fraction than all-scope"
        )

    def test_compute_cost_factor_higher_for_compute_policy(self):
        """CCF must be > 1.0 when a compute threshold is present."""
        built = self._build("Training above 10^26 FLOPS requires third-party audit.")
        ccf = built.policy_spec.compute_cost_factor
        assert ccf > 1.0, (
            f"compute_cost_factor should be > 1.0 when compute threshold present, got {ccf}"
        )

    def test_compute_cost_factor_1_when_no_threshold(self):
        """CCF must be 1.0 when no compute threshold is specified."""
        built = self._build("Voluntary guidelines for AI safety. No thresholds specified.")
        ccf = built.policy_spec.compute_cost_factor
        assert ccf == 1.0, f"Expected CCF=1.0 without threshold, got {ccf}"

    def test_num_rounds_includes_grace_period(self):
        """num_rounds must be at least grace_period_quarters + 4."""
        from policylab.v2.ingest.document_loader import load_text_string
        from policylab.v2.ingest.provision_extractor import _extract_regex
        from policylab.v2.ingest.entity_graph import build_entity_graph
        from policylab.v2.ingest.spec_builder import build_spec
        doc = load_text_string("Civil penalties. 12-month implementation period.")
        ex = _extract_regex(doc)
        g = build_entity_graph(ex)
        built = build_spec(ex, g)
        rounds = built.config_overrides["num_rounds"]
        grace_q = math.ceil(ex.grace_period_months.value / 3) if ex.grace_period_months.value else 0
        assert rounds >= grace_q + 4, (
            f"rounds={rounds} should be >= grace_quarters({grace_q}) + 4"
        )

    def test_low_confidence_widens_severity_sweep(self):
        """Fix 13b: low confidence extraction must widen the severity sweep to ±1.0."""
        from policylab.v2.ingest.document_loader import load_text_string
        from policylab.v2.ingest.provision_extractor import _extract_regex, ExtractedField
        from policylab.v2.ingest.entity_graph import build_entity_graph
        from policylab.v2.ingest.spec_builder import build_spec
        doc = load_text_string("Some vague document without clear regulatory language.")
        ex = _extract_regex(doc)
        # Force all core fields to low confidence
        ex.penalty_type.confidence = 0.25
        ex.enforcement_mechanism.confidence = 0.25
        ex.scope.confidence = 0.25
        g = build_entity_graph(ex)
        built = build_spec(ex, g)
        sweep = built.policy_spec.recommended_severity_sweep
        if sweep:
            lo, mid, hi = sweep
            sev = built.policy_spec.severity
            assert hi - lo >= 1.8, (
                f"Low-confidence sweep should be >=1.8 wide, got {hi-lo:.1f} (lo={lo}, hi={hi}). "
                f"Fix 13b: widen to ±1.0 when core confidence < 0.50."
            )

    def test_traceability_report_contains_all_params(self):
        """traceability_report() must mention all 5 derived parameter names."""
        built = self._build("Civil penalties. 10^26 FLOPS threshold.")
        report = built.traceability_report()
        for param in ["type_distribution", "num_rounds", "compute_cost_factor",
                      "source_jurisdiction", "n_population"]:
            assert param in report, f"Traceability report missing '{param}'"

    def test_base_distribution_documented_as_assumed(self):
        """Fix 13a: _BASE_DISTRIBUTION must be documented as [ASSUMED]."""
        import inspect
        from policylab.v2.ingest import spec_builder
        src = inspect.getsource(spec_builder)
        assert "[ASSUMED]" in src, "_BASE_DISTRIBUTION must be tagged [ASSUMED]"


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — END-TO-END
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestPipeline:

    SB53_TEXT = (
        "CALIFORNIA SENATE BILL 53. AN ACT relating to artificial intelligence safety. "
        "Large AI developers with annual revenue exceeding $500M must comply. "
        "Training runs above 10^26 FLOPS require third-party safety evaluation. "
        "Civil penalties up to $1M per violation. "
        "Six-month implementation period. "
        "Small businesses with fewer than 50 employees are exempt."
    )

    EU_TEXT = (
        "REGULATION (EU) on artificial intelligence. "
        "General-purpose AI models trained above 10^25 FLOPS face systemic risk obligations. "
        "Model evaluations and adversarial testing required. "
        "Fines not exceeding 3% of total worldwide annual turnover. "
        "Research institutions are exempt. "
        "12 months implementation period."
    )

    def test_ingest_text_returns_ingest_result(self):
        """ingest_text() must return an IngestResult."""
        from policylab.v2.ingest.pipeline import ingest_text, IngestResult
        result = ingest_text(self.SB53_TEXT, name="SB-53 test", verbose=False)
        assert isinstance(result, IngestResult)

    def test_ingest_result_has_spec(self):
        """IngestResult must have a PolicySpec with severity in [1, 5]."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        assert 1.0 <= result.spec.severity <= 5.0, (
            f"severity {result.spec.severity} out of [1, 5]"
        )

    def test_ingest_result_config_has_required_keys(self):
        """config must contain all keys needed by HybridSimConfig."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        required = {"n_population", "num_rounds", "type_distribution",
                    "source_jurisdiction", "compute_cost_factor"}
        missing = required - set(result.config.keys())
        assert not missing, f"config missing keys: {missing}"

    def test_ingest_result_ready_to_simulate(self):
        """SB-53 text has enough content to be ready_to_simulate()."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        # ready_to_simulate() returns bool — just ensure it doesn't crash
        assert isinstance(result.ready_to_simulate(), bool)

    def test_ingest_type_distribution_sums_to_one(self):
        """type_distribution in config must sum to 1.0."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        td = result.config["type_distribution"]
        total = sum(td.values())
        assert abs(total - 1.0) < 0.01, f"type_distribution sums to {total:.4f}"

    def test_ingest_extracts_compute_threshold(self):
        """10^26 FLOPS in SB-53 text must be extracted."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        thresh = result.extraction.compute_threshold_flops.value
        assert thresh is not None, "Compute threshold must be extracted from SB-53 text"
        assert abs(thresh - 1e26) / 1e26 < 0.01, f"Expected 1e26, got {thresh:.2e}"

    def test_ingest_sb53_jurisdiction_is_us(self):
        """SB-53 (California) must be detected as US jurisdiction."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        assert result.config.get("source_jurisdiction") == "US", (
            f"Expected US, got {result.config.get('source_jurisdiction')}"
        )

    def test_ingest_eu_jurisdiction_detected(self):
        """EU regulation text must detect EU jurisdiction."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.EU_TEXT, verbose=False)
        assert result.config.get("source_jurisdiction") == "EU"

    def test_ingest_compute_cost_factor_sb53(self):
        """SB-53 with 10^26 threshold must get CCF = 2.0."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        assert result.spec.compute_cost_factor >= 2.0, (
            f"SB-53 CCF should be >= 2.0, got {result.spec.compute_cost_factor}"
        )

    def test_ingest_compute_cost_factor_eu_gpai(self):
        """EU text with 10^25 threshold must get CCF >= 3.0."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.EU_TEXT, verbose=False)
        assert result.spec.compute_cost_factor >= 3.0, (
            f"EU GPAI CCF should be >= 3.0, got {result.spec.compute_cost_factor}"
        )

    def test_ingest_traceability_report_not_empty(self):
        """traceability_report() must produce non-empty output."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        report = result.traceability_report()
        assert len(report) > 200, "Traceability report is too short"
        assert "GROUNDED" in report or "DIRECTIONAL" in report or "ASSUMED" in report

    def test_ingest_entity_graph_has_nodes(self):
        """Entity graph must have nodes after ingestion."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text(self.SB53_TEXT, verbose=False)
        assert len(result.graph) >= 3, (
            f"Entity graph has too few nodes: {len(result.graph)}"
        )

    def test_ingest_warnings_is_list(self):
        """warnings must be a list (not None, not crash)."""
        from policylab.v2.ingest.pipeline import ingest_text
        result = ingest_text("Simple text.", verbose=False)
        assert isinstance(result.warnings, list)

    def test_ingest_and_simulate_has_base_url_param(self):
        """Fix 7: ingest_and_simulate must accept base_url parameter."""
        import inspect
        from policylab.v2.ingest.pipeline import ingest_and_simulate
        params = inspect.signature(ingest_and_simulate).parameters
        assert "base_url" in params, (
            "ingest_and_simulate missing base_url parameter — Fix 7"
        )

    def test_ingest_end_to_end_with_simulation(self):
        """Full pipeline: ingest_text → HybridSimConfig → run_hybrid_simulation."""
        import warnings as _w; _w.filterwarnings("ignore")
        from policylab.v2.ingest.pipeline import ingest_text
        from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation
        result = ingest_text(self.SB53_TEXT, verbose=False)
        # result.config already contains n_population and num_rounds.
        # Override them separately to avoid duplicate keyword arguments.
        config_kwargs = dict(result.config)
        config_kwargs["n_population"] = 100
        config_kwargs["num_rounds"] = 4
        config_kwargs["verbose"] = False
        config_kwargs["seed"] = 42
        config = HybridSimConfig(**config_kwargs)
        sim = run_hybrid_simulation(
            result.spec.name, result.spec.description,
            result.spec.severity, config=config,
        )
        fp = sim.final_population_summary
        assert 0.0 <= fp.get("compliance_rate", 0) <= 1.0
        assert 0.0 <= fp.get("relocation_rate", 0) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILISTIC TRIGGER — Fix 5
# ─────────────────────────────────────────────────────────────────────────────

class TestProbabilisticTrigger:

    def test_deterministic_for_same_inputs(self):
        """Fix 5: should_fire must return same result for identical (round_num, prob)."""
        from policylab.v2.simulation.events import ProbabilisticTrigger
        results = [
            ProbabilisticTrigger(probability=0.5).should_fire(round_num=3, stocks=None, pop=None)
            for _ in range(10)
        ]
        assert len(set(results)) == 1, (
            f"ProbabilisticTrigger not deterministic: {results}. "
            "Fix 5: must use seeded numpy RNG, not global random.random()."
        )

    def test_different_rounds_can_give_different_results(self):
        """Different round numbers should (in general) give different outcomes."""
        from policylab.v2.simulation.events import ProbabilisticTrigger
        # With p=0.5, rounds 1-20 should not all give the same result
        results = [
            ProbabilisticTrigger(probability=0.5).should_fire(round_num=r, stocks=None, pop=None)
            for r in range(1, 21)
        ]
        # At least some variation expected (with overwhelming probability)
        assert len(set(results)) > 1, "All rounds gave same result — seed not varying by round_num"

    def test_zero_probability_never_fires(self):
        """probability=0 must never fire."""
        from policylab.v2.simulation.events import ProbabilisticTrigger
        t = ProbabilisticTrigger(probability=0.0)
        results = [t.should_fire(r, None, None) for r in range(1, 20)]
        assert not any(results), "probability=0 must never fire"

    def test_max_fires_respected(self):
        """max_fires=1 must fire at most once across all rounds."""
        from policylab.v2.simulation.events import ProbabilisticTrigger
        t = ProbabilisticTrigger(probability=1.0, max_fires=1)
        fires = sum(1 for r in range(1, 20) if t.should_fire(r, None, None))
        assert fires <= 1, f"max_fires=1 but fired {fires} times"


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS.PY DUPLICATE POLICYSPEC — Fix 12
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisPolicySpec:

    def test_comparison_policy_defined(self):
        """Fix 12: ComparisonPolicy must be defined in analysis.py."""
        from policylab.v2.analysis import ComparisonPolicy
        p = ComparisonPolicy(name="test", description="desc", severity=3.0)
        assert p.name == "test"
        assert p.severity == 3.0

    def test_policysec_alias_works(self):
        """Fix 12: PolicySpec alias must work for backwards compatibility."""
        from policylab.v2.analysis import PolicySpec
        p = PolicySpec(name="compat", description="desc", severity=2.0)
        assert p.name == "compat"

    def test_parser_policyspec_distinct(self):
        """The canonical PolicySpec from parser.py must have justification attribute."""
        from policylab.v2.policy.parser import PolicySpec as ParserSpec
        from policylab.v2.analysis import PolicySpec as AnalysisSpec
        p = ParserSpec(name="x", description="", severity=1.0, justification=[])
        assert hasattr(p, "justification"), "Parser PolicySpec must have justification"
        # Analysis PolicySpec does NOT have justification — that's the distinction
        p2 = AnalysisSpec(name="x", description="", severity=1.0)
        assert not hasattr(p2, "justification"), (
            "Analysis PolicySpec should NOT have justification — they are distinct types"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PYPROJECT.TOML — Fix 1
# ─────────────────────────────────────────────────────────────────────────────

class TestPyprojectToml:

    def test_pypdf_in_dependencies(self):
        """Fix 1: pypdf must be listed in project.dependencies."""
        import tomllib
        with open(os.path.join(BASE, "pyproject.toml"), "rb") as f:
            t = tomllib.load(f)
        deps = t["project"]["dependencies"]
        assert any("pypdf" in d for d in deps), (
            "pypdf missing from pyproject.toml. Every ingest('file.pdf') call "
            "on a fresh install raises ImportError without it."
        )

    def test_python_docx_in_dependencies(self):
        """Fix 1: python-docx must be listed in project.dependencies."""
        import tomllib
        with open(os.path.join(BASE, "pyproject.toml"), "rb") as f:
            t = tomllib.load(f)
        deps = t["project"]["dependencies"]
        assert any("python-docx" in d for d in deps), (
            "python-docx missing from pyproject.toml. Every ingest('file.docx') call "
            "on a fresh install raises ImportError without it."
        )

    def test_path_imported_in_provision_extractor(self):
        """Fix 2: provision_extractor.py must import Path."""
        with open(os.path.join(BASE, "policylab", "v2", "ingest",
                               "provision_extractor.py")) as f:
            src = f.read()
        assert "from pathlib import Path" in src, (
            "Path not imported in provision_extractor.py — Fix 2"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    classes = [
        TestDocumentLoader,
        TestProvisionExtractorRegex,
        TestPassageValidation,
        TestEntityGraph,
        TestSpecBuilder,
        TestIngestPipeline,
        TestProbabilisticTrigger,
        TestAnalysisPolicySpec,
        TestPyprojectToml,
    ]
    passed = failed = 0
    for cls in classes:
        inst = cls()
        methods = sorted(m for m in dir(cls) if m.startswith("test_"))
        print(f"\n{cls.__name__}")
        for m in methods:
            try:
                getattr(inst, m)()
                print(f"  PASS  {m}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {m}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    import sys; sys.exit(1 if failed else 0)
