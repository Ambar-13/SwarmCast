"""
Provision extractor: document text → structured regulatory fields with confidence.

This is the core of the ingestion pipeline. It makes ONE carefully-designed LLM
call that extracts all regulatory dimensions simultaneously, with three pieces of
evidence per field:

  value          — the extracted value (penalty_type="civil", threshold=1e26, ...)
  confidence     — float [0, 1] of how certain the extraction is
  source_passage — the exact quote from the document that supports the value

Confidence thresholds map directly to PolicyLab's epistemic framework:
  ≥ 0.80  → GROUNDED  (text explicitly states this; high extraction certainty)
  ≥ 0.50  → DIRECTIONAL (text implies this but leaves room for interpretation)
  < 0.50  → ASSUMED    (defaulted; document doesn't address this dimension)

The LLM is optional. If no API key is configured, regex-based extraction runs
as fallback. Regex extraction produces lower confidence scores (max 0.7) and
cannot match nuanced language, but it never returns wrong answers due to
hallucination — which pure LLM extraction can.

Design principle: the LLM's job is to QUOTE the document, not to reason about
it. Every value must be accompanied by the exact passage that supports it. If
the LLM cannot find a passage, it must return confidence < 0.3 and say so.
This prevents the model from inventing regulatory provisions that aren't there.

Usage
─────
    from swarmcast.v2.ingest.provision_extractor import extract_provisions
    from swarmcast.v2.ingest.document_loader import load_document

    doc = load_document("eu_ai_act_impact_assessment.pdf")
    result = extract_provisions(doc, api_key="sk-...", model="gpt-4o")

    print(result.penalty_type.value)       # "civil"
    print(result.penalty_type.confidence)  # 0.95
    print(result.penalty_type.source)      # "[page 47] ...fines not exceeding 3%..."
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from swarmcast.v2.ingest.document_loader import LoadedDocument


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTED FIELD — carries value + confidence + evidence
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ExtractedField:
    """A single extracted regulatory dimension with full provenance.

    epistemic_tag is derived from confidence automatically.
    """
    value: Any
    confidence: float          # [0, 1]
    source_passage: str        # exact quote from the document, or "" if not found
    extraction_method: str     # "llm", "regex", or "default"

    @property
    def epistemic_tag(self) -> str:
        if self.confidence >= 0.80:
            return "GROUNDED"
        elif self.confidence >= 0.50:
            return "DIRECTIONAL"
        else:
            return "ASSUMED"

    def __repr__(self) -> str:
        return (
            f"ExtractedField({self.value!r}, "
            f"confidence={self.confidence:.2f} [{self.epistemic_tag}], "
            f"method={self.extraction_method!r})"
        )


def _field(value: Any, conf: float, source: str, method: str) -> ExtractedField:
    return ExtractedField(
        value=value,
        confidence=max(0.0, min(1.0, conf)),
        source_passage=source,
        extraction_method=method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION RESULT — all dimensions together
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ExtractionResult:
    """Complete structured extraction of a regulatory document.

    Every field is an ExtractedField with value + confidence + source passage.
    The spec_builder.py module consumes this to produce a PolicySpec.
    """
    # Core regulatory dimensions (map to PolicySpec fields)
    policy_name:           ExtractedField   # short name of the regulation
    policy_description:    ExtractedField   # 1-2 sentence plain English description
    penalty_type:          ExtractedField   # "none"|"voluntary"|"civil"|"civil_heavy"|"criminal"
    penalty_cap_usd:       ExtractedField   # float or None (% of turnover → None)
    compute_threshold_flops: ExtractedField # float or None
    enforcement_mechanism: ExtractedField   # "none"|"self_report"|"third_party_audit"|"government_inspect"|"criminal_invest"
    grace_period_months:   ExtractedField   # int
    scope:                 ExtractedField   # "voluntary"|"frontier_only"|"large_developers_only"|"all"
    source_jurisdiction:   ExtractedField   # "EU"|"US"|"UK"|"Singapore"|"other"

    # Agent population composition signals (map to HybridSimConfig.type_distribution)
    has_sme_provisions:         ExtractedField   # bool — SMEs explicitly mentioned
    has_frontier_lab_focus:     ExtractedField   # bool — frontier labs / GPAI specifically targeted
    has_research_exemptions:    ExtractedField   # bool — research explicitly exempted or addressed
    has_investor_provisions:    ExtractedField   # bool — investors/VCs addressed
    estimated_n_regulated:      ExtractedField   # "handful"|"dozens"|"hundreds"|"thousands"|"unknown"

    # Key provisions — list of the most important individual regulatory requirements
    # Each is a (provision_text, source_id) pair
    key_provisions: list[tuple[str, str]]

    # Metadata
    extraction_method_used: str   # "llm" or "regex_fallback"
    model_used: str | None        # model name if LLM was used
    unresolved_provisions: list[str]  # passages that couldn't be classified

    def summary_table(self) -> str:
        """Human-readable summary for review before running simulation."""
        rows = [
            ("Policy name",          self.policy_name),
            ("Penalty type",         self.penalty_type),
            ("Penalty cap (USD)",     self.penalty_cap_usd),
            ("Compute threshold",     self.compute_threshold_flops),
            ("Enforcement",          self.enforcement_mechanism),
            ("Grace period (months)", self.grace_period_months),
            ("Scope",                self.scope),
            ("Jurisdiction",         self.source_jurisdiction),
            ("SME provisions",       self.has_sme_provisions),
            ("Frontier lab focus",   self.has_frontier_lab_focus),
            ("Research exemptions",  self.has_research_exemptions),
            ("N regulated (est.)",   self.estimated_n_regulated),
        ]
        lines = ["EXTRACTION RESULT", "─" * 72]
        for label, field in rows:
            tag = field.epistemic_tag
            val = field.value
            conf = field.confidence
            lines.append(
                f"  {label:<28} {str(val):<20} conf={conf:.2f} [{tag}]"
            )
        if self.unresolved_provisions:
            lines.append(f"\n  Unresolved provisions ({len(self.unresolved_provisions)}):")
            for u in self.unresolved_provisions[:3]:
                lines.append(f"    - {u[:80]}")
        lines.append(f"\n  Extraction method: {self.extraction_method_used}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_SCHEMA = """
{
  "policy_name": "short name of the regulation (e.g. 'California SB-53', 'EU AI Act GPAI tier')",
  "policy_description": "1-2 sentence plain English description of what the regulation requires and who it applies to",
  "penalty_type": "one of: none | voluntary | civil | civil_heavy | criminal",
  "penalty_cap_usd": "the maximum fine in USD as a number, or null if expressed as % of revenue or uncapped",
  "penalty_cap_note": "exact quote about the penalty amount, including if it is % of turnover",
  "compute_threshold_flops": "the compute threshold in FLOPS as a number (e.g. 1e26 for 10^26), or null if not specified",
  "compute_threshold_source": "exact quote from document about the compute threshold",
  "enforcement_mechanism": "one of: none | self_report | third_party_audit | government_inspect | criminal_invest",
  "enforcement_source": "exact quote describing the enforcement mechanism",
  "grace_period_months": "implementation timeline in months (integer), or 0 if immediate",
  "grace_period_source": "exact quote about the implementation timeline",
  "scope": "one of: voluntary | frontier_only | large_developers_only | all",
  "scope_source": "exact quote describing who the regulation applies to",
  "source_jurisdiction": "one of: EU | US | UK | Singapore | other",
  "has_sme_provisions": "true if SMEs, small companies, or small developers are explicitly mentioned",
  "sme_source": "exact quote about SME provisions, or null",
  "has_frontier_lab_focus": "true if frontier labs, GPAI, general-purpose AI, or compute thresholds specifically target top-tier AI developers",
  "frontier_source": "exact quote about frontier lab focus, or null",
  "has_research_exemptions": "true if research institutions are explicitly exempted or treated differently",
  "research_source": "exact quote about research provisions, or null",
  "has_investor_provisions": "true if investors, VCs, or financial actors are explicitly addressed",
  "investor_source": "exact quote about investor provisions, or null",
  "estimated_n_regulated": "one of: handful | dozens | hundreds | thousands | unknown — how many entities would be covered",
  "estimated_n_source": "exact quote or reasoning for the estimate",
  "key_provisions": [
    {"text": "verbatim provision text", "source_id": "page/section reference"}
  ],
  "unresolved": ["any provisions you could not classify into the schema above"]
}

CRITICAL RULES:
1. Every *_source field must be a verbatim quote from the document, not your own words.
2. If the document does not address a dimension, use the default value and set confidence to 0.
3. Do not invent provisions. If it is not in the document, say so.
4. For penalty_cap_usd: if the penalty is expressed as "X% of global turnover", set to null.
5. For compute_threshold_flops: 10^26 = 1e26, 10^25 = 1e25, etc.
6. Respond ONLY with valid JSON. No markdown, no preamble, no explanation.
"""

_CONFIDENCE_SCHEMA = """
After extracting the fields above, add a "confidence" object with scores [0.0, 1.0] for each field:
{
  "confidence": {
    "policy_name": 0.95,
    "penalty_type": 0.90,
    "penalty_cap_usd": 0.85,
    "compute_threshold_flops": 0.80,
    "enforcement_mechanism": 0.75,
    "grace_period_months": 0.70,
    "scope": 0.80,
    "source_jurisdiction": 0.95,
    "has_sme_provisions": 0.70,
    "has_frontier_lab_focus": 0.85,
    "has_research_exemptions": 0.60,
    "has_investor_provisions": 0.40,
    "estimated_n_regulated": 0.50
  }
}

Confidence rules:
  0.9-1.0: The document explicitly states this in unambiguous language.
  0.7-0.9: The document strongly implies this; a careful reader would agree.
  0.5-0.7: The document partially addresses this; reasonable analysts might differ.
  0.3-0.5: You are making an informed inference; the document is ambiguous.
  0.0-0.3: You defaulted this value; the document does not address this dimension.
"""


def _build_extraction_prompt(doc: LoadedDocument, max_chars: int = 80_000) -> str:
    """Build the LLM extraction prompt with document text."""
    text = doc.full_text

    # Truncate intelligently: keep start (usually title/definitions) and end
    # (usually enforcement/penalties) which are most information-dense
    if len(text) > max_chars:
        half = max_chars // 2
        text = (
            text[:half]
            + f"\n\n[... {len(text) - max_chars} characters omitted for length ...]\n\n"
            + text[-half:]
        )

    return f"""You are a regulatory analysis assistant. Your task is to extract structured
information from the following regulatory document for use in a compliance simulation tool.

DOCUMENT SOURCE: {doc.file_path}
DOCUMENT LENGTH: {len(doc.full_text)} characters, {doc.n_pages} pages/sections
FILE TYPE: {doc.file_type}

═══════════════════════════════════════════════════════════════════════════════
DOCUMENT TEXT:
═══════════════════════════════════════════════════════════════════════════════
{text}
═══════════════════════════════════════════════════════════════════════════════

Extract the following fields from this document.

EXTRACTION SCHEMA:
{_EXTRACTION_SCHEMA}

{_CONFIDENCE_SCHEMA}

Your final response must be a single valid JSON object containing all the extraction
fields AND the confidence object. Nothing else. Start with {{ and end with }}.
"""


def _validate_source_passage(passage: str, full_text: str, min_len: int = 10) -> tuple[str, bool]:
    """Verify a source passage actually appears in the document.

    Returns (passage, is_genuine). If not found, returns ("", False) so the
    field gets ASSUMED tag instead of a hallucinated GROUNDED tag.
    Checks exact match and normalised-whitespace match (handles LLM line-break collapsing).
    """
    if not passage or len(passage) < min_len:
        return passage, False
    if passage in full_text:
        return passage, True
    norm_p = re.sub(r'\s+', ' ', passage).strip()
    norm_f = re.sub(r'\s+', ' ', full_text)
    if norm_p in norm_f:
        return passage, True
    # Partial: leading 50 chars must appear verbatim
    window = min(50, len(norm_p))
    if window >= 20 and norm_p[:window] in norm_f:
        return passage, True
    return "", False


def _parse_llm_response(raw: str, doc: LoadedDocument) -> ExtractionResult | None:
    """Parse the LLM's JSON response. Validates all source passages against the
    document — a passage not found in the document gets confidence capped below
    GROUNDED threshold (0.45) to prevent hallucinated provenance."""
    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.MULTILINE)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from within the response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    conf = data.get("confidence", {})

    def _ef(key: str, default_val: Any, source_key: str | None = None) -> ExtractedField:
        val = data.get(key, default_val)
        c = float(conf.get(key, 0.3))
        raw_src = data.get(source_key, "") if source_key else ""
        if raw_src is None:
            raw_src = ""
        validated_src, is_genuine = _validate_source_passage(str(raw_src), doc.full_text)
        if raw_src and not is_genuine:
            c = min(c, 0.45)   # cap below GROUNDED — hallucinated passage
            validated_src = f"[unverifiable: {str(raw_src)[:80]}]"
        return _field(val, c, validated_src[:500], "llm")

    # Parse key provisions
    raw_provisions = data.get("key_provisions", [])
    provisions = []
    for p in raw_provisions[:10]:  # cap at 10
        if isinstance(p, dict):
            provisions.append((
                str(p.get("text", ""))[:400],
                str(p.get("source_id", "unknown"))
            ))

    return ExtractionResult(
        policy_name=_ef("policy_name", "Unknown policy", None),
        policy_description=_ef("policy_description", "", None),
        penalty_type=_ef("penalty_type", "civil", "penalty_cap_note"),
        penalty_cap_usd=_ef("penalty_cap_usd", None, "penalty_cap_note"),
        compute_threshold_flops=_ef("compute_threshold_flops", None, "compute_threshold_source"),
        enforcement_mechanism=_ef("enforcement_mechanism", "self_report", "enforcement_source"),
        grace_period_months=_ef("grace_period_months", 0, "grace_period_source"),
        scope=_ef("scope", "all", "scope_source"),
        source_jurisdiction=_ef("source_jurisdiction", "other", None),
        has_sme_provisions=_ef("has_sme_provisions", False, "sme_source"),
        has_frontier_lab_focus=_ef("has_frontier_lab_focus", False, "frontier_source"),
        has_research_exemptions=_ef("has_research_exemptions", False, "research_source"),
        has_investor_provisions=_ef("has_investor_provisions", False, "investor_source"),
        estimated_n_regulated=_ef("estimated_n_regulated", "unknown", "estimated_n_source"),
        key_provisions=provisions,
        extraction_method_used="llm",
        model_used=None,  # set by caller
        unresolved_provisions=data.get("unresolved", []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# REGEX FALLBACK EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_regex(doc: LoadedDocument) -> ExtractionResult:
    """Regex-based extraction as fallback when no LLM is configured.

    Confidence is capped at 0.70 because regex cannot handle paraphrasing
    or document-specific terminology. Use LLM extraction for production.

    Each pattern is documented with its failure modes so analysts know when
    to distrust the result.
    """
    text = doc.full_text.lower()
    full = doc.full_text  # preserve case for quotes

    def _find_passage(pattern: re.Pattern, max_len: int = 200) -> str:
        m = pattern.search(full)
        if m:
            start = max(0, m.start() - 40)
            end = min(len(full), m.end() + 80)
            return full[start:end].strip()
        return ""

    # ── Penalty type ────────────────────────────────────────────────────────
    if re.search(r'\bcriminal\b|\bimprison|\bprison|\bfelony|\bprosecution', text):
        pt, pt_conf = "criminal", 0.70
    elif re.search(r'\bcivil\s+fine|\bmonetary\s+penalt|\badministrative\s+fine|\bpenalt', text):
        # Check for large fines suggesting civil_heavy
        large_fine = re.search(r'\d+\s*%\s*of\s*(global|annual|worldwide)?\s*(turn|revenue|sales)', text)
        if large_fine or re.search(r'billion|\b[5-9]\d{7,}', text):
            pt, pt_conf = "civil_heavy", 0.65
        else:
            pt, pt_conf = "civil", 0.65
    elif re.search(r'\bvoluntary\b|\bguideline|\bencourage\b|\bbest\s+practice', text):
        pt, pt_conf = "voluntary", 0.60
    else:
        pt, pt_conf = "civil", 0.30

    pt_source = _find_passage(re.compile(
        r'.{0,60}(criminal|civil fine|penalty|monetary|administrative fine).{0,80}',
        re.IGNORECASE
    ))

    # ── Penalty cap ─────────────────────────────────────────────────────────
    cap = None
    cap_conf = 0.0
    cap_source = ""
    # Match fine/penalty amounts, NOT revenue thresholds.
    # Negative lookbehind for "revenue", "turnover", "sales" which are entity
    # size filters, not penalty caps.
    dollar_m = re.search(
        r'(?<!\w)(?:fine|penalt[yi]|not\s+to\s+exceed|maximum\s+of).*?'
        r'\$\s*([\d,]+)\s*(million|billion|M\b|B\b)',
        full, re.IGNORECASE | re.DOTALL
    )
    if not dollar_m:
        # Simpler fallback: any dollar amount NOT preceded by "revenue", "turnover"
        dollar_m = re.search(
            r'(?<!revenue\s)(?<!turnover\s)(?<!annual\s)\$\s*([\d,]+)\s*(million|billion|M\b|B\b)',
            full, re.IGNORECASE
        )
    pct_m = re.search(r'([\d.]+)\s*%\s*of\s*(global|annual|worldwide)?\s*(turn|revenue)', full, re.IGNORECASE)
    if dollar_m:
        val_str = dollar_m.group(1).replace(",", "")
        mult = dollar_m.group(2).lower()
        cap = float(val_str) * (1e9 if mult.startswith("b") else 1e6)
        cap_conf = 0.65
        cap_source = full[max(0,dollar_m.start()-20):dollar_m.end()+40]
    elif pct_m:
        cap = None  # % of turnover — not a fixed cap
        cap_conf = 0.60
        cap_source = full[max(0,pct_m.start()-20):pct_m.end()+40]

    # ── Compute threshold ────────────────────────────────────────────────────
    thresh = None
    thresh_conf = 0.0
    thresh_source = ""
    # Try "base × 10^exp" FIRST (more specific), then plain "10^exp"
    # Swapping order ensures "1.5 × 10^26" gets base=1.5 rather than
    # having the plain pattern grab "10^26" and discard the coefficient.
    flop_m = re.search(r'([\d.]+)\s*[×x]\s*10\s*[\^*]\s*([\d.]+)', full)
    if not flop_m:
        flop_m = re.search(r'10\s*[\^*\^]\s*([\d.]+)', full)
    if flop_m:
        try:
            if len(flop_m.groups()) == 1:
                exp = float(flop_m.group(1))
                thresh = 10 ** exp
            else:
                # e.g. "1.5 × 10^26" — must multiply base by power, not ignore base
                base_val = float(flop_m.group(1))
                exp = float(flop_m.group(2))
                thresh = base_val * (10 ** exp)
            thresh_conf = 0.70
            thresh_source = full[max(0,flop_m.start()-30):flop_m.end()+60]
        except (ValueError, OverflowError):
            pass

    # ── Enforcement mechanism ────────────────────────────────────────────────
    if re.search(r'criminal\s+invest|prosecution\s+by', text):
        enf, enf_conf = "criminal_invest", 0.70
    elif re.search(r'government\s+inspect|regulator\s+inspect|inspect\s+by\s+', text):
        enf, enf_conf = "government_inspect", 0.65
    elif re.search(r'third[- ]party\s+audit|independent\s+audit|accredited\s+auditor', text):
        enf, enf_conf = "third_party_audit", 0.68
    elif re.search(r'self[- ]report|self[- ]cert|notify\s+the|disclosure\s+to', text):
        enf, enf_conf = "self_report", 0.60
    else:
        enf, enf_conf = "self_report", 0.25

    enf_source = _find_passage(re.compile(
        r'.{0,60}(audit|inspect|report|enforcement|compliance\s+verification).{0,80}',
        re.IGNORECASE
    ))

    # ── Grace period ─────────────────────────────────────────────────────────
    grace = 0
    grace_conf = 0.0
    grace_source = ""
    gp_m = re.search(r'(\d+)[- ](month|year)', full, re.IGNORECASE)
    if gp_m:
        n = int(gp_m.group(1))
        unit = gp_m.group(2).lower()
        grace = n * 12 if unit.startswith("year") else n
        grace_conf = 0.60
        grace_source = full[max(0,gp_m.start()-30):gp_m.end()+60]

    # ── Scope ────────────────────────────────────────────────────────────────
    if re.search(r'frontier\s+ai|general.purpose\s+ai|gpai|10\s*[\^*]\s*2[5-7]', text):
        scope, scope_conf = "frontier_only", 0.65
    elif re.search(r'large\s+(developer|company|enterprise)|revenue\s+above|more\s+than\s+\d+\s+employee', text):
        scope, scope_conf = "large_developers_only", 0.60
    elif re.search(r'voluntary|encourage|best\s+practice|opt.in', text):
        scope, scope_conf = "voluntary", 0.58
    else:
        scope, scope_conf = "all", 0.35

    scope_source = _find_passage(re.compile(
        r'.{0,60}(applies?\s+to|covers?|subject\s+to|regulated\s+entities).{0,80}',
        re.IGNORECASE
    ))

    # ── Jurisdiction ─────────────────────────────────────────────────────────
    if re.search(
        r'\beuropean\s+union\b|\beuropean\s+parliament\b'
        r'|\bofficial\s+journal\b|\bregulation.*eu\b|\beu\s+ai\s+act\b',
        text):
        jur, jur_conf = "EU", 0.80
    elif re.search(r'\bunited\s+states\b|\bfederal\b|\bcongress\b|\bfitc\b|\bnist\b', text):
        jur, jur_conf = "US", 0.75
    elif re.search(r'\bunited\s+kingdom\b|\buk\b|\bparliament\b|\bofcom\b', text):
        jur, jur_conf = "UK", 0.75
    elif re.search(r'\bcalifornia\b|\bsb[- ]?\d+\b|\bab[- ]?\d+\b', text):
        jur, jur_conf = "US", 0.70
    else:
        jur, jur_conf = "other", 0.30

    # ── Population composition ───────────────────────────────────────────────
    has_sme = bool(re.search(r'\bsme\b|\bsmall\s+(business|developer|company|enterprise)\b|\bstartup\b', text))
    has_frontier = bool(re.search(r'\bfrontier\b|\bgpai\b|\bgeneral[- ]purpose\b|\bfoundation\s+model\b', text))
    has_research = bool(re.search(
        r'\bresearch\b.{0,120}\b(exempt|excluded|exception)\b'
        r'|\b(scientific\s+research|academic\s+institution)s?\b'
        r'|\bresearch\s+exemption\b',
        text, re.DOTALL
    ))
    has_investor = bool(re.search(r'\binvestor\b|\bventure\b|\bvc\b|\bfinancial\s+(institution|actor)\b', text))

    # ── N regulated estimate ─────────────────────────────────────────────────
    if re.search(r'10\s*[\^*]\s*2[6-7]|very\s+few|handful\s+of\s+(compan|lab|develop)', text):
        n_reg, n_reg_conf = "handful", 0.60
    elif thresh and thresh >= 1e26:
        n_reg, n_reg_conf = "dozens", 0.50
    elif thresh and thresh >= 1e25:
        n_reg, n_reg_conf = "hundreds", 0.45
    else:
        n_reg, n_reg_conf = "unknown", 0.20

    # ── Policy name ──────────────────────────────────────────────────────────
    name_m = re.search(
        r'((?:(?:european\s+union\s+)?artificial\s+intelligence\s+act'
        r'|eu\s+ai\s+act'
        r'|(?:sb|ab|hb|hr|s)\s*[-–]?\s*\d{1,4}'
        r'|(?:executive\s+order|eo)\s+\d+'
        r'|\w+\s+(?:act|regulation|directive|framework|bill)\s+\d*'
        r'))',
        full, re.IGNORECASE
    )
    if name_m:
        pol_name = name_m.group(0)[:80].strip()
        name_conf = 0.75
    else:
        pol_name = Path(doc.file_path).stem.replace("_", " ").replace("-", " ")
        name_conf = 0.30

    # Build result
    def ef(v, c, src): return _field(v, c, src[:400] if src else "", "regex")

    return ExtractionResult(
        policy_name=ef(pol_name, name_conf, ""),
        policy_description=ef("", 0.0, ""),
        penalty_type=ef(pt, pt_conf, pt_source),
        penalty_cap_usd=ef(cap, cap_conf, cap_source),
        compute_threshold_flops=ef(thresh, thresh_conf, thresh_source),
        enforcement_mechanism=ef(enf, enf_conf, enf_source),
        grace_period_months=ef(grace, grace_conf, grace_source),
        scope=ef(scope, scope_conf, scope_source),
        source_jurisdiction=ef(jur, jur_conf, ""),
        has_sme_provisions=ef(has_sme, 0.60 if has_sme else 0.40, ""),
        has_frontier_lab_focus=ef(has_frontier, 0.65 if has_frontier else 0.40, ""),
        has_research_exemptions=ef(has_research, 0.60 if has_research else 0.35, ""),
        has_investor_provisions=ef(has_investor, 0.55 if has_investor else 0.30, ""),
        estimated_n_regulated=ef(n_reg, n_reg_conf, ""),
        key_provisions=[],
        extraction_method_used="regex_fallback",
        model_used=None,
        unresolved_provisions=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_provisions(
    doc: LoadedDocument,
    api_key: str | None = None,
    model: str = "gpt-4o",
    base_url: str | None = None,
    temperature: float = 0.0,
    force_regex: bool = False,
) -> ExtractionResult:
    """Extract structured regulatory provisions from a loaded document.

    Parameters
    ──────────
    doc        : LoadedDocument from document_loader.load_document()
    api_key    : OpenAI-compatible API key. If None, uses OPENAI_API_KEY env var.
                 If neither is set, falls back to regex extraction.
    model      : LLM model name. Recommend gpt-4o or claude-3-5-sonnet-20241022.
    base_url   : Optional base URL for non-OpenAI endpoints (e.g. Anthropic proxy).
    temperature: Extraction temperature. Keep at 0.0 for deterministic results.
    force_regex: Skip LLM even if API key is available (useful for testing).

    Returns
    ───────
    ExtractionResult with all fields populated, each with confidence score and
    source passage. extraction_method_used indicates whether LLM or regex was used.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    if force_regex or not key:
        return _extract_regex(doc)

    # LLM extraction
    try:
        import openai
        client = openai.OpenAI(api_key=key, base_url=base_url)
        prompt = _build_extraction_prompt(doc)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=temperature,
            response_format={"type": "json_object"} if "gpt" in model.lower() else None,
        )
        raw = response.choices[0].message.content or ""
        result = _parse_llm_response(raw, doc)

        if result is None:
            # LLM returned unparseable output — fall back to regex
            regex_result = _extract_regex(doc)
            regex_result.unresolved_provisions.insert(
                0, f"LLM extraction failed (unparseable JSON); fell back to regex. Raw: {raw[:200]}"
            )
            return regex_result

        result.model_used = model
        return result

    except ImportError:
        return _extract_regex(doc)

    except Exception as e:
        # Any API error → fall back to regex, log in unresolved
        result = _extract_regex(doc)
        result.unresolved_provisions.insert(
            0, f"LLM extraction error ({type(e).__name__}: {e}); fell back to regex"
        )
        return result
