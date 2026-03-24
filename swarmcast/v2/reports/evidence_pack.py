"""
Evidence pack generator — produces a 3-page staff-ready PDF brief.

The target reader is a legislative staff person preparing for a markup
session on compute threshold legislation. They need:
  1. A one-paragraph description of what the policy does and who it covers
  2. Three charts they can hand to their member: compliance trajectory,
     relocation trajectory, and investment impact
  3. A table showing which numbers came from data vs assumption, so they
     can defend the analysis if challenged

Usage
─────
    from swarmcast.v2.reports.evidence_pack import generate_evidence_pack
    from swarmcast.v2.policy.parser import california_sb53

    spec = california_sb53()
    pack = generate_evidence_pack(
        policy_spec=spec,
        output_path="./sb53_evidence_pack.pdf",
        n_population=2000,
        num_rounds=16,
    )

The output is a self-contained PDF — no external dependencies, no internet
connection required, no special fonts needed.
"""

from __future__ import annotations

import io
import os
import warnings
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from swarmcast.v2.policy.parser import PolicySpec
    from swarmcast.v2.simulation.hybrid_loop import HybridSimResult


# ─────────────────────────────────────────────────────────────────────────────
# COLOURS  (muted, professional — suitable for government documents)
# ─────────────────────────────────────────────────────────────────────────────

C_NAVY    = (0.13, 0.19, 0.33)
C_SLATE   = (0.38, 0.45, 0.55)
C_LIGHT   = (0.91, 0.93, 0.96)
C_WHITE   = (1.0,  1.0,  1.0)
C_GREEN   = (0.18, 0.55, 0.34)
C_AMBER   = (0.80, 0.55, 0.10)
C_RED     = (0.72, 0.20, 0.18)
C_BORDER  = (0.75, 0.78, 0.82)

GROUNDED_COL  = C_GREEN
DIRECTIONAL_COL = C_AMBER
ASSUMED_COL   = C_RED


def _rgb(*t):
    """reportlab Color from (r, g, b) 0-1 tuple."""
    from reportlab.lib.colors import Color
    return Color(*t)


# ─────────────────────────────────────────────────────────────────────────────
# CORE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_evidence_pack(
    policy_spec: "PolicySpec",
    output_path: str = "./evidence_pack.pdf",
    n_population: int = 2000,
    num_rounds: int = 16,
    n_ensemble: int = 3,
    seed: int = 42,
    verbose: bool = True,
    ingest_result=None,
) -> str:
    """Run simulation and render a PDF evidence pack.

    Returns the path to the written PDF.

    The pages:
      Page 1 — Policy description, severity rationale, coverage scope
      Page 2 — Compliance S-curve, relocation trajectory, investment impact
      Page 3 — Epistemic status table, limitations, citation
      Page 4 — (Optional) Ingest traceability: extraction confidence table and
                derived parameters. Only included when ingest_result is provided.

    Parameters
    ──────────
    ingest_result : IngestResult from swarmcast.v2.ingest.pipeline, or None.
                   When provided, a 4th page is added showing extraction
                   confidence for each field and the derived parameter chain.

    Runs n_ensemble simulations and shows mean ± 1 SD on charts to give
    a sense of Monte Carlo variance.
    """
    warnings.filterwarnings("ignore")

    if verbose:
        print(f"[evidence pack] Running {n_ensemble} simulations for '{policy_spec.name}'...")

    results = _run_sims(policy_spec, n_population, num_rounds, n_ensemble, seed, verbose)
    _render_pdf(policy_spec, results, output_path, num_rounds, n_population, n_ensemble, verbose,
                ingest_result=ingest_result)
    return output_path


def _run_sims(spec, n_pop, n_rounds, n_ensemble, seed, verbose):
    """Run n_ensemble simulations and collect trajectory arrays."""
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from swarmcast.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation

    # Also run ±0.5 severity scenarios for the comparison band
    severities = {
        "low":  max(1.0, spec.severity - 0.5),
        "mid":  spec.severity,
        "high": min(5.0, spec.severity + 0.5),
    }

    all_compliance   = {k: [] for k in severities}
    all_relocation   = {k: [] for k in severities}
    all_investment   = {k: [] for k in severities}
    all_enforcement  = {k: [] for k in severities}
    final_summaries  = []

    for i in range(n_ensemble):
        for label, sev in severities.items():
            config = HybridSimConfig(
                n_population=n_pop, num_rounds=n_rounds,
                verbose=False, seed=seed + i,
            )
            r = run_hybrid_simulation(
                spec.name, spec.description, sev, config=config
            )
            all_compliance[label].append([
                s.get("compliance_rate", 0) for s in r.round_summaries
            ])
            all_relocation[label].append([
                s.get("relocation_rate", 0) for s in r.round_summaries
            ])
            all_investment[label].append([
                s.get("ai_investment_index", 50) for s in r.stock_history
            ])
            all_enforcement[label].append([
                s.get("enforcement_contact_rate", 0) for s in r.round_summaries
            ])
            if label == "mid" and i == 0:
                final_summaries.append(r.final_population_summary)
                final_summaries.append(r.final_stocks)

        if verbose:
            print(f"  Ensemble {i+1}/{n_ensemble} done")

    # Convert to numpy arrays: shape (n_ensemble, n_rounds)
    def _arr(d, k):
        return np.array(d[k])

    return {
        "compliance":  {k: _arr(all_compliance, k)  for k in severities},
        "relocation":  {k: _arr(all_relocation, k)  for k in severities},
        "investment":  {k: _arr(all_investment, k)  for k in severities},
        "enforcement": {k: _arr(all_enforcement, k) for k in severities},
        "final_pop":   final_summaries[0] if final_summaries else {},
        "final_stocks": final_summaries[1] if len(final_summaries) > 1 else {},
        "severities":  severities,
        "n_rounds":    n_rounds,
    }


def _render_pdf(spec, data, output_path, n_rounds, n_population, n_ensemble, verbose,
                ingest_result=None):
    """Render all three pages into a PDF using reportlab Platypus.

    All table cells use Paragraph objects so text wraps automatically.
    Chart Drawing objects use truncated strings to prevent overflow.
    All column widths are computed from the actual usable page width.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.graphics.shapes import Drawing, Rect, Line, String
    from reportlab.graphics import renderPDF
    from reportlab.lib.colors import Color

    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm
    USABLE_W = PAGE_W - 2 * MARGIN   # ~174mm = ~494 points

    def _c(*t):
        return Color(*t)

    # ── Paragraph styles ─────────────────────────────────────────────────────
    def _sty(name, **kw):  # Build a ParagraphStyle with Navy defaults, overridden by kw
        base = dict(fontName="Helvetica", fontSize=9, leading=12,
                    textColor=_c(*C_NAVY), spaceAfter=3, wordWrap="CJK")
        base.update(kw)
        return ParagraphStyle(name, **base)

    S = {
        "title":  _sty("T",  fontSize=16, fontName="Helvetica-Bold", spaceAfter=4),
        "h1":     _sty("H1", fontSize=11, fontName="Helvetica-Bold", spaceAfter=3),
        "h2":     _sty("H2", fontSize=9,  fontName="Helvetica-Bold", spaceAfter=2),
        "body":   _sty("B",  fontSize=8.5, leading=13, spaceAfter=5),
        "small":  _sty("S",  fontSize=7,  textColor=_c(*C_SLATE)),
        "caveat": _sty("CV", fontSize=7.5, textColor=_c(*C_SLATE), leading=11, spaceAfter=3),
        # Table cell styles — smaller font, no extra spaceAfter
        "cell":   _sty("CE", fontSize=7.5, leading=10, spaceAfter=0),
        "cell_b": _sty("CB", fontSize=7.5, leading=10, spaceAfter=0, fontName="Helvetica-Bold"),
        "cell_hd":_sty("CH", fontSize=7.5, leading=10, spaceAfter=0,
                        fontName="Helvetica-Bold", textColor=_c(*C_WHITE)),
        "cell_sm":_sty("CS", fontSize=6.5, leading=9, spaceAfter=0,
                        textColor=_c(*C_SLATE)),
    }

    def P(text, style="cell", **kw):  # Paragraph wrapper for ReportLab
        """Paragraph with automatic word wrap."""
        return Paragraph(str(text), S[style])

    def _green(t):  return Paragraph(t, ParagraphStyle("G", parent=S["cell"],  # Render text in GROUNDED green
                        fontName="Helvetica-Bold", textColor=_c(*GROUNDED_COL)))
    def _amber(t):  return Paragraph(t, ParagraphStyle("A", parent=S["cell"],  # Render text in DIRECTIONAL amber
                        fontName="Helvetica-Bold", textColor=_c(*DIRECTIONAL_COL)))
    def _red(t):    return Paragraph(t, ParagraphStyle("R", parent=S["cell"],  # Render text in ASSUMED red
                        fontName="Helvetica-Bold", textColor=_c(*ASSUMED_COL)))

    import html as _html
    story = []
    HR  = lambda: HRFlowable(width="100%", thickness=0.4, color=_c(*C_BORDER), spaceAfter=3)
    BR  = lambda n=1: Spacer(1, n * 4 * mm)

    # ── Common table style ────────────────────────────────────────────────────
    def _tbl_style(header_color=C_NAVY):
        return TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  _c(*header_color)),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c(*C_WHITE), _c(*C_LIGHT)]),
            ("GRID",          (0,0), (-1,-1), 0.25, _c(*C_BORDER)),
            ("LEFTPADDING",   (0,0), (-1,-1), 4),
            ("RIGHTPADDING",  (0,0), (-1,-1), 4),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ])

    # ── Chart builder ─────────────────────────────────────────────────────────
    lo = max(1.0, spec.severity - 0.5)
    mid_sev = spec.severity
    hi  = min(5.0, spec.severity + 0.5)
    if spec.recommended_severity_sweep:
        lo, mid_sev, hi = spec.recommended_severity_sweep

    CHART_H = 58 * mm

    def _chart(title_str, series_dict, note="", full_width=False):
        """Build a ReportLab Drawing with mean lines and ±1 SD bands for low/mid/high scenarios.
        Returns a Drawing object ready for insertion into a Platypus story.
        """
        dw = USABLE_W if full_width else (USABLE_W - 4*mm) / 2
        dh = CHART_H + 10 * mm
        d  = Drawing(float(dw), float(dh))
        d.add(Rect(0, 0, dw, dh, fillColor=_c(*C_WHITE), strokeColor=None))

        # Title — truncate to fit, centred
        max_title = int(dw / 5)
        d.add(String(dw/2, dh - 9, title_str[:max_title],
                     fontName="Helvetica-Bold", fontSize=8,
                     textAnchor="middle", fillColor=_c(*C_NAVY)))

        # Sub-note — truncate to fit
        if note:
            max_note = int(dw / 4.2)
            d.add(String(dw/2, dh - 17, note[:max_note],
                         fontName="Helvetica", fontSize=6,
                         textAnchor="middle", fillColor=_c(*C_SLATE)))

        # Plot area
        px0 = 30.0
        py0 = 14.0
        pw  = float(dw) - px0 - 6
        ph  = float(CHART_H) - 8

        # Grid + Y labels
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            y = py0 + frac * ph
            d.add(Line(px0, y, px0+pw, y, strokeColor=_c(*C_BORDER), strokeWidth=0.25))
            d.add(String(px0 - 3, y - 3, f"{frac:.0%}",
                         fontName="Helvetica", fontSize=5.5,
                         textAnchor="end", fillColor=_c(*C_SLATE)))

        # Axes
        d.add(Line(px0, py0, px0, py0+ph, strokeColor=_c(*C_SLATE), strokeWidth=0.5))
        d.add(Line(px0, py0, px0+pw, py0, strokeColor=_c(*C_SLATE), strokeWidth=0.5))

        colours = {"low": _c(0.2,0.6,0.9), "mid": _c(*C_NAVY), "high": _c(*C_RED)}
        n_x = n_rounds

        for key, arr in series_dict.items():
            if arr.shape[0] == 0: continue
            mean_v = arr.mean(axis=0)
            col = colours.get(key, _c(*C_SLATE))
            pts = []
            for xi, yv in enumerate(mean_v):
                x = px0 + xi / max(1, n_x - 1) * pw
                y = py0 + min(1.0, max(0.0, float(yv))) * ph
                pts.append((x, y))
            for i in range(len(pts) - 1):
                d.add(Line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                           strokeColor=col,
                           strokeWidth=1.4 if key=="mid" else 0.8,
                           strokeDashArray=None if key=="mid" else [3,2]))

        # X labels
        for xi in range(0, n_x, 4):
            x = px0 + xi / max(1, n_x - 1) * pw
            d.add(Line(x, py0-2, x, py0+2, strokeColor=_c(*C_SLATE), strokeWidth=0.3))
            d.add(String(x, py0 - 9, f"Y{xi//4+1}",
                         fontName="Helvetica", fontSize=5.5,
                         textAnchor="middle", fillColor=_c(*C_SLATE)))

        # Legend — fit inside dw
        legend_items = [
            (f"Sev {lo:.1f}", colours["low"]),
            (f"Sev {mid_sev:.1f}", colours["mid"]),
            (f"Sev {hi:.1f}", colours["high"]),
        ]
        step = min(float(dw) / 4, 55.0)
        for i, (lbl, col) in enumerate(legend_items):
            x = px0 + i * step
            d.add(Rect(x, 3, 7, 4, fillColor=col, strokeColor=None))
            d.add(String(x + 9, 3, lbl,
                         fontName="Helvetica", fontSize=5.5,
                         fillColor=_c(*C_SLATE)))
        return d

    # ── PAGE 1 ────────────────────────────────────────────────────────────────
    story.append(Paragraph("PolicyLab Evidence Pack", S["small"]))
    story.append(Paragraph(_html.escape(spec.name), S["title"]))
    story.append(Paragraph(
        f"Prepared {date.today().strftime('%B %d, %Y')}  ·  "
        f"Severity: <b>{spec.severity:.1f} / 5.0</b>  ·  "
        f"Horizon: {n_rounds} quarters ({n_rounds//4} years)  ·  "
        f"Population: {n_population:,} agents",
        S["small"]
    ))
    story.append(HR())
    story.append(BR(0.4))

    story.append(Paragraph("Policy Description", S["h1"]))
    desc = spec.description or "(no description provided)"
    story.append(Paragraph(_html.escape(desc[:500]), S["body"]))
    story.append(BR(0.3))

    # Severity breakdown
    story.append(Paragraph("Severity scoring", S["h1"]))
    story.append(Paragraph(
        "Each dimension below contributed a sub-score; total scaled to 1–5. "
        "1 = voluntary guidelines, 3 = civil fines (GDPR/EU AI Act level), 5 = criminal liability.",
        S["body"]
    ))

    # Column widths summing exactly to USABLE_W
    W1, W2, W3 = USABLE_W * 0.53, USABLE_W * 0.15, USABLE_W * 0.32
    just_rows = [[P("Dimension","cell_hd"), P("Score","cell_hd"), P("Notes","cell_hd")]]
    for j in spec.justification:
        if "raw_total" in j:
            import re
            m = re.search(r'severity=(\d+\.?\d*)', j)
            sev_str = f"→ {m.group(1)}" if m else ""
            just_rows.append([P("Final severity score","cell_b"), P(sev_str,"cell"), P("Weighted sum of sub-scores, scaled to 1–5","cell_sm")])
        elif j.startswith("[") and "→" in j:
            bracket_end = j.find("]")
            if bracket_end == -1:
                bracket_end = j.find("→") - 1
            dim_raw = j[1:bracket_end].strip()
            after = j.split("→", 1)
            score_raw = after[1].strip() if len(after) > 1 else ""
            import re
            score_match = re.match(r'^[\-\+]?\d+\.?\d*', score_raw)
            score_str = score_match.group(0) if score_match else score_raw[:8]
            if "=" in dim_raw:
                key, _, val = dim_raw.partition("=")
                dim_raw = f"{key.replace('_',' ').strip().title()}: {val.replace('_',' ').strip()}"
            just_rows.append([P(dim_raw,"cell"), P(score_str,"cell"), P("","cell")])
        else:
            just_rows.append([P("","cell"), P("","cell"), P(j[:160],"cell_sm")])

    if len(just_rows) > 1:
        t = Table(just_rows, colWidths=[W1, W2, W3])
        t.setStyle(_tbl_style())
        story.append(t)
        story.append(BR(0.4))

    # Structured parameters
    story.append(Paragraph("Bill parameters", S["h1"]))
    WA, WB, WC = USABLE_W * 0.35, USABLE_W * 0.38, USABLE_W * 0.27
    params = [
        [P("Parameter","cell_hd"), P("Value","cell_hd"), P("Source","cell_hd")],
        [P("Penalty type"), P(spec.penalty_type.replace("_"," ").title()), P("Bill text")],
        [P("Penalty cap"),
         P(f"${spec.penalty_cap_usd/1e6:.1f}M" if spec.penalty_cap_usd else "Uncapped / % turnover"),
         P("Bill text")],
        [P("Compute threshold"),
         P(f"10^{round(len(str(int(spec.compute_threshold_flops or 0)))-1)} FLOPS"
           if spec.compute_threshold_flops else "Not specified"),
         P("Bill text")],
        [P("Enforcement"),       P(spec.enforcement_mechanism.replace("_"," ").title()), P("Bill text")],
        [P("Grace period"),      P(f"{spec.grace_period_months} months"),              P("Bill text")],
        [P("Scope"),             P(spec.scope.replace("_"," ").title()),               P("Bill text")],
    ]
    t2 = Table(params, colWidths=[WA, WB, WC])
    t2.setStyle(_tbl_style(C_SLATE))
    story.append(t2)
    story.append(BR(0.3))

    story.append(Paragraph(
        f"<b>Sensitivity sweep:</b> Page 2 shows severity {lo} (relaxed), "
        f"{mid_sev} (as-written), and {hi} (strict).",
        S["caveat"]
    ))
    story.append(PageBreak())

    # ── PAGE 2 ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Simulation Results", S["title"]))
    story.append(Paragraph(
        f"Population: {n_population:,} agents  ·  Horizon: {n_rounds} qtrs  ·  "
        f"Ensemble: {n_ensemble} runs  ·  Mean shown; dashed = ±0.5 severity scenarios.",
        S["small"]
    ))
    story.append(HR())
    story.append(BR(0.3))

    # Side-by-side charts — each takes exactly half the usable width
    half_w = (USABLE_W - 3*mm) / 2
    c_comp  = _chart("Compliance rate", data["compliance"],
                     "Fraction of active companies compliant")
    c_reloc = _chart("Relocation rate",  data["relocation"],
                     "Fraction that left the source jurisdiction")
    inv_norm = {k: data["investment"][k] / 100.0 for k in data["investment"]}
    c_inv   = _chart("AI investment index (100 = baseline)", inv_norm,
                     "Indexed to baseline. Proportional to domestic AI R&D spend.",
                     full_width=True)

    # Two charts side by side — zero padding in the table cells
    top_charts = Table([[c_comp, c_reloc]], colWidths=[half_w, half_w])
    top_charts.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
    ]))
    story.append(top_charts)
    story.append(BR(0.2))

    # Investment chart full width
    inv_tbl = Table([[c_inv]], colWidths=[USABLE_W])
    inv_tbl.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story.append(inv_tbl)
    story.append(BR(0.2))

    # Final-state summary
    fp = data["final_pop"]
    fs = data["final_stocks"]
    story.append(Paragraph("Final state (as-written scenario)", S["h2"]))
    C1, C2, C3 = USABLE_W*0.38, USABLE_W*0.24, USABLE_W*0.38
    summ = [
        [P("Metric","cell_hd"),          P("Value","cell_hd"),                                    P("Unit","cell_hd")],
        [P("Compliance rate"),           P(f"{fp.get('compliance_rate',0):.0%}"),                 P("of active companies")],
        [P("Relocation rate"),           P(f"{fp.get('relocation_rate',0):.0%}"),                 P("of original companies")],
        [P("Enforcement contacts"),      P(f"{fp.get('enforcement_contact_rate',0):.0%}"),        P("fraction ever contacted")],
        [P("Ever lobbied"),              P(f"{fp.get('ever_lobbied_rate',0):.0%}"),               P("of companies")],
        [P("AI investment index"),       P(f"{fs.get('ai_investment_index',0):.0f} / 100"),       P("100 = baseline")],
        [P("Regulatory burden"),         P(f"{fs.get('regulatory_burden',0):.0f} / 100"),         P("compliance overhead")],
    ]
    st = Table(summ, colWidths=[C1, C2, C3])
    st.setStyle(_tbl_style())
    story.append(st)
    story.append(PageBreak())

    # ── PAGE 3 ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Parameter Sources and Assumptions", S["title"]))
    story.append(Paragraph(
        "GROUNDED = derived from published empirical data.  "
        "DIRECTIONAL = direction supported, magnitude uncertain.  "
        "ASSUMED = no calibration target — sweep before citing.",
        S["body"]
    ))
    story.append(HR())
    story.append(BR(0.3))

    # Epistemic table — proportional column widths
    EP1 = USABLE_W * 0.28   # parameter name
    EP2 = USABLE_W * 0.13   # value
    EP3 = USABLE_W * 0.15   # status
    EP4 = USABLE_W * 0.44   # source

    def _status_p(txt):
        fn = {"GROUNDED": _green, "DIRECTIONAL": _amber, "ASSUMED": _red}
        return fn.get(txt, P)(txt)

    ep_header = [P("Parameter","cell_hd"), P("Value","cell_hd"),
                 P("Status","cell_hd"),    P("Source / Notes","cell_hd")]
    ep_rows = [ep_header] + [
        [P(a), P(b), _status_p(c), P(d, "cell_sm")]
        for a, b, c, d in [
            ("Compliance lambda (large)",       "3.32 rounds", "GROUNDED",
             "DLA Piper 2020 GDPR: 91% large-firm compliance at 24mo"),
            ("Compliance lambda (SME)",         "10.9 rounds", "GROUNDED",
             "DLA Piper 2020 GDPR: 52% SME compliance at 24mo"),
            ("Compliance lambda (frontier lab)","1.5 rounds",  "ASSUMED",
             "No direct data. Frontier labs have dedicated policy teams. Sweep [1.0–2.5]"),
            ("Relocation max rate",             "0.0318/qtr",  "DIRECTIONAL",
             "EU AI Act Transparency Register — self-reported threats, not observed relocations"),
            ("Relocation threshold (large)",    "72.0 burden", "ASSUMED",
             "Calibrated to EU AI Act Transparency Register. Sweep [60–85]"),
            ("Relocation threshold (frontier)", "45.0 burden", "ASSUMED",
             "Substantially lower than large companies; no direct data. Sweep [35–60]"),
            ("Severity-cubed scaling",          "(sev/3)^3",   "DIRECTIONAL",
             "Nonlinear threshold response grounded; cubic exponent assumed. Sweep [linear, quadratic, cubic]"),
            ("Enforcement contact rate",        "0.005/qtr/sev","GROUNDED",
             "DLA Piper 2020: 6%/yr GDPR ÷ 4 qtrs ÷ 3 severity units"),
            ("Ongoing burden per severity",     "1.5 pts/qtr", "DIRECTIONAL",
             "Monitoring costs scale with severity; 1.5 is estimated. Sweep [0.5–4.0]"),
            ("SME resource cap",                "60% max",     "ASSUMED",
             "IAPP 2019: 40% of SMEs had no dedicated compliance staff at 24mo"),
            ("R&amp;D investment elasticity",       "-0.003/qtr",  "GROUNDED",
             "Springler 2023 OECD PMR elasticity e=-0.197, converted to quarterly"),
            ("Innovation coefficient",          "+0.000345/qtr","GROUNDED",
             "Ugur 2016 meta-analysis: R&amp;D to TFP elasticity e=0.138"),
        ]
    ]
    ep_t = Table(ep_rows, colWidths=[EP1, EP2, EP3, EP4])
    ep_t.setStyle(_tbl_style())
    story.append(ep_t)
    story.append(BR(0.4))

    # Limitations
    story.append(Paragraph("What this analysis cannot tell you", S["h1"]))
    LW1, LW2 = USABLE_W * 0.30, USABLE_W * 0.70
    lim_rows = [
        [P("[1] No frontier lab validation","cell_b"),
         P("Frontier lab parameters are assumed. No published compliance survey exists for this group.", "cell_sm")],
        [P("[2] Relocation = stated intent","cell_b"),
         P("12% calibration point is from self-reported relocation threats, not observed moves.", "cell_sm")],
        [P("[3] Compliance ≠ risk reduction","cell_b"),
         P("The model tracks formal compliance, not whether the regulation reduces AI safety risk.", "cell_sm")],
        [P("[4] Rational actor assumption","cell_b"),
         P("Relocation is cost-benefit. Mission-driven resistance and national identity are not modelled.", "cell_sm")],
        [P("[5] Single policy","cell_b"),
         P("The model runs one policy at a time. Overlapping federal/state/international rules are not captured.", "cell_sm")],
    ]
    lim_t = Table(lim_rows, colWidths=[LW1, LW2])
    lim_t.setStyle(TableStyle([
        ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 4),
        ("LINEBELOW",    (0,0), (-1,-2), 0.25, _c(*C_BORDER)),
    ]))
    story.append(lim_t)
    story.append(BR(0.4))

    # Citation
    story.append(HR())
    story.append(Paragraph("Citation", S["h2"]))
    story.append(Paragraph(
        f"PolicyLab v2 (2025). Agent-based simulation of AI governance policy responses. "
        f"Generated {date.today().isoformat()}. "
        f"Calibrated against: DLA Piper (2020) GDPR Enforcement Survey; "
        f"Springler (2023) OECD PMR; Ugur (2016) R&amp;D meta-analysis; "
        f"EC Impact Assessment SWD(2021)84; EU AI Act Transparency Register. "
        f"github.com/Ambar-13/PolicyLab",
        S["caveat"]
    ))

    # ── PAGE 4 (optional) — Ingest traceability ──────────────────────────────
    if ingest_result is not None:
        story.append(PageBreak())
        story.append(Paragraph("Ingest Traceability", S["title"]))
        story.append(Paragraph(
            "Extraction confidence and derived parameter provenance from document ingestion. "
            "GROUNDED = high confidence (≥80%).  DIRECTIONAL = moderate (50–79%).  "
            "ASSUMED = low confidence (<50%) — verify before citing.",
            S["body"]
        ))
        story.append(HR())
        story.append(BR(0.3))

        # ── Extraction confidence table ───────────────────────────────────────
        story.append(Paragraph("Extraction confidence", S["h1"]))
        EC1 = USABLE_W * 0.22   # field name
        EC2 = USABLE_W * 0.18   # value
        EC3 = USABLE_W * 0.10   # confidence
        EC4 = USABLE_W * 0.12   # epistemic tag
        EC5 = USABLE_W * 0.38   # source passage

        ex = ingest_result.extraction
        field_map = [
            ("policy_name",             "Policy name"),
            ("penalty_type",            "Penalty type"),
            ("penalty_cap_usd",         "Penalty cap (USD)"),
            ("compute_threshold_flops", "Compute threshold"),
            ("enforcement_mechanism",   "Enforcement"),
            ("grace_period_months",     "Grace period (months)"),
            ("scope",                   "Scope"),
            ("source_jurisdiction",     "Jurisdiction"),
            ("has_sme_provisions",      "SME provisions"),
            ("has_frontier_lab_focus",  "Frontier lab focus"),
            ("has_research_exemptions", "Research exemptions"),
            ("estimated_n_regulated",   "N entities regulated"),
        ]

        ec_header = [
            P("Field",          "cell_hd"),
            P("Value",          "cell_hd"),
            P("Confidence",     "cell_hd"),
            P("Status",         "cell_hd"),
            P("Source passage", "cell_hd"),
        ]
        ec_rows = [ec_header]
        for attr, label in field_map:
            fld = getattr(ex, attr, None)
            if fld is None:
                continue
            raw_val = fld.value
            if isinstance(raw_val, float) and raw_val > 1e6:
                val_str = f"{raw_val:.2e}"
            elif isinstance(raw_val, float):
                val_str = f"{raw_val:.2f}"
            else:
                val_str = str(raw_val)[:40]
            conf_str = f"{fld.confidence:.0%}"
            tag = fld.epistemic_tag if hasattr(fld, "epistemic_tag") else ""
            passage = ""
            if hasattr(fld, "source_passage") and fld.source_passage:
                sp = fld.source_passage
                if not sp.startswith("[unverif"):
                    passage = _html.escape(sp[:120])
            ec_rows.append([
                P(label,    "cell"),
                P(val_str,  "cell"),
                P(conf_str, "cell"),
                _status_p(tag),
                P(passage,  "cell_sm"),
            ])

        ec_t = Table(ec_rows, colWidths=[EC1, EC2, EC3, EC4, EC5])
        ec_t.setStyle(_tbl_style())
        story.append(ec_t)
        story.append(BR(0.4))

        # ── Derived parameters table ──────────────────────────────────────────
        built = getattr(ingest_result, "built_spec", None)
        derived = getattr(built, "derived_params", None) if built is not None else None
        if derived:
            story.append(Paragraph("Derived parameters", S["h1"]))
            DP1 = USABLE_W * 0.22   # parameter name
            DP2 = USABLE_W * 0.14   # value
            DP3 = USABLE_W * 0.10   # confidence
            DP4 = USABLE_W * 0.12   # epistemic tag
            DP5 = USABLE_W * 0.42   # justification

            dp_header = [
                P("Parameter",    "cell_hd"),
                P("Value",        "cell_hd"),
                P("Confidence",   "cell_hd"),
                P("Status",       "cell_hd"),
                P("Justification","cell_hd"),
            ]
            dp_rows = [dp_header]
            for pname, dp in derived.items():
                raw_val = dp.value
                if isinstance(raw_val, dict):
                    val_str = ", ".join(f"{k}:{v:.0%}" for k, v in list(raw_val.items())[:3])
                elif isinstance(raw_val, float):
                    val_str = f"{raw_val:.2f}"
                else:
                    val_str = str(raw_val)[:40]
                dp_rows.append([
                    P(pname.replace("_", " "),  "cell"),
                    P(val_str,                  "cell_sm"),
                    P(f"{dp.confidence:.0%}",   "cell"),
                    _status_p(dp.epistemic_tag),
                    P(_html.escape(dp.justification[:200]), "cell_sm"),
                ])

            dp_t = Table(dp_rows, colWidths=[DP1, DP2, DP3, DP4, DP5])
            dp_t.setStyle(_tbl_style())
            story.append(dp_t)
            story.append(BR(0.3))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"PolicyLab — {spec.name}",
        author="PolicyLab",
    )
    doc.build(story)
    if verbose:
        print(f"[evidence pack] Written → {output_path}")