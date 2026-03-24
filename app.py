"""
PolicyLab — AI governance policy simulator.

Run with:  streamlit run app.py

Four modes:
  Upload a document  — PDF/txt/md/docx → AI-extracted parameters → simulation
  Analyze a bill     — paste text or fill structured parameters manually
  Compare policies   — up to 4 policies side-by-side
  Influence scenario — test network resilience by injecting false beliefs into agent reasoning
"""

import os, sys, io, json, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="PolicyLab", page_icon="⚖️", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "PolicyLab — github.com/Ambar-13/PolicyLab"},
)

st.markdown("""<style>
[data-testid="stAppViewContainer"] { font-family: "Inter", "Segoe UI", sans-serif; }
h1 { color: #0d3155; font-weight: 700; }
h2 { color: #1b4f8a; font-weight: 600; font-size: 1.15rem; margin-top: 1.2rem; }
[data-testid="stSidebar"] { background: #0d3155; }
[data-testid="stSidebar"] * { color: #e8eef6 !important; }
.tag-grounded    { background:#d4edda; color:#155724; padding:2px 8px; border-radius:4px; font-size:.75rem; font-weight:700; font-family:monospace; }
.tag-directional { background:#fff3cd; color:#856404; padding:2px 8px; border-radius:4px; font-size:.75rem; font-weight:700; font-family:monospace; }
.tag-assumed     { background:#f8d7da; color:#721c24; padding:2px 8px; border-radius:4px; font-size:.75rem; font-weight:700; font-family:monospace; }
.extract-row     { display:flex; align-items:center; padding:5px 0; border-bottom:1px solid #f0f2f5; gap:10px; }
.extract-label   { font-size:.82rem; color:#495057; min-width:160px; }
.extract-value   { font-size:.88rem; font-weight:600; color:#0d3155; flex:1; }
.evidence-box    { background:#f8f9fa; border-left:3px solid #1b4f8a; padding:6px 10px; font-size:.78rem; color:#495057; font-style:italic; border-radius:0 4px 4px 0; margin:2px 0 6px 0; }
.low-conf-warn   { background:#fff3cd; border:1px solid #ffc107; border-radius:6px; padding:10px 14px; font-size:.85rem; color:#856404; margin:8px 0; }
.poldiv          { border:none; border-top:2px solid #dee2e6; margin:18px 0; }
</style>""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ PolicyLab")
    st.caption("AI governance policy simulator")
    st.divider()
    mode = st.radio("Mode", [
        "📄 Upload document",
        "✏️ Analyze a bill",
        "⚖️ Compare policies",
        "🧪 Influence scenario",
    ])
    st.divider()
    st.subheader("Simulation")
    n_population = st.select_slider("Population", [200,500,1000,2000,5000], value=1000)
    num_rounds   = st.select_slider("Horizon (quarters)", [4,8,12,16,20], value=8)
    n_ensemble   = st.select_slider("Ensemble runs", [1,3,5], value=3)
    seed         = st.number_input("Seed", value=42, step=1)
    st.divider()
    api_key  = st.text_input("API key (optional)", type="password", placeholder="sk-...",
                              help="OpenAI-compatible key for LLM document extraction. "
                                   "Without it, regex extraction runs.")
    llm_model = st.selectbox("Extraction model",
                              ["gpt-4o","gpt-4o-mini","gpt-4-turbo","claude-3-5-sonnet-20241022"])
    st.caption("Calibrated: DLA Piper 2020 · EU AI Act · OECD PMR · Ugur 2016")

# ── Shared helpers ────────────────────────────────────────────────────────────
@st.cache_resource
def _load_presets():
    from policylab.v2.policy.parser import california_sb53, eu_ai_act_gpai, ny_raise_act, hypothetical_compute_ban
    return {
        "California SB-53 (2025)":          california_sb53(),
        "EU AI Act — GPAI systemic risk":   eu_ai_act_gpai(),
        "NY RAISE Act (proposed)":          ny_raise_act(),
        "Hypothetical compute ban (10²⁶)":  hypothetical_compute_ban(1e26),
        "Custom": None,
    }

@st.cache_data(show_spinner=False)
def _run_sim(name, desc, sev, _n_pop, _rounds, _ens, _seed,
             _type_dist=None, _ccf=1.0, _jur="EU", _hk=1.0):
    from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation
    out = {}
    for lbl, sv in [("low",max(1.0,sev-.5)),("mid",sev),("high",min(5.0,sev+.5))]:
        cr, rr, ir = [], [], []
        for i in range(_ens):
            cfg = HybridSimConfig(n_population=_n_pop, num_rounds=_rounds,
                verbose=False, seed=_seed+i, type_distribution=_type_dist,
                compute_cost_factor=_ccf, source_jurisdiction=_jur, hk_epsilon=_hk)
            r = run_hybrid_simulation(name, desc, sv, config=cfg)
            cr.append([s.get("compliance_rate",0) for s in r.round_summaries])
            rr.append([s.get("relocation_rate",0) for s in r.round_summaries])
            ir.append([s.get("ai_investment_index",50) for s in r.stock_history])
            if lbl=="mid" and i==0:
                out["final_pop"]    = r.final_population_summary
                out["final_stocks"] = r.final_stocks
                out["jurisdiction"] = r.jurisdiction_summary
        out[f"cr_{lbl}"]=np.array(cr); out[f"rr_{lbl}"]=np.array(rr); out[f"ir_{lbl}"]=np.array(ir)
    out["sev"]=sev; out["nr"]=_rounds
    return out

def _badge(tag):
    cls = {"GROUNDED":"tag-grounded","DIRECTIONAL":"tag-directional","ASSUMED":"tag-assumed"}.get(tag,"tag-assumed")
    return f'<span class="{cls}">{tag}</span>'

def _sev_gauge(sev):
    pct = (sev-1)/4*100
    col = "#198754" if sev<=2 else "#fd7e14" if sev<=3 else "#dc3545" if sev<=4 else "#6f42c1"
    lbl = {1:"Voluntary",2:"Reporting",3:"Civil fines",4:"Strict",5:"Criminal"}.get(round(sev),"")
    st.markdown(
        f'<div style="margin:6px 0"><div style="display:flex;justify-content:space-between;'
        f'font-size:.78rem;color:#6c757d"><span>1 Voluntary</span><span>5 Criminal</span></div>'
        f'<div style="background:#e9ecef;border-radius:6px;height:10px">'
        f'<div style="width:{pct}%;height:10px;border-radius:6px;background:{col}"></div></div>'
        f'<div style="font-size:.9rem;color:{col};font-weight:700;margin-top:3px">'
        f'Severity {sev:.2f}/5.0 — {lbl}</div></div>', unsafe_allow_html=True)

def _extract_table(ex):
    fields = [
        ("Policy name",          ex.policy_name),
        ("Penalty type",         ex.penalty_type),
        ("Penalty cap (USD)",    ex.penalty_cap_usd),
        ("Compute threshold",    ex.compute_threshold_flops),
        ("Enforcement",          ex.enforcement_mechanism),
        ("Grace period (months)",ex.grace_period_months),
        ("Scope",                ex.scope),
        ("Jurisdiction",         ex.source_jurisdiction),
        ("SME provisions",       ex.has_sme_provisions),
        ("Frontier lab focus",   ex.has_frontier_lab_focus),
        ("Research exemptions",  ex.has_research_exemptions),
        ("N entities regulated", ex.estimated_n_regulated),
    ]
    html = ""
    for lbl, fld in fields:
        v = fld.value
        vs = f"{v:.2e}" if isinstance(v, float) and v > 1e6 else str(v)
        badge = _badge(fld.epistemic_tag)
        src = ""
        if fld.source_passage and not fld.source_passage.startswith("[unverif"):
            src = f'<div class="evidence-box">📍 {fld.source_passage[:180]}</div>'
        html += (f'<div class="extract-row">'
                 f'<span class="extract-label">{lbl}</span>'
                 f'<span class="extract-value">{vs[:50]}</span>'
                 f'<span style="font-size:.72rem;color:#adb5bd;min-width:40px;text-align:right">{fld.confidence:.0%}</span>'
                 f'{badge}</div>{src}')
    st.markdown(html, unsafe_allow_html=True)

def _graph_view(graph):
    icons = {"FRONTIER_LAB":"🔬","DEVELOPER":"🏢","SME":"🏪","STARTUP":"🚀",
             "RESEARCHER":"📚","INVESTOR":"💰","CIVIL_SOCIETY":"🤝",
             "ENFORCEMENT_BODY":"⚖️","COMPUTE_THRESHOLD":"💻","REQUIREMENT":"📋"}
    html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0">'
    for node in graph._nodes.values():
        if node.confidence < 0.30: continue
        icon = icons.get(node.node_type, "📌")
        html += (f'<div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:5px;'
                 f'padding:3px 8px;font-size:.78rem">{icon} {node.label[:35]} '
                 f'<span style="color:#adb5bd;font-size:.68rem">{node.confidence:.0%}</span></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def _plot(data, key, title, ytitle, pct=True, h=270):
    try: import plotly.graph_objects as go
    except: return
    nr = data.get("nr", 8)
    rounds = list(range(1, nr+1))
    fig = go.Figure()
    colors = {"low":"rgba(30,100,200,{a})","mid":"rgba(13,49,85,{a})","high":"rgba(184,50,46,{a})"}
    names  = {"low":f"Sev {max(1.0,data['sev']-.5):.1f}","mid":f"Sev {data['sev']:.1f} (written)","high":f"Sev {min(5.0,data['sev']+.5):.1f}"}
    for k in ["low","mid","high"]:
        arr = data.get(f"{key}_{k}")
        if arr is None or arr.shape[0]==0: continue
        mean = arr.mean(0); std = arr.std(0); sc = 100 if pct else 1
        fig.add_trace(go.Scatter(x=rounds+rounds[::-1],
            y=list((mean+std)*sc)+list((mean-std)[::-1]*sc),
            fill="toself", fillcolor=colors[k].format(a="0.10"),
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rounds, y=mean*sc, name=names[k],
            line=dict(color=colors[k].format(a="1"),
                      width=2.5 if k=="mid" else 1.2,
                      dash="solid" if k=="mid" else "dot")))
    fig.update_layout(
        title=dict(text=title, font_size=13, font_color="#0d3155"),
        yaxis=dict(title=ytitle, tickformat=".0%" if pct else ".0f", gridcolor="#f0f2f5"),
        xaxis=dict(title="Quarter", tickvals=list(range(1,nr+1,4)),
                   ticktext=[f"Y{i//4+1}" for i in range(0,nr,4)], gridcolor="#f0f2f5"),
        height=h, margin=dict(l=10,r=10,t=36,b=24),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
        plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

def _metrics(data):
    fp = data.get("final_pop",{}); fs = data.get("final_stocks",{})
    c = st.columns(5)
    c[0].metric("Compliance",  f"{fp.get('compliance_rate',0):.0%}")
    c[1].metric("Relocation",  f"{fp.get('relocation_rate',0):.0%}")
    c[2].metric("AI investment", f"{fs.get('ai_investment_index',0):.0f}/100",
                delta=f"{fs.get('ai_investment_index',0)-100:.0f}")
    c[3].metric("Burden",      f"{fs.get('regulatory_burden',0):.0f}/100")
    c[4].metric("Lobbied",     f"{fp.get('ever_lobbied_rate',0):.0%}")

def _charts(data):
    c1, c2 = st.columns(2)
    with c1:
        _plot(data,"cr","Compliance rate","Compliance (%)"); _plot(data,"rr","Relocation rate","Relocation (%)")
    with c2:
        inv = {**data,**{f"ir_{k}":data[f"ir_{k}"]/100 for k in ["low","mid","high"]}}
        _plot(inv,"ir","AI investment index (100=baseline)","Index",pct=False)

def _jurisdiction(data):
    jur = data.get("jurisdiction",{})
    if not jur: return
    st.subheader("Relocation destinations")
    sj = sorted(jur.items(), key=lambda x:-x[1].get("company_count",0))
    cols = st.columns(len(sj))
    for i,(name,info) in enumerate(sj):
        cols[i].metric(name, f"{info.get('company_count',0)} cos.", f"burden {info.get('burden',0):.0f}")

def _pdf_dl(spec, n_pop, rounds, ens, seed_v, ingest_result=None):
    st.subheader("Evidence pack PDF")
    ci, cb = st.columns([3,1])
    pages_note = "4-page brief (includes ingest traceability)" if ingest_result is not None \
        else "3-page brief: severity rationale, compliance/relocation/investment charts, epistemic table."
    ci.caption(pages_note)
    if cb.button("📄 Generate", key=f"pdf_{spec.name[:10]}"):
        with st.spinner("Rendering..."):
            from policylab.v2.reports.evidence_pack import generate_evidence_pack
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                generate_evidence_pack(spec, tmp.name, n_population=n_pop,
                    num_rounds=rounds, n_ensemble=ens, seed=seed_v, verbose=False,
                    ingest_result=ingest_result)
                tp = tmp.name
        with open(tp,"rb") as f:
            st.download_button("⬇️ Download", f,
                file_name=f"policylab_{spec.name[:25].replace(' ','_')}.pdf",
                mime="application/pdf")

# ── MODE 1: Upload document ───────────────────────────────────────────────────
if mode == "📄 Upload document":
    st.title("Upload a regulatory document")
    st.markdown(
        "Drop in any regulatory document — bill text, impact assessment, white paper. "
        "PolicyLab extracts regulatory parameters, builds an entity graph, and runs a "
        "calibrated simulation. Every parameter is traced to its source passage and tagged "
        "**GROUNDED**, **DIRECTIONAL**, or **ASSUMED**."
    )

    file_tab, paste_tab = st.tabs(["📁 Upload file", "📋 Paste text"])
    uploaded_file = None; pasted_text = ""; doc_name = "Pasted document"
    with file_tab:
        uploaded_file = st.file_uploader("PDF, txt, md, or docx",
            type=["pdf","txt","md","markdown","docx"],
            help="Text-layer PDFs only. Scanned image PDFs need OCR.")
        if uploaded_file:
            st.success(f"{uploaded_file.name} ({uploaded_file.size/1024:.0f} KB)")
    with paste_tab:
        pasted_text = st.text_area("Paste bill or document text", height=200,
            placeholder="Even a summary paragraph is enough for regex extraction.")
        doc_name = st.text_input("Document name", "Pasted document")

    col_r, col_o = st.columns([2,1])
    with col_o:
        force_regex = st.checkbox("Regex only (no LLM)", value=not bool(api_key),
            help="Faster, no hallucination risk, but lower confidence.")
    with col_r:
        run_ingest = st.button("🔍 Extract & simulate", type="primary",
                               disabled=not (uploaded_file or pasted_text.strip()))

    if run_ingest:
        from policylab.v2.ingest.pipeline import ingest, ingest_text as _ingest_text

        with st.spinner("Extracting provisions..."):
            try:
                if uploaded_file:
                    suffix = os.path.splitext(uploaded_file.name)[1] or ".txt"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.read()); tp = tmp.name
                    result = ingest(tp, api_key=api_key if not force_regex else None,
                                    model=llm_model, verbose=False)
                    os.unlink(tp)
                else:
                    result = _ingest_text(pasted_text, name=doc_name,
                                          api_key=api_key if not force_regex else None,
                                          model=llm_model, verbose=False)
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

        st.markdown("<hr class='poldiv'>", unsafe_allow_html=True)
        ec, gc = st.columns([3,2])

        with ec:
            st.subheader("Extracted provisions")
            meth = "LLM" if result.extraction.extraction_method_used == "llm" else "Regex (fallback)"
            st.caption(f"{meth} · method={result.extraction.extraction_method_used}")
            core_conf = (result.extraction.penalty_type.confidence +
                         result.extraction.enforcement_mechanism.confidence +
                         result.extraction.scope.confidence) / 3
            if core_conf < 0.50:
                st.markdown(
                    f'<div class="low-conf-warn">⚠️ <strong>Low confidence ({core_conf:.0%})</strong> — ' +
                    f'many parameters are ASSUMED. Verify before citing. '
                    f'An API key enables LLM extraction with higher accuracy.</div>',
                    unsafe_allow_html=True)
            _extract_table(result.extraction)
            if result.extraction.key_provisions:
                with st.expander(f"Key provisions ({len(result.extraction.key_provisions)})"):
                    for txt, src in result.extraction.key_provisions:
                        st.markdown(f"**[{src}]** {txt}")

        with gc:
            st.subheader("Entity graph")
            st.caption(f"{len(result.graph)} nodes · regulated: {', '.join(result.graph.regulated_entity_types()) or 'none'}")
            _graph_view(result.graph)
            st.subheader("Derived parameters")
            for pname, dp in result.built_spec.derived_params.items():
                vs = str(dp.value)
                if isinstance(dp.value, dict):
                    vs = ", ".join(f"{k}:{v:.0%}" for k,v in list(dp.value.items())[:3])
                elif isinstance(dp.value, float):
                    vs = f"{dp.value:.2f}"
                st.markdown(f'<div class="extract-row"><span class="extract-label">{pname.replace("_"," ")}</span>'
                            f'<span class="extract-value" style="font-size:.78rem">{vs[:55]}</span>'
                            f'{_badge(dp.epistemic_tag)}</div>', unsafe_allow_html=True)

        st.markdown("<hr class='poldiv'>", unsafe_allow_html=True)
        sc, rc = st.columns([1,2])
        with sc:
            st.subheader(f"Severity {result.spec.severity:.2f}/5.0")
            _sev_gauge(result.spec.severity)
            lo,mid,hi = result.spec.recommended_severity_sweep or (
                max(1.0,result.spec.severity-.5), result.spec.severity, min(5.0,result.spec.severity+.5))
            st.caption(f"Sweep: {lo:.1f} / {mid:.1f} / {hi:.1f}")
            st.caption(f"compute_cost_factor = {result.spec.compute_cost_factor:.1f} [ASSUMED]")
            with st.expander("Scoring breakdown"):
                for j in result.spec.justification:
                    st.text(j)
        with rc:
            with st.spinner("Simulating..."):
                td = result.config.get("type_distribution")
                sim_data = _run_sim(
                    result.spec.name, result.spec.description, result.spec.severity,
                    result.config.get("n_population", n_population),
                    result.config.get("num_rounds", num_rounds),
                    n_ensemble, seed,
                    _type_dist=td, _ccf=result.spec.compute_cost_factor,
                    _jur=result.config.get("source_jurisdiction","EU"))
            _metrics(sim_data)

        _charts(sim_data)
        _jurisdiction(sim_data)
        with st.expander("Full traceability report"):
            st.code(result.traceability_report(), language=None)
        _pdf_dl(result.spec, result.config.get("n_population",n_population),
                result.config.get("num_rounds",num_rounds), n_ensemble, seed,
                ingest_result=result)
        if result.warnings:
            with st.expander(f"Extraction warnings ({len(result.warnings)})"):
                for w in result.warnings: st.warning(w)

# ── MODE 2: Analyze a bill ────────────────────────────────────────────────────
elif mode == "✏️ Analyze a bill":
    st.title("Analyze a policy")
    presets = _load_presets()
    pc = st.selectbox("Start from preset", list(presets.keys()))
    p  = presets[pc]
    cl, cr = st.columns([1,1])
    with cl:
        bill_text = st.text_area("Policy text", value=p.description if p else "", height=140)
        pft = st.checkbox("Extract parameters from text (regex)", value=(p is None))
    with cr:
        from policylab.v2.policy.parser import parse_bill, parse_bill_text
        pt=p.penalty_type if p else "civil"; pcap=p.penalty_cap_usd or 1_000_000 if p else 1_000_000
        pfl=p.compute_threshold_flops or 1e26 if p else 1e26; pe=p.enforcement_mechanism if p else "third_party_audit"
        pg=p.grace_period_months if p else 12; ps=p.scope if p else "large_developers_only"
        penalty_type = st.selectbox("Penalty type", ["voluntary","civil","civil_heavy","criminal"],
                                     index=["voluntary","civil","civil_heavy","criminal"].index(pt))
        penalty_cap  = st.number_input("Penalty cap (USD, 0=uncapped)", value=int(pcap), step=1_000_000, min_value=0)
        flops_exp    = st.slider("Compute threshold (10^N FLOPS)", 23, 28, value=int(round(len(str(int(pfl)))-1)) if pfl else 26)
        enf_mech     = st.selectbox("Enforcement", ["self_report","third_party_audit","government_inspect","criminal_invest"],
                                     index=["self_report","third_party_audit","government_inspect","criminal_invest"].index(pe))
        grace_mo     = st.slider("Grace period (months)", 0, 36, value=pg)
        scope        = st.selectbox("Scope", ["voluntary","frontier_only","large_developers_only","all"],
                                     index=["voluntary","frontier_only","large_developers_only","all"].index(ps))

    if st.button("▶ Score and simulate", type="primary"):
        with st.spinner("Scoring..."):
            if pft and bill_text:
                spec = parse_bill_text(bill_text, name=pc if p else "Custom")
            else:
                spec = parse_bill(name=pc if p else "Custom", description=bill_text,
                    penalty_type=penalty_type, penalty_cap_usd=penalty_cap if penalty_cap>0 else None,
                    compute_threshold_flops=10**flops_exp, enforcement_mechanism=enf_mech,
                    grace_period_months=grace_mo, scope=scope)
        _sev_gauge(spec.severity)
        with st.expander("Scoring breakdown"):
            for j in spec.justification: st.text(j)
        with st.spinner("Simulating..."):
            data = _run_sim(spec.name, spec.description, spec.severity,
                            n_population, num_rounds, n_ensemble, seed,
                            _ccf=spec.compute_cost_factor)
        st.divider(); _metrics(data); st.divider()
        _charts(data); _jurisdiction(data)
        _pdf_dl(spec, n_population, num_rounds, n_ensemble, seed)

# ── MODE 3: Compare policies ──────────────────────────────────────────────────
elif mode == "⚖️ Compare policies":
    st.title("Compare policies")
    presets = _load_presets()
    n_p = st.slider("Number of policies", 2, 4, 3)
    cols = st.columns(n_p); specs = []
    for i, col in enumerate(cols):
        with col:
            st.subheader(f"Policy {i+1}")
            ch = st.selectbox("Preset", list(presets.keys()), index=min(i,len(presets)-1), key=f"p{i}")
            pr = presets[ch]
            if pr:
                import dataclasses as _dc
                sev = st.slider("Severity", 1.0, 5.0, pr.severity, 0.1, key=f"sv{i}")
                spec = _dc.replace(pr, severity=sev)
            else:
                sev  = st.slider("Severity", 1.0, 5.0, 3.0, 0.1, key=f"sv{i}")
                nm   = st.text_input("Name", f"Policy {i+1}", key=f"nm{i}")
                from policylab.v2.policy.parser import PolicySpec as _PS
                spec = _PS(name=nm, description="", severity=sev, justification=[])
            specs.append(spec); _sev_gauge(spec.severity)

    if st.button("▶ Compare", type="primary"):
        # TODO: Use compare_policies(specs, baseline_seed) from analysis.py for cross-policy ranking. Requires a shared random seed across all simulations. Current per-policy simulation uses independent seeds, which is less sound for relative comparisons.
        all_data = []
        for spec in specs:
            with st.spinner(f"Simulating {spec.name}..."):
                d = _run_sim(spec.name, spec.description, spec.severity,
                             n_population, num_rounds, n_ensemble, seed,
                             _ccf=getattr(spec,"compute_cost_factor",1.0))
            all_data.append((spec, d))
        st.subheader("Results")
        import pandas as pd
        metrics_def = [
            ("Severity",   lambda d: f"{d['sev']:.1f}/5"),
            ("Compliance", lambda d: f"{d['final_pop'].get('compliance_rate',0):.0%}"),
            ("Relocation", lambda d: f"{d['final_pop'].get('relocation_rate',0):.0%}"),
            ("Investment", lambda d: f"{d['final_stocks'].get('ai_investment_index',0):.0f}/100"),
            ("Burden",     lambda d: f"{d['final_stocks'].get('regulatory_burden',0):.0f}/100"),
            ("Lobbied",    lambda d: f"{d['final_pop'].get('ever_lobbied_rate',0):.0%}"),
        ]
        rows = [[mn]+[fn(d) for _,d in all_data] for mn,fn in metrics_def]
        st.dataframe(pd.DataFrame(rows, columns=["Metric"]+[s.name for s,_ in all_data]),
                     use_container_width=True, hide_index=True)
        try:
            import plotly.graph_objects as go
            fig = go.Figure(); palette=["#0D3155","#1b5e99","#e05555","#2a8f5c"]
            rs = list(range(1,num_rounds+1))
            for i,(spec,d) in enumerate(all_data):
                fig.add_trace(go.Scatter(x=rs, y=d["cr_mid"].mean(0)*100,
                    name=spec.name, line=dict(color=palette[i],width=2)))
            fig.update_layout(title="Compliance trajectories", yaxis_title="Compliance (%)",
                xaxis=dict(title="Quarter",tickvals=list(range(1,num_rounds+1,4)),
                           ticktext=[f"Y{i//4+1}" for i in range(0,num_rounds,4)]),
                height=350, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        except: pass

# ── MODE 4: Influence scenario ────────────────────────────────────────────────
elif mode == "🧪 Influence scenario":
    st.title("Influence scenario")
    st.caption("Test policy robustness by simulating coordinated false narratives in agent reasoning. All injection parameters (rate, direction, magnitude) are [ASSUMED] — no empirical calibration target exists.")
    presets = _load_presets()
    ch = st.selectbox("Policy", list(presets.keys()))
    pr = presets.get(ch)
    sev = st.slider("Severity", 1.0, 5.0, pr.severity if pr else 3.0, 0.1)
    pol_name = pr.name if pr else "Custom"; pol_desc = pr.description if pr else ""
    c1,c2,c3,c4 = st.columns(4)
    inj_rate  = c1.slider("Injection rate/round", 0.01, 0.30, 0.05, 0.01)
    inj_dir   = c2.selectbox("Direction", ["+1.0 anti-regulation","-1.0 pro-compliance"])
    inj_dir_v = 1.0 if "+1" in inj_dir else -1.0
    inj_mag   = c3.slider("Magnitude per injection", 0.02, 0.20, 0.08, 0.01)
    inj_start = c4.slider("Start round", 1, num_rounds, 1)

    if st.button("▶ Run", type="primary"):
        with st.spinner("Running baseline and injected simulations..."):
            from policylab.v2.influence.adversarial import run_with_injection
            result = run_with_injection(pol_name, pol_desc, sev,
                injection_rate=inj_rate, injection_direction=inj_dir_v,
                injection_magnitude=inj_mag, injection_start_round=inj_start,
                n_population=n_population, num_rounds=num_rounds, seed=seed)
        m1,m2,m3 = st.columns(3)
        m1.metric("Compliance Δ", f"{result.compliance_delta:+.1%}")
        m2.metric("Relocation Δ", f"{result.relocation_delta:+.1%}")
        m3.metric("Network resilience", f"{result.resilience_score:.2f}",
                  delta="resistant" if result.resilience_score>0.7 else "vulnerable")
        try:
            import plotly.graph_objects as go
            rs = list(range(1,len(result.round_compliance_baseline)+1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rs,y=[v*100 for v in result.round_compliance_baseline],
                name="Baseline",line=dict(color="#0D3155",width=2)))
            fig.add_trace(go.Scatter(x=rs,y=[v*100 for v in result.round_compliance_injected],
                name=f"Injected ({inj_rate:.0%}/round)",
                line=dict(color="#e05555",width=2,dash="dot")))
            if inj_start>1:
                fig.add_vline(x=inj_start,line_dash="dash",line_color="#adb5bd",
                              annotation_text="Injection starts")
            fig.update_layout(title="Compliance: baseline vs injected",
                yaxis_title="Compliance (%)", height=350, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        except: pass
        st.info("⚠️ All injection parameters are [ASSUMED]. Treat as directional scenario analysis.")
