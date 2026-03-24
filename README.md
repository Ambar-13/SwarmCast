# PolicyLab

**Author: Ambar**

PolicyLab stress-tests AI regulation proposals through multi-agent simulation. It models how companies, regulators, investors, and civil society respond to regulatory pressure — producing behavioral predictions (relocation rates, coalition patterns, compliance timelines, evasion frequencies) and directional indicator changes grounded in regulatory economics literature.

---

## Table of Contents

1. [Install](#install)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Epistemic Framework](#epistemic-framework)
5. [v2 Engine — Vectorized Population Simulation](#v2-engine--vectorized-population-simulation)
   - [Agent Types and Calibration](#agent-types-and-calibration)
   - [Compliance and Relocation Dynamics](#compliance-and-relocation-dynamics)
   - [SMM Calibration Moments](#smm-calibration-moments)
   - [System Dynamics Layer](#system-dynamics-layer)
   - [Stock-Flow Accounting](#stock-flow-accounting)
   - [Events System](#events-system)
   - [International Jurisdictions](#international-jurisdictions)
6. [Policy Parser](#policy-parser)
7. [Evidence Pack (PDF Reports)](#evidence-pack-pdf-reports)
8. [Influence Scenario](#influence-scenario)
9. [v1 Engine — LLM Strategic Agents](#v1-engine--llm-strategic-agents)
   - [Stress Tester](#stress-tester)
   - [War Game](#war-game)
   - [Blind Spot Finder](#blind-spot-finder)
   - [Ensemble Runner](#ensemble-runner)
   - [Backtester](#backtester)
10. [Hybrid Loop (v1 + v2 Combined)](#hybrid-loop-v1--v2-combined)
11. [CLI Reference](#cli-reference)
12. [Streamlit App](#streamlit-app)
13. [Interpreting Results](#interpreting-results)
14. [Known Limitations](#known-limitations)
15. [References](#references)

---

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. Dependencies (from `pyproject.toml`):

```
gdm-concordia>=2.4.0   # LLM agent framework (v1 engine)
openai>=1.0.0
numpy
scipy>=1.10.0
scikit-learn>=1.3.0
sentence-transformers
streamlit>=1.30.0
plotly>=5.0.0
reportlab>=4.0.0
pandas>=2.0.0
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
# or for Anthropic:
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick Start

**Run a v2 simulation from Python:**

```python
from policylab.v2.simulation.hybrid_loop import run_v2_simulation
from policylab.v2.policy.parser import parse_bill

policy = parse_bill("eu_ai_act_gpai")          # built-in preset, severity 2.39
result = run_v2_simulation(policy, rounds=12, n_agents=10_000, seed=42)

print(result["summary"])
print(result["relocation_rate"])               # fraction of population that relocated
print(result["compliance_rate"])               # fraction compliant at end
```

**Parse a real bill text:**

```python
from policylab.v2.policy.parser import parse_bill_text

severity = parse_bill_text(
    penalty="civil_heavy",
    enforcement="government_inspect",
    flop_threshold=1e26,
    grace_period_months=12,
    scope="large_devs_only",
)
print(severity)   # float in [1.0, 5.0]
```

**Run the CLI:**

```bash
python -m policylab.cli v2-stress-test --preset eu_ai_act_gpai --rounds 12
python -m policylab.cli stress-test --policy "Mandatory safety audits for all AI systems"
python -m policylab.cli war-game --scenario "major_incident"
python -m policylab.cli blind-spots --policy "Compute threshold licensing"
python -m policylab.cli backtest --case eu_ai_act
python -m policylab.cli demo
```

**Streamlit UI:**

```bash
streamlit run app.py
```

Opens three modes: Analyze bill / Compare policies / Influence scenario.

---

## Architecture Overview

PolicyLab has two simulation engines that can run separately or in a hybrid loop:

```
┌─────────────────────────────────────────────────────────────┐
│  v2 Engine (vectorized, ~2ms/round at n=10,000)             │
│  ├── PopulationArray  — all agent state as float32 arrays   │
│  ├── vectorized_round() — DeGroot → compliance → relocation │
│  ├── SMM calibration   — 6 moments, DLA Piper + EC data     │
│  ├── GovernanceStocks  — company/burden/innovation stocks   │
│  ├── EventQueue        — policy amendments, shocks          │
│  └── Jurisdictions     — 5 destinations with burden routing │
│                                                             │
│  v1 Engine (LLM-backed Concordia agents, ~200s/round)       │
│  ├── StressTester      — 6 named agents, multi-round        │
│  ├── WarGame           — 8 incident templates               │
│  ├── BlindSpotFinder   — randomized parameter sweeps        │
│  ├── EnsembleRunner    — Lempert 2003 perturbation          │
│  └── Backtester        — EU AI Act + US EO 14110 cases      │
│                                                             │
│  Policy Parser         — bill text → severity score [1–5]   │
│  Evidence Pack         — 3-page PDF for legislative staff   │
│  Influence Scenario    — belief injection on network        │
└─────────────────────────────────────────────────────────────┘
```

---

## Epistemic Framework

Every parameter and result in PolicyLab carries one of three epistemic labels. Read these before citing any number.

| Label | Meaning | How to use |
|---|---|---|
| **GROUNDED** | Derived from empirical data with documented elasticity and SE | Cite direction and approximate magnitude |
| **DIRECTIONAL** | Direction known from theory; magnitude uncertain | Cite direction only |
| **ASSUMED** | No calibration target; plausible but invented | Sweep in sensitivity analysis; do not cite as predictions |

**What is valid to cite:**

```
VALID:   "In 100% of runs, at least one company relocated."
VALID:   "Regulatory burden shows directional suppression of investment
          consistent with OECD PMR literature."
VALID:   "Relocation appears in 94% of sensitivity configurations."

INVALID: "Innovation will drop by 37 points."
INVALID: "Public trust will fall to 36.5 ± 10.8."
INVALID: "The tipping point was reached." (ASSUMED threshold)
```

Three output layers:
1. **Behavioral** (action frequencies, relocation rates, failure modes) — most reliable; arises from LLM or vectorized agent reasoning
2. **Indicator tendencies** (directional changes in innovation_rate, ai_investment_index, etc.) — use direction, not magnitude
3. **Sensitivity results** — ROBUST if present in rigorous baseline and >80% of configs; NON-ROBUST if only under ASSUMED parameters

---

## v2 Engine — Vectorized Population Simulation

The v2 engine represents the agent population as parallel numpy float32 arrays. One round runs in ~2ms at n=10,000 agents.

### Core Data Structure: `PopulationArray`

```python
from policylab.v2.engine.population import PopulationArray

pop = PopulationArray(n=10_000, seed=42)
```

All state is stored as `(n,)` float32 arrays:

| Array | Description |
|---|---|
| `beliefs` | Current belief that regulation is legitimate [0, 1] |
| `lambdas` | Weibull mean time-to-compliance in rounds [GROUNDED/ASSUMED] |
| `sizes` | Company size proxy [0, 1] |
| `thresholds` | Belief threshold above which relocation is triggered [ASSUMED] |
| `stubbornness` | Resistance to social influence [0, 1] [ASSUMED] |
| `memory` | Shape (n, 8, 4) — 8 recent rounds × 4 belief sources |
| `type_ids` | Integer agent type index (0–6) |
| `complied` | Boolean: has agent complied this episode |
| `relocated` | Boolean: has agent relocated |

### `vectorized_round()`

Each simulation round applies these steps in order:

1. **DeGroot belief update** — weighted average of neighbors' beliefs via row-stochastic influence matrix W_norm (Barabasi-Albert network, m=3). `belief_new = (1 - stubbornness) × W_norm @ beliefs + stubbornness × beliefs`
2. **Memory update** — store current belief in rolling 8-round buffer
3. **Compliance hazard** — Weibull hazard rate `h = 1/λ` per round (NOT CDF). Agents comply when `U(0,1) < h`
4. **Relocation sigmoid** — `p_relocate = σ((belief - threshold) / temperature)`. Frontier labs get an additional factor: `p_relocate_fl = p_relocate × (1 + 0.30 × belief)` [DIRECTIONAL]
5. **Evasion draw** — non-compliant agents draw against evasion probability [ASSUMED]
6. **Lobbying draw** — agents draw against lobbying probability scaled by size [GROUNDED coalition bonus]
7. **State update** — write complied/relocated/evaded flags back to arrays

### Influence Network

```python
from policylab.v2.engine.network import build_influence_matrix

W_raw, W_norm = build_influence_matrix(n=10_000, m=3, seed=42)
# W_raw: sparse CSR, unnormalized (for degree computation)
# W_norm: sparse CSR, row-stochastic (for DeGroot update)
```

Barabasi-Albert preferential attachment, m=3 edges per new node. Pure numpy construction, no networkx dependency. The degree distribution's Gini coefficient feeds the resilience score in the influence scenario.

---

### Agent Types and Calibration

Seven agent types. Lambdas marked GROUNDED come from DLA Piper's GDPR compliance survey (2020); others are interpolated or assumed.

| Type ID | Name | λ (rounds) | Source | Threshold | Notes |
|---|---|---|---|---|---|
| 0 | large_company | 3.32 | **GROUNDED** DLA Piper | 72.0 | |
| 1 | mid_company | 7.1 | ASSUMED (interpolated) | 65.0 | |
| 2 | startup | 10.9 | **GROUNDED** DLA Piper | 55.0 | 60% cumulative cap |
| 3 | researcher | 15.0 | ASSUMED | 60.0 | |
| 4 | investor | 5.0 | ASSUMED | 70.0 | |
| 5 | civil_society | 8.0 | ASSUMED | 999 | never relocates |
| 6 | frontier_lab | 1.5 | DIRECTIONAL | 45.0 [ASSUMED] | belief-accelerated relocation |

**On frontier labs:** The model assumes frontier_lab agents are already multinational (no domestic anchor), mission-driven (belief-accelerated relocation when trust is low), and face global scrutiny even before a policy passes. This is a structural assumption. The model is better suited to estimating aggregate industry behavior across the broader population of AI developers. Specific firms like Anthropic, Google DeepMind, or OpenAI are not modeled — their actual responses depend on internal governance, mission commitments, and strategic positioning that falls outside aggregate statistical calibration.

---

### Compliance and Relocation Dynamics

**Periodization:** 1 round = 91.25 days (quarterly), matching `ROUNDS_PER_YEAR = 4` in `calibration.py`. A 12-round simulation covers 3 years.

**Severity scale:** [1.0, 5.0]. The severity score from the policy parser feeds:
- compliance hazard (higher severity → faster hazard rate)
- relocation threshold adjustment
- calibration moment 3 (relocation_rate adjusted via `for_severity()`)

**Weibull hazard:** `h = 1/λ` per round. NOT the CDF `1 - exp(-t/λ)`. Using the instantaneous hazard rate corrects the compliance timing — at the calibrated λ=10.9 for startups, 24-month (8-round) compliance rate lands at 52% matching the DLA Piper moment.

---

### SMM Calibration Moments

Six moments calibrated against GDPR 2018–2020 data. Run calibration:

```python
from policylab.game_master.calibration import run_smm_calibration, print_calibration_report

result = run_smm_calibration()
print_calibration_report()
```

| Moment | Target | Source | Label | SE | Weight |
|---|---|---|---|---|---|
| m1: ever_lobbied_rate | 0.85 | EU Transparency Register | **GROUNDED** | 0.05 | 400 |
| m2: compliance_rate_y1 | 0.23 | EC Impact Assessment × 0.50 announcement factor | **GROUNDED** | 0.08 | 156 |
| m3: relocation_rate | 0.12 | Adjusted by `for_severity()` | DIRECTIONAL | 0.04 | 625 |
| m4: sme_compliance_24mo | 0.52 | DLA Piper 2020 | **GROUNDED** | 0.06 | 278 |
| m5: large_compliance_24mo | 0.91 | DLA Piper 2020 | **GROUNDED** | 0.04 | 625 |
| m6: enforcement_contact_rate | 0.06 | DLA Piper 2020 | **GROUNDED** | 0.01 | 10000 |

Calibration uses scipy differential evolution to minimize the weighted sum-of-squared deviations between simulated and target moments.

---

### System Dynamics Layer

After each round's agent actions resolve, `apply_indicator_feedback()` updates five economic indicators through coupled feedback loops.

```python
from policylab.game_master.indicator_dynamics import apply_indicator_feedback, DynamicsConfig

report = apply_indicator_feedback(world_state)   # uses DEFAULT_DYNAMICS
# or customize:
cfg = DynamicsConfig(burden_to_investment=-0.010, damping=0.9)
report = apply_indicator_feedback(world_state, config=cfg)
```

**Feedback links and their epistemic status:**

| Link | Coefficient | Direction | Source | Label |
|---|---|---|---|---|
| burden → investment | −0.00295/round/pt above 30 | Negative | Springler 2023, OECD PMR ε=−0.197 annual, 100/6 scale conversion | **GROUNDED** |
| R&D → innovation | +0.000345/round/pt above 50 | Positive | Ugur 2016 meta-analysis, ε=0.138, SE=0.012 | **GROUNDED** |
| concentration → innovation | inverted-U peak at 40 | Both | Aghion, Bloom et al. 2005 QJE | **GROUNDED** |
| trust ↔ burden | ±0.006/round | Both | Legitimacy theory | DIRECTIONAL |
| burden → trust | −0.004/round above 60 | Negative | Regulatory backlash literature | DIRECTIONAL |
| innovation → trust | +0.003/round above 70 | Positive | TAM (Davis 1989) | DIRECTIONAL |

**Tipping point thresholds** [ASSUMED — invented, not calibrated]:
- `investment < 20`: capital-flight cascade, multiplier 1.5× on adverse dynamics
- `trust < 20`: regulatory legitimacy collapse
- `innovation < 15`: domestic frontier capability gone

These fire a `TippingPointReport`. Do not cite tipping point results as policy predictions — all three thresholds are assumed.

**DynamicsConfig defaults** (all overridable):

```python
@dataclasses.dataclass
class DynamicsConfig:
    investment_to_innovation: float = 0.015   # GROUNDED direction
    innovation_to_investment: float = 0.010   # GROUNDED direction
    burden_to_investment: float = -0.008      # GROUNDED
    concentration_innovation_peak: float = 40.0  # GROUNDED
    concentration_innovation_strength: float = 0.005
    trust_to_burden: float = -0.006           # DIRECTIONAL
    burden_to_trust: float = -0.004           # DIRECTIONAL
    innovation_to_trust: float = 0.003        # DIRECTIONAL
    tipping_investment_cascade: float = 20.0  # ASSUMED
    tipping_trust_collapse: float = 20.0      # ASSUMED
    tipping_innovation_death: float = 15.0    # ASSUMED
    tipping_cascade_multiplier: float = 1.5   # ASSUMED
    damping: float = 0.8
```

Set DIRECTIONAL parameters to 0.0 for a rigorous baseline; sweep them for sensitivity analysis.

**Dimensional anchors** (for interpreting indicator values):
- `innovation_rate = 50` ↔ 2.5% TFP/year (historical US baseline)
- `ai_investment_index = 50` ↔ $50B/year (approximate 2022 global AI investment)

---

### Stock-Flow Accounting

`GovernanceStocks` tracks conservation-law quantities that the indicator layer does not capture.

```python
from policylab.v2.stocks.governance_stocks import GovernanceStocks

stocks = GovernanceStocks(initial_companies=500)
stocks.company.add_relocated(n=10)
stocks.burden.add_policy_burden(severity=2.5)
stocks.burden.discharge_compliance(n_compliant=50)
stocks.innovation.update(investment=60.0, round_num=3)
print(stocks.company.domestic_count())    # companies remaining in jurisdiction
```

**CompanyStock** — conservation: `dN/dt = entrants − relocated − failed`
- `total`, `in_pipeline`, `relocated`, `failed`, `new_entrants`
- `domestic_count() = total − relocated − failed`

**BurdenStock** — level + cumulative; outflow via `discharge_compliance()`, inflow via `add_policy_burden()`

**InnovationCapacity** — Aghion-Howitt forward-looking expectations, 0.3 update weight [ASSUMED]

**RelocationPipeline** — 2–4 round delay [ASSUMED], 30% cancellation when `policy_softened()` is called

---

### Events System

Events fire policy changes mid-simulation, modeling amendments, trust shocks, enforcement surges, and destination policy changes.

```python
from policylab.v2.simulation.events import (
    EventQueue, RoundTrigger, ThresholdTrigger, ProbabilisticTrigger,
    PolicyAmendmentEffect, TrustShockEffect, InvestmentSurgeEffect,
    EnforcementSurgeEffect, DestinationPolicyChangeEffect, LLMProposalEffect,
)

queue = EventQueue()

# Fire on round 4:
queue.add(RoundTrigger(round_num=4, effect=PolicyAmendmentEffect(severity_delta=-0.5)))

# Fire when relocation rate crosses 0.15:
queue.add(ThresholdTrigger(
    indicator="relocation_rate",
    threshold=0.15,
    effect=TrustShockEffect(delta=-10.0),
))

# Fire with 20% probability each round:
queue.add(ProbabilisticTrigger(prob=0.20, effect=EnforcementSurgeEffect(multiplier=1.5)))

# Use deep_copy() before ensemble runs to isolate queue state:
queue_copy = queue.deep_copy()
```

**Effect types:**

| Effect | Parameters | What it does |
|---|---|---|
| `PolicyAmendmentEffect` | `severity_delta` | Shifts policy severity up/down |
| `TrustShockEffect` | `delta` | Immediately changes public_trust indicator |
| `InvestmentSurgeEffect` | `delta` | Immediately changes ai_investment_index |
| `EnforcementSurgeEffect` | `multiplier` | Scales enforcement contact rate for N rounds |
| `DestinationPolicyChangeEffect` | `jurisdiction, burden_delta` | Changes a destination jurisdiction's burden |
| `LLMProposalEffect` | `prompt` | Calls LLM to generate and apply an amendment |

---

### International Jurisdictions

Five destination jurisdictions for relocating companies. Softmax routing assigns probability mass based on attractiveness.

```python
from policylab.v2.international.jurisdictions import JurisdictionRouter

router = JurisdictionRouter()
destinations = router.route(n_relocating=50, policy_severity=2.5)
# Returns dict: {"EU": 3, "US": 18, "UK": 12, "Singapore": 11, "UAE": 6}
```

**Jurisdiction parameters:**

| Jurisdiction | Regulatory burden | Corporate tax rate |
|---|---|---|
| EU | 55 | 25% |
| US | 25 | 21% |
| UK | 20 | 19% |
| Singapore | 10 | 17% |
| UAE | 5 | 9% |

Routing uses softmax with temperature=0.1 [ASSUMED], which concentrates flow to lowest-burden destinations. Log congestion penalty: `5.0 × log(1 + n/capacity)` [ASSUMED] — prevents unrealistic pile-up.

When companies relocate, their R&D capacity moves with them rather than disappearing — the domestic innovation stock decreases, not the total innovation capacity.

---

## Document Ingestion Pipeline

The primary entry point for PolicyLab. Upload any regulatory document — bill text,
impact assessment, white paper, PDF — and the pipeline automatically extracts
regulatory provisions, builds an entity graph, and populates all simulation
parameters with full epistemic traceability.

Every derived parameter carries a confidence score and the exact source passage
that grounded it, mapped automatically to **GROUNDED**, **DIRECTIONAL**, or
**ASSUMED**.

### Quick start

```python
from policylab.v2.ingest import ingest, ingest_text

# From a file (PDF, txt, md, docx)
result = ingest("eu_ai_act_impact_assessment.pdf")

# From pasted text (e.g. from the Streamlit UI)
result = ingest_text("""
    REGULATION (EU). General-purpose AI models trained above 10^25 FLOPS
    must conduct adversarial testing and report incidents. Fines up to 3%
    of global turnover. 12-month implementation period. Research exempt.
""", name="EU AI Act GPAI")

# Every parameter is traced to its source
print(result.spec.severity)           # 2.39
print(result.spec.compute_cost_factor) # 3.0 [ASSUMED — no empirical data]
print(result.extraction.penalty_type.epistemic_tag)   # DIRECTIONAL
print(result.extraction.penalty_type.source_passage)  # verbatim quote from document
print(result.extraction.compute_threshold_flops.confidence)  # 0.70

# Run simulation with all ingest-derived parameters
from policylab.v2.simulation.hybrid_loop import HybridSimConfig, run_hybrid_simulation
sim = run_hybrid_simulation(
    result.spec.name, result.spec.description, result.spec.severity,
    config=HybridSimConfig(**result.config),
)
```

### CLI

```bash
# Extract + simulate
policylab ingest eu_ai_act.pdf

# Extract only, write JSON for inspection
policylab ingest bill.txt --no-simulate --output-json result.json

# With LLM extraction (higher confidence, requires API key)
policylab ingest impact_assessment.pdf --api-key sk-... --model gpt-4o

# Full traceability report
policylab ingest sb53.pdf --traceability
```

### Pipeline architecture

```
policylab/v2/ingest/
├── document_loader.py     PDF/txt/md/docx → LoadedDocument with per-chunk provenance
├── provision_extractor.py One LLM call → all fields with (value, confidence, source_passage)
├── entity_graph.py        In-memory graph: who is regulated, exempted, what triggers
├── spec_builder.py        Graph + extraction → PolicySpec + HybridSimConfig overrides
└── pipeline.py            ingest() / ingest_text() / ingest_and_simulate()
```

### Epistemic confidence

| confidence | Epistemic tag | Meaning |
|---|---|---|
| ≥ 0.80 | **GROUNDED** | Document explicitly states this in clear language |
| 0.50–0.79 | **DIRECTIONAL** | Document implies this; reasonable analysts would agree |
| < 0.50 | **ASSUMED** | Defaulted; document does not address this dimension |

LLM extraction: validates every source passage against the actual document.
A passage the LLM cannot find is rejected and confidence is capped at 0.45
(below GROUNDED threshold), preventing hallucinated provenance.

Regex fallback (no API key): confidence capped at 0.70. No hallucination risk.
Lower accuracy on nuanced regulatory language.

### Derived parameters

The spec builder derives five additional parameters beyond the direct
extraction fields:

| Parameter | Derived from | Epistemic tag |
|---|---|---|
| `type_distribution` | scope, threshold, SME/research provisions, entity graph | DIRECTIONAL |
| `num_rounds` | grace_period_months + 4-quarter equilibration window | DIRECTIONAL |
| `compute_cost_factor` | threshold level × enforcement mechanism | ASSUMED |
| `source_jurisdiction` | document language and reference patterns | DIRECTIONAL |
| `n_population` | estimated_n_regulated (handful/dozens/hundreds/thousands) | ASSUMED |

All are passed directly to `HybridSimConfig` via `result.config`.

### Low-confidence behaviour

When core extraction confidence is below 0.50 (many ASSUMED fields),
the pipeline:
- Adds a warning to `result.warnings`
- Widens `spec.recommended_severity_sweep` to ±1.0 (instead of ±0.5)
- Adds a note to `spec.justification` flagging the low confidence

This prevents the model from producing false precision from uncertain extractions.

---

## Policy Parser

Converts bill provisions into a severity score in [1.0, 5.0].

```python
from policylab.v2.policy.parser import parse_bill, parse_bill_text

# Built-in presets:
policy = parse_bill("california_sb53")          # → severity 2.39
policy = parse_bill("eu_ai_act_gpai")           # → severity 2.39
policy = parse_bill("ny_raise_act")             # → severity 2.11
policy = parse_bill("hypothetical_compute_ban") # → severity 5.0

# Parse from provisions:
severity = parse_bill_text(
    penalty="civil_heavy",         # "none", "voluntary", "civil", "civil_heavy", "criminal"
    enforcement="government_inspect",  # "none", "self_report", "third_party_audit",
                                       # "government_inspect", "criminal_invest"
    flop_threshold=1e26,           # FLOP compute threshold (None = no threshold)
    grace_period_months=12,        # 0, ≤6, 7-12, 13-24, >24
    scope="large_devs_only",       # "voluntary", "frontier_only", "large_devs_only", "all"
)
```

**Scoring weights** [all ASSUMED]:

| Dimension | Value | Score contribution |
|---|---|---|
| penalty: none | — | 0.0 |
| penalty: voluntary | — | 0.3 |
| penalty: civil | — | 1.0 |
| penalty: civil_heavy | — | 1.8 |
| penalty: criminal | — | 3.0 |
| enforcement: none | — | 0.0 |
| enforcement: self_report | — | 0.3 |
| enforcement: third_party_audit | — | 0.8 |
| enforcement: government_inspect | — | 1.2 |
| enforcement: criminal_invest | — | 2.0 |
| threshold ≥ 10²⁷ FLOPs | — | 0.2 |
| threshold ≥ 10²⁶ FLOPs | — | 0.5 |
| threshold ≥ 10²⁵ FLOPs | — | 0.8 |
| threshold ≥ 10²⁴ FLOPs | — | 1.1 |
| threshold < 10²⁴ FLOPs | — | 1.4 |
| grace: 0 months | — | 0.8 |
| grace: ≤ 6 months | — | 0.5 |
| grace: 7–12 months | — | 0.2 |
| grace: 13–24 months | — | 0.0 |
| grace: > 24 months | — | −0.2 |
| scope: voluntary | — | 0.0 |
| scope: frontier_only | — | 0.3 |
| scope: large_devs_only | — | 0.5 |
| scope: all | — | 0.8 |

Raw score is linearly scaled from [RAW_MIN=0.8, RAW_MAX=8.0] → [1.0, 5.0].

---

## Evidence Pack (PDF Reports)

Generates a 3-page PDF suitable for legislative staff briefings.

```python
from policylab.v2.reports.evidence_pack import generate_evidence_pack

pdf_path = generate_evidence_pack(
    policy_name="EU AI Act GPAI",
    severity=2.39,
    simulation_result=result,
    output_path="briefing.pdf",
)
```

Three pages:
1. **Severity rationale** — which provisions drive the score, with epistemic labels
2. **Charts** — compliance curve, relocation over time, indicator trajectories
3. **Epistemic table** — which results are GROUNDED / DIRECTIONAL / ASSUMED

---

## Influence Scenario

Models belief injection into the regulatory stakeholder network. Useful for understanding how concentrated information campaigns affect compliance and relocation outcomes.

```python
from policylab.v2.influence.adversarial import run_with_injection, inject_beliefs

# Matched-pair design: same seed, clean causal estimate
clean_result, injected_result = run_with_injection(
    policy=policy,
    n_agents=10_000,
    rounds=12,
    seed=42,
    n_targets=50,               # number of hubs to target
    injection_magnitude=0.08,   # belief shift per injection [ASSUMED]
    direction="pro_compliance",  # or "anti_compliance"
)

delta_relocation = injected_result["relocation_rate"] - clean_result["relocation_rate"]
print(f"Injection effect on relocation: {delta_relocation:+.3f}")
```

**How injection works:**

`inject_beliefs()` identifies the top-degree hub nodes in the Barabasi-Albert influence network (degree centrality as a proxy for media or lobbying reach). It draws a pool of 4× n_targets candidates — modeling an imperfect adversary that cannot perfectly identify all hubs — then injects a belief shift of ±magnitude to the selected nodes. These beliefs then propagate through the DeGroot update in subsequent rounds.

**Parameters and labels:**

| Parameter | Default sweep | Label |
|---|---|---|
| injection_rate | [0.01, 0.05, 0.10] fraction of population | ASSUMED |
| injection_magnitude | [0.03, 0.08, 0.15] belief units | ASSUMED |
| hub targeting pool multiplier | 4× | ASSUMED |
| resilience_score = 1 − Gini(degree) | computed | DIRECTIONAL |

The resilience score measures how evenly influence is distributed across the network. A network where all nodes have similar degree is more resilient to targeted injection than one with extreme hub concentration. This is directionally motivated but not calibrated against empirical data.

**When to use this:** The influence scenario is useful for sensitivity analysis — does the simulation's behavioral output change substantially when 5% of the highest-reach nodes are given shifted beliefs? If relocation and compliance rates shift significantly, the result is sensitive to information environment assumptions and should be labeled accordingly.

---

## v1 Engine — LLM Strategic Agents

The v1 engine uses the Concordia framework to run named LLM agents through structured game-master interactions. Each round takes ~200 seconds. Use this when you need the full strategic reasoning and coalition dynamics that the vectorized engine does not capture.

**Six agents:**
- Government — drafts and defends regulation
- Large AI Company — seeks to minimize compliance burden
- Startup — sensitive to compliance costs, may relocate
- Regulator — enforces and monitors compliance
- Civil Society — advocates for public interest
- **Safety-First Corp (SAFETY_FIRST_CORP)** — a contrarian large company that supports the regulation; corrects LLM herd-behavior bias (OASIS 2024: LLM agents over-coordinate compared to real humans)

### Stress Tester

```python
from policylab.features.stress_tester import StressTester

tester = StressTester(
    policy="Mandatory safety audits for all frontier AI systems",
    severity=3.0,
    n_rounds=5,
    model="gpt-4o",
)
result = tester.run()
print(result["failure_modes"])      # list of detected failure patterns
print(result["coalition_pattern"])  # which agents aligned
print(result["action_frequencies"])
```

**Key fields in result:**
- `failure_modes`: e.g. `["relocation", "regulatory_capture", "evasion", "trust_collapse"]`
- `coalition_pattern`: dict of agent → stance
- `action_frequencies`: how often each action type appeared
- `baseline_comparison`: counterfactual without policy for causal identification

The stress tester runs one baseline (no policy) and one treatment simulation, comparing action distributions. The presence of SAFETY_FIRST_CORP ensures not all large companies defect simultaneously.

### War Game

```python
from policylab.features.war_game import WarGame

game = WarGame(
    policy="EU AI Act GPAI provisions",
    scenario="major_incident",
    framework="eu_ai_act",
)
result = game.run()
```

**8 incident templates:**
`major_incident`, `gradual_erosion`, `regulatory_arbitrage`, `lobbying_blitz`, `whistleblower`, `international_pressure`, `court_challenge`, `public_backlash`

**3 governance framework presets:**
`eu_ai_act`, `us_eo_14110`, `light_touch`

### Blind Spot Finder

Randomizes key parameters across their plausible ranges to surface failure modes that only appear under specific conditions.

```python
from policylab.features.blind_spot_finder import BlindSpotFinder

finder = BlindSpotFinder(policy="Compute threshold licensing", n_runs=20)
result = finder.run()
print(result["blind_spots"])   # conditions under which policy fails unexpectedly
```

Parameters swept [all ASSUMED ranges]:
- `severity`: [0.5, 5.0]
- `time_pressure`: [low, medium, high]
- `actor_capability`: [low, medium, high]
- `trust`: [20, 80]
- `investment`: [20, 80]
- `regulator_capacity`: [low, medium, high]

### Ensemble Runner

Applies Lempert (2003) deep uncertainty perturbation — runs many simulations with slightly different parameters to identify results that are robust vs. fragile.

```python
from policylab.features.ensemble_runner import EnsembleRunner

runner = EnsembleRunner(policy="...", n_ensemble=50)
result = runner.run()
print(result["robust_findings"])   # appear in >80% of runs
print(result["fragile_findings"])  # appear in <20% of runs
```

### Backtester

Tests whether the simulation reproduces documented historical outcomes for known policies.

```python
from policylab.features.backtester import Backtester

bt = Backtester()
result = bt.run(case="eu_ai_act")    # or "us_eo_14110"
print(result["hit_rate"])            # fraction of historical behaviors reproduced
```

Minimum hit rate threshold: `MIN_HIT_RATE = 0.30`. Historical matching uses `SemanticMatcher` (cosine threshold=0.45) via sentence-transformers, not keyword matching. `ContaminationGuard` checks whether the model has likely seen a given policy text in training data.

**Backtest epistemic status:** Hit rates measure behavioral consistency with documented history, not predictive validity. Known limitations: LLM training-data contamination for high-profile policies; 6-agent configuration cannot represent hundreds of real stakeholders; historical outcomes reflect one realized path.

---

## Hybrid Loop (v1 + v2 Combined)

The hybrid loop runs the v2 vectorized engine for population dynamics and calls v1 LLM agents for strategic decision points (major policy shifts, coalition negotiations, incident response).

```python
from policylab.v2.simulation.hybrid_loop import run_hybrid_simulation

result = run_hybrid_simulation(
    policy=policy,
    n_agents=10_000,
    rounds=12,
    llm_rounds=[3, 6, 9],    # call LLM agents on these rounds
    seed=42,
)
```

LLM agents fire on the specified rounds, their outputs are parsed into belief shifts and action frequencies, and those feed back into the next vectorized round.

---

## CLI Reference

```bash
python -m policylab.cli <command> [options]
```

| Command | What it does | Key flags |
|---|---|---|
| `v2-stress-test` | Run vectorized v2 simulation | `--preset`, `--rounds`, `--n-agents`, `--seed` |
| `stress-test` | Run v1 LLM stress tester | `--policy`, `--severity`, `--rounds`, `--model` |
| `war-game` | Run v1 war game | `--policy`, `--scenario`, `--framework` |
| `blind-spots` | Sweep parameters for failure modes | `--policy`, `--n-runs` |
| `backtest` | Test against historical case | `--case` (`eu_ai_act` or `us_eo_14110`) |
| `demo` | Run a quick demo with built-in policy | — |
| `ingest` | Ingest a document and simulate | `--api-key`, `--model`, `--base-url`, `--no-simulate`, `--output-json`, `--traceability` |

**v2-stress-test examples:**

```bash
# EU AI Act preset, 12 rounds, 10k agents
python -m policylab.cli v2-stress-test --preset eu_ai_act_gpai --rounds 12 --n-agents 10000

# Custom severity
python -m policylab.cli v2-stress-test --severity 3.5 --rounds 8

# With a policy amendment on round 6
python -m policylab.cli v2-stress-test --preset california_sb53 --amend-round 6 --amend-delta -0.5
```

**Calibration report:**

```bash
python -m policylab.cli calibration-report
```

---

## Streamlit App

```bash
streamlit run app.py
```

Three modes accessible from the sidebar:

**Analyze bill** — parse a bill's provisions using the policy parser, display severity score with breakdown by dimension, run a v2 simulation, show compliance curve and relocation trajectory.

**Compare policies** — run two policies side-by-side, compare relocation rates, compliance timelines, indicator trajectories, and failure mode frequencies.

**Influence scenario** — configure a belief injection: choose direction (pro/anti compliance), number of targets, injection magnitude; run matched-pair simulation; display delta plots showing the injection's effect on relocation and compliance.

---

## Interpreting Results

**Reading the simulation output:**

```python
result = run_v2_simulation(policy, rounds=12, n_agents=10_000)

result["relocation_rate"]         # fraction relocated by end [0, 1]
result["compliance_rate"]         # fraction compliant at end [0, 1]
result["evasion_rate"]            # fraction actively evading [0, 1]
result["lobbying_rate"]           # fraction that lobbied at least once [0, 1]
result["compliance_curve"]        # list of 12 per-round compliance fractions
result["relocation_curve"]        # list of 12 per-round relocation fractions
result["indicators"]              # final dict of economic indicators
result["indicator_history"]       # list of 12 per-round indicator dicts
result["tipping_reports"]         # list of 12 TippingPointReport objects
result["summary"]                 # human-readable summary string
```

**Ranking policies:** Compare `relocation_rate` and `compliance_rate` across policies at the same `n_agents` and `rounds`. The ranking is more reliable than the absolute values.

**What drives compliance timing:** The dominant factor is agent type (lambda calibrated from DLA Piper). Large companies comply faster (λ=3.32), startups slower (λ=10.9). Severity adjusts the hazard rate upward.

**What drives relocation:** Belief threshold crossings. Frontier labs have a lower threshold (45.0) and an additional belief-proportional relocation factor. The 12% baseline relocation target is calibrated to self-reported GDPR data and is the weakest calibration moment — treat with caution.

**Indicators:** Read directional changes only. "Investment shows downward pressure consistent with OECD PMR literature" is a valid citation. "Investment will fall by 18 points" is not.

---

## Known Limitations

**Recent bug fixes (not yet reflected in published calibration):**
- `observed_enforcement_frac` denominator was `len(llm_agents or [1]) = 1` in
  population-only mode, sending binary noise into agent memory. Fixed: now divides
  by `config.n_population`. SMM moment m6 (enforcement contact rate) may shift
  slightly from previously-published runs — re-run calibration if citing m6.
- `CompanyStock.relocated` was double-counted. Fixed: single accounting point
  in `update()`. SMM moment m3 (relocation rate) unaffected (driven by agent
  arrays, not stock accounting).

**Calibration coverage:**
- Relocation moment (m3 = 0.12) is based on self-reported intent-to-relocate surveys, not actual movements. This is the weakest calibration moment.
- m4 (SME compliance 24mo) and m5 (large compliance 24mo) share a partially collinear fallback path in the SMM objective — treat their individual fit with caution.
- All frontier_lab parameters are ASSUMED. There is no published compliance survey data for the small number of firms that would qualify as frontier labs globally.
- GDPR calibration targets are used as a proxy for AI regulation. AI-specific compliance data does not yet exist.

**Population structure:**
- LLM agents over-coordinate compared to real humans (OASIS, arXiv:2411.11581). The v1 engine partially corrects this with SAFETY_FIRST_CORP, but full correction requires hundreds of heterogeneous agents.
- 6 LLM agents cannot represent the diversity of real stakeholder populations.

**Stock accounting:**
- ~~`CompanyStock.relocated` double-count~~ **Fixed**: the direct `+=` in `hybrid_loop.py` was removed; `update()` is now the single accounting point. Stock totals and SMM moments m3/m6 are unaffected by this bug.
- `new_entrants` are announced but do not feed back into `domestic_count()` — entry dynamics are tracked but the conservation law is not yet closed.
- The `failed` accumulator is declared but has no inflow path — failed companies do not yet reduce the domestic stock.

**Dynamics:**
- The Ugur R&D→innovation coefficient (~0.017 pts/round maximum) is numerically small relative to the relocation effect. In practice, relocation dynamics dominate innovation stock changes more than the R&D investment feedback does.
- `ongoing_burden_per_severity` (DIRECTIONAL, 1.5 pts/round/severity unit) is the largest single driver of the burden indicator in multi-round simulations.
- The public trust feedback loop has no return path: trust falls when companies leave but has no mechanism to recover except through the innovation→trust link, which is weak.

**International routing:**
- Softmax temperature=0.1 produces strongly concentrated flows to UAE/Singapore. Real relocation is less concentrated due to talent markets, legal infrastructure, and customer proximity.
- Log congestion penalty parameters are assumed.

---

## References

Aghion, P. & Howitt, P. (1992). A model of growth through creative destruction. *Econometrica* 60(2): 323–351.

Aghion, P., Bloom, N., Blundell, R., Griffith, R. & Howitt, P. (2005). Competition and innovation: An inverted-U relationship. *QJE* 120(2): 701–728.

Baumgartner, F.R. & Leech, B.L. (1998). *Basic Interests: The Importance of Groups in Politics and in Political Science*. Princeton UP. [Coalition lobbying bonus: 30–50% higher success]

Davis, F.D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly* 13(3): 319–340.

DLA Piper (2020). GDPR Fines and Data Breach Survey. [Compliance rates m4, m5, m6]

European Commission (2021). Impact Assessment SWD(2021) 84 final. [Compliance rate m2]

Lamperti, F., Roventini, A. & Sani, A. (2018). Agent-based model calibration using machine learning surrogates. *JEDC* 90: 366–389.

Lempert, R.J. (2003). *Shaping the Next One Hundred Years: New Methods for Quantitative, Long-Term Policy Analysis*. RAND.

OASIS / CAMEL-AI (2024). Open Agents Social Interaction Simulations. arXiv:2411.11581. [LLM herd-behavior bias]

Springler, E. et al. (2023). Product market regulation and FDI: Evidence from OECD countries. *Empirica*. [burden→investment ε=−0.197]

Ugur, M., Trushin, E., Solomon, E. & Guidi, F. (2016). R&D and productivity in OECD firms and industries. *R&D Management* 46(3): 461–469. [R&D→innovation ε=0.138, SE=0.012]
