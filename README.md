<p align="center">
  <img src="swarmcast_logo.svg" alt="SwarmCast" width="620" />
</p>

<p align="center">
  <b>v1.0.0</b> &nbsp;·&nbsp; Author: Ambar<br/>
  <a href="#1-quickstart--web-app">Quickstart</a> &nbsp;·&nbsp;
  <a href="#2-optional-swarm-llm-intelligence">Swarm Mode</a> &nbsp;·&nbsp;
  <a href="#6-epistemic-framework">Epistemic Framework</a> &nbsp;·&nbsp;
  <a href="#7-python-api">Python API</a>
</p>

---

SwarmCast is a macro behavioral simulation engine. It models how large populations of heterogeneous agents — companies, investors, regulators, consumers, traders — respond to shocks, rules, and incentives over time. It produces population-level behavioral forecasts (adoption curves, defection rates, coalition formation, jurisdiction flight, compliance timelines) with an epistemic layer that tracks exactly how grounded each output is.

Built for any domain where aggregate behavior emerges from individual decisions:

| Domain | What you simulate |
|---|---|
| **Regulatory policy** | Compliance rates, evasion, jurisdiction flight, lobbying coalitions |
| **Macroeconomics** | Market adoption, capital reallocation, investment suppression |
| **Financial markets** | Sentiment propagation, herding, sector rotation under shock |
| **Geopolitics** | Alliance formation, sanctions response, technology decoupling |
| **Public health** | Behavior change under mandates, vaccine adoption, non-compliance |
| **ESG / climate** | Corporate adaptation, greenwashing, regulatory arbitrage |

---

## Table of Contents

1. [Quickstart — Web App](#1-quickstart--web-app)
2. [Optional: Swarm LLM Intelligence](#2-optional-swarm-llm-intelligence)
3. [Architecture](#3-architecture)
4. [Simulation Engine](#4-simulation-engine)
5. [15 Jurisdictions](#5-15-jurisdictions)
6. [Epistemic Framework](#6-epistemic-framework)
7. [Python API](#7-python-api)
8. [CLI Reference](#8-cli-reference)
9. [Interpreting Results](#9-interpreting-results)
10. [Known Limitations](#10-known-limitations)
11. [References](#11-references)

---

## 1. Quickstart — Web App

The fastest way to use SwarmCast is the web interface. You get an interactive simulation environment, real-time charts, agent network visualizations, and an epistemic transparency layer — no code required.

### Step 1 — Install Python dependencies

```bash
cd policylab_m3
pip install -e ".[dev]"
```

Requires Python 3.10+.

### Step 2 — Start the API server

```bash
cd api
uvicorn main:app --reload --port 8000
```

The API will be running at `http://localhost:8000`. Check `http://localhost:8000/docs` for the interactive Swagger UI.

### Step 3 — Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

### Step 4 — Run your first simulation

1. Go to **Analyze** in the top nav
2. Pick a preset scenario (EU AI Act, California SB-53, etc.) or define your own parameters
3. Set population size and number of rounds in **Simulation Config**
4. Click **Run Simulation**
5. Results appear within 2–4 seconds: compliance trajectory, relocation rates, investment index, enforcement contact rate, jurisdiction flow diagram, and simulated moments table

> **Tip:** The web app works for any domain. The preset names are AI governance examples but the underlying engine is domain-agnostic — just change the severity, penalty structure, and agent population.

---

## 2. Optional: Swarm LLM Intelligence

By default, SwarmCast runs a fast vectorized simulation (~2s for n=1,000 agents). Optionally, you can layer in **swarm LLM intelligence** — a set of 23 persona agents (frontier lab, SME, investor, regulator, civil society…) that reason about the scenario using a language model and produce behaviorally-calibrated priors that seed the population simulation.

Swarm mode is **off by default** and requires an OpenAI API key.

### Enable swarm intelligence

**In the web app:**

The **Swarm Intelligence** toggle is in the Simulation Config panel (right side, below population size). Toggle it on before clicking Run. The simulation will take ~20–40 seconds instead of 2–4 seconds while the LLM personas reason in parallel.

**Via environment variable:**

```bash
# api/.env  (no restart needed — read on every request)
OPENAI_API_KEY=sk-...

# or with the SwarmCast-namespaced key:
SWARMCAST_OPENAI_API_KEY=sk-...
```

You can also set it in the web app's **OpenAI API Key** field without touching environment files.

### What swarm intelligence adds

When swarm is enabled, each of the 23 persona agents:
1. Reads the scenario description
2. Reasons about likely behavioral responses from their perspective
3. Votes on compliance probability, relocation pressure, and lobbying intensity

Their votes are aggregated into a behavioral prior that replaces the default GDPR-fitted parameter baseline. Results are tagged `[SWARM-ELICITED]` in the epistemic layer.

After the run, the **Agent Network** panel at the bottom of the results page shows:
- **Swarm personas tab**: hub-and-spoke network graph. Click any node to read that agent's full reasoning.
- **Population model tab**: 400-dot cloud colored by behavioral outcome (compliant / relocating / evading / lobbying), plus a plain-English explanation of the ABM mechanics.

### When to use swarm mode

| Use case | Recommendation |
|---|---|
| Quick scenario exploration | Off — vectorized baseline is faster and sufficient |
| Presenting to stakeholders | On — the reasoning chain adds interpretability |
| Novel domain with no calibration data | On — LLM priors substitute for missing empirical targets |
| Sensitivity sweeps (many runs) | Off — LLM calls add latency and cost |
| Evidence pack generation | On — included automatically in the confidence bands |

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Web App  (Next.js 14, port 3000)                              │
│  ├── /analyze      — run & visualize a scenario                │
│  ├── /compare      — side-by-side multi-scenario comparison    │
│  ├── /influence    — adversarial belief injection              │
│  └── /upload       — parse a PDF/text document into a scenario │
└───────────────────────┬────────────────────────────────────────┘
                        │ REST
┌───────────────────────▼────────────────────────────────────────┐
│  FastAPI  (port 8000)                                          │
│  ├── POST /simulate        — run single scenario               │
│  ├── POST /simulate/upload — ingest document + simulate        │
│  ├── POST /compare         — parallel multi-scenario run       │
│  ├── POST /evidence-pack   — async confidence band job         │
│  ├── GET  /evidence-pack/{id} — poll job status               │
│  └── POST /inject          — adversarial influence run         │
└───────────────────────┬────────────────────────────────────────┘
                        │ Python
┌───────────────────────▼────────────────────────────────────────┐
│  Simulation Engine  (policylab/v2/)                            │
│  ├── v2 Vectorized engine  — ~2ms/round at n=10,000            │
│  │   ├── PopulationArray   — agent state as float32 arrays     │
│  │   ├── DeGroot influence — network belief propagation        │
│  │   ├── SMM calibration   — 6 moments, GDPR empirical data    │
│  │   ├── GovernanceStocks  — system dynamics layer             │
│  │   ├── EventQueue        — shocks, amendments, triggers      │
│  │   └── 15 Jurisdictions  — softmax routing with burden       │
│  │                                                             │
│  ├── v1 LLM engine  — Concordia agents, ~200s/round            │
│  │   ├── StressTester, WarGame, BlindSpotFinder                │
│  │   └── EnsembleRunner, Backtester                            │
│  │                                                             │
│  └── Swarm Elicitation  — 23 LLM personas → behavioral priors  │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Simulation Engine

### Vectorized population model (v2)

The core engine represents a population of N heterogeneous agents (default 1,000; configurable up to 100,000+) as float32 arrays for speed. Each round (~quarterly cycle by default):

1. **Belief update** — DeGroot averaging over a Barabasí–Albert network propagates beliefs through the population
2. **Compliance decision** — each agent weighs expected penalty, peer compliance, and type-specific cost sensitivity against a sigmoid response function
3. **Relocation decision** — agents above a threshold consider leaving; softmax routing sends them to the least-burdened destination jurisdiction
4. **Lobbying / evasion** — non-compliant agents either evade (low-cost types) or lobby (high-cost, politically connected types)
5. **Stock update** — GovernanceStocks layer updates regulatory burden, innovation rate, AI investment index, and public trust using system dynamics equations

**Agent types** (population mix is configurable):

| Type | Default share | Baseline compliance sensitivity |
|---|---|---|
| Large company | 15% | Low — compliance is budget item |
| Mid company | 25% | Medium |
| Startup | 30% | High — compliance is existential |
| Researcher | 10% | High |
| Investor | 10% | Tracks compliance signal |
| Civil society | 5% | Norm-based |
| Frontier lab | 5% | Strategic — lobbies heavily |

### SMM calibration

Six simulated moments are calibrated against empirical targets:

| Moment | Epistemic tier | Source |
|---|---|---|
| Compliance rate year 1 | **GROUNDED** | DLA Piper GDPR survey 2020, n=200 |
| Relocation rate | DIRECTIONAL | EU AI Act impact assessment |
| Lobbying rate | DIRECTIONAL | EU transparency register |
| SME compliance 24mo | ASSUMED | Structural assumption |
| Large firm compliance 24mo | DIRECTIONAL | GDPR large-firm follow-up |
| Enforcement contact rate | DIRECTIONAL | DPC annual report |

---

## 5. 15 Jurisdictions

SwarmCast models company relocation across 15 destinations. Each jurisdiction has a regulatory burden score, corporate tax rate, innovation subsidy, and initial company count.

| Jurisdiction | Burden | Tax | Stance | Notes |
|---|---|---|---|---|
| EU | 55 | 25% | Strict | Post-AI Act + GDPR |
| US | 25 | 21% | Moderate | Pre-comprehensive federal AI law |
| UK | 20 | 19% | Moderate | Pro-innovation DSIT approach |
| Singapore | 10 | 17% | Permissive | MAS/IMDA voluntary framework |
| UAE | 5 | 9% | Permissive | ADGM/DIFC free-zone model |
| China | 60 | 25% | Strict | AIGC + data localisation |
| Canada | 30 | 15% | Moderate | AIDA framework |
| Japan | 22 | 23% | Moderate | METI soft-law guidelines |
| Switzerland | 15 | 12% | Permissive | nFADP only, low cantonal tax |
| Australia | 28 | 30% | Moderate | Voluntary AI Ethics Framework |
| India | 12 | 22% | Permissive | No AI regulation, IndiaAI Mission |
| Russia | 65 | 20% | Strict | Sovereign AI + data localisation |
| South Korea | 25 | 22% | Moderate | AI Basic Act 2024 |
| France | 50 | 25% | Strict | EU AI Act + active CNIL |
| Germany | 58 | 30% | Strict | EU AI Act + BSI/BaFin guidance |

Relocation is routed via softmax discrete choice over `attractiveness = 100 - (burden × 0.7 + tax × 0.3) + subsidy`. Companies cluster in the cheapest available destination — low temperature (0.1) means near-deterministic routing, matching observed EU AI Act patterns.

---

## 6. Epistemic Framework

Every output carries one of three labels. Read this before citing any number.

| Label | Meaning | How to use |
|---|---|---|
| **GROUNDED** | Calibrated against empirical data with documented source | Cite direction and approximate magnitude |
| **DIRECTIONAL** | Direction known from theory or literature; magnitude uncertain | Cite direction only |
| **ASSUMED** | No calibration target; structurally plausible but invented | Sensitivity sweep only; never cite as prediction |

**Valid uses of output:**

```
✓  "In 100% of runs, at least one company relocated."
✓  "Regulatory burden shows directional suppression of investment
    consistent with OECD PMR literature."
✓  "Relocation appears in 94% of sensitivity configurations."

✗  "Innovation will drop by 37 points."
✗  "The tipping point occurs at severity 3.2."
✗  "Public trust will fall to 36.5." (ASSUMED threshold)
```

ASSUMED rows are visually dimmed in the web app. GROUNDED rows are the only ones safe to cite in reports.

---

## 7. Python API

### Run a simulation directly

```python
from policylab.v2.simulation.hybrid_loop import run_hybrid_simulation, HybridSimConfig

result = run_hybrid_simulation(
    policy_name="Carbon border adjustment",
    policy_description="Tariff on carbon-intensive imports proportional to embedded emissions",
    policy_severity=2.5,           # 1.0 (light) to 5.0 (extreme)
    config=HybridSimConfig(
        n_population=5000,
        num_rounds=16,
        source_jurisdiction="EU",
        destination_jurisdictions=["US", "UK", "Switzerland", "Singapore"],
        seed=42,
    )
)

print(result.simulated_moments)
print(result.jurisdiction_summary)
print(result.elapsed_seconds)
```

### Run with swarm intelligence

```python
result = run_hybrid_simulation(
    policy_name="Central bank digital currency mandate",
    policy_description="All retail payments above $500 must use CBDC by 2027",
    policy_severity=3.8,
    config=HybridSimConfig(
        n_population=2000,
        num_rounds=12,
        run_llm_strategic=True,        # enables swarm elicitation
        llm_model="gpt-4o",            # any OpenAI-compatible model
        seed=42,
    )
)
```

### Parse a document into simulation parameters

```python
from policylab.v2.ingest.pipeline import ingest_text

result = ingest_text(
    text="""
    All exchanges operating in the EU must hold 30% of customer assets
    in domestic custodians by Q1 2026. Violations subject to civil fines
    up to 5% of annual revenue. Six-month grace period from enactment.
    """,
    name="EU Exchange Custody Rule",
    api_key="sk-...",
)

print(result.spec)           # extracted PolicySpec
print(result.config)         # ready-to-use HybridSimConfig dict
```

### Adversarial influence injection

```python
from policylab.v2.influence.adversarial import run_with_injection

result = run_with_injection(
    policy_name="Mandatory AI safety audits",
    policy_description="...",
    policy_severity=2.8,
    injection_fraction=0.05,       # 5% of agents receive the injection
    injection_belief="This regulation is unenforceable and will be repealed",
    injection_start_round=3,
)

print(result.resilience_score)     # 0–1; higher = more resistant to manipulation
print(result.compliance_delta)     # change vs uninflected baseline
```

---

## 8. CLI Reference

```bash
# v2 simulation
python -m policylab.cli v2-stress-test --preset eu_ai_act_gpai --rounds 12

# stress test with LLM agents (v1)
python -m policylab.cli stress-test --policy "Mandatory safety audits for all AI systems"

# war game a specific incident scenario
python -m policylab.cli war-game --scenario "major_incident"

# find blind spots in a policy design
python -m policylab.cli blind-spots --policy "Compute threshold licensing"

# backtest against a historical case
python -m policylab.cli backtest --case eu_ai_act

# demo run
python -m policylab.cli demo
```

---

## 9. Interpreting Results

### Compliance trajectory

The round-by-round compliance rate chart shows early adopters, late adopters, and a steady state. Year-1 compliance (in the Simulated Moments table) is the mean over rounds 1–4 — it will be higher than the round-1 snapshot because early adopters join during the year.

### Relocation

Relocation rate is the fraction of the original population that has permanently left the source jurisdiction. Watch the Jurisdiction Flow Diagram for where they went and how that changes destination burden over time.

### SMM distance

The SMM distance is a weighted Euclidean distance between simulated moments and their GDPR-era calibration targets. Lower = closer to the empirically-grounded baseline. Values below 0.10 indicate the simulation is behaving within the historical reference range.

### Confidence bands

Run the Evidence Pack (Analyze page → Evidence Pack button) to generate 9-run ensemble bands. Shaded areas on the trajectory charts show p10–p90 spread. Wide bands = high model uncertainty; interpret direction only.

---

## 10. Known Limitations

- **No equilibrium guarantee.** The simulation runs for a fixed number of rounds. It does not check for steady-state convergence.
- **Calibrated to GDPR, not general.** The default parameters were fit to EU regulatory compliance data. For other domains (financial markets, public health), treat all moments as DIRECTIONAL or ASSUMED until recalibrated.
- **Network is static.** The social influence graph is initialised once at simulation start and does not rewire as agents relocate or change behavior.
- **Jurisdiction burden is endogenous but one-directional.** Regulatory burden in destination jurisdictions rises as companies arrive (logarithmic saturation), but source jurisdiction burden does not fall as companies leave.
- **Swarm personas are AI governance–tuned.** The 23 LLM persona prompts are written for regulatory compliance scenarios. For other domains, the persona reasoning will still be directionally useful but less precisely calibrated.
- **No financial contagion.** The investment index is an aggregate sentiment indicator. For asset pricing applications, pair outputs with a quantitative pricing model — SwarmCast gives you the behavioral layer, not the discount rate.

---

## 11. References

- DLA Piper, *GDPR Fines and Data Breach Survey* (2020) — compliance rate calibration
- European Commission, *AI Act Impact Assessment* (2021) — relocation and burden estimates
- Wilson (1989), *Bureaucracy* — regulatory capacity model for jurisdiction saturation
- Lempert (2003), *Shaping the Next One Hundred Years* — ensemble perturbation methodology
- DeGroot (1974), *Reaching a Consensus* — belief propagation model
- OECD, *Product Market Regulation indicators* — investment suppression direction
- Barabási & Albert (1999), *Emergence of Scaling in Random Networks* — social graph topology
- McFadden (1974), *Conditional Logit Analysis* — softmax discrete choice for relocation

---

---

*SwarmCast v1.0.0 · All ASSUMED-tier parameters should be swept in sensitivity analysis before outputs are presented as evidence. GROUNDED and DIRECTIONAL outputs are calibrated against empirical sources cited above.*
