# PolicyLab v2 — Hybrid Governance Simulation Engine

*Author: Ambar*

## What's New in v2

v2 is a ground-up rebuild of the simulation layer. It keeps the v1 LLM
strategic agents but adds a calibrated population layer beneath them.

### Architecture

```
v2 Hybrid Simulation
├── 5 LLM Strategic Agents (--llm flag, optional)
│   ├── Government Official  — counter-policies, enforcement priorities
│   ├── Regulator            — enforcement execution, capacity allocation  
│   ├── Industry Association — lobbying coordination, exemption negotiation
│   ├── Civil Society Leader — public campaigns, coalition formation
│   └── Safety-First Corp    — early compliance, regulatory moat strategy
│
└── 50-200 Population Agents (always active, rule-based)
    ├── Compliance: Weibull model fitted to DLA Piper 2020 GDPR data
    │   λ_large=3.32 rounds (91% at 24mo), λ_sme=10.9 rounds (52% at 24mo)
    ├── Relocation: two-regime sigmoid
    │   • Normal (sev≤3): calibrated to 12% at burden=70 (EU AI Act)
    │   • Criminal (sev=5): ~78% over 16 rounds (dissolution + imprisonment)
    ├── Social learning: DeGroot (1974) belief updating via network
    └── Network: Barabási-Albert scale-free topology (Newman 2003)
```

### Structural Improvements vs v1

| Issue | v1 | v2 |
|---|---|---|
| No innovation→investment feedback | acyclic | Aghion-Howitt expectation |
| No domestic company stock | no stock | CompanyStock with conservation |
| Burden ratchets only | no outflow | compliance discharge outflow |
| 8-round horizon | 8 rounds | 16 rounds (4 years) default |
| Instantaneous relocation | instant | 2–4 round pipeline delay |
| Relocation destroys innovation | 100% loss | spillover_factor=0.5 |
| Unitless indicators | no anchors | dimensional anchors (TFP%, $B) |

## CLI Usage

```bash
# Population-only (fast, no API key needed)
python -m policylab v2-stress-test \
  --policy "EU AI Act" \
  --description "Mandatory risk assessment for high-risk AI..." \
  --severity 3 --n-population 100 --rounds 16 --n-ensemble 5

# Full hybrid (population + 5 LLM strategic agents)
python -m policylab v2-stress-test \
  --policy "Total AI Development Moratorium" \
  --description "3-year moratorium. Dissolution. 10 years imprisonment." \
  --severity 5 --n-population 200 --rounds 16 --n-ensemble 3 --llm
```

## Python API

```python
# Ensemble stress test
from policylab.v2.stress_test_v2 import HybridStressTest
tester = HybridStressTest(n_population=100, num_rounds=16)
report = tester.run("Policy Name", "Description...", n_ensemble=5)
print(report.summary())

# Counterfactual (policy vs clean no-regulation baseline)
from policylab.v2.analysis import run_counterfactual_v2
cf = run_counterfactual_v2("Name", "Description", severity=5.0)
print(cf.summary())  # shows deltas: Δinvestment, Δinnovation, Δburden

# Policy comparison/ranking
from policylab.v2.analysis import compare_policies, PolicySpec
ranking = compare_policies([
    PolicySpec("Voluntary Guidelines", "...", 1.0),
    PolicySpec("EU AI Act", "...", 3.0),
    PolicySpec("Criminal Ban", "...", 5.0),
])
print(ranking.summary())  # ranked table + composite score

# Sensitivity analysis
from policylab.v2.analysis import run_sensitivity_v2
sens = run_sensitivity_v2("Policy", "...", severity=5.0,
    parameter="spillover_factor", values=[0.2, 0.5, 0.8])
print(sens.summary())  # ROBUST / DIRECTIONAL-DEPENDENT / NON-ROBUST per indicator
```

## Epistemic Status

| Component | Status | Source |
|---|---|---|
| Compliance Weibull λ | GROUNDED | DLA Piper 2020 GDPR data |
| Relocation sev≤3 | GROUNDED | EU AI Act Transparency Register |
| Relocation sev=5 | DIRECTIONAL | Extrapolation from EU AI Act |
| Criminal threshold bonus | DIRECTIONAL | OpenAI EU exit threat (sev~2 baseline) |
| DeGroot belief updating | GROUNDED direction | DeGroot 1974; Golub & Jackson 2010 |
| Aghion-Howitt coupling | GROUNDED direction | Aghion & Howitt 1992 |
| Spillover factor 0.5 | ASSUMED | Sweep [0.2, 0.5, 0.8] |
| Ongoing burden 1.5/round | DIRECTIONAL | Sweep [0.5, 1.5, 3.0] |
| SMM calibration | FRAMEWORK READY | Lamperti et al. 2018 |

## Next Steps for Full Calibration

1. Run 500+ population-only simulations for SMM surrogate training
2. Train MLSurrogate on (θ, moments) pairs
3. Run Nelder-Mead on surrogate (~20 restarts)
4. Verify top-5 θ* with full hybrid simulation
5. See `policylab/v2/calibration/smm_framework.py`
