"""Epistemic status disclaimers for all PolicyLab reports.

Every report outputs three kinds of content. This module documents their
different epistemic statuses so readers know what to trust.

─────────────────────────────────────────────────────────────────────────────
LAYER 1 — BEHAVIORAL OUTPUT  [Most reliable]
─────────────────────────────────────────────────────────────────────────────
Action frequencies, coalition patterns, relocation rates, failure mode
detection. These arise from LLM agents reasoning about the policy and their
objectives. Direction and pattern are meaningful.

Known limitation: LLM agents over-coordinate compared to real humans.
Source: OASIS (arXiv:2411.11581) — "agents are more susceptible to herd
behavior than humans." PolicyLab partially corrects this by including one
contrarian agent (a company that supports the regulation), but full
correction would require hundreds of heterogeneous agents.

─────────────────────────────────────────────────────────────────────────────
LAYER 2 — INDICATOR TENDENCIES  [Use direction, not magnitude]
─────────────────────────────────────────────────────────────────────────────
Indicator changes (innovation_rate, ai_investment_index, public_trust, etc.)
computed by the system dynamics layer. Divided into three sub-layers:

  GROUNDED (4 effects with empirical derivations):
  • burden → investment: ε = −0.00295/round [OECD PMR FDI gravity, Springler
    et al. 2023, −0.197 annual elasticity converted to per-round per-index-pt]
  • investment → innovation: ε = +0.000345/round [Ugur et al. 2016 meta-
    analysis, R&D-to-productivity elasticity 0.138, per-quarter-per-index-pt]
  • market concentration → innovation: inverted-U peak at concentration=40
    [Aghion, Bloom et al. 2005 QJE, Lerner index peak ≈ 0.4 → 40 on 0-100]
  • coalition lobbying bonus: 1.30-1.50× [Baumgartner & Leech 1998,
    30-50% higher success for organized coalitions in US federal rulemaking]

  DIRECTIONAL (direction known; set to 0.0 in RIGOROUS_BASELINE):
  • trust ↔ regulatory burden (legitimacy theory; no empirical magnitude)
  • passive burden accumulation (conceptual; no empirical magnitude)
  • innovation → public trust (TAM; no empirical magnitude)

  ASSUMED (disabled in RIGOROUS_BASELINE; SENSITIVITY_ASSUMED only):
  • tipping-point thresholds — completely invented
  • cascade multiplier — completely invented
  • ongoing relocation drain — was calibrated to desired output (worst bias)

─────────────────────────────────────────────────────────────────────────────
LAYER 3 — SENSITIVITY ANALYSIS  [Label results as ROBUST / NOT-ROBUST]
─────────────────────────────────────────────────────────────────────────────
The sensitivity_layer module sweeps all DIRECTIONAL and ASSUMED parameters
across their plausible ranges and classifies results:

  ROBUST:             appears in RIGOROUS_BASELINE and >80% of configs
  DIRECTIONAL-DEPENDENT: only appears when directional params are non-zero
  ASSUMED-DEPENDENT:  only appears in SENSITIVITY_ASSUMED (non-robust)
  NOT-ROBUST:         doesn't appear reliably in any configuration

─────────────────────────────────────────────────────────────────────────────
HOW TO CITE RESULTS
─────────────────────────────────────────────────────────────────────────────
  VALID (behavioral):   "In 100% of runs, at least one company relocated."
  VALID (grounded):     "Regulatory burden shows directional suppression of
                         investment consistent with OECD PMR literature."
  VALID (sensitivity):  "Capital flight is a robust result (appears in
                         RIGOROUS_BASELINE and 94% of sensitivity configs)."
  INVALID:              "Innovation will drop by 37 points."
  INVALID:              "Public trust will fall to 36.5 ± 10.8."
  INVALID:              "The tipping point was reached." (ASSUMED-LAYER only)

─────────────────────────────────────────────────────────────────────────────
FUTURE CALIBRATION PATH
─────────────────────────────────────────────────────────────────────────────
Proper calibration requires Simulated Method of Moments (SMM) or Bayesian
estimation against historical data:

  • EU AI Act 2021-2024: EU Transparency Register, EC Impact Assessment
  • GDPR 2018-2020: IAPP survey, DLA Piper DPA annual reports
  • SB 1047 2023: California legislature records, OpenSecrets filings

ML surrogate approach (Lamperti, Roventini & Sani 2018) is recommended
given the ~200s per simulation run cost.

─────────────────────────────────────────────────────────────────────────────
REFERENCES
─────────────────────────────────────────────────────────────────────────────
  Aghion & Howitt (1992). Econometrica 60(2): 323-351. [Aghion & Howitt, 2025 Nobel Prize in Economic Sciences]
  Aghion, Bloom et al. (2005). QJE 120(2): 701-728. [Inverted-U]
  Baumgartner & Leech (1998). Basic Interests. Princeton UP.
  Springler et al. (2023). Empirica. [OECD PMR FDI gravity, ε=−0.197]
  Ugur et al. (2016). R&D Management. [Meta-analysis ε=0.138, SE=0.012]
  OASIS / CAMEL-AI (2024). arXiv:2411.11581. [LLM herd-behavior bias]
  Lamperti, Roventini & Sani (2018). JEDC 90: 366-389. [ML surrogates]
  Platt (2020). JEDC 113: 103859. [ABM calibration comparison]
"""

_SEPARATOR = "─" * 64

INDICATOR_DISCLAIMER = (
    f"\n{_SEPARATOR}\n"
    "EPISTEMIC STATUS — READ BEFORE INTERPRETING NUMBERS\n"
    f"{_SEPARATOR}\n"
    "THREE LAYERS OF OUTPUT (all are ORDINAL comparisons, not point estimates):\n"
    "  1. BEHAVIORAL (action frequencies, failure modes)  ← most reliable\n"
    "  2. INDICATOR TENDENCIES (directional, not magnitudes)\n"
    "     Grounded: burden→investment [OECD PMR ε=−0.197],\n"
    "               R&D→innovation [Ugur 2016 ε=0.138].\n"
    "     Ungrounded: trust/burden feedback, passive accumulation.\n"
    "  3. SENSITIVITY RESULTS (ASSUMED-LAYER only; labeled NON-ROBUST)\n"
    "\n"
    "VALID:   rank policies by behavioral impact severity\n"
    "VALID:   cite failure modes (relocation, evasion, trust collapse)\n"
    "VALID:   report indicator DIRECTION with explicit uncertainty\n"
    "INVALID: cite specific indicator values as point predictions\n"
    "INVALID: treat ± ranges as calibrated confidence intervals\n"
    "INVALID: cite [ASSUMED-LAYER] tipping results as policy predictions\n"
    "\n"
    "V1 LIMITATIONS (applies to the LLM-based stress-test engine only):\n"
    "  \u2022 Innovation responds to investment (\u03b5=0.000345/round) but investment does not respond back \u2014 one-way coupling only.\n"
    "  \u2022 Company population is implicit; relocation has no conservation law.\n"
    "  \u2022 Regulatory burden only increases; no compliance-driven discharge.\n"
    "  \u2022 Default simulation horizon is 8 rounds (2 years).\n"
    "  \u2022 Relocation is instantaneous; no pipeline delay.\n"
    "  \u2022 Indicators are unitless 0\u2013100 indices with no empirical anchor.\n"
    "  \u2022 B6: Model treats all companies as globally mobile by default;\n"
    "    this overstates exit rates for firms with deep domestic ties\n"
    "    (talent, customers, infrastructure). Treat relocation rate as an\n"
    "    upper bound for jurisdictions with high switching costs.\n"
    "  Staff scaling (0.3) and severity scoring coefficients: [ASSUMED],\n"
    "  no calibration target \u2014 run sweeps across plausible ranges.\n"
    "\n"
    "THREE [ASSUMED] PARAMETERS TO SWEEP IN ANY EVIDENCE PACK:\n"
    "  compute_cost_factor [1.0, 2.0, 4.0]\n"
    "    1.0=GDPR-equivalent (calibration baseline), 2.0=AI-specific,\n"
    "    4.0=broad compute coverage. Affects: compliance S-curve timing.\n"
    "  hk_epsilon [0.3, 0.5, 1.0]\n"
    "    1.0=pure DeGroot (baseline), 0.5=bounded confidence,\n"
    "    0.3=polarised camps. Affects: lobbying rate trajectory (m1).\n"
    "  severity_cubed_relocation exponent [1, 2, 3]\n"
    "    Cubic is assumed. Linear and quadratic equally defensible.\n"
    "    Affects: relocation rate at severity >= 4.\n"
    "\n"
    "For calibration methodology (SMM/Bayesian), see calibration.py\n"
    f"{_SEPARATOR}"
)

BACKTEST_DISCLAIMER = (
    f"\n{_SEPARATOR}\n"
    "BACKTEST EPISTEMIC STATUS\n"
    f"{_SEPARATOR}\n"
    "Hit rates measure whether LLM agents generate action descriptions that match documented historical responses. This is a proxy for behavioral consistency, not predictive accuracy of future policies. Known limitations:\n"
    "  1. LLM training-data contamination for high-profile policies.\n"
    "  2. 6-agent config cannot represent hundreds of real stakeholders.\n"
    "  3. Keyword matching penalizes semantically-equivalent responses.\n"
    "  4. Historical outcomes reflect one realized path; many paths exist.\n"
    f"{_SEPARATOR}"
)


def indicator_disclaimer() -> str:
    return INDICATOR_DISCLAIMER


def backtest_disclaimer() -> str:
    return BACKTEST_DISCLAIMER
