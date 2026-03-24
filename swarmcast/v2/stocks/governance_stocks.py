"""Stock-flow governance model with proper conservation laws.

Each variable is a stock with explicit inflows and outflows. Stocks track:
company population, regulatory burden, innovation capacity, and a relocation
pipeline that delays departure by 2–4 rounds.

Dimensional anchors map the 0–100 indices to real-world units so outputs can
be compared against empirical benchmarks.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarmcast.v2.population.agents import PopulationAgent


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONAL ANCHORS
# ─────────────────────────────────────────────────────────────────────────────

class DimensionalAnchors:
    """Maps 0-100 index scale to real-world units for interpretable outputs.

    Without dimensional anchors, every output is an arbitrary index and the
    model cannot be calibrated against empirical data.
    Each anchor is grounded in a specific published reference level.

    innovation_rate=50 ↔ 2.5% TFP growth/year: the US Bureau of Labor
    Statistics multi-factor productivity series averaged ~2.5%/year for the
    non-farm business sector across 2000-2019. This is the "normal" innovation
    rate a well-functioning AI jurisdiction should sustain.

    ai_investment=50 ↔ $50B/year: Stanford AI Index (2023) estimated global
    frontier AI R&D investment at roughly $40-60B in 2022, making $50B a
    reasonable mid-point anchor for the pre-GPT-4 era.
    """
    # innovation_rate = 50 ↔ TFP growth = 2.5%/year (US average 2000-2019, BLS)
    INNOVATION_RATE_AT_50 = 2.5  # % TFP growth per year

    # ai_investment = 50 ↔ $50B AI R&D (rough global frontier AI ~2022)
    AI_INVESTMENT_AT_50 = 50.0   # $ billions per year

    # regulatory_burden = 50 ↔ 50% of R&D budget devoted to compliance
    # (this is extreme; moderate regulation ~10-20%, total ban ~80%+)
    BURDEN_AT_50 = 50.0          # % of R&D budget

    # public_trust = 50 ↔ 50% trust AI to be beneficial (Eurobarometer scale)
    # EU AI trust ~50-60% in 2022 (Eurobarometer 98)
    TRUST_AT_50 = 50.0           # % of population

    @staticmethod
    def innovation_to_tfp(idx: float) -> float:
        """Convert InnovationCapacity stock value to TFP growth rate. 50 → 2.5%/yr."""
        return idx * DimensionalAnchors.INNOVATION_RATE_AT_50 / 50.0

    @staticmethod
    def investment_to_billions(idx: float) -> float:
        """Convert ai_investment stock value to USD billions/yr. 50 → $50B/yr."""
        return idx * DimensionalAnchors.AI_INVESTMENT_AT_50 / 50.0


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY STOCK
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CompanyStock:
    """Stock of AI companies remaining in the jurisdiction.

    Conservation equation: dN/dt = new_entrants - relocated - failed
    Every company that enters must exit through one of three paths: it
    relocates (enters the RelocationPipeline), fails/closes, or remains.
    There is no stock creation ex nihilo and no deletion without accounting.

    total starts at 100 (normalized) so that domestic_fraction() is directly
    comparable across scenarios with different absolute company counts.
    Calibration target for absolute scale: ~3,000 AI companies in the EU
    pre-AI-Act (Dealroom 2022 estimate), so each model unit ≈ 30 companies.

    Dimensional unit: number of companies (integer-valued stock).
    """
    total: int = 100          # normalized to 100 for comparison across scenarios
    in_pipeline: int = 0      # companies in relocation pipeline (not yet left)
    relocated: int = 0        # cumulative companies that have left
    failed: int = 0           # cumulative companies that closed (not relocated)
    new_entrants: int = 0     # cumulative new entries

    def domestic_fraction(self) -> float:
        """Current companies as a fraction of the original stock.

        Numerator includes new_entrants (see domestic_count docstring).
        Can exceed 1.0 if entry outpaces exit — a sign the jurisdiction
        is attracting companies even under regulation.
        """
        return max(0.0, self.domestic_count() / self.total)

    def domestic_count(self) -> int:
        """Current number of companies active in the jurisdiction.

        Conservation law: initial_total - departed + entrants.
        new_entrants are companies that form or enter after the policy is in
        force — they build compliance in from day 1 and are less likely to
        relocate than incumbents facing a sudden transition cost.

        Previously this method ignored new_entrants, creating a one-way drain
        where domestic_count could only fall. That made long-horizon runs at
        high severity look catastrophic purely as an accounting artifact.
        [GROUNDED direction] Entry responds to regulatory cost (IO literature);
        [ASSUMED magnitude] entry_rate_base=2.0 companies/round at baseline (Poisson draw).
        """
        return max(0, self.total - self.relocated - self.failed + self.new_entrants)

    def update(
        self,
        n_relocating_this_round: int,
        n_arriving_from_pipeline: int,
        burden: float,
        innovation_expectation: float,
        rng=None,
    ) -> None:
        """Update company stock for one round.

        n_relocating_this_round: companies entering the pipeline
        n_arriving_from_pipeline: companies completing relocation (leaving)
        """
        # Companies leaving
        self.in_pipeline += n_relocating_this_round
        self.relocated += n_arriving_from_pipeline
        self.in_pipeline -= n_arriving_from_pipeline
        self.in_pipeline = max(0, self.in_pipeline)

        # New entry: only if expected profits exceed regulatory costs
        # Entry rate drops sharply when burden is high (regulatory deterrence)
        # [DIRECTIONAL] Shape grounded (entry responds to regulatory cost);
        # coefficient is assumed
        # Entry rate: Poisson draw so fractional rates produce stochastic
        # but correct-on-average entry rather than always rounding to zero.
        #
        # entry_rate_base = 2.0 [ASSUMED]: ~2 new AI companies enter a major
        # jurisdiction per quarter at low-burden baseline. Calibration anchor:
        # the EU had ~3,000 AI companies pre-Act (Dealroom 2022). Entry at
        # ~2/quarter means the domestic stock turns over ~2.7% per year at
        # baseline, consistent with typical high-growth tech sector churn.
        #
        # burden_deterrence: linear decline from 1.0 (no burden) to 0.0
        # (burden=100). [DIRECTIONAL] direction grounded in IO entry literature;
        # linear form is [ASSUMED].
        #
        # innovation_attraction: higher domestic innovation attracts entrants.
        # [DIRECTIONAL] direction grounded; coefficient [ASSUMED].
        import numpy as _np_entry
        entry_rate_base = 2.0
        burden_deterrence = max(0.0, 1.0 - burden / 100.0)
        innovation_attraction = min(1.0, innovation_expectation / 100.0)
        rate = entry_rate_base * burden_deterrence * innovation_attraction
        # Poisson draw using module-level RNG (unseeded per-call) so each
        # invocation gets genuine stochastic variation. Reproducibility at the
        # simulation level comes from PopulationArray.generate(seed=...), not
        # from stock accounting which has no meaningful ordering constraint.
        # At rate=0.72 (burden=40, innov=60) this produces on average ~0.72
        # entrants/round = ~5.8 over 8 rounds vs always-0 from round(0.5×...).
        if rate > 0:
            if rng is not None:
                new_this_round = int(rng.poisson(rate))
            else:
                # Deterministic fallback: seed from burden + innovation so standalone
                # calls (tests, single runs) are reproducible without global state bleed.
                seed = hash((round(burden, 1), round(innovation_expectation, 1))) & 0xFFFFFFFF
                new_this_round = int(_np_entry.random.default_rng(seed).poisson(rate))
        else:
            new_this_round = 0
        self.new_entrants += new_this_round


# ─────────────────────────────────────────────────────────────────────────────
# RELOCATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class RelocationPipeline:
    """Companies that have decided to relocate but have not yet departed.

    Real-world corporate relocation involves legal entity changes, staff
    transitions, and regulatory de-registration — a process that typically
    takes 18-36 months. The 2-4 round delay (MIN=2, MAX=4, where 1 round ≈
    3 months) is a compressed proxy for this: it is shorter than the real
    timeline but preserves the key structural property that relocation is not
    instantaneous.

    This delay matters for policy dynamics because:
      1. Regulators can respond and soften policy during the pipeline window.
      2. If policy softens, companies already in-pipeline can cancel (~30%
         cancellation rate — see process() docstring).
      3. Short-run and long-run relocation counts diverge, enabling the model
         to distinguish policy shock from steady-state capital flight.

    [ASSUMED] MIN_DELAY=2, MAX_DELAY=4 rounds. The real 18-36 month range would
    map to 6-12 rounds at 1 round = 3 months; 2-4 is used as a conservative
    lower bound that avoids excessive pipeline accumulation in short simulations.
    """
    queue: deque = dataclasses.field(default_factory=deque)
    MIN_DELAY = 2   # rounds (≈ 6 months at 1 round = 3 months) [ASSUMED]
    MAX_DELAY = 4   # rounds (≈ 12 months) [ASSUMED]

    def add(
        self,
        company_id: str,
        current_round: int,
        reversible: bool = True,
        rng=None,
    ) -> None:
        """Add a company to the relocation pipeline."""
        if rng is not None:
            delay = int(rng.integers(self.MIN_DELAY, self.MAX_DELAY + 1))
        else:
            # Fallback: deterministic seed from company_id + round so standalone
            # calls (tests, non-ensemble use) are reproducible without global state bleed.
            import numpy as _np_rl
            seed = hash((company_id, current_round)) & 0xFFFFFFFF
            delay = int(_np_rl.random.default_rng(seed).integers(self.MIN_DELAY, self.MAX_DELAY + 1))
        completion_round = current_round + delay
        self.queue.append({
            "company_id": company_id,
            "decision_round": current_round,
            "completion_round": completion_round,
            "reversible": reversible,
        })

    def process(
        self,
        current_round: int,
        policy_softened: bool = False,
        rng=None,
    ) -> tuple[list[str], list[str]]:
        """Process pipeline: return (departing, cancelled) company IDs.

        If policy_softened and relocation is reversible, a fraction of companies
        in the pipeline cancel rather than complete relocation. This models the
        real-world phenomenon where regulatory rollback during the relocation window
        causes some companies to reverse course (e.g., firms that threatened to
        leave the EU during GDPR implementation then stayed when enforcement proved
        lighter than feared).

        cancellation_rate_on_softening=0.30 means 30% of reversible in-pipeline
        companies cancel when policy softens. [ASSUMED] — the direction
        (softening → some cancellations) is grounded; 30% is a heuristic.
        """
        departing = []
        cancelled = []
        remaining = deque()

        for entry in self.queue:
            if entry["completion_round"] <= current_round:
                if policy_softened and entry["reversible"]:
                    if rng is not None:
                        roll = rng.random()
                    else:
                        import numpy as _np_rl
                        seed = hash((entry["company_id"], current_round, "cancel")) & 0xFFFFFFFF
                        roll = float(_np_rl.random.default_rng(seed).random())
                    if roll < 0.30:
                        cancelled.append(entry["company_id"])
                        continue
                departing.append(entry["company_id"])
            else:
                remaining.append(entry)

        self.queue = remaining
        return departing, cancelled


# ─────────────────────────────────────────────────────────────────────────────
# BURDEN STOCK
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BurdenStock:
    """Regulatory burden as a proper stock with inflows and outflows.

    In v1, burden had only inflows: every policy action increased it and nothing
    reduced it. This meant burden monotonically ratcheted to 100 over any
    sufficiently long run, making all policy scenarios eventually indistinguishable.

    Real-world regulation has outflows. The EU GDPR is the best-documented case:
    compliance burden for large companies fell from approximately 60% of IT budgets
    in 2018 (the year of implementation) to approximately 20% by 2021, as firms
    completed certification, built internal tooling, and legal templates standardised
    across the industry (IAPP Annual Privacy Governance Report surveys). That is a
    ~40-point drop over ~12 rounds (3 years).

    Inflows:
      + New policy enacted (severity-weighted, see add_policy_burden())
      + Enforcement actions (each action adds compliance overhead)

    Outflows:
      - Compliance certification (as firms get certified, marginal overhead falls)
      - Policy reform / amendment (lobbying success reduces burden)
      - Regulatory exemptions (carve-outs reduce burden for in-scope firms)

    Dimensional unit: % of R&D budget devoted to regulatory compliance.
    """
    level: float = 0.0
    cumulative_inflow: float = 0.0
    cumulative_outflow: float = 0.0

    # Compliance discharge rate: fraction of burden discharged per compliant firm
    # Calibrated from GDPR: burden dropped ~40 points over 3 years as firms complied
    # With 8 rounds = 2 years, that's ~40/(8) = 5 points/round discharge at full compliance
    DISCHARGE_RATE_PER_COMPLIANT_FIRM = 0.05  # [DIRECTIONAL] magnitude assumed

    def add_policy_burden(self, severity: float) -> None:
        """Add burden from a newly enacted policy."""
        # Burden from policy = severity-weighted amount
        # Severity 5 adds 20 points; severity 3 adds 10; severity 1 adds 3
        increment = {1: 3.0, 2: 6.0, 3: 10.0, 4: 15.0, 5: 20.0}.get(
            round(severity), severity * 4.0
        )
        self.level += increment
        self.cumulative_inflow += increment
        self.level = min(100.0, self.level)

    def add_enforcement_burden(self, n_enforcement_actions: int) -> None:
        """Enforcement actions add compliance overhead."""
        increment = n_enforcement_actions * 1.5  # [ASSUMED]
        self.level += increment
        self.cumulative_inflow += increment
        self.level = min(100.0, self.level)

    def discharge_compliance(self, compliance_rate: float) -> float:
        """Discharge burden as firms complete compliance programs.

        compliance_rate: fraction of population currently in compliance [0,1].
        Returns amount discharged this round.
        """
        # Maximum discharge when 100% compliance: 5% of current burden/round
        discharge = self.level * compliance_rate * self.DISCHARGE_RATE_PER_COMPLIANT_FIRM
        discharge = min(discharge, self.level)
        self.level -= discharge
        self.cumulative_outflow += discharge
        return discharge

    def discharge_reform(self, reform_magnitude: float) -> float:
        """Discharge burden from successful lobbying / policy reform.

        reform_magnitude: 0-1 (fraction of burden to discharge).
        [DIRECTIONAL] Direction grounded (reforms reduce burden); magnitude assumed.
        """
        discharge = self.level * reform_magnitude
        self.level -= discharge
        self.cumulative_outflow += discharge
        return discharge


# ─────────────────────────────────────────────────────────────────────────────
# INNOVATION STOCK
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class InnovationCapacity:
    """Domestic innovation capacity as a stock.

    Includes innovation→investment feedback via expectation mechanism.
    High innovation → high expected future rents (Aghion-Howitt 1992) →
    attracts investment → increases R&D → sustains innovation.

    Conservation: dI/dt = R&D_investment_inflow - natural_depreciation
                          + spillover_inflows - direct_destruction_from_policy
    """
    level: float = 100.0
    expected_future_level: float = 100.0  # forward-looking (Aghion-Howitt)

    # Natural depreciation: frontier AI knowledge depreciates fast
    # (new methods obsolete old ones in 1-2 years)
    # [DIRECTIONAL] 12.5%/round depreciation → half-life ~4 rounds (1 year)
    # Roughly consistent with AI research publication pace
    DEPRECIATION_RATE = 0.125

    def update_expectation(
        self,
        current_level: float,
        investment: float,
        burden: float,
        domestic_company_fraction: float,
    ) -> None:
        """Update expected future innovation using Aghion-Howitt forward-looking logic.

        The key insight from Aghion & Howitt (1992) is that R&D investment decisions
        are forward-looking: investors commit capital today based on the expected
        value of future innovation rents, not the current innovation stock. This
        creates the reinforcing loop that was absent from v1:

          high expected future innovation
            → attracts investment today
            → R&D spending increases
            → innovation stock grows
            → validates the expectation

        And the destructive loop when regulation suppresses expectations:

          high burden reduces investment expectations
            → less investment today
            → innovation declines
            → expectations revised down further

        The adaptive blending (0.7 × old + 0.3 × new signal) prevents
        expectations from jumping discontinuously on a single round's data.
        [DIRECTIONAL] The 0.3 update weight is assumed; the forward-looking
        structure is grounded in Aghion-Howitt (1992) and subsequent growth theory.
        """
        # Expected future level: what agents believe innovation will be
        # Simple adaptive expectations with adjustment for current conditions
        potential_innovation = investment * domestic_company_fraction
        regulation_discount = max(0.0, 1.0 - burden / 150.0)
        expected_next = potential_innovation * regulation_discount

        # Adaptive: blend current expectation with new signal
        self.expected_future_level = (
            0.7 * self.expected_future_level + 0.3 * expected_next
        )

    def apply_rd_investment(self, investment: float) -> float:
        """Net innovation change from investment deviation from the steady-state baseline.

        [GROUNDED] Ugur et al. (2016) meta-analysis of 1,955 R&D-productivity
        estimates: average elasticity ε = 0.138 (own R&D → own productivity).
        Converted to per-round index-scale coefficient:
          0.138 / 4 quarters / 100 scale = 0.000345 per round per index point.

        The formula uses (investment - 50), not max(0, investment - 50), making
        the effect symmetric around the baseline:
          investment > 50 → positive delta → innovation grows
          investment = 50 → zero delta → innovation is stable (steady state)
          investment < 50 → negative delta → innovation declines

        This is the correct interpretation of the Ugur elasticity. In steady
        state, replacement investment exactly offsets depreciation, so the
        Ugur coefficient captures NET changes from the baseline investment
        level, not additive growth on top of an already-growing stock. A model
        that only adds innovation when investment is positive (and ignores the
        symmetric decline) would systematically overstate resilience under
        regulatory shocks that suppress investment.

        Under a moratorium collapsing investment to 0: delta = 0.000345 × (0 - 50)
        = -0.017/round → approximately -14% over 16 rounds. This is far more
        conservative than the previous 12.5%/round depreciation, which produced
        near-zero innovation after 8 rounds regardless of policy content.
        """
        from swarmcast.game_master.calibration import RD_TO_INNOVATION
        # Symmetric: investment above baseline grows innovation; below shrinks it
        delta = RD_TO_INNOVATION.value * (investment - 50.0)
        self.level = max(0.0, self.level + delta)
        return delta

    def apply_depreciation(self) -> float:
        """Knowledge depreciation — absorbed into the net R&D investment model.

        In the Ugur (2016) meta-analysis, the R&D-to-productivity elasticity
        (ε=0.138) measures the NET effect of investment changes on innovation.
        It already incorporates the baseline replacement rate.

        Therefore: depreciation is NOT applied separately. Innovation is stable
        at baseline investment (=50) and changes proportionally to investment
        deviations from baseline via apply_rd_investment().

        Setting depreciation=0 makes the model internally consistent with
        the Ugur coefficient. Apply only if investment falls below replacement
        threshold (modeled by negative investment→innovation coupling in
        apply_rd_investment at investment < 50).
        """
        # No separate depreciation — it is implicit in the net Ugur coupling.
        # Innovation still declines when investment << 50 via apply_rd_investment.
        return 0.0

    def apply_relocation_effect(
        self,
        n_relocated_this_round: int,
        spillover_factor: float = 0.5,
    ) -> tuple[float, float]:
        """Companies relocating reduce DOMESTIC innovation but NOT global.

        Relocating companies keep their research — it just moves.
        domestic_loss = innovation_share * (1 - spillover_factor)
        international_gain = innovation_share * spillover_factor (outside model)

        spillover_factor: fraction of innovation that stays accessible globally
        [DIRECTIONAL] 0.5 is assumed; some research (papers, open source)
        remains globally accessible, some (internal methods) is lost.
        Calibration target: OECD patent mobility data (not yet integrated).
        """
        if n_relocated_this_round == 0:
            return 0.0, 0.0

        # Each relocated company contributes ~1/100 of innovation capacity
        per_company_share = self.level / 100.0
        domestic_loss = per_company_share * n_relocated_this_round * (1.0 - spillover_factor)
        global_preservation = per_company_share * n_relocated_this_round * spillover_factor

        self.level = max(0.0, self.level - domestic_loss)
        return domestic_loss, global_preservation


# ─────────────────────────────────────────────────────────────────────────────
# GOVERNANCE STOCKS — unified container
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class GovernanceStocks:
    """All governance stocks in one place.

    This replaces the v1 economic_indicators dict with proper stock-flow
    accounting. Every quantity has inflows, outflows, and dimensional units.
    """
    companies: CompanyStock = dataclasses.field(default_factory=CompanyStock)
    burden: BurdenStock = dataclasses.field(default_factory=BurdenStock)
    innovation: InnovationCapacity = dataclasses.field(default_factory=InnovationCapacity)
    relocation_pipeline: RelocationPipeline = dataclasses.field(
        default_factory=RelocationPipeline
    )

    # Investment: modeled as a flow (influenced by stocks)
    # Not a stock itself — investment is a rate ($/year), not an accumulated quantity
    ai_investment_rate: float = 100.0  # 0-100 index (50 ↔ $50B/yr)
    public_trust: float = 60.0        # % population trusting AI is beneficial

    def compute_investment_rate(self) -> float:
        """Compute AI investment rate from current stocks — structural fix for v1 loop.

        The v1 formula was multiplicative: raw = investment × innovation_attraction.
        At innovation_attraction=0.9 (a plausible value under mild regulation),
        this decays investment by 10% per round even with no policy at all —
        producing near-zero investment after 16 rounds in every scenario. The
        loop was self-defeating: it imposed structural decay that dominated any
        policy signal.

        The corrected Aghion-Howitt structure is additive:

          investment = BASELINE × innovation_factor - burden_suppression

        BASELINE = 100 (full investment in a stable, unregulated economy)
        innovation_factor = expected_future_innovation / 100, scaled by the
          fraction of domestic companies remaining (if all companies relocated,
          there is no domestic capital to attract)
        burden_suppression = OECD_coeff × max(0, burden - 30) × 100
          (burden below 30 — mild compliance overhead — does not suppress
          investment; above 30, each additional burden point reduces investment
          by the OECD coefficient)

        Steady-state check (no regulation, innovation_expect=100, no relocation):
          investment = 100 × 1.0 - 0 = 100 (stable) ✓

        Severe regulation check (burden=90, innovation_expect=20):
          investment = 100 × 0.2 - 0.00295 × 60 × 100 = 20 - 17.7 = 2.3 ✓

        [GROUNDED] burden→investment suppression: Springler (2023) reviewing
          OECD product market regulation data, PMR elasticity ε = −0.197.
          Per-round coefficient: −0.197 / (100/6 policy-scale) / 4 quarters = −0.00295.
        [GROUNDED] innovation→investment attraction: Aghion & Howitt (1992) —
          investment responds to expected future rents proxied by innovation expectation.
        """
        from swarmcast.game_master.calibration import BURDEN_TO_INVESTMENT

        # Forward-looking innovation expectation (Aghion-Howitt 1992)
        # High expected future innovation → capital attracted to the jurisdiction
        innovation_factor = (
            self.innovation.expected_future_level / 100.0
        ) * self.companies.domestic_fraction()

        # Burden suppresses investment above mild-regulation baseline (OECD grounded)
        # Per-round coefficient: −0.00295 per index point above 30
        # Multiply by 100 to convert from proportion to index-point scale
        burden_suppression = abs(BURDEN_TO_INVESTMENT.value) * max(
            0.0, self.burden.level - 30.0
        ) * 100.0

        # Structural equation: investment = potential * attraction - suppression
        # Potential = 100 (full baseline investment in a stable economy)
        raw = 100.0 * innovation_factor - burden_suppression
        self.ai_investment_rate = max(0.0, min(100.0, raw))
        return self.ai_investment_rate

    def to_indicators_dict(self) -> dict[str, float]:
        """Export as v1-compatible economic_indicators dict.

        Allows hybrid simulation to report in the same format as v1.
        """
        return {
            "ai_investment_index": self.ai_investment_rate,
            "innovation_rate": self.innovation.level,
            "public_trust": self.public_trust,
            "regulatory_burden": self.burden.level,
            "market_concentration": max(0.0, min(100.0,
                50.0 + (100 - self.companies.domestic_fraction() * 100) * 0.5
            )),
            "domestic_companies": float(self.companies.domestic_count()),
            "relocation_pipeline_size": float(self.companies.in_pipeline),
            "expected_future_innovation": self.innovation.expected_future_level,
        }
