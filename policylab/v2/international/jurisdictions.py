"""Multi-jurisdiction relocation model.

Companies that decide to relocate are routed to a destination jurisdiction via
softmax discrete choice over regulatory burden and tax environment. Each
jurisdiction tracks its company count; high inflows increase burden logarithmically
to model congestion effects.

Relocating companies retain their R&D capacity — innovation moves with them rather
than disappearing from the model.
"""

from __future__ import annotations

import dataclasses
import numpy as np
from typing import Literal


RegStance = Literal["permissive", "moderate", "strict"]


@dataclasses.dataclass
class Jurisdiction:
    """A regulatory jurisdiction that companies can relocate to or from."""
    name: str
    burden: float           # 0-100, regulatory compliance overhead
    corporate_tax: float    # 0-100, effective corporate tax rate
    ai_stance: RegStance    # regulatory philosophy
    company_count: int      # current number of AI companies
    innovation_subsidy: float = 0.0  # active R&D subsidies (positive = attractive)

    # History
    burden_history: list = dataclasses.field(default_factory=list)
    company_history: list = dataclasses.field(default_factory=list)

    def net_cost(self, burden_weight: float = 0.7, tax_weight: float = 0.3) -> float:
        """Effective cost of operating in this jurisdiction."""
        return (
            self.burden * burden_weight
            + self.corporate_tax * tax_weight
            - self.innovation_subsidy
        )

    def attractiveness(self) -> float:
        """0-100 score — higher = more attractive for AI companies."""
        return max(0.0, 100.0 - self.net_cost())

    def record_round(self) -> None:
        self.burden_history.append(self.burden)
        self.company_history.append(self.company_count)

    def update_burden_on_arrival(self, n_arriving: int) -> None:
        """Burden increases as AI companies arrive, but with diminishing returns.

        The previous flat +0.3/company formula caused burden overflow: in a
        mass-exodus scenario 2,000 companies × 0.3 = 600 burden points, pushing
        every destination to burden=100 within a few rounds and making all
        jurisdictions equally unattractive. The model then produced pathological
        no-destination equilibria regardless of policy content.

        The corrected formula reflects that administrative capacity grows slowly.
        Beyond the first wave of arrivals, each additional company requires
        proportionally less new regulatory effort — courts, templates, and
        precedents already exist. Logarithmic saturation captures this:

          Δburden = base_sensitivity × log(1 + n_arriving / regulatory_capacity)

        regulatory_capacity is set per jurisdiction to reflect bureaucratic scale:
        EU/US can absorb more AI companies before their regulators are stretched;
        Singapore/UAE have smaller civil services and saturate sooner.

        [DIRECTIONAL] Logarithmic saturation grounded in administrative capacity
        literature (Wilson 1989, Bureaucracy). Magnitude: [ASSUMED] see sweep range.

        Calibration target: UAE absorbing 500 AI companies over 1 year should
        produce a moderate burden increase (~+10-15), not +150.
        With capacity=50: log(1 + 500/50) = log(11) ≈ 2.4 → 5.0 × 2.4 = 12.0 ✓
        """
        import math
        # Regulatory capacity by jurisdiction type (in terms of AI companies)
        CAPACITY = {
            "EU": 500, "US": 400, "UK": 200, "Singapore": 80, "UAE": 50,
        }
        capacity = CAPACITY.get(self.name, 100)
        # base_sensitivity: how much each doubling of companies stresses the regulator
        # [ASSUMED] 5.0 → doubling the company count adds ~3.5 burden points
        base_sensitivity = 5.0  # [ASSUMED] sweep [2.0, 5.0, 10.0]
        delta = base_sensitivity * math.log(1.0 + n_arriving / max(1.0, capacity))
        self.burden = min(100.0, self.burden + delta)


# ─────────────────────────────────────────────────────────────────────────────
# PRESET JURISDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_eu() -> Jurisdiction:
    """EU — strict/high-burden; approximates the post-AI-Act regime.

    burden=55 reflects the cumulative compliance overhead of GDPR + AI Act
    for high-risk AI systems. corporate_tax=25 approximates the EU average
    effective rate post-BEPS pillar-two. No innovation subsidy: the EU's
    Horizon programme is not modelled as a direct per-company subsidy here.
    [ASSUMED] Initial company_count=100 (normalized reference jurisdiction).
    """
    return Jurisdiction(
        name="EU", burden=55.0, corporate_tax=25.0,
        ai_stance="strict", company_count=100,
    )

def make_us() -> Jurisdiction:
    """US — moderate/lower-burden; approximates the pre-2024 US federal stance.

    burden=25 reflects lighter federal AI regulation with no comprehensive
    AI Act equivalent. corporate_tax=21 is the statutory rate post-TCJA.
    innovation_subsidy=5 represents IRA-era R&D tax credits and CHIPS Act
    funding flowing to AI-adjacent compute infrastructure. [ASSUMED] magnitudes.
    """
    return Jurisdiction(
        name="US", burden=25.0, corporate_tax=21.0,
        ai_stance="moderate", company_count=150,
        innovation_subsidy=5.0,
    )

def make_uk() -> Jurisdiction:
    """UK — moderate/light-touch; approximates the post-Brexit "pro-innovation" stance.

    burden=20 reflects the DSIT/DSIT white paper approach: principles-based
    rather than rules-based AI governance. corporate_tax=19 was the rate before
    the 2023 increase to 25%; set conservatively. innovation_subsidy=3 models
    UKRI AI programmes. [ASSUMED] all magnitudes.
    """
    return Jurisdiction(
        name="UK", burden=20.0, corporate_tax=19.0,
        ai_stance="moderate", company_count=40,
        innovation_subsidy=3.0,
    )

def make_singapore() -> Jurisdiction:
    """Singapore — permissive/pro-innovation; MAS/IMDA AI governance sandbox model.

    burden=10 reflects Singapore's Model AI Governance Framework (2020):
    voluntary, principles-based, with no mandatory compliance burden.
    corporate_tax=17 is the standard rate. innovation_subsidy=8 models
    the National AI Strategy 2.0 targeted grants. Small initial company_count
    reflects realistic market size, not regulatory quality. [ASSUMED] magnitudes.
    """
    return Jurisdiction(
        name="Singapore", burden=10.0, corporate_tax=17.0,
        ai_stance="permissive", company_count=20,
        innovation_subsidy=8.0,
    )

def make_uae() -> Jurisdiction:
    """UAE — permissive/minimal-tax; approximates the ADGM/Dubai DIFC free-zone model.

    burden=5 is the lowest preset: the UAE has no federal AI regulation and
    ADGM/DIFC operate as independent financial zones with light-touch oversight.
    corporate_tax=9 reflects the UAE's 2023 corporate tax introduction (from 0);
    many free-zone companies remain exempt. innovation_subsidy=10 models the
    UAE AI 100 programme and Abu Dhabi Falcon fund. [ASSUMED] magnitudes.
    """
    return Jurisdiction(
        name="UAE", burden=5.0, corporate_tax=9.0,
        ai_stance="permissive", company_count=10,
        innovation_subsidy=10.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RELOCATION ROUTER — softmax over destination attractiveness
# ─────────────────────────────────────────────────────────────────────────────

def route_relocating_companies(
    n_relocating: int,
    source: Jurisdiction,
    destinations: list[Jurisdiction],
    temperature: float = 0.1,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """Distribute n_relocating companies across destination jurisdictions.

    Uses softmax (multinomial logit) over attractiveness scores — the standard
    discrete choice model for agents selecting from a menu of alternatives.
    Companies prefer cheaper/less-burdened destinations, but the softmax
    temperature introduces dispersion that models real-world friction:
    information asymmetry, pre-existing legal relationships, language barriers,
    and idiosyncratic preferences.

    temperature=0.1 is intentionally low (near-deterministic): most companies
    will concentrate in the single cheapest option rather than spreading evenly.
    This matches the observed pattern in EU AI Act: the UK and Singapore attracted
    disproportionately more relocations than a uniform model would predict.
    A higher temperature (e.g., 0.5) would produce a more uniform spread.

    [DIRECTIONAL] Softmax (multinomial logit) is a standard discrete-choice model
    and the direction (lower-burden destinations attract more companies) is grounded.
    Temperature=0.1 is [ASSUMED]. Sweep [0.05, 0.10, 0.20].

    Returns dict: {jurisdiction_name: n_companies_arriving}
    """
    if n_relocating == 0 or not destinations:
        return {}

    if rng is None:
        rng = np.random.default_rng(42)

    # Softmax over attractiveness
    attractiveness = np.array([d.attractiveness() for d in destinations], dtype=float)
    logits = attractiveness / (temperature * 100.0 + 1e-9)
    logits -= logits.max()  # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()

    # Multinomial draw
    counts = rng.multinomial(n_relocating, probs)

    # Update destination company counts
    result = {}
    for d, count in zip(destinations, counts):
        if count > 0:
            d.company_count += count
            d.update_burden_on_arrival(count)
            result[d.name] = int(count)
            # Source loses companies
            source.company_count -= count

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-JURISDICTION STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MultiJurisdictionState:
    """Tracks all jurisdictions and their evolution over time."""
    source: Jurisdiction
    destinations: list[Jurisdiction]
    relocation_log: list[dict] = dataclasses.field(default_factory=list)

    def process_relocations(
        self,
        n_relocating: int,
        round_num: int,
        rng: np.random.Generator | None = None,
        temperature: float = 0.1,
    ) -> dict[str, int]:
        """Route companies to destinations and log the flows."""
        flows = route_relocating_companies(
            n_relocating, self.source, self.destinations, temperature, rng
        )
        if flows:
            self.relocation_log.append({
                "round": round_num,
                "total_leaving": n_relocating,
                "flows": flows,
            })
        return flows

    def record_round(self) -> None:
        self.source.record_round()
        for d in self.destinations:
            d.record_round()

    def destination_summary(self) -> dict:
        """Summary of where companies ended up."""
        return {
            d.name: {
                "company_count": d.company_count,
                "burden": d.burden,
                "attractiveness": d.attractiveness(),
            }
            for d in self.destinations
        }

    def source_fraction_remaining(self) -> float:
        """Fraction of original company count still in source jurisdiction."""
        original = 100  # normalized
        return max(0.0, self.source.company_count / original)

    def global_innovation_report(
        self,
        domestic_innovation: float,
        spillover_factor: float = 0.5,
    ) -> dict:
        """Report domestic vs global innovation, accounting for relocated companies.

        Relocated companies keep their research capacity — it moves jurisdictions,
        not disappears — their R&D capacity moves with them. This method partitions total
        innovation into what remains domestically attributable and what has moved
        to destination jurisdictions.

        spillover_factor controls how much of the relocated research remains
        globally accessible to the source jurisdiction (via publications, open
        source, patent licensing, researcher mobility). At spillover_factor=0.5:
          - half the relocated innovation is permanently lost to the source
          - half remains accessible globally (counted as destination innovation)

        domestic_innovation: the GovernanceStocks innovation level for the source.
        dest_innovation: relocated_fraction × 100 × spillover_factor.
        global_total: sum of both — the globally accessible frontier.

        [ASSUMED] spillover_factor=0.5 default. Calibration target:
        OECD patent mobility data (not yet integrated). Sweep [0.3, 0.5, 0.7].
        """
        relocated_frac = 1.0 - self.source_fraction_remaining()
        domestic = domestic_innovation
        # Innovation in destination jurisdictions (partially accessible globally)
        dest_innovation = relocated_frac * 100.0 * spillover_factor
        global_total = domestic + dest_innovation

        return {
            "domestic_innovation": domestic,
            "destination_innovation": dest_innovation,
            "global_innovation": global_total,
            "domestic_fraction": domestic / max(1.0, global_total),
            "top_destination": max(
                self.destinations, key=lambda d: d.company_count
            ).name if self.destinations else "none",
        }

    def summary(self) -> str:
        lines = [f"\nJURISDICTION FLOWS (source: {self.source.name})"]
        lines.append(f"  Source remaining: {self.source.company_count}/100 companies")
        for d in sorted(self.destinations, key=lambda x: -x.company_count):
            base = d.company_history[0] if d.company_history else d.company_count
            arrived = d.company_count - base
            if arrived > 0:
                lines.append(
                    f"  → {d.name:<12} +{arrived:3d} companies  "
                    f"burden={d.burden:.0f}  attract={d.attractiveness():.0f}"
                )
        if self.relocation_log:
            total = sum(e["total_leaving"] for e in self.relocation_log)
            lines.append(f"  Total relocated: {total} companies over {len(self.relocation_log)} rounds")
        return "\n".join(lines)
