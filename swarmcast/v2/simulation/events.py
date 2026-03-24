"""Policy event injection system.

Real governance does not run on a fixed track for 16 rounds without anything
happening. Policies get amended, incidents occur, enforcement escalates, and
companies successfully lobby for exemptions. This module provides:

  PolicyEvent        — a named event pairing a trigger with an effect
  EventQueue         — collection of pending events; processes all triggers each round
  Trigger classes    — when the event fires (RoundTrigger, ThresholdTrigger,
                       ProbabilisticTrigger)
  Effect classes     — what changes when the event fires (burden, trust,
                       investment, enforcement, destination burden, LLM proposals)

SUPPORTED EVENT TYPES
─────────────────────
  policy_amendment    — modify the active policy (relax/tighten requirements)
  enforcement_surge   — regulator escalates enforcement capacity
  trust_shock         — public incident changes trust sharply
  investment_surge    — breakthrough attracts capital
  exemption_granted   — specific company type gets carved out
  compliance_deadline — grace period extension or contraction
  jurisdiction_policy — destination jurisdiction changes its rules
  llm_proposal        — LLM strategic agent successfully proposes an amendment

USAGE
─────
  queue = EventQueue()
  queue.add(PolicyEvent(
      name="EU AI Act Amendment",
      trigger=RoundTrigger(round=6),
      effect=PolicyAmendmentEffect(burden_delta=-15, description="SME exemptions added"),
  ))

  # In simulation loop:
  fired = queue.process(round_num, current_stocks, current_pop_summary)
  for event in fired:
      event.apply(stocks, policy, jurisdictions)

ENSEMBLE RUNS
─────────────
  Always call EventQueue.deep_copy() before starting each ensemble run.
  fired=True flags on events and already_fired/n_fired state on triggers
  persist on the original queue object. Reusing the same queue across runs
  means events that fired in run 1 silently skip in run 2.
"""

from __future__ import annotations

import dataclasses
import heapq
from typing import Callable, Any


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class RoundTrigger:
    """Fire exactly once at a specific simulation round number.

    Usage: RoundTrigger(round=6) fires when round_num == 6 and never again.
    Suitable for scheduled policy events with known implementation dates,
    such as the EU AI Act's fixed compliance deadlines.
    """
    round: int
    def should_fire(self, round_num: int, stocks: Any, pop: Any, rng=None) -> bool:
        return round_num == self.round


@dataclasses.dataclass
class ThresholdTrigger:
    """Fire once when a named indicator crosses a threshold in the specified direction.

    Usage: ThresholdTrigger(indicator="public_trust", threshold=30.0, direction="below")
    fires the first time public_trust drops to or below 30, then sets already_fired=True
    so it never fires again. Suitable for reactive policy — enforcement surges when
    trust collapses, or market interventions when relocation_rate exceeds tolerance.

    The indicator is looked up first in stocks.to_indicators_dict(), then in the
    pop_summary dict if not found there. Returns False silently if the indicator
    is absent in both sources.

    GOTCHA: already_fired persists on the object across rounds. EventQueue.deep_copy()
    resets it between ensemble runs; without deep_copy(), the trigger is permanently
    exhausted after the first run.

    Note: already_fired is instance state. EventQueue.deep_copy() must be called
    before each ensemble run to reset fired flags.
    """
    indicator: str       # e.g. "regulatory_burden", "relocation_rate"
    threshold: float
    direction: str = "above"  # "above" or "below"
    already_fired: bool = False

    def should_fire(self, round_num: int, stocks: Any, pop: Any, rng=None) -> bool:
        if self.already_fired:
            return False
        val = None
        if hasattr(stocks, "to_indicators_dict"):
            val = stocks.to_indicators_dict().get(self.indicator)
        if val is None and pop is not None:
            val = pop.get(self.indicator)
        if val is None:
            return False
        if self.direction == "above" and val >= self.threshold:
            self.already_fired = True
            return True
        if self.direction == "below" and val <= self.threshold:
            self.already_fired = True
            return True
        return False


@dataclasses.dataclass
class ProbabilisticTrigger:
    """Fire with probability p each round, up to max_fires times total.

    Usage: ProbabilisticTrigger(probability=0.15, min_round=3, max_fires=1)
    models a random AI safety incident that has a 15% chance of occurring
    each round from round 3 onward, firing at most once. Suitable for
    stochastic shocks (incidents, breakthroughs) whose timing is uncertain
    but whose frequency can be calibrated against historical base rates.

    GOTCHA: n_fired persists on the object. EventQueue.deep_copy() resets it
    to 0 between ensemble runs so each run gets an independent draw.
    """
    probability: float
    min_round: int = 1    # don't fire before this round
    max_fires: int = 1    # fire at most this many times total
    n_fired: int = 0

    def should_fire(self, round_num: int, stocks: Any, pop: Any, rng=None) -> bool:
        if round_num < self.min_round or self.n_fired >= self.max_fires:
            return False
        if rng is not None:
            # Caller provides a seeded RNG that includes the run seed — use it
            # directly so each ensemble member gets distinct outcomes.
            roll = rng.random()
        else:
            # Standalone call (tests, pre-loop code): derive a deterministic
            # seed from the trigger's own parameters so the same (round_num,
            # probability, min_round) always gives the same result — no
            # global random state bleed across test runs or ensemble members.
            import numpy as _np_pt
            seed = hash((round_num, self.probability, self.min_round)) & 0xFFFFFFFF
            roll = _np_pt.random.default_rng(seed).random()
        if roll < self.probability:
            self.n_fired += 1
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# EVENT EFFECTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PolicyAmendmentEffect:
    """Discharge or increase regulatory burden when a policy is amended.

    Usage: PolicyAmendmentEffect(burden_delta=-12, description="SME exemptions added")
    reduces BurdenStock by calling discharge_reform() with the fractional reduction;
    a positive burden_delta adds directly to stocks.burden.level.

    Compliance certification triggers burden reduction via discharge_compliance(). Without this discharge, burden would only increase over time.
    Real legislative processes include
    both tightening (positive burden_delta) and reform/discharge events (negative
    burden_delta). Setting burden_delta negative models a reform — SME exemptions,
    grace-period extensions, or burden-discharge provisions. The absolute value is
    passed to discharge_reform() as a fraction of 100 so the BurdenStock can apply
    its own discharge logic (e.g., proportional rather than absolute reduction).

    affects_types limits which agent types observe the amendment in their memory;
    None means the change applies to all agents' operating environment.
    """
    burden_delta: float = 0.0      # negative = relaxation, positive = tightening
    description: str = ""
    affects_types: list[str] | None = None  # None = all types affected

    def apply(self, stocks: Any, policy: Any, jurisdictions: Any) -> str:
        if stocks is not None and hasattr(stocks, "burden"):
            stocks.burden.discharge_reform(abs(self.burden_delta) / 100.0
                                            if self.burden_delta < 0 else 0.0)
            if self.burden_delta > 0:
                stocks.burden.level = min(100.0, stocks.burden.level + self.burden_delta)
        return f"Policy amendment: {self.description} (burden Δ{self.burden_delta:+.0f})"


@dataclasses.dataclass
class TrustShockEffect:
    """Apply a sudden change to the public_trust stock.

    Usage: TrustShockEffect(delta=-25, description="Major AI system failure")
    drops public_trust by 25 points (clamped to [0, 100]).

    Positive delta models trust-building events (transparency announcements,
    successful audits); negative delta models trust collapses (safety incidents,
    enforcement failures). The change is instantaneous and permanent within the
    run — there is no built-in trust recovery; use a subsequent positive-delta
    event to model gradual restoration.
    """
    delta: float     # positive = trust increase, negative = collapse
    description: str = ""

    def apply(self, stocks: Any, policy: Any, jurisdictions: Any) -> str:
        if stocks is not None:
            stocks.public_trust = max(0.0, min(100.0, stocks.public_trust + self.delta))
        return f"Trust shock: {self.description} (trust Δ{self.delta:+.0f})"


@dataclasses.dataclass
class InvestmentSurgeEffect:
    """Apply an exogenous increase to the ai_investment_rate stock.

    Usage: InvestmentSurgeEffect(delta=15.0, description="Government AI subsidy")
    raises ai_investment_rate by 15 points (clamped to 100).

    Models capital inflows from technology breakthroughs, government subsidies,
    or investor sentiment shifts. Affects the investment stock immediately;
    downstream effects on compliance capacity and relocation probability are
    mediated by the population response functions in subsequent rounds.
    """
    delta: float
    description: str = ""

    def apply(self, stocks: Any, policy: Any, jurisdictions: Any) -> str:
        if stocks is not None:
            stocks.ai_investment_rate = min(100.0, stocks.ai_investment_rate + self.delta)
        return f"Investment event: {self.description} (investment Δ{self.delta:+.0f})"


@dataclasses.dataclass
class EnforcementSurgeEffect:
    """Add enforcement burden via BurdenStock.add_enforcement_burden().

    Usage: EnforcementSurgeEffect(enforcement_actions=3, description="Emergency response")
    calls stocks.burden.add_enforcement_burden(3), which adds 3 enforcement actions'
    worth of burden to BurdenStock. This increases the regulatory pressure on
    all non-compliant agents in subsequent rounds.

    Models regulator capacity surges following public trust collapses or political
    mandates. Typically paired with a ThresholdTrigger on public_trust in the
    make_ai_incident_scenario() preset.
    """
    enforcement_actions: int = 1
    description: str = ""

    def apply(self, stocks: Any, policy: Any, jurisdictions: Any) -> str:
        if stocks is not None and hasattr(stocks, "burden"):
            stocks.burden.add_enforcement_burden(self.enforcement_actions)
        return f"Enforcement surge: {self.description} ({self.enforcement_actions} new actions)"


@dataclasses.dataclass
class DestinationPolicyChangeEffect:
    """Adjust the burden of a named destination jurisdiction.

    Usage: DestinationPolicyChangeEffect(jurisdiction_name="Singapore", burden_delta=+20,
    description="Singapore enacts mandatory safety assessments") raises Singapore's
    burden by 20 points in jurisdictions.destinations, making it less attractive
    as a relocation target in subsequent rounds.

    Searches jurisdictions.destinations by name; no-op if the name is not found.
    burden_delta can be positive (tightening) or negative (loosening). This models
    the strategic interdependence between jurisdictions: as firms relocate to
    Singapore, Singapore may respond by tightening its own rules, reducing the
    incentive for further relocation.
    """
    jurisdiction_name: str
    burden_delta: float
    description: str = ""

    def apply(self, stocks: Any, policy: Any, jurisdictions: Any) -> str:
        if jurisdictions is not None:
            for jur in jurisdictions.destinations:
                if jur.name == self.jurisdiction_name:
                    jur.burden = max(0.0, min(100.0, jur.burden + self.burden_delta))
        return (f"Jurisdiction change: {self.jurisdiction_name} "
                f"burden Δ{self.burden_delta:+.0f}. {self.description}")


@dataclasses.dataclass
class LLMProposalEffect:
    """An LLM strategic agent successfully lobbied for an amendment.

    Usage: LLMProposalEffect(proposer="BigTech LLM", amendment_text="Narrow scope of
    high-risk definition", burden_delta=-8, trust_delta=2.0) — the named LLM agent's
    proposal is accepted, reducing burden by 8 points and slightly increasing trust.

    This effect closes the LLM-to-population feedback loop. In the hybrid simulation,
    LLM agents make strategic choices (lobby, propose, comply) each round. When an LLM
    agent's proposal succeeds, wrapping it in an LLMProposalEffect and scheduling it
    as a PolicyEvent ensures the proposal actually modifies BurdenStock. The population
    array observes the updated burden in subsequent rounds via its memory window, so
    agents' compliance and relocation decisions respond to the LLM agent's intervention.
    Without this mechanism, LLM proposals would be purely rhetorical — they would be
    logged but would not affect population behavior.

    burden_delta < 0 calls discharge_reform() (reform/relaxation).
    burden_delta > 0 adds directly to stocks.burden.level (tightening).
    trust_delta modifies stocks.public_trust (can be positive or negative).
    """
    proposer: str           # which LLM agent proposed this
    amendment_text: str     # human-readable description
    burden_delta: float     # mechanical effect
    trust_delta: float = 0.0

    def apply(self, stocks: Any, policy: Any, jurisdictions: Any) -> str:
        if stocks is not None:
            if self.burden_delta < 0 and hasattr(stocks, "burden"):
                stocks.burden.discharge_reform(abs(self.burden_delta) / 100.0)
            elif self.burden_delta > 0 and hasattr(stocks, "burden"):
                stocks.burden.level = min(100.0, stocks.burden.level + self.burden_delta)
            if self.trust_delta != 0:
                stocks.public_trust = max(0.0, min(100.0,
                    stocks.public_trust + self.trust_delta))
        return (f"[LLM PROPOSAL by {self.proposer}] {self.amendment_text} "
                f"(burden Δ{self.burden_delta:+.0f}, trust Δ{self.trust_delta:+.0f})")


# ─────────────────────────────────────────────────────────────────────────────
# POLICY EVENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PolicyEvent:
    """A scheduled or triggered event that modifies simulation state."""
    name: str
    trigger: RoundTrigger | ThresholdTrigger | ProbabilisticTrigger
    effect: Any   # one of the *Effect classes above
    fired: bool = False
    fired_round: int | None = None

    def check_and_fire(
        self,
        round_num: int,
        stocks: Any,
        pop_summary: Any,
        policy: Any = None,
        jurisdictions: Any = None,
        rng=None,
    ) -> str | None:
        """Check trigger; if fires, apply effect and return description."""
        if self.fired:
            return None
        if self.trigger.should_fire(round_num, stocks, pop_summary, rng=rng):
            self.fired = True
            self.fired_round = round_num
            try:
                return self.effect.apply(stocks, policy, jurisdictions)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"PolicyEvent '{self.name}' fired at round {round_num} but "
                    f"effect.apply() raised {type(e).__name__}: {e}. "
                    f"Event is marked fired to prevent double-fire; stocks may be partially updated.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return f"[ERROR] {self.name}: effect failed ({e})"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EVENT QUEUE
# ─────────────────────────────────────────────────────────────────────────────

class EventQueue:
    """Manages all pending policy events for a simulation run."""

    def __init__(self):
        self.events: list[PolicyEvent] = []
        self.fired_events: list[dict] = []

    def add(self, event: PolicyEvent) -> None:
        self.events.append(event)

    def deep_copy(self) -> "EventQueue":
        """Return a fresh queue with all events and trigger state reset to unfired.

        CRITICAL for ensemble runs. PolicyEvent.fired and PolicyEvent.fired_round
        are instance state that persists across calls. If the same EventQueue is
        reused across runs without deep_copy(), any event that fired in run 1
        (fired=True) will pass the early-return check in check_and_fire() and
        silently skip in every subsequent run — no error, no warning, just missing
        events. The same applies to ThresholdTrigger.already_fired and
        ProbabilisticTrigger.n_fired.

        Returns a new EventQueue with deep copies of all events, with fired=False,
        fired_round=None, already_fired=False, and n_fired=0 reset on every event
        and trigger. The fired_events log on the new queue starts empty.

        Always call this method before launching each ensemble run:
          template_queue = make_eu_ai_act_amendment_scenario()
          for seed in seeds:
              run_queue = template_queue.deep_copy()
              run_simulation(run_queue, seed=seed)
        """
        import copy
        new_q = EventQueue()
        for event in self.events:
            new_event = copy.deepcopy(event)
            new_event.fired = False
            new_event.fired_round = None
            # Reset already_fired on threshold triggers
            if hasattr(new_event.trigger, "already_fired"):
                new_event.trigger.already_fired = False
            if hasattr(new_event.trigger, "n_fired"):
                new_event.trigger.n_fired = 0
            new_q.events.append(new_event)
        return new_q

    def process(
        self,
        round_num: int,
        stocks: Any,
        pop_summary: dict,
        policy: Any = None,
        jurisdictions: Any = None,
        rng=None,
    ) -> list[str]:
        """Check all events; fire those whose trigger conditions are met.

        Returns list of human-readable descriptions of fired events.
        """
        fired_descriptions = []
        for event in self.events:
            desc = event.check_and_fire(
                round_num, stocks, pop_summary, policy, jurisdictions, rng=rng
            )
            if desc is not None:
                fired_descriptions.append(desc)
                self.fired_events.append({
                    "round": round_num,
                    "name": event.name,
                    "description": desc,
                })
        return fired_descriptions

    def summary(self) -> str:
        if not self.fired_events:
            return "No events fired."
        lines = ["EVENTS:"]
        for e in self.fired_events:
            lines.append(f"  R{e['round']:02d}: [{e['name']}] {e['description']}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PRESET EVENT SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

def make_eu_ai_act_amendment_scenario() -> EventQueue:
    """Preset: EU AI Act amendment cycle with SME exemptions and enforcement budget cut.

    The EU AI Act was amended 47 times in committee before passage. This scenario
    models two of the most consequential post-passage amendments:

    Round 6 (18 months): SME Exemption Package
      Small enterprises (<10 employees) are carved out of the high-risk provisions.
      burden_delta=-12 via PolicyAmendmentEffect — a moderate discharge that reduces
      burden accumulated in the first 18 months and slows relocation among startups.

    Round 10 (30 months): Enforcement Budget Cut
      Political pushback leads to reduced regulator capacity. Modeled as a trust
      shock (delta=-8) rather than a direct burden change, because the primary
      observable effect was declining compliance confidence, not formal rule changes.

    Returns an unstarted EventQueue. Call deep_copy() before each ensemble run.
    """
    q = EventQueue()
    q.add(PolicyEvent(
        name="SME Exemption Package",
        trigger=RoundTrigger(round=6),
        effect=PolicyAmendmentEffect(
            burden_delta=-12,
            description="SMEs with <10 employees exempt from high-risk provisions"
        ),
    ))
    q.add(PolicyEvent(
        name="Enforcement Budget Cut",
        trigger=RoundTrigger(round=10),
        effect=TrustShockEffect(
            delta=-8,
            description="Budget cuts reduce regulator capacity; compliance confidence drops"
        ),
    ))
    return q


def make_ai_incident_scenario() -> EventQueue:
    """Preset: stochastic AI safety incident followed by emergency enforcement surge.

    Models the regulatory dynamics triggered by a major AI system failure:

    AI Safety Incident (probabilistic, p=0.15, from round 3):
      Each round from round 3 onward there is a 15% chance of a major incident.
      When it fires (at most once), public_trust drops by 25 points via TrustShockEffect.
      The 15% per-round rate corresponds to roughly a 70% chance of at least one
      incident occurring over an 8-round run — consistent with historical AI incident
      frequency for frontier models in production.

    Emergency Enforcement Surge (threshold-triggered):
      Fires once when public_trust falls below 30. Adds 3 enforcement actions via
      EnforcementSurgeEffect, representing emergency regulator capacity activation.
      The threshold coupling between these two events models the reactive policy
      dynamic: incidents cause trust collapse, which crosses the enforcement threshold,
      which increases burden, which may cause further relocation.

    Returns an unstarted EventQueue. Call deep_copy() before each ensemble run.
    """
    q = EventQueue()
    q.add(PolicyEvent(
        name="AI Safety Incident",
        trigger=ProbabilisticTrigger(probability=0.15, min_round=3, max_fires=1),
        effect=TrustShockEffect(
            delta=-25,
            description="Major AI system failure causes public trust collapse"
        ),
    ))
    q.add(PolicyEvent(
        name="Emergency Enforcement Surge",
        trigger=ThresholdTrigger(
            indicator="public_trust", threshold=30.0, direction="below"
        ),
        effect=EnforcementSurgeEffect(
            enforcement_actions=3,
            description="Government responds to trust collapse with enforcement surge"
        ),
    ))
    return q


def make_destination_tightening_scenario() -> EventQueue:
    """Preset: Singapore tightens AI regulation at round 8 in response to inbound firms.

    Models the strategic endogeneity of relocation destinations. As firms relocate
    to Singapore to escape EU AI Act burden, Singapore observes the inflow and
    responds by enacting its own mandatory safety assessment framework.

    Round 8 (24 months): Singapore AI Act
      DestinationPolicyChangeEffect raises Singapore's burden by +20, reducing
      its net attractiveness to firms that have not yet relocated. Firms that
      already relocated face higher ongoing compliance costs in their new home.

    This scenario tests whether the model's relocation dynamics are sensitive to
    destination-side feedback — a question relevant to regulatory arbitrage and
    the race-to-the-bottom literature. Comparing runs with and without this
    scenario shows how destination-jurisdiction response dampens relocation.

    Returns an unstarted EventQueue. Call deep_copy() before each ensemble run.
    """
    q = EventQueue()
    q.add(PolicyEvent(
        name="Singapore AI Act",
        trigger=RoundTrigger(round=8),
        effect=DestinationPolicyChangeEffect(
            jurisdiction_name="Singapore",
            burden_delta=+20,
            description="Singapore enacts mandatory safety assessments for frontier AI"
        ),
    ))
    return q
