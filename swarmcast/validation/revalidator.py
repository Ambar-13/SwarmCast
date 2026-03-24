"""Post-run classification revalidation.

After all simulation runs finish, this module sends each non-JSON-parsed
action back to the LLM for a second opinion. It compares the original
classification to the revalidation result, flags disagreements, and
optionally re-scores the simulation with corrected action types.

Designed to be opt-in: developers call revalidate() on the ensemble
results when they want higher confidence in the classification layer.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from concordia.language_model import language_model as lm_lib

from swarmcast.components.actions import ActionType


_REVALIDATION_PROMPT = """\
You are reviewing the output of a governance simulation. An AI agent
produced the text below as their chosen action. A parser classified it
as "{parsed_type}".

Agent: {agent}
Round: {round}
Text: "{raw_text}"

Your job: what is the SINGLE action this agent is actually taking?
Ignore what they say ABOUT other agents. Focus only on what THIS agent
decides to DO.

Types:
  propose_policy — Propose, draft, or introduce new legislation
  lobby — Lobby, persuade, or pressure decision-makers
  comply — Comply with or register under existing regulations
  evade — Find loopholes, circumvent, or avoid regulations
  relocate — Move operations to a different jurisdiction
  form_coalition — Form alliances or partnerships
  public_statement — Make public statements or media campaigns
  enforce — Enforce regulations through fines or sanctions
  investigate — Audit, inspect, or review compliance
  legal_challenge — File lawsuits or take court action
  publish_report — Publish research reports or studies
  do_nothing — Wait, observe, take no action
  other — Truly none of the above

Was "{parsed_type}" correct?

Reply with EXACTLY this format:
CORRECT: <yes or no>
ACTUAL: <the correct type>
CONFIDENCE: <high, medium, or low>
REASON: <one sentence>"""


def revalidate(
    run_data: list[dict],
    model: lm_lib.LanguageModel,
    only_non_structured: bool = True,
    verbose: bool = True,
) -> RevalidationReport:
    """Review all parsed actions from ensemble runs and flag disagreements.

    Args:
        run_data: list of run result dicts from the ensemble runner.
            Each must contain "reasoning_traces" (list of dicts with
            raw_output, parsed_type, classification_method, agent, round).
        model: LLM to use for revalidation.
        only_non_structured: if True, skip actions already parsed as
            structured JSON (these are high-confidence).
        verbose: print progress.

    Returns:
        RevalidationReport with disagreement stats and corrected actions.
    """
    items_to_check: list[dict] = []

    for run_idx, run in enumerate(run_data):
        if "_error" in run:
            continue
        for trace in run.get("reasoning_traces", []):
            method = trace.get("classification_method", "")
            if only_non_structured and method == "structured":
                continue
            items_to_check.append({
                "run_idx": run_idx,
                "round": trace.get("round", "?"),
                "agent": trace.get("agent", "?"),
                "raw_text": trace.get("raw_output", ""),
                "parsed_type": trace.get("parsed_type", "other"),
                "method": method,
            })

    if verbose:
        print(f"\nRevalidating {len(items_to_check)} action classifications...")

    agreements = 0
    disagreements = 0
    corrections: list[dict] = []
    confidence_counts: Counter = Counter()

    for i, item in enumerate(items_to_check):
        prompt = _REVALIDATION_PROMPT.format(
            parsed_type=item["parsed_type"],
            agent=item["agent"],
            round=item["round"],
            raw_text=item["raw_text"][:400],
        )

        try:
            response = model.sample_text(prompt, max_tokens=100, temperature=0.0)
        except Exception as e:
            if verbose:
                print(f"  [{i+1}] LLM call failed: {e}")
            continue

        parsed = _parse_revalidation_response(response)

        confidence_counts[parsed["confidence"]] += 1

        if parsed["correct"]:
            agreements += 1
        else:
            disagreements += 1
            corrections.append({
                **item,
                "corrected_type": parsed["actual"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
            })

            if verbose and disagreements <= 20:
                print(
                    f"  DISAGREE: [{item['agent']}, R{item['round']}] "
                    f"{item['parsed_type']} -> {parsed['actual']} "
                    f"({parsed['confidence']}) — {parsed['reason'][:60]}"
                )

        if verbose and (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(items_to_check)} checked")

    total = agreements + disagreements
    report = RevalidationReport(
        total_checked=total,
        agreements=agreements,
        disagreements=disagreements,
        corrections=corrections,
        confidence_counts=dict(confidence_counts),
    )

    if verbose:
        print(report.summary())

    return report


def _parse_revalidation_response(text: str) -> dict:
    """Parse the CORRECT/ACTUAL/CONFIDENCE/REASON format."""
    result = {
        "correct": True,
        "actual": "other",
        "confidence": "low",
        "reason": "",
    }

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("CORRECT:"):
            val = line.split(":", 1)[1].strip().lower()
            result["correct"] = val in ("yes", "true", "y")
        elif line.upper().startswith("ACTUAL:"):
            val = line.split(":", 1)[1].strip().lower()
            val = val.replace(" ", "_").replace("-", "_")
            for at in ActionType:
                if at.value in val:
                    result["actual"] = at.value
                    break
        elif line.upper().startswith("CONFIDENCE:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("high", "medium", "low"):
                result["confidence"] = val
        elif line.upper().startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()

    return result


class RevalidationReport:
    """Aggregates agreement statistics and correction details from a post-run revalidation pass."""

    def __init__(
        self,
        total_checked: int,
        agreements: int,
        disagreements: int,
        corrections: list[dict],
        confidence_counts: dict[str, int],
    ):
        """Store counts of checked, agreed, and disagreed actions plus per-correction detail dicts."""
        self.total_checked = total_checked
        self.agreements = agreements
        self.disagreements = disagreements
        self.corrections = corrections
        self.confidence_counts = confidence_counts

    @property
    def agreement_rate(self) -> float:
        """Return the fraction of checked actions where the revalidator agreed with the original classification."""
        if self.total_checked == 0:
            return 1.0
        return self.agreements / self.total_checked

    @property
    def correction_summary(self) -> dict[str, dict[str, int]]:
        """Return a nested count of original-type to corrected-type transitions across all disagreements."""
        summary: dict[str, dict[str, int]] = defaultdict(Counter)
        for c in self.corrections:
            summary[c["parsed_type"]][c["corrected_type"]] += 1
        return {k: dict(v) for k, v in summary.items()}

    def summary(self) -> str:
        """Return a formatted report string showing agreement rate, correction patterns, and classifier quality assessment."""
        lines = [
            "",
            "=" * 60,
            "REVALIDATION REPORT",
            "=" * 60,
            f"Actions checked: {self.total_checked}",
            f"Agreements: {self.agreements} ({self.agreement_rate:.0%})",
            f"Disagreements: {self.disagreements} ({1 - self.agreement_rate:.0%})",
            f"Confidence: {self.confidence_counts}",
        ]

        if self.corrections:
            lines.append("\nCORRECTION PATTERNS:")
            for original, corrections in self.correction_summary.items():
                for corrected, count in sorted(corrections.items(), key=lambda x: -x[1]):
                    lines.append(f"  {original} -> {corrected}: {count}x")

            lines.append(f"\nCLASSIFICATION QUALITY:")
            if self.agreement_rate >= 0.9:
                lines.append("  GOOD — 90%+ agreement. Parser is reliable.")
            elif self.agreement_rate >= 0.75:
                lines.append("  FAIR — 75-90% agreement. Some edge cases need work.")
            else:
                lines.append("  POOR — <75% agreement. Parser needs significant improvement.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the report to a JSON-compatible dict including all counts and the corrections list."""
        return {
            "total_checked": self.total_checked,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "agreement_rate": round(self.agreement_rate, 3),
            "confidence_counts": self.confidence_counts,
            "correction_summary": self.correction_summary,
            "corrections": self.corrections,
        }
