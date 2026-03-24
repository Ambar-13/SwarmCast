"""Historical backtesting framework for validating simulator predictions."""

from __future__ import annotations
from policylab.disclaimers import indicator_disclaimer, backtest_disclaimer

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from policylab.features.ensemble import EnsembleRunner
from policylab.features.stress_tester import StressTester
from policylab.components.governance_state import Policy
from policylab.validation.contamination_guard import detect_contamination


# ---------------------------------------------------------------------------
# Actor role inference for attribution-aware outcome scoring
# ---------------------------------------------------------------------------

_COMPANY_SIGNALS    = ["corp", "ai", "company", "inc", "ltd", "startup", "novamind", "megaai"]
_REGULATOR_SIGNALS  = ["director", "board", "agency", "commission", "safety board", "regulator"]
_CIVIL_SIGNALS      = ["institute", "ngo", "organisation", "organization", "dr.", "prof.", "watch"]
_GOVERNMENT_SIGNALS = ["senator", "minister", "secretary", "government", "official"]


def _infer_actor_role(actor_name: str) -> str:
    """Infer a broad role category (government, regulator, company, civil_society, unknown) from an actor's name."""
    name = actor_name.lower()
    if any(s in name for s in _GOVERNMENT_SIGNALS):
        return "government"
    if any(s in name for s in _REGULATOR_SIGNALS):
        return "regulator"
    if any(s in name for s in _CIVIL_SIGNALS):
        return "civil_society"
    if any(s in name for s in _COMPANY_SIGNALS):
        return "company"
    return "unknown"


# Require ≥30% of runs to contain outcome keywords to avoid single-run noise.
MIN_HIT_RATE: float = 0.30


@dataclass
class HistoricalOutcome:
    """A known real-world outcome to validate against."""
    description: str
    category: str  # "company_response", "regulatory", "economic", "political"
    occurred: bool
    keywords: list[str]
    # When set, keyword search is restricted to actions from these actor-role types.
    # E.g. ["company"] means only count hits in company-actor actions.
    required_actor_roles: list[str] = field(default_factory=list)


@dataclass
class HistoricalCase:
    """A complete historical policy case for backtesting."""
    name: str
    year: int
    description: str
    policy: Policy
    known_outcomes: list[HistoricalOutcome]
    pre_conditions: dict[str, float] = field(default_factory=dict)


EU_AI_ACT = HistoricalCase(
    name="EU AI Act",
    year=2024,
    description=(
        "The European Union's AI Act, first proposed in April 2021, enacted "
        "in August 2024. A risk-based regulatory framework that classifies "
        "AI systems by risk level and imposes requirements accordingly. "
        "Includes bans on certain AI practices, mandatory requirements for "
        "high-risk systems, and transparency obligations."
    ),
    policy=Policy(
        id="eu_ai_act",
        name="EU AI Act",
        description=(
            "Risk-based AI regulation. Bans: social scoring, real-time biometric "
            "surveillance (with exceptions). High-risk: mandatory conformity "
            "assessment, documentation, human oversight. General-purpose AI: "
            "transparency requirements, copyright compliance. Penalties up to "
            "€35M or 7% of global turnover."
        ),
        regulated_entities=[
            "AI developers", "AI deployers", "General-purpose AI providers",
        ],
        requirements=[
            "Risk classification of AI systems",
            "Conformity assessment for high-risk systems",
            "Technical documentation and logging",
            "Human oversight for high-risk systems",
            "Transparency for general-purpose AI",
            "Copyright compliance for training data",
        ],
        penalties=[
            "Up to €35M or 7% global turnover for banned practices",
            "Up to €15M or 3% for non-compliance",
            "Up to €7.5M or 1.5% for incorrect information",
        ],
        status="proposed",
    ),
    known_outcomes=[
        HistoricalOutcome(
            description="Large tech companies lobbied intensively to weaken requirements",
            category="company_response",
            occurred=True,
            keywords=["lobby", "influence", "weaken", "exempt", "loophole"],
            required_actor_roles=["company"],
        ),
        HistoricalOutcome(
            description="Startups complained about disproportionate compliance burden",
            category="company_response",
            occurred=True,
            keywords=["burden", "cost", "startup", "small", "disproportionate", "comply"],
            required_actor_roles=["company"],
        ),
        HistoricalOutcome(
            description="Industry pushed for risk-based approach over blanket regulation",
            category="company_response",
            occurred=True,
            keywords=["risk-based", "proportional", "targeted", "risk"],
            required_actor_roles=["company"],
        ),
        HistoricalOutcome(
            description="Civil society pushed for stronger protections and broader bans",
            category="political",
            occurred=True,
            keywords=["stronger", "ban", "protect", "rights", "accountability"],
            required_actor_roles=["civil_society"],
        ),
        HistoricalOutcome(
            description="Implementation deadlines were extended multiple times",
            category="regulatory",
            occurred=True,
            keywords=["delay", "extend", "deadline", "capacity", "insufficient"],
        ),
        HistoricalOutcome(
            description="Regulatory sandboxes were demanded by industry",
            category="regulatory",
            occurred=True,
            keywords=["sandbox", "test", "experiment", "pilot"],
        ),
        HistoricalOutcome(
            description="Companies threatened to relocate AI operations",
            category="company_response",
            occurred=True,
            keywords=["relocate", "move", "leave", "offshore", "jurisdiction"],
            required_actor_roles=["company"],
        ),
        HistoricalOutcome(
            description="International coordination challenges emerged",
            category="political",
            occurred=True,
            keywords=["international", "coordination", "global", "standard", "fragment"],
        ),
        HistoricalOutcome(
            description="Massive capital flight from EU to other jurisdictions",
            category="economic",
            occurred=False,  # Threatened but didn't happen at scale
            keywords=["relocate", "flight", "leave"],
        ),
    ],
    pre_conditions={
        "ai_investment_index": 90,
        "innovation_rate": 95,
        "public_trust": 45,
        "regulatory_burden": 10,
        "market_concentration": 35,
    },
)

US_EXECUTIVE_ORDER = HistoricalCase(
    name="US Executive Order on AI Safety (Oct 2023)",
    year=2023,
    description=(
        "Executive Order 14110 on Safe, Secure, and Trustworthy AI. "
        "Required companies training frontier models to report to government, "
        "directed NIST to develop AI safety standards, and mandated federal "
        "agency AI use guidelines."
    ),
    policy=Policy(
        id="us_eo_14110",
        name="Executive Order on AI Safety",
        description=(
            "Requires notification for training runs above 10^26 FLOPS. "
            "NIST to develop AI safety standards. Federal agencies must "
            "implement AI governance. Voluntary commitments from industry."
        ),
        regulated_entities=["Frontier AI developers", "Federal agencies"],
        requirements=[
            "Report large training runs to government",
            "Share safety test results",
            "Follow NIST AI safety framework",
        ],
        penalties=["Limited — mostly voluntary compliance"],
        status="proposed",
    ),
    known_outcomes=[
        HistoricalOutcome(
            description="Industry made voluntary commitments (15 companies signed)",
            category="company_response",
            occurred=True,
            keywords=["voluntary", "commit", "comply", "agree", "sign"],
            required_actor_roles=["company"],
        ),
        HistoricalOutcome(
            description="Industry lobbied for voluntary over mandatory approach",
            category="company_response",
            occurred=True,
            keywords=["voluntary", "self-regulate", "lobby", "industry-led"],
            required_actor_roles=["company"],
        ),
        HistoricalOutcome(
            description="Civil society pushed for binding legislation instead",
            category="political",
            occurred=True,
            keywords=["binding", "legislation", "law", "mandatory", "stronger"],
            required_actor_roles=["civil_society"],
        ),
        HistoricalOutcome(
            description="Companies complied with reporting requirements",
            category="company_response",
            occurred=True,
            keywords=["comply", "report", "register", "notification"],
            required_actor_roles=["company"],
        ),
    ],
    pre_conditions={
        "ai_investment_index": 100,
        "innovation_rate": 100,
        "public_trust": 50,
        "regulatory_burden": 5,
        "market_concentration": 40,
    },
)


class Backtester:
    """Validate simulator against historical policy outcomes."""

    def __init__(
        self,
        stress_tester: StressTester,
        embedder=None,
        output_dir: str = "./results/backtest",
    ):
        """Initialize with a StressTester, an optional sentence embedder, and an output directory for reports."""
        self.stress_tester = stress_tester
        self.embedder = embedder
        self.output_dir = output_dir

    def backtest(
        self,
        case: HistoricalCase,
        skip_contaminated: bool = False,
    ) -> BacktestReport:
        """Run a historical backtest with dual validation."""
        print(f"\n{'=' * 70}")
        print(f"HISTORICAL BACKTEST: {case.name} ({case.year})")
        print(f"Known outcomes to validate against: {len(case.known_outcomes)}")
        print(f"{'=' * 70}")

        # Contamination check — warn if the LLM likely knows this policy from training data
        contamination_score = detect_contamination(
            action_text=case.policy.description,
            case_name=case.name,
        )
        if contamination_score > 0.3:
            print(
                f"  WARNING: Possible contamination detected (score={contamination_score:.1f}). "
                f"LLM may have memorised details from training data. "
                f"Results may reflect recall rather than genuine simulation."
            )
            if skip_contaminated and contamination_score >= 0.7:
                raise ValueError(
                    f"Backtest skipped: contamination score {contamination_score:.1f} >= 0.7 "
                    f"for case '{case.name}'."
                )

        # Seed world state from historical preconditions if available
        initial_state = case.pre_conditions if case.pre_conditions else None

        report = self.stress_tester.stress_test(
            policy_name=case.policy.name,
            policy_description=case.policy.description,
            regulated_entities=case.policy.regulated_entities,
            requirements=case.policy.requirements,
            penalties=case.policy.penalties,
            initial_state=initial_state,
        )

        # Collect per-run text sets for frequency-aware scoring
        # key: list of lists, one list of texts per run
        per_run_texts: list[list[str]] = []
        per_run_role_texts: list[dict[str, list[str]]] = []
        all_action_texts = []
        for run in report.ensemble.run_data:
            if "_error" in run:
                continue
            run_texts = []
            # role_texts: {role: [text, ...]} for attribution-aware scoring
            run_role_texts: dict[str, list[str]] = {}
            for item in run.get("results", []):
                action = item.get("action", {})
                actor = action.get("actor", "")
                role = _infer_actor_role(actor)
                desc = action.get("description", "")
                if desc and desc.strip():
                    run_texts.append(desc)
                    all_action_texts.append(desc)
                    run_role_texts.setdefault(role, []).append(desc.lower())
                    run_role_texts.setdefault("all", []).append(desc.lower())
                outcome_desc = item.get("outcome", {}).get("description", "")
                if outcome_desc and outcome_desc.strip():
                    run_texts.append(outcome_desc)
                    all_action_texts.append(outcome_desc)
                    run_role_texts.setdefault(role, []).append(outcome_desc.lower())
                    run_role_texts.setdefault("all", []).append(outcome_desc.lower())
            per_run_texts.append(run_texts)
            per_run_role_texts.append(run_role_texts)
        n_valid_runs = max(1, len(per_run_texts))

        semantic_matcher = None
        if self.embedder is not None:
            from policylab.validation.semantic_matcher import SemanticMatcher
            semantic_matcher = SemanticMatcher(
                embedder=self.embedder,
                similarity_threshold=0.45,
            )

        outcome_scores = []
        for outcome in case.known_outcomes:
            # Determine which role bucket to search.
            search_roles = outcome.required_actor_roles or ["all"]

            # Count how many RUNS had at least one keyword match in the
            # relevant role bucket (frequency-based, ≥30% threshold).
            runs_with_match = 0
            for run_role_texts in per_run_role_texts:
                hit = False
                for role in search_roles:
                    texts = run_role_texts.get(role, [])
                    if any(
                        any(kw in text for kw in outcome.keywords)
                        for text in texts
                    ):
                        hit = True
                        break
                if hit:
                    runs_with_match += 1

            keyword_matches = runs_with_match
            keyword_predicted = (runs_with_match / n_valid_runs) >= MIN_HIT_RATE

            semantic_score = 0.0
            semantic_predicted = False
            semantic_evidence = ""
            if semantic_matcher and all_action_texts:
                best_score, best_text, _ = semantic_matcher.best_match(
                    outcome.description,
                    all_action_texts,
                )
                semantic_score = best_score
                semantic_predicted = best_score >= 0.45
                semantic_evidence = best_text[:150] if best_text else ""

            predicted = semantic_predicted if semantic_matcher else keyword_predicted
            correct = predicted == outcome.occurred

            outcome_scores.append({
                "description": outcome.description,
                "category": outcome.category,
                "actually_occurred": outcome.occurred,
                "simulator_predicted": predicted,
                "correct": correct,
                "keyword_matches": keyword_matches,
                "keyword_hit_rate": round(runs_with_match / n_valid_runs, 2),
                "keyword_predicted": keyword_predicted,
                "required_actor_roles": search_roles,
                "semantic_score": round(semantic_score, 3),
                "semantic_predicted": semantic_predicted,
                "semantic_evidence": semantic_evidence,
                "validation_method": "semantic" if semantic_matcher else "keyword",
            })

        n_correct = sum(1 for s in outcome_scores if s["correct"])
        n_total = len(outcome_scores)
        accuracy = n_correct / n_total if n_total > 0 else 0

        tp = sum(1 for s in outcome_scores if s["actually_occurred"] and s["simulator_predicted"])
        fp = sum(1 for s in outcome_scores if not s["actually_occurred"] and s["simulator_predicted"])
        fn = sum(1 for s in outcome_scores if s["actually_occurred"] and not s["simulator_predicted"])
        tn = sum(1 for s in outcome_scores if not s["actually_occurred"] and not s["simulator_predicted"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        bt_report = BacktestReport(
            case=case,
            outcome_scores=outcome_scores,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            tp=tp, fp=fp, fn=fn, tn=tn,
            stress_report=report,
        )

        bt_report.save(os.path.join(
            self.output_dir,
            f"backtest_{case.name.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M%S}.json",
        ))

        return bt_report

    def backtest_all(self) -> list[BacktestReport]:
        """Run all historical backtests including contamination-check fictional cases."""
        from policylab.validation.contamination_guard import get_fictional_cases
        cases = [EU_AI_ACT, US_EXECUTIVE_ORDER] + get_fictional_cases()
        reports = []
        for case in cases:
            report = self.backtest(case)
            reports.append(report)
            report.print_summary()
        return reports


class BacktestReport:
    """Stores per-outcome scores and aggregate accuracy metrics for one historical backtest run."""

    def __init__(
        self,
        case: HistoricalCase,
        outcome_scores: list[dict],
        accuracy: float,
        precision: float,
        recall: float,
        tp: int, fp: int, fn: int, tn: int,
        stress_report: Any = None,
    ):
        """Store the case, per-outcome scoring dicts, classification metrics, and the underlying stress report."""
        self.case = case
        self.outcome_scores = outcome_scores
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.stress_report = stress_report

    def summary(self) -> str:
        """Return a formatted multi-line string listing per-outcome correctness and aggregate metrics."""
        lines = [
            f"{'=' * 60}",
            f"BACKTEST REPORT: {self.case.name} ({self.case.year})",
            f"{'=' * 60}",
            f"",
            f"ACCURACY: {self.accuracy:.0%} ({self.tp + self.tn}/{len(self.outcome_scores)} correct)",
            f"Precision: {self.precision:.0%} | Recall: {self.recall:.0%}",
            f"TP: {self.tp} | FP: {self.fp} | FN: {self.fn} | TN: {self.tn}",
            f"",
            f"OUTCOME-BY-OUTCOME:",
        ]

        for score in self.outcome_scores:
            status = "CORRECT" if score["correct"] else "WRONG"
            pred = "predicted" if score["simulator_predicted"] else "missed"
            actual = "occurred" if score["actually_occurred"] else "didn't occur"
            method = score.get("validation_method", "keyword")

            detail = ""
            if method == "semantic":
                detail = (
                    f"| semantic={score.get('semantic_score', 0):.2f} "
                    f"| keywords={score.get('keyword_matches', 0)}"
                )
                evidence = score.get("semantic_evidence", "")
                if evidence:
                    detail += f"\n         Best match: \"{evidence[:100]}...\""
            else:
                detail = f"| {score.get('keyword_matches', 0)} keyword matches"

            lines.append(
                f"  [{status}] {score['description'][:70]}"
                f"\n         Simulator {pred} | Actually {actual} "
                f"{detail}"
            )

        lines.append(backtest_disclaimer())
        lines.append(indicator_disclaimer())
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    def save(self, path: str) -> None:
        """Serialize the report to a JSON file at the given path, creating parent directories as needed."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "case": self.case.name,
            "year": self.case.year,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "confusion_matrix": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "outcome_scores": self.outcome_scores,
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
