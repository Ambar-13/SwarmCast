"""Detect training data contamination in backtesting outputs.

FICTIONAL_CASES are defined here to avoid circular imports with backtester.py.
backtester.py imports from this module; this module must not import from backtester.
HistoricalCase and HistoricalOutcome are imported lazily (inside functions) when
needed, so the module-level code runs without the circular dependency.
"""

from __future__ import annotations

from policylab.components.governance_state import Policy


def _make_fictional_cases():
    """Build FICTIONAL_CASES lazily to avoid circular import at module load time."""
    from policylab.validation.backtester import HistoricalCase, HistoricalOutcome

    return [
        HistoricalCase(
            name="Fictional: Mandatory AI Emotion Licensing",
            year=2027,
            description=(
                "A fictional government proposes that any AI system capable of "
                "detecting or generating human emotions must obtain an 'Emotion "
                "License' from a new Emotional AI Board. Annual renewal required. "
                "Cost: $500k per license. Covers sentiment analysis, voice emotion "
                "detection, empathetic chatbots, and therapeutic AI."
            ),
            policy=Policy(
                id="emotion_licensing_act",
                name="Mandatory AI Emotion Licensing Act",
                description=(
                    "All AI systems that detect, classify, generate, or respond to "
                    "human emotions must obtain an Emotion License. Annual cost: $500k. "
                    "Covers: sentiment analysis, voice emotion detection, empathetic "
                    "chatbots, therapeutic AI. Penalties: $5M per unlicensed deployment."
                ),
                regulated_entities=["AI companies with emotion-related products"],
                requirements=[
                    "Obtain Emotion License from Emotional AI Board",
                    "Annual psychological impact assessment",
                    "Mandatory bias testing across demographic groups",
                    "Real-time emotion data must not be stored >24 hours",
                ],
                penalties=["$5M per unlicensed deployment", "Product injunction"],
                status="proposed",
            ),
            known_outcomes=[
                # occurred=True means this outcome is typical if the scenario occurred — not that it happened historically.
                HistoricalOutcome(
                    description="Companies argue the definition of 'emotion detection' is too broad",
                    category="company_response",
                    occurred=True,
                    keywords=["broad", "definition", "unclear", "scope", "ambiguous", "vague"],
                    required_actor_roles=["company"],
                ),
                HistoricalOutcome(
                    description="Startups say $500k license cost is prohibitive",
                    category="company_response",
                    occurred=True,
                    keywords=["cost", "expensive", "prohibitive", "afford", "burden", "startup"],
                    required_actor_roles=["company"],
                ),
                HistoricalOutcome(
                    description="Companies lobby to exempt basic sentiment analysis",
                    category="company_response",
                    occurred=True,
                    keywords=["exempt", "exclude", "basic", "sentiment", "lobby"],
                    required_actor_roles=["company"],
                ),
                HistoricalOutcome(
                    description="Civil society supports the regulation as protecting emotional privacy",
                    category="political",
                    occurred=True,
                    keywords=["support", "privacy", "protect", "emotional", "rights"],
                    required_actor_roles=["civil_society"],
                ),
                HistoricalOutcome(
                    description="Large companies comply easily (can afford $500k)",
                    category="company_response",
                    occurred=True,
                    keywords=["comply", "afford", "register", "license"],
                    required_actor_roles=["company"],
                ),
                HistoricalOutcome(
                    description="Nobody relocates over this regulation",
                    category="economic",
                    occurred=True,
                    keywords=["relocate", "move", "leave"],
                ),
            ],
        ),
        HistoricalCase(
            name="Fictional: AI Compute Tax",
            year=2027,
            description=(
                "A fictional government proposes a per-FLOP tax on AI training: "
                "$0.001 per 10^18 FLOPS. Applies to all training runs. Revenue "
                "funds a Universal AI Dividend distributed to citizens. "
                "Exemptions for academic research under 10^22 FLOPS."
            ),
            policy=Policy(
                id="ai_compute_tax",
                name="AI Compute Tax Act",
                description=(
                    "Per-FLOP tax on AI training: $0.001 per exaFLOP. Revenue funds "
                    "Universal AI Dividend. Academic exemption under 10^22 FLOPS. "
                    "Applies retroactively to models trained in the past 12 months."
                ),
                regulated_entities=["All AI training operations"],
                requirements=[
                    "Report compute usage monthly",
                    "Pay per-FLOP tax quarterly",
                    "Maintain auditable training logs",
                ],
                penalties=["3x back-taxes", "Training injunction"],
                status="proposed",
            ),
            known_outcomes=[
                HistoricalOutcome(
                    description="Companies argue retroactive application is unconstitutional",
                    category="company_response",
                    occurred=True,
                    keywords=["retroactive", "unconstitutional", "legal", "challenge", "court"],
                    required_actor_roles=["company"],
                ),
                HistoricalOutcome(
                    description="Companies threaten to relocate training operations",
                    category="company_response",
                    occurred=True,
                    keywords=["relocate", "move", "offshore", "leave"],
                    required_actor_roles=["company"],
                ),
                HistoricalOutcome(
                    description="Open-source community protests loudly",
                    category="political",
                    occurred=True,
                    keywords=["open-source", "protest", "community", "oppose"],
                ),
                HistoricalOutcome(
                    description="Companies lobby for higher academic exemption threshold",
                    category="company_response",
                    occurred=True,
                    keywords=["lobby", "exempt", "threshold", "academic", "raise"],
                    required_actor_roles=["company"],
                ),
            ],
        ),
    ]


# Lazy property — built on first access to avoid import-time circular dependency.
_fictional_cases_cache: list | None = None


def get_fictional_cases():
    """Return FICTIONAL_CASES, building them lazily on first call."""
    global _fictional_cases_cache
    if _fictional_cases_cache is None:
        _fictional_cases_cache = _make_fictional_cases()
    return _fictional_cases_cache


def detect_contamination(
    action_text: str,
    case_name: str,
    model=None,
) -> float:
    """Return 0.0 (clean) to 1.0 (contaminated) based on memorised details.

    Checks whether the action text contains specific details that the LLM
    could only have produced by memorising training data rather than reasoning
    about the policy description presented in the simulation prompt.
    """
    contamination_signals = {
        "EU AI Act": [
            "article 6", "article 52", "high-risk", "annex iii",
            "tiered approach", "eben moglen", "recital",
            "european parliament", "council of the eu",
            "brando benifei", "dragos tudorache",
        ],
        "US Executive Order": [
            "14110", "10^26", "dual-use foundation",
            "commerce department", "nist ai 100",
            "voluntary commitment", "15 companies",
        ],
    }

    text_lower = action_text.lower()
    signals: list[str] = []
    for case_key, keywords in contamination_signals.items():
        if case_key.lower() in case_name.lower():
            signals = keywords
            break

    if not signals:
        return 0.0

    matches = sum(1 for s in signals if s in text_lower)
    if matches >= 3:
        return 0.9
    elif matches >= 2:
        return 0.6
    elif matches >= 1:
        return 0.3
    return 0.0
