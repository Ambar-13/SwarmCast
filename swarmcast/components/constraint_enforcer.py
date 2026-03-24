"""Constraint violation detection and enforcement for governance agents.

Provides rule-based checks (keyword matching, action-type checks) followed by
an optional LLM fallback for subtle violations. Use enforce_constraints() as
the single entry point for action filtering.
"""

from __future__ import annotations

from swarmcast.components.actions import GovernanceAction, ActionType
from swarmcast.components.objectives import Objective


def check_constraint_violation(
    action: GovernanceAction,
    objective: Objective,
    model=None,
) -> tuple[bool, str]:
    """Check if action violates any of the objective's constraints. Returns (violated, reason)."""
    if not objective.cannot_accept:
        return False, ""

    description = action.description.lower()

    # Rule-based checks
    for constraint in objective.cannot_accept:
        constraint_lower = constraint.lower()

        constraint_keywords = _extract_keywords(constraint_lower)
        support_keywords = {"support", "accept", "agree", "endorse", "advocate for", "push for"}

        for ckw in constraint_keywords:
            if ckw in description:
                for skw in support_keywords:
                    if skw in description:
                        return True, f"Action supports '{ckw}' which violates: {constraint}"

        if "relocate" in constraint_lower and action.action_type == ActionType.RELOCATE:
            return True, f"Relocation violates: {constraint}"
        if "self-regulat" in constraint_lower and "self-regulat" in description:
            return True, f"Self-regulation violates: {constraint}"
        if "open-sourc" in constraint_lower and "open-sourc" in description:
            return True, f"Open-sourcing violates: {constraint}"

    # LLM check for subtle violations
    if model is not None:
        try:
            prompt = (
                f"Does this action violate any of these constraints?\n\n"
                f"Action: {action.description[:300]}\n\n"
                f"Constraints (CANNOT):\n"
                + "\n".join(f"- {c}" for c in objective.cannot_accept)
                + "\n\nAnswer ONLY 'yes' or 'no'."
            )
            result = model.sample_text(prompt, max_tokens=5, temperature=0.0)
            if "yes" in result.lower():
                return True, f"LLM detected constraint violation"
        except Exception:
            pass

    return False, ""


def _extract_keywords(text: str) -> list[str]:
    """Extract content words (length > 3, stop words removed) plus bigrams."""
    stop_words = {
        "any", "that", "the", "a", "an", "of", "in", "to", "for", "with",
        "without", "not", "no", "cannot", "must", "should", "would", "could",
        "framework", "accept", "give", "gives", "giving",
    }
    words = text.split()
    keywords = []
    for i, word in enumerate(words):
        if word not in stop_words and len(word) > 3:
            keywords.append(word)
        if i < len(words) - 1:
            bigram = f"{word} {words[i + 1]}"
            if word not in stop_words or words[i + 1] not in stop_words:
                keywords.append(bigram)

    return keywords


def enforce_constraints(
    action: GovernanceAction,
    objective: Objective,
    model=None,
) -> GovernanceAction:
    """Return the action unchanged, or DO_NOTHING with explanation if a constraint is violated."""
    violated, reason = check_constraint_violation(action, objective, model)

    if violated:
        return GovernanceAction(
            action_type=ActionType.DO_NOTHING,
            actor=action.actor,
            description=(
                f"[CONSTRAINT VIOLATION] Original action rejected: {reason}. "
                f"Agent forced to reconsider."
            ),
            classification_method="constraint_enforced",
            metadata={"original_action": action.to_dict(), "violation_reason": reason},
        )

    return action
