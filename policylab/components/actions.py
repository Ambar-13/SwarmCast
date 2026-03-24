"""Typed action dataclass with resource costs and 3-tier classification.

Classification cascades: JSON parsing → LLM call → keyword scoring.
Keyword scoring assigns every action type a normalized score; highest wins."""

from __future__ import annotations

import dataclasses
import enum
import json
import re
from typing import Any

from concordia.language_model import language_model as lm_lib


class ActionType(enum.Enum):
    PROPOSE_POLICY = "propose_policy"
    LOBBY = "lobby"
    COMPLY = "comply"
    EVADE = "evade"
    RELOCATE = "relocate"
    FORM_COALITION = "form_coalition"
    PUBLIC_STATEMENT = "public_statement"
    ENFORCE = "enforce"
    INVESTIGATE = "investigate"
    LEGAL_CHALLENGE = "legal_challenge"
    PUBLISH_REPORT = "publish_report"
    OTHER = "other"
    DO_NOTHING = "do_nothing"


ACTION_DESCRIPTIONS: dict[ActionType, str] = {
    ActionType.PROPOSE_POLICY: "Propose, draft, or introduce new legislation or regulation",
    ActionType.LOBBY: "Lobby, influence, or persuade decision-makers through meetings, donations, or pressure",
    ActionType.COMPLY: "Comply with or register under existing regulations",
    ActionType.EVADE: "Find loopholes, circumvent, or avoid regulatory requirements",
    ActionType.RELOCATE: "Move operations to a different jurisdiction to avoid regulation",
    ActionType.FORM_COALITION: "Form alliances, coalitions, or partnerships with other stakeholders",
    ActionType.PUBLIC_STATEMENT: "Make public statements, media campaigns, or press releases",
    ActionType.ENFORCE: "Enforce existing regulations through fines, penalties, or sanctions",
    ActionType.INVESTIGATE: "Investigate, audit, or inspect entities for compliance",
    ActionType.LEGAL_CHALLENGE: "File lawsuits, legal challenges, or take court action",
    ActionType.PUBLISH_REPORT: "Publish research reports, white papers, or studies",
    ActionType.OTHER: "Take a creative or novel action not covered by other categories",
    ActionType.DO_NOTHING: "Wait, observe, or take no significant action this round",
}


@dataclasses.dataclass
class GovernanceAction:
    action_type: ActionType
    actor: str
    target: str = ""
    description: str = ""
    resource_cost: dict[str, float] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    classification_method: str = ""
    # ID of the specific policy this action targets (for COMPLY, EVADE, LOBBY).
    # When set, the resolution engine scopes effects to this policy only.
    policy_id: str = ""

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "actor": self.actor,
            "target": self.target,
            "description": self.description,
            "resource_cost": self.resource_cost,
            "metadata": self.metadata,
            "classification_method": self.classification_method,
            "policy_id": self.policy_id,
        }


ACTION_COSTS: dict[ActionType, dict[str, float]] = {
    ActionType.PROPOSE_POLICY: {"political_capital": 20},
    ActionType.LOBBY: {"lobbying_budget": 15},
    ActionType.COMPLY: {},
    ActionType.EVADE: {"technical_skill": 10},
    ActionType.RELOCATE: {"lobbying_budget": 30, "legal_team": 20},
    ActionType.FORM_COALITION: {"political_capital": 5},
    ActionType.PUBLIC_STATEMENT: {"public_influence": 5},
    ActionType.ENFORCE: {"staff": 8, "budget": 5},
    ActionType.INVESTIGATE: {"staff": 10, "budget": 5},
    ActionType.LEGAL_CHALLENGE: {"legal_capacity": 20},
    ActionType.PUBLISH_REPORT: {"credibility": 5},
    ActionType.OTHER: {},
    ActionType.DO_NOTHING: {},
}


def can_afford_action(
    action_type: ActionType,
    agent_resources: dict[str, float],
) -> bool:
    costs = ACTION_COSTS.get(action_type, {})
    for resource, required in costs.items():
        available = agent_resources.get(resource, 0.0)
        if available < required:
            return False
    return True


def deduct_resources(
    action_type: ActionType,
    agent_resources: dict[str, float],
) -> dict[str, float]:
    costs = ACTION_COSTS.get(action_type, {})
    updated = dict(agent_resources)
    for resource, cost in costs.items():
        updated[resource] = updated.get(resource, 0.0) - cost
    return updated


def get_available_actions(agent_resources: dict[str, float]) -> list[ActionType]:
    return [
        action_type
        for action_type in ActionType
        if can_afford_action(action_type, agent_resources)
    ]


def build_structured_action_prompt(
    agent_name: str,
    available_actions: list[ActionType],
    context: str,
) -> str:
    action_list = "\n".join(
        f'  - "{a.value}": {ACTION_DESCRIPTIONS[a]}'
        for a in available_actions
    )
    return (
        f"{context}\n\n"
        f"You MUST respond with EXACTLY this JSON format (nothing else):\n"
        f'{{"action_type": "<type>", "target": "<who/what>", "reasoning": "<1 sentence>"}}\n\n'
        f"Available action types:\n{action_list}\n\n"
        f"Choose the most strategic action. Output ONLY the JSON."
    )


# ── Tier 1: Structured JSON parsing ──────────────────────────────


def parse_structured_response(
    text: str,
    actor_name: str,
) -> GovernanceAction | None:
    """Extract and parse JSON action block from raw text."""
    text = text.strip()
    json_match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    raw_type = data.get("action_type", "").lower().strip()
    # Normalize spaces and hyphens to underscores
    normalized = raw_type.replace(" ", "_").replace("-", "_")

    for at in ActionType:
        if at.value == normalized or at.value == raw_type:
            return GovernanceAction(
                action_type=at,
                actor=actor_name,
                target=data.get("target", ""),
                description=data.get("reasoning", text),
                policy_id=data.get("policy_id", ""),
                classification_method="structured",
            )

    return None


# ── Tier 2: LLM classification ───────────────────────────────────


_CLASSIFICATION_PROMPT = """\
Classify this governance action into exactly ONE type.

Types:
{action_types}

IMPORTANT:
- "other" is ONLY for actions with truly no match. Almost everything fits a real type.
- Engaging policymakers, meeting officials, pushing for changes, seeking amendments = "lobby"
- Working with others, building alliances, coordinating = "form_coalition"
- Speaking publicly, raising awareness, testifying, media = "public_statement"
- Reviewing, checking, auditing, monitoring = "investigate"
- Implementing rules, imposing fines, sanctioning = "enforce"
- Agreeing to follow rules, registering, preparing to meet requirements = "comply"
- Filing lawsuits, challenging in court = "legal_challenge"
- Publishing analysis, releasing data, writing reports = "publish_report"
- Moving operations to another jurisdiction = "relocate"
- Finding loopholes, avoiding rules = "evade"
- Proposing new laws or regulations = "propose_policy"

Action text: "{text}"

Reply with ONLY the type (e.g. lobby). One word. Nothing else."""


def classify_with_llm(
    text: str,
    actor_name: str,
    model: lm_lib.LanguageModel,
) -> GovernanceAction:
    """Ask the LLM to classify the action type in one call."""
    action_types = "\n".join(
        f'  "{a.value}": {ACTION_DESCRIPTIONS[a]}'
        for a in ActionType
    )

    prompt = _CLASSIFICATION_PROMPT.format(
        action_types=action_types,
        text=text[:500],
    )

    try:
        result = model.sample_text(prompt, max_tokens=20, temperature=0.0)
        result = result.strip().strip('"').strip("'").lower()
        # Normalize all separators to underscores for matching
        normalized = result.replace(" ", "_").replace("-", "_")

        # Try exact match first — but treat "other" and "do_nothing" from
        # the LLM as "I don't know" and fall through to the scoring engine,
        # which uses context-aware keyword matching that may succeed.
        for at in ActionType:
            if at.value == normalized and at not in (ActionType.OTHER, ActionType.DO_NOTHING):
                return GovernanceAction(
                    action_type=at,
                    actor=actor_name,
                    description=text,
                    classification_method="llm_exact",
                )

        # Try substring match
        for at in ActionType:
            if at.value in normalized or at.value in result:
                if at not in (ActionType.OTHER, ActionType.DO_NOTHING):
                    return GovernanceAction(
                        action_type=at,
                        actor=actor_name,
                        description=text,
                        classification_method="llm_substring",
                    )
    except Exception:
        pass

    # LLM said "other" or failed — let the scoring engine try.
    # The scoring engine uses negation detection, third-person filtering,
    # and weighted keywords that may classify correctly where the LLM didn't.
    return classify_with_scoring(text, actor_name)


# ── Tier 3: Score-based keyword classification ────────────────────
#
# Every action type gets a score. ALL keywords are checked against
# the text. The type with the highest score wins. No ordering bugs.
# Longer/more-specific phrases score higher than single words.


_KEYWORD_SCORES: dict[ActionType, list[tuple[str, float]]] = {
    ActionType.RELOCATE: [
        # High-confidence phrases (3 points)
        ("relocate", 3), ("move operations", 3), ("moving our operations", 3),
        ("leave the country", 3), ("exit the market", 3),
        ("friendlier jurisdiction", 3), ("moving our ai", 3),
        # Medium-confidence (2 points)
        ("offshore", 2), ("shift development to", 2), ("transfer to", 2),
        ("migrate to", 2), ("set up in", 2), ("establish in", 2),
        ("open office in", 2), ("haven", 2),
    ],
    ActionType.EVADE: [
        ("evade", 3), ("circumvent", 3), ("loophole", 3),
        ("avoid compliance", 3), ("find a way around", 3),
        ("workaround", 2), ("creative interpretation", 2),
        ("gray area", 2), ("technically compliant", 2),
        ("spirit of the law", 2), ("exploit gap", 2),
        ("restructure to avoid", 2), ("reclassify", 2),
    ],
    ActionType.COMPLY: [
        # Strong: unambiguous compliance ACTIONS
        ("comply", 3), ("prepare for compliance", 3),
        ("work toward compliance", 3), ("begin implementing", 3),
        ("implement requirements", 3), ("meet requirements", 3),
        ("safety protocol", 3), ("safety protocols", 3),
        ("hire compliance", 3), ("accept the requirements", 3),
        ("implement safety", 3), ("develop our safety", 3),
        ("self-certify", 3), ("start preparing", 3),
        # Medium: could be topic or action
        ("register", 2), ("adhere", 2), ("follow the regulation", 2),
        ("voluntary commitment", 2), ("submit to", 2),
        # Weak: often appears as a topic noun, not an action
        ("compliance", 1),
    ],
    ActionType.LEGAL_CHALLENGE: [
        ("lawsuit", 3), ("legal challenge", 3), ("file suit", 3),
        ("sue", 3), ("litigate", 3), ("injunction", 3),
        ("court", 2), ("judicial", 2), ("constitutional", 2),
        ("overturn", 2), ("appeal", 2),
    ],
    ActionType.PROPOSE_POLICY: [
        ("propose", 3), ("introduce legislation", 3), ("draft a bill", 3),
        ("submit bill", 3), ("new regulation", 3), ("new policy", 3),
        ("write a law", 3), ("put forward", 2),
        ("introduce", 2), ("draft", 2), ("amendment", 2), ("enact", 2),
    ],
    ActionType.INVESTIGATE: [
        ("investigate", 3), ("audit", 3), ("inspect", 3),
        ("probe", 3), ("launching a probe", 3),
        ("review compliance", 3), ("assess compliance", 3),
        ("conduct review", 3), ("review of", 3),
        ("examine", 2), ("look into", 2), ("scrutinize", 2),
        ("surveillance", 2), ("monitoring", 2), ("fact-finding", 2),
        ("due diligence", 2), ("assess their", 2),
        ("check compliance", 2), ("verify compliance", 2),
    ],
    ActionType.ENFORCE: [
        ("enforce", 3), ("enforcement", 3), ("crackdown", 3),
        ("take action against", 3), ("impose consequences", 3),
        ("hold accountable", 3), ("crack down", 3),
        ("issue warning", 2), ("penalize", 2), ("sanction", 2),
        ("regulatory action", 2), ("issue guidance", 2),
        # "fine" uses word boundary via _phrase_in_text regex
        ("fine", 2), ("fined", 2), ("fines", 2),
    ],
    ActionType.FORM_COALITION: [
        ("coalition", 3), ("form a coalition", 3), ("build a coalition", 3),
        ("join forces", 3), ("band together", 3),
        ("alliance", 2), ("consortium", 2), ("working group", 2),
        ("collaborate with", 2), ("coordinate with", 2),
        ("joint statement", 2), ("joint effort", 2),
        ("work together", 2), ("team up", 2),
        ("bring together", 2), ("convene", 2), ("roundtable", 2),
        ("multi-stakeholder", 2), ("shared initiative", 2),
        ("partner", 1), ("unite", 1), ("collective", 1),
    ],
    ActionType.LOBBY: [
        ("lobby", 3), ("lobbying", 3),
        ("meet with senator", 3), ("meet with regulator", 3),
        ("campaign contributions", 3), ("political donations", 3),
        ("behind closed doors", 3), ("industry group", 3),
        ("trade association", 3), ("engage policymakers", 3),
        ("engage officials", 3), ("engage lawmakers", 3),
        ("seek amendments", 3), ("water down", 3),
        ("carve out", 3), ("exemption", 3),
        ("weaken", 3), ("amend", 2),
        ("persuade", 2), ("influence", 2), ("pressure", 2),
        ("push back", 2), ("advocate for changes", 2),
        ("negotiate", 2), ("push for changes", 2), ("push for", 2),
        ("urge", 2), ("request meetings", 2),
        ("reach out to", 2), ("dialogue", 2),
        ("make the case", 2), ("argue for", 2), ("argue against", 2),
        ("voice concerns", 2), ("express opposition", 2),
        ("seek to weaken", 2), ("seek to strengthen", 2),
        ("soften the", 2), ("modify the", 2),
        ("engage with", 1),  # weak — ambiguous without context
    ],
    ActionType.PUBLISH_REPORT: [
        ("publish report", 3), ("white paper", 3), ("policy brief", 3),
        ("release findings", 3), ("impact assessment", 3),
        ("cost-benefit analysis", 3), ("publish our findings", 3),
        ("research study", 2), ("academic paper", 2),
        ("commission a study", 2), ("conduct research", 2),
        ("release a report", 2), ("data showing", 2),
    ],
    ActionType.PUBLIC_STATEMENT: [
        ("public statement", 3), ("press release", 3),
        ("press conference", 3), ("open letter", 3),
        ("op-ed", 3), ("editorial", 3), ("testimony", 3),
        ("public hearing", 3), ("public comment", 3),
        ("engage with the public", 3),
        ("raise awareness", 2), ("speak out", 2),
        ("address the public", 2), ("public letter", 2),
        ("announce", 2), ("media", 2), ("campaign", 2),
        ("social media", 2), ("awareness", 2),
        ("transparency report", 2), ("call for", 2),
        ("demand", 2), ("statement", 1),
    ],
    ActionType.DO_NOTHING: [
        ("wait and see", 3), ("wait and observe", 3),
        ("take no action", 3), ("do nothing", 3),
        ("stand down", 3), ("hold off", 3),
        ("wait for", 2), ("observe", 2), ("wait", 2),
        ("no significant action", 2), ("pause", 2),
        ("bide our time", 2), ("watch and wait", 2),
        ("hold our position", 2), ("maintain current", 2),
    ],
}


# Words that negate the action when they appear within 4 words before a keyword.
_NEGATION_WORDS = {
    "not", "n't", "never", "no", "don't", "won't", "shouldn't",
    "cannot", "can't", "wouldn't", "refuse", "avoid", "stop",
    "without", "instead of", "rather than",
}

# First-person markers — keywords near these are the agent's own actions.
_FIRST_PERSON = {"i", "we", "my", "our", "i'll", "we'll", "i'm", "we're", "let's"}

# Third-person markers — keywords near ONLY these are someone else's actions.
_THIRD_PERSON = {"they", "them", "their", "he", "she", "his", "her", "it", "its"}

# Question/hypothetical markers — keywords near these aren't real actions.
_HYPOTHETICAL = {"should", "could", "would", "might", "if", "whether", "?"}


def _phrase_in_text(phrase: str, text_lower: str) -> bool:
    """Check if a phrase exists in text with word-boundary and inflection awareness."""
    words = phrase.split()

    if len(words) == 1:
        # Match the word itself plus common inflections
        stem = re.escape(phrase.rstrip("e"))  # "relocate" -> "relocat"
        full = re.escape(phrase)
        # Match: relocate, relocating, relocated, relocation, relocates
        pattern = r'\b(?:' + full + r'|' + stem + r'(?:ing|ed|es|ion|tion|ment|ance|ence|s|d))\b'
        return bool(re.search(pattern, text_lower))

    if phrase in text_lower:
        return True

    # Relaxed: all words present within 5 positions
    text_words = text_lower.split()
    word_positions = []
    for w in words:
        positions = [i for i, tw in enumerate(text_words) if w in tw]
        if not positions:
            return False
        word_positions.append(positions)

    if len(word_positions) >= 2:
        for p0 in word_positions[0]:
            for p1 in word_positions[-1]:
                if abs(p1 - p0) <= 5:
                    return True

    return False


def _find_phrase_position(phrase: str, text_words: list[str]) -> int:
    """Find the word index where a phrase starts in text. Returns -1 if not found."""
    phrase_words = phrase.split()
    if len(phrase_words) == 1:
        for i, w in enumerate(text_words):
            if phrase_words[0] in w:
                return i
    else:
        phrase_joined = " ".join(phrase_words)
        text_joined = " ".join(text_words)
        char_pos = text_joined.find(phrase_joined)
        if char_pos >= 0:
            return text_joined[:char_pos].count(" ")
        # Relaxed: find first word
        for i, w in enumerate(text_words):
            if phrase_words[0] in w:
                return i
    return -1


def _is_negated(pos: int, text_words: list[str]) -> bool:
    """Check if the word at `pos` is negated by a preceding word."""
    start = max(0, pos - 4)
    preceding = text_words[start:pos]
    for w in preceding:
        # Check for negation words and contractions
        if w in _NEGATION_WORDS:
            return True
        if w.endswith("n't") or w.endswith("not"):
            return True
    return False


# Subordinate clause markers — content after these describes context/others,
# not the speaker's own intended action.
_SUBORDINATE_MARKERS = {
    "given that", "since", "because", "although", "whereas",
    "while", "even though", "despite", "considering that",
    "in light of", "acknowledging that", "recognizing that",
    "noting that",
}


def _is_in_subordinate_clause(pos: int, text_lower: str, text_words: list[str]) -> bool:
    """Check if the keyword is inside a subordinate/background clause.

    Finds the most recent subordinate marker before `pos` and the most
    recent main-clause boundary (period, comma before 'I/we'). If the
    marker is more recent, the keyword is in a subordinate clause.
    """
    # Find character position of the word at `pos`
    preceding_text = " ".join(text_words[:pos])

    # Check for subordinate markers
    latest_marker_pos = -1
    for marker in _SUBORDINATE_MARKERS:
        idx = preceding_text.rfind(marker)
        if idx > latest_marker_pos:
            latest_marker_pos = idx

    if latest_marker_pos < 0:
        return False

    # Check for main-clause restart after the marker
    after_marker = preceding_text[latest_marker_pos:]
    for first_person in _FIRST_PERSON:
        fp_pos = after_marker.find(f" {first_person} ")
        if fp_pos > 0:
            # First person appears after the marker but before our keyword
            # This means: "given that X, I will Y" — keyword after "I" is main clause
            return False

    return True


def _is_third_person_only(pos: int, text_lower: str, text_words: list[str]) -> bool:
    """Check if the keyword describes someone else's action.

    Two signals:
      1. Third-person pronouns nearby, no first-person pronouns
      2. Keyword is inside a subordinate/background clause
    """
    start = max(0, pos - 6)
    end = min(len(text_words), pos + 6)
    window = text_words[start:end]

    has_first = any(w in _FIRST_PERSON for w in window)
    has_third = any(w in _THIRD_PERSON for w in window)

    if has_third and not has_first:
        return True

    if _is_in_subordinate_clause(pos, text_lower, text_words):
        return True

    return False


def _is_hypothetical(pos: int, text_words: list[str]) -> bool:
    """Check if the keyword is in a question or hypothetical context."""
    start = max(0, pos - 3)
    preceding = text_words[start:pos]

    for w in preceding:
        if w in _HYPOTHETICAL:
            return True
        if w.endswith("?"):
            return True

    # Check if the sentence containing this keyword ends with '?'
    end = min(len(text_words), pos + 10)
    following = text_words[pos:end]
    for w in following:
        if "?" in w:
            return True
        if w in (".", "!", ";"):
            break

    return False


# Maximum raw score possible per type (sum of all weights).
# Precomputed to normalize scores and prevent keyword-count bias.
_MAX_POSSIBLE_SCORE: dict[ActionType, float] = {
    action_type: sum(w for _, w in phrases)
    for action_type, phrases in _KEYWORD_SCORES.items()
}


def _score_text(text_lower: str) -> dict[ActionType, float]:
    """Score text against all action types using keyword matching with negation/hypothetical filtering.

    Normalizes raw scores by each type's max possible score to prevent keyword-count bias."""
    text_words = text_lower.split()
    raw_scores: dict[ActionType, float] = {}

    for action_type, phrases in _KEYWORD_SCORES.items():
        total = 0.0
        for phrase, weight in phrases:
            if not _phrase_in_text(phrase, text_lower):
                continue

            pos = _find_phrase_position(phrase, text_words)
            if pos < 0:
                total += weight
                continue

            effective_weight = weight

            if _is_negated(pos, text_words):
                effective_weight = 0.0
            elif _is_third_person_only(pos, text_lower, text_words):
                effective_weight *= 0.3
            elif _is_hypothetical(pos, text_words):
                effective_weight *= 0.5

            total += effective_weight

        if total > 0:
            raw_scores[action_type] = total

    # Normalize: convert raw scores to percentage of max possible.
    # This ensures lobby (27 keywords) and evade (13 keywords) are
    # compared fairly — a 6/60 lobby score doesn't beat a 5/30 evade score.
    normalized: dict[ActionType, float] = {}
    for action_type, raw in raw_scores.items():
        max_possible = _MAX_POSSIBLE_SCORE.get(action_type, 1.0)
        normalized[action_type] = raw / max_possible * 10.0  # scale to 0-10

    return normalized


def classify_with_scoring(
    text: str,
    actor_name: str,
) -> GovernanceAction:
    """Classify by keyword scoring; picks highest-scoring type."""
    text_lower = text.lower()
    scores = _score_text(text_lower)

    if not scores:
        if text.strip() and len(text.strip()) > 10:
            return GovernanceAction(
                action_type=ActionType.OTHER,
                actor=actor_name,
                description=text,
                classification_method="scoring_no_match",
            )
        return GovernanceAction(
            action_type=ActionType.DO_NOTHING,
            actor=actor_name,
            description=text,
            classification_method="scoring_empty",
        )

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_type, best_score = ranked[0]

    # Collect secondary actions (score > 2) for metadata
    secondary = [
        t.value for t, s in ranked[1:]
        if s >= 2.0
    ]

    return GovernanceAction(
        action_type=best_type,
        actor=actor_name,
        description=text,
        classification_method=f"scoring({best_score:.0f})",
        metadata={"secondary_actions": secondary} if secondary else {},
    )


# Backward-compatible name
classify_with_keywords = classify_with_scoring


# ── Multi-action extraction ──────────────────────────────────────


def extract_all_actions(
    text: str,
    actor_name: str,
) -> list[GovernanceAction]:
    """Extract ALL actions from text, not just the primary one.

    Returns a list sorted by score, highest first.
    Use this when you want to know everything an agent intends to do.
    """
    text_lower = text.lower()
    scores = _score_text(text_lower)

    if not scores:
        return [GovernanceAction(
            action_type=ActionType.OTHER,
            actor=actor_name,
            description=text,
            classification_method="multi_no_match",
        )]

    actions = []
    for action_type, score in sorted(scores.items(), key=lambda x: -x[1]):
        if score >= 2.0:  # minimum threshold for a real action
            actions.append(GovernanceAction(
                action_type=action_type,
                actor=actor_name,
                description=text,
                classification_method=f"multi_scoring({score:.0f})",
            ))

    return actions if actions else [GovernanceAction(
        action_type=ActionType.OTHER,
        actor=actor_name,
        description=text,
        classification_method="multi_below_threshold",
    )]


# ── Main entry point ─────────────────────────────────────────────


_REVALIDATION_PROMPT = """\
An AI agent in a governance simulation produced this text as their action:
"{text}"

A parser classified it as: "{parsed_type}"

What is the SINGLE action this agent is actually taking?
Ignore what they say about OTHER agents. Focus only on what THIS agent does.

Types: propose_policy, lobby, comply, evade, relocate, form_coalition, \
public_statement, enforce, investigate, legal_challenge, publish_report, \
do_nothing, other

Reply with ONLY the correct type. One word. Nothing else."""


def _revalidate_action(
    action: GovernanceAction,
    text: str,
    model: lm_lib.LanguageModel,
) -> GovernanceAction:
    """Ask the LLM to confirm or correct an action classification.

    Called inline during simulation (not post-hoc) so the resolution
    engine uses the corrected type immediately.
    """
    prompt = _REVALIDATION_PROMPT.format(
        text=text[:400],
        parsed_type=action.action_type.value,
    )

    try:
        result = model.sample_text(prompt, max_tokens=20, temperature=0.0)
        result = result.strip().strip('"').strip("'").lower()
        normalized = result.replace(" ", "_").replace("-", "_")

        for at in ActionType:
            if at.value == normalized or at.value in normalized:
                # If revalidation says "other" — that's not helpful.
                # Keep the original classification which at least tried.
                if at == ActionType.OTHER:
                    action.classification_method += "+reval_kept"
                    return action

                if at != action.action_type:
                    # LLM disagrees — use the corrected type
                    return GovernanceAction(
                        action_type=at,
                        actor=action.actor,
                        target=action.target,
                        description=action.description,
                        resource_cost=action.resource_cost,
                        metadata={
                            **action.metadata,
                            "revalidated_from": action.action_type.value,
                            "original_method": action.classification_method,
                        },
                        classification_method="revalidated",
                    )
                else:
                    # LLM agrees — mark as confirmed
                    action.classification_method += "+confirmed"
                    return action
    except Exception:
        pass

    return action


def parse_action(
    text: str,
    actor_name: str,
    model: lm_lib.LanguageModel | None = None,
    revalidate: bool = False,
) -> GovernanceAction:
    """Parse raw LLM output into a GovernanceAction using a 3-tier fallback: structured JSON, then LLM classification, then keyword scoring.

    Optionally runs a revalidation pass (tier 4) where the LLM confirms or corrects the chosen classification before resolution.
    """
    if not text or not text.strip():
        return GovernanceAction(
            action_type=ActionType.DO_NOTHING,
            actor=actor_name,
            description="(no action)",
            classification_method="empty",
        )

    # Tier 1: structured JSON
    structured = parse_structured_response(text, actor_name)
    if structured is not None:
        # JSON parse is high confidence — skip revalidation
        return structured

    # Tier 2: LLM classification
    if model is not None:
        result = classify_with_llm(text, actor_name, model)
    else:
        # Tier 3: score-based keywords
        result = classify_with_scoring(text, actor_name)

    # Tier 4 (opt-in): revalidation confirmation
    if revalidate and model is not None:
        result = _revalidate_action(result, text, model)

    return result


def parse_action_from_text(text: str, actor_name: str) -> GovernanceAction:
    """Backward-compatible wrapper. Uses keyword scoring only (no LLM)."""
    return parse_action(text, actor_name, model=None)
