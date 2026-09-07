"""Canonical meanings and local effects for the seven-dimensional STATE schema."""

from __future__ import annotations

from dataclasses import dataclass

from ibd.schemas import STATE_ANCHOR_FIELDS, StateField


@dataclass(frozen=True)
class StateFieldGuide:
    definition: str
    values: dict[str, str]
    isolation_rule: str
    permitted_local_effects: tuple[str, ...]
    prohibited_local_effects: tuple[str, ...]


STATE_FIELD_GUIDES: dict[StateField, StateFieldGuide] = {
    "dominant_emotion": StateFieldGuide(
        definition=(
            "The emotional family that is most dominant for the seeker now; it does "
            "not encode intensity, support need, or action capacity."
        ),
        values={
            "sadness_loss": "Loss, low mood, grief, or dejection is central; use hurt_disappointment when an unmet expectation is central.",
            "fear_anxiety": "Threat, worry, uncertainty, or anticipated danger is central; ordinary stress alone is insufficient.",
            "anger_frustration": "Obstruction, unfairness, offence, or outward blame is central; inward blame may indicate shame_guilt.",
            "shame_guilt": "Inadequacy, self-condemnation, responsibility, or regret is central; an ordinary mistake is insufficient.",
            "hurt_disappointment": "Being ignored, let down, betrayed, or having an expectation unmet is central.",
            "loneliness": "Lack of connection, companionship, or belonging is central; being alone is insufficient evidence.",
            "overwhelm": "Demands exceed the seeker's current felt ability to bear them; this is not the same as high distress.",
            "relief": "Relaxation follows a reduction in pressure or threat.",
            "hope_positive": "The seeker expresses positive expectation, confidence, or gladness about improvement; politeness is insufficient.",
            "neutral": "The seeker explicitly indicates calm or no notable emotional load; missing evidence is unknown, not neutral.",
            "mixed": "Two or more emotional families are equally dominant and no single one can reasonably be selected.",
            "other": "A clearly evidenced emotion is present but does not belong to another listed family.",
            "unknown": "There is insufficient evidence to identify the current dominant emotion.",
        },
        isolation_rule="Do not encode emotional intensity, support need, or action capacity in this field.",
        permitted_local_effects=("emotion recognition wording", "reflection wording", "validation wording"),
        prohibited_local_effects=("advice permission", "action steps"),
    ),
    "distress_level": StateFieldGuide(
        definition=(
            "How much distress currently occupies attention and limits reflection or "
            "regulation; it is not a measure of safety risk."
        ),
        values={
            "low": "Discomfort is present, but the seeker can narrate and reflect steadily without notable disruption.",
            "moderate": "Distress is clear and sustained, but the seeker can still express preferences, understand questions, or consider options.",
            "high": "Distress dominates the expression and clearly limits information processing, reflection, or action organization.",
            "unknown": "There is insufficient evidence to judge the level of distress.",
        },
        isolation_rule="Do not infer a specific emotion from this field; topic severity and message length alone do not determine distress.",
        permitted_local_effects=("pace", "length", "information density", "progression intensity"),
        prohibited_local_effects=("emotion category", "primary support need"),
    ),
    "primary_support_need": StateFieldGuide(
        definition=(
            "The foremost need the seeker wants this interaction to meet; it is not "
            "the supporter's next-response strategy. Prefer the latest explicit "
            "request, then an explicitly expressed lack, then a repeatedly expressed "
            "need; use unknown when no single need is supported."
        ),
        values={
            "validation": "The seeker primarily needs confirmation that a feeling or reaction is understandable.",
            "esteem_support": "The seeker primarily needs support for confidence, dignity, competence, or self-worth.",
            "sensemaking": "The seeker primarily wants to understand a reaction, problem structure, or meaning.",
            "information": "The seeker primarily needs facts, rules, resources, or explanatory knowledge.",
            "decision_support": "The seeker primarily needs to compare options, clarify preferences, or work through trade-offs.",
            "action_support": "The seeker primarily needs a next step, practice, plan, or execution help.",
            "connection": "The seeker primarily needs less isolation, greater belonging, or access to real-world support.",
            "unknown": "No single foremost support need can be reliably identified.",
        },
        isolation_rule="Do not encode a supporter strategy or response goal in this field.",
        permitted_local_effects=("which seeker need the response addresses first",),
        prohibited_local_effects=("the seeker's attitude toward advice",),
    ),
    "advice_receptivity": StateFieldGuide(
        definition=(
            "Whether the seeker currently welcomes external suggestions, options, or "
            "guidance; it does not indicate readiness or ability to act."
        ),
        values={
            "closed": "The seeker explicitly rejects advice or says they only want to be heard.",
            "hesitant": "The seeker expresses reservations, resistance, conditions, or concern about receiving advice.",
            "open": "The seeker explicitly welcomes ideas but has not directly requested advice.",
            "requested": "The seeker directly asks for advice, options, information, or steps.",
            "unknown": "The seeker has not expressed an attitude toward advice; continuing the conversation alone does not imply open.",
        },
        isolation_rule="Do not encode willingness to change or ability to act in this field.",
        permitted_local_effects=("whether to offer advice", "whether to ask permission first", "degree of directiveness"),
        prohibited_local_effects=("the seeker's action capacity",),
    ),
    "action_intent": StateFieldGuide(
        definition=(
            "Whether the seeker wants to change their situation, behavior, or relationship "
            "pattern; it does not indicate whether they can do so."
        ),
        values={
            "not_considering": "The seeker explicitly does not want change, chooses the status quo, or wants expression without considering action.",
            "ambivalent": "Genuine reasons for both change and maintaining the status quo are simultaneously present.",
            "considering": "The seeker is exploring the possibility of change without a settled intention.",
            "committed": "The seeker has decided or clearly intends to act but has not started.",
            "acting": "The seeker has begun or is continuing concrete action.",
            "unknown": "There is insufficient evidence about the seeker's intention to act.",
        },
        isolation_rule="Do not encode whether the seeker is able to act in this field.",
        permitted_local_effects=("exploration", "deliberation", "commitment", "action progression"),
        prohibited_local_effects=("whether advice is welcome",),
    ),
    "action_capacity": StateFieldGuide(
        definition=(
            "The seeker's current perceived confidence, energy, control, and usable "
            "resources; it is not an outside judgment of objective conditions."
        ),
        values={
            "blocked": "The seeker sees no feasible path or cannot carry even a minimal step.",
            "limited": "Some action is possible, but confidence, energy, skill, or resources are notably constrained.",
            "adequate": "The seeker sees at least one realistic next step and ordinary support is sufficient.",
            "strong": "The seeker shows clear ability, resources, and confidence and mainly needs confirmation, refinement, or selection.",
            "unknown": "The seeker has not expressed whether they can act.",
        },
        isolation_rule="Do not encode whether the seeker wants to change in this field.",
        permitted_local_effects=("step size", "scaffolding", "practical burden"),
        prohibited_local_effects=("whether the seeker wants change",),
    ),
    "continuation_intent": StateFieldGuide(
        definition=(
            "Whether and how actively the seeker wants the current support conversation "
            "to continue; it does not encode advice preference or real-world action "
            "intent. Check a pending question or explicit continuation request first, "
            "then explicit closure, then substantive new content, then a minimal reply."
        ),
        values={
            "closing": "The seeker explicitly ends the exchange, or gives only terminal thanks without an unfinished request.",
            "passive_open": "The seeker does not close but offers only a minimal response without elaboration or a request.",
            "engaged": "The seeker adds substantive content, answers seriously, or corrects an interpretation without explicitly requesting another turn.",
            "explicitly_continuing": "The seeker asks a pending question, requests further exploration, asks to continue, or says more remains to be shared.",
            "unknown": "The expression is incomplete or it is not possible to tell whether the seeker is ending or continuing.",
        },
        isolation_rule="Do not encode advice attitude or real-world action intent in this field.",
        permitted_local_effects=("whether to ask a question", "openness to another turn", "natural closure"),
        prohibited_local_effects=("real-world action progression",),
    ),
}


def render_state_label_guide() -> str:
    """Render the canonical label definitions and boundaries for analyzer prompts."""

    sections = [
        "General rules:",
        "- Describe the seeker now, not what the supporter should do next.",
        "- Prioritize the latest seeker turn and explicit statements over inference.",
        "- Use unknown only for insufficient evidence, never as a low or middle value.",
        "- Do not infer state from demographics, relationship roles, or event severity stereotypes.",
        "- Absence of an advice request does not imply closed.",
        "- Wanting to act but feeling unable is not ambivalent; combine committed with limited or blocked when supported.",
        "- A pending direct question or continuation request takes precedence over closing language.",
    ]
    for field in STATE_ANCHOR_FIELDS:
        guide = STATE_FIELD_GUIDES[field]
        sections.extend((f"\n{field}: {guide.definition}", "Allowed values:"))
        sections.extend(f"- {value}: {meaning}" for value, meaning in guide.values.items())
        sections.append(f"Isolation boundary: {guide.isolation_rule}")
        sections.append(
            "Permitted local response effects: "
            + ", ".join(guide.permitted_local_effects)
        )
        sections.append(
            "Do not use this field to change: "
            + ", ".join(guide.prohibited_local_effects)
        )
    return "\n".join(sections)
