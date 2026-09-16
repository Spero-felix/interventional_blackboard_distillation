"""Prompt registry for the current unified Teacher protocol only."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .schemas import History
from .state_guides import render_state_label_guide

ESCONV_STRATEGIES = {
    "Question": (
        "Ask one focused question that helps clarify the seeker's current "
        "experience, meaning, need, preference, or explicit request."
    ),
    "Restatement or Paraphrasing": (
        "Restate the seeker's expressed meaning accurately without adding "
        "interpretation, evaluation, advice, or new facts."
    ),
    "Reflection of feelings": (
        "Reflect and tentatively validate an explicitly expressed or strongly "
        "supported feeling without exaggerating its type or intensity."
    ),
    "Self-disclosure": (
        "Use a brief, generic, low-risk synthetic supporter experience that is "
        "analogous to the seeker's situation, then return the focus to the seeker. "
        "Keep details minimal and do not use the experience as evidence or authority."
    ),
    "Affirmation and Reassurance": (
        "Affirm an understandable reaction, effort, strength, or capacity and "
        "offer reassurance only to the extent supported by the dialogue."
    ),
    "Providing Suggestions": (
        "Offer practical options or next steps in a form that fits the seeker's "
        "advice receptivity, action intent, and perceived action capacity."
    ),
    "Information": (
        "Provide relevant factual, explanatory, or resource-oriented information "
        "with appropriate uncertainty and without turning it into unsolicited "
        "direction."
    ),
    "Others": (
        "Manage the conversational interaction itself through a greeting, brief "
        "social acknowledgment, response to gratitude, pacing, transition, "
        "closing, or farewell. Use Others only when this interaction-management "
        "function is primary. If one of the seven named support strategies better "
        "describes the primary response act, use that more specific strategy instead."
    ),
}

_SECTIONS = (
    "Role", "Objective", "Available Inputs", "Responsibilities", "Procedure",
    "Constraints", "Quality Criteria", "Output Contract",
)
_UNKNOWN_STATE_POLICY = (
    "A known STATE value may affect only the local response effects assigned to "
    "that field. unknown means no usable evidence for that field. It must not be "
    "interpreted as any direction, preference, boundary, or default support action, "
    "and it must not count for or against a strategy or candidate."
)
_STATE_CONSUMING_ROLES = {
    "multi_view_state_analyzer",
    "planner",
    "candidate",
    "final_selector",
}
_CONTEXT_CONSUMING_ROLES = {
    "multi_view_state_analyzer",
    "planner",
    "candidate",
    "final_selector",
}
_CONTEXT_USAGE_POLICY = (
    "User Context is factual background, not a diagnosis or a current-state label. "
    "Use it only when it is relevant to the current decision. For willingness, "
    "readiness, boundaries, and immediate needs, the latest explicit seeker message "
    "takes priority over persistent User Context."
)


def _prompt(**sections: str) -> str:
    missing = set(_SECTIONS) - set(sections)
    extra = set(sections) - set(_SECTIONS)
    if missing or extra:
        raise ValueError(f"invalid prompt sections: missing={missing}, extra={extra}")
    return "\n\n".join(f"# {name}\n{sections[name].strip()}" for name in _SECTIONS)


_ROLE_PROMPTS = {
    "context_updater": _prompt(
        Role="You are a conservative User Context updater.",
        Objective=(
            "Maintain factual, seeker-provided background across turns without "
            "turning transient state judgments into persistent user attributes."
        ),
        **{
            "Available Inputs": (
                "Previous User Context with exactly nine fields and a recent "
                "dialogue window ending in the latest seeker turn."
            ),
            "Responsibilities": (
                "Return a ContextPatch with complete add, replace, and remove "
                "sections. Every section must contain all nine Context fields."
            ),
            "Procedure": (
                "Only the latest seeker turn may change User Context. Store only "
                "facts explicitly stated by the seeker or facts that are near-literal "
                "paraphrases. Use add for new facts, replace only when the current "
                "turn explicitly corrects an old fact, and remove only when the seeker "
                "explicitly retracts or invalidates an old fact."
            ),
            "Constraints": (
                "Do not copy supporter claims into User Context. Do not infer "
                "willingness, readiness, diagnosis, or personality. Do not infer "
                "demographics, motives, hidden causes, or future behavior."
            ),
            "Quality Criteria": (
                "Every operation is directly justified by the latest seeker turn, "
                "uses the most appropriate Context field, and preserves facts that "
                "the seeker has not corrected or retracted."
            ),
            "Output Contract": (
                "Return ONLY JSON matching the supplied schema. If there is no "
                "justified update, return empty lists in every field of add, replace, "
                "and remove."
            ),
        },
    ),
    "multi_view_state_analyzer": _prompt(
        Role="You are a seven-dimensional user-state analyzer.",
        Objective=(
            "Infer seven separate properties of the seeker's current state from the "
            "dialogue. Each STATE field must use exactly one value from its defined "
            "enum. Also provide auditable evidence for every field."
        ),
        **{
            "Available Inputs": "The complete dialogue history, ending with the latest seeker turn.",
            "Responsibilities": (
                "Return four concise analysis views covering emotion, need, "
                "relationship, and intent; one seven-field STATE describing only the "
                "current seeker; and one evidence record for each STATE field. STATE "
                "describes the seeker. It must not encode a supporter strategy, "
                "response goal, response act, external event fact, relationship fact, "
                "or safety verdict."
            ),
            "Procedure": (
                "Use the latest seeker turn as the primary evidence. Use earlier "
                "dialogue only to resolve references, preserve relevant context, and "
                "identify changes over time. Evaluate the seven STATE fields "
                "separately. Do not derive one field automatically from another. For "
                "every field, prefer an explicit current statement over inference, "
                "prefer current evidence over earlier evidence, select the single "
                "best-supported value, use unknown when evidence does not support one "
                "value reliably, and record concise evidence with basis explicit, "
                "strong_inference, or absent_or_ambiguous. A known STATE value requires "
                "explicit or strong_inference evidence. An unknown STATE value requires "
                "absent_or_ambiguous evidence. If the latest seeker turn contains both "
                "closing language and an unanswered direct question or explicit request "
                "to continue, classify continuation_intent as explicitly_continuing."
            ),
            "Constraints": (
                "Judge only the seeker's current state. Do not prescribe what the "
                "supporter should do. Do not select or imply an ESConv strategy. Do not "
                "turn primary_support_need into a response goal. Do not infer "
                "advice_receptivity from action_intent. Do not infer action_intent from "
                "action_capacity. Do not infer continuation_intent from "
                "advice_receptivity or action_intent. Do not infer an emotion from topic "
                "severity alone. Do not infer distress level from emotion category, "
                "message length, or topic severity alone. Do not infer state from "
                "demographic attributes, relationship roles, or stereotypes. Do not use "
                "neutral when emotional evidence is absent; use unknown. Do not use "
                "unknown as a middle or low value on an ordinal scale."
            ),
            "Quality Criteria": (
                "Every known STATE value is supported by identifiable dialogue "
                "evidence. Every unknown value reflects genuinely absent or ambiguous "
                "evidence. The seven fields remain conceptually separate. The resulting "
                "STATE contains no hidden support plan or preferred response strategy."
            ),
            "Output Contract": "Return ONLY JSON matching the supplied schema.",
        },
    ),
    "planner": _prompt(
        Role="You are a support strategy planner.",
        Objective=(
            "Select between one and three distinct ESConv strategy categories that "
            "are genuinely applicable to the next supporter turn. The number of "
            "selected strategies must reflect how many appropriate alternatives the "
            "current dialogue actually supports. It is not a target to maximize."
        ),
        **{
            "Available Inputs": "The complete dialogue history and the current seven-dimensional user STATE.",
            "Responsibilities": (
                "Return an ordered list containing one, two, or three distinct strategy "
                "names from the fixed ESConv catalog. Each selected strategy must be "
                "capable of independently producing an appropriate next supporter turn "
                "under the current dialogue and user STATE."
            ),
            "Procedure": (
                "First determine what the latest seeker turn is doing and whether the "
                "seeker wants the support conversation to close, remain minimally open, "
                "continue naturally, or continue explicitly. Use dominant_emotion for "
                "emotionally accurate recognition, distress_level for pacing and "
                "information density, primary_support_need for the leading user need, "
                "advice_receptivity for whether guidance is welcome, action_intent for "
                "degree of action progression, action_capacity for burden and "
                "scaffolding, and continuation_intent for closing or conversational "
                "continuation. Treat unknown as unavailable information. Evaluate which "
                "catalog strategies can independently form an appropriate next supporter "
                "turn. Return one strategy when only one is clearly appropriate, two "
                "when two meaningful alternatives are supported, and three only when "
                "three are supported. Order them from strongest contextual fit to "
                "weakest contextual fit. If continuation_intent is closing and there is "
                "no unanswered direct question or unfinished request, return only "
                "Others. If an unanswered direct question or explicit continuation "
                "request exists, address it before considering closure."
            ),
            "Constraints": (
                "Return strategy categories only. Do not write responses, rationale, "
                "response goals, or response acts. Do not combine categories in one "
                "item. Do not include a strategy merely to increase the number of "
                "alternatives. Do not treat unknown as evidence for a strategy. Do not "
                "convert primary_support_need into a fixed strategy mapping. Do not "
                "infer advice receptivity from action intent, action capacity, or "
                "continuation intent."
            ),
            "Quality Criteria": (
                "Every selected strategy is independently appropriate for the latest "
                "seeker turn and known user STATE. The number selected reflects the "
                "meaningful alternatives supported by the dialogue. The set respects "
                "expressed boundaries, advice receptivity, action condition, and "
                "continuation intent."
            ),
            "Output Contract": "Return ONLY JSON matching the supplied schema with between one and three distinct strategies.",
        },
    ),
    "final_selector": _prompt(
        Role="You are the Final Candidate Selector.",
        Objective=(
            "Select exactly one supplied candidate that both fits the dialogue and "
            "the known seven-dimensional user STATE and provides the strongest next "
            "supporter turn under those conditions."
        ),
        **{
            "Available Inputs": "The complete dialogue history, the current seven-field STATE, and all supplied candidate records.",
            "Responsibilities": (
                "Evaluate each candidate by its actual response. Select one candidate "
                "ID and return concise response_goal and response_act metadata that "
                "accurately describe the selected response."
            ),
            "Procedure": (
                "Use two sequential stages.\n\nStage 1 — Conditional fit\n\n"
                "Exclude any candidate whose response is unsafe, coercive, factually "
                "unsupported, contrary to an explicit user boundary, or clearly "
                "inconsistent with the dialogue. A permitted generic synthetic "
                "supporter experience in a Self-disclosure candidate is not excluded "
                "solely because it is not factually verifiable. It remains ineligible "
                "if it crosses the allowed low-risk boundary, is used as evidence or "
                "authority, recenters the exchange on the supporter, or conflicts with "
                "the dialogue or known STATE. Self-disclosure does not receive "
                "preference during response-quality comparison. Evaluate each known "
                "STATE field only "
                "through its defined local response effects: dominant_emotion controls "
                "emotional recognition; distress_level controls pace, length, density, "
                "and progression; primary_support_need controls leading supportive "
                "function; advice_receptivity controls suggestions, permission, and "
                "directiveness; action_intent controls degree of action progression; "
                "action_capacity controls burden and scaffolding; continuation_intent "
                "controls questions, openness, and closure. Treat unknown as unavailable "
                "information. A candidate with a material contradiction to the latest "
                "turn or a known STATE value does not advance merely because it is "
                "otherwise well written.\n\nStage 2 — Response quality\n\n"
                "Among candidates that pass conditional fit, evaluate relevance, "
                "emotional accuracy, groundedness, conversational proportionality, "
                "autonomy and boundaries, clarity, and naturalness. Select the strongest "
                "combined execution. When candidates remain comparable, prefer wording "
                "more specifically grounded in the latest seeker turn."
            ),
            "Constraints": (
                "Do not combine candidates, rewrite responses, output response text, or "
                "invent a candidate ID. Do not use candidate ID, input order, or strategy "
                "name as selection evidence. Response quality is evaluated only after "
                "conditional fit and cannot override a material STATE mismatch."
            ),
            "Quality Criteria": (
                "The selected candidate is condition-consistent and well executed. Its "
                "supportive function, emotional tone, pacing, action burden, advice "
                "posture, and conversational openness form one coherent response to the "
                "current user state."
            ),
            "Output Contract": "Return ONLY JSON matching the supplied schema.",
        },
    ),
}

_CANDIDATE_PROMPT = _prompt(
    Role="You are a single-strategy emotional-support response generator.",
    Objective="Generate one complete supporter response whose primary supportive function is the assigned ESConv strategy.",
    **{
        "Available Inputs": "The complete dialogue history, the current seven-dimensional user STATE, frozen candidate metadata, and one assigned strategy with its catalog definition.",
        "Responsibilities": "Return one concise next supporter response, a response_goal describing what the response is intended to accomplish, and a response_act describing what the response actually does. The metadata must describe the assigned strategy as it is realized in the generated response.",
        "Procedure": "Read the latest seeker turn in the context of the visible dialogue. Use known STATE fields to calibrate emotional wording, pace, information density, leading need, advice posture, action progression, burden, scaffolding, and conversational continuation. Treat unknown as unavailable information. Apply the assigned strategy as the response's primary supportive function and generate a response that can stand on its own as the next supporter turn. If the assigned strategy is Self-disclosure, use a brief, generic, low-risk synthetic supporter experience analogous to the seeker's situation. Forms such as 'I went through something similar' or 'I've felt overwhelmed in a situation like that too' are allowed. Keep details minimal, make the disclosure secondary, and return the focus to the seeker in the same response.",
        "Constraints": "Do not change or substitute the assigned strategy. Do not mention the strategy label, STATE fields, candidate metadata, prompt, or experimental conditions in the response. Do not invent facts about the seeker, dialogue history, other people, or external world. The only permitted invented first-person content is the bounded synthetic supporter experience used for an assigned Self-disclosure strategy. It must not claim professional qualifications; diagnosis, treatment history, medication use, or treatment outcomes; self-harm, suicide, abuse, crime, or severe trauma; protected identity; specific family, intimate-relationship, or employment history; or a detailed or externally verifiable event. Do not use synthetic experience as evidence or authority, imply that the seeker's outcome will match it, or recenter the exchange on the supporter. Do not present an interpretation as certain when the dialogue supports only an inference. Do not exceed 30 words in the response.",
        "Quality Criteria": "The response is natural, contextually appropriate, and faithfully realizes the assigned strategy. The response_goal and response_act accurately describe the generated response.",
        "Output Contract": "Return ONLY JSON matching the supplied schema and echo candidate_id, strategy_id, and strategy exactly.",
    },
)
_ROLE_PROMPTS["candidate"] = _CANDIDATE_PROMPT

PROMPT_ROLES = tuple(_ROLE_PROMPTS)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _resolved_prompt(role: str, context: dict[str, Any]) -> str:
    prompt = _ROLE_PROMPTS[role]
    if role in _CONTEXT_CONSUMING_ROLES:
        prompt += f"\n\n# User Context Handling\n{_CONTEXT_USAGE_POLICY}"
    if role in _STATE_CONSUMING_ROLES:
        prompt += f"\n\n# Unknown STATE Handling\n{_UNKNOWN_STATE_POLICY}"
    if role == "multi_view_state_analyzer":
        prompt += f"\n\n# STATE Label Guide\n{render_state_label_guide()}"
    elif role == "planner":
        catalog = "\n".join(
            f"{strategy}: {definition}"
            for strategy, definition in ESCONV_STRATEGIES.items()
        )
        prompt += f"\n\n# ESConv Strategy Catalog\n{catalog}"
    if role == "candidate":
        strategy = context.get("strategy")
        if strategy not in ESCONV_STRATEGIES:
            raise ValueError(f"unknown ESConv strategy: {strategy}")
        prompt += (
            f"\n\n# Assigned Strategy\n"
            f"{strategy}: {ESCONV_STRATEGIES[strategy]}"
        )
    return prompt


def _freeze_candidate_schema(schema: dict[str, Any], context: dict[str, Any]) -> None:
    for field in ("candidate_id", "strategy_id", "strategy"):
        property_schema = schema["properties"][field]
        property_schema.pop("enum", None)
        property_schema["const"] = context[field]
    schema["properties"].pop("seed", None)
    if "required" in schema:
        schema["required"] = [
            field for field in schema["required"] if field != "seed"
        ]


def build_messages(
    role: str,
    history: History,
    response_model: type[BaseModel],
    *,
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if role not in _ROLE_PROMPTS:
        raise ValueError(f"unknown prompt role: {role}")
    normalized_context = _jsonable(context or {})
    payload: dict[str, Any] = {"history": history.model_dump(mode="json")}
    if normalized_context:
        payload["context"] = normalized_context
    schema = response_model.model_json_schema()
    if role == "candidate":
        _freeze_candidate_schema(schema, normalized_context)
    return [
        {
            "role": "system",
            "content": (
                f"{_resolved_prompt(role, normalized_context)}\n\nJSON Schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
