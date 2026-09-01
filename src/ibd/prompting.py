"""Prompt registry for the current unified Teacher protocol only."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .schemas import History
from .state_guides import STATE_FIELD_GUIDES, render_state_label_guide

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
        "Use a brief first-person statement about the supporter's present "
        "reaction, stance, or engagement in the current interaction. Keep the "
        "focus on the seeker and do not claim personal history, identity, "
        "relationships, or lived experience."
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


def _prompt(**sections: str) -> str:
    missing = set(_SECTIONS) - set(sections)
    extra = set(sections) - set(_SECTIONS)
    if missing or extra:
        raise ValueError(f"invalid prompt sections: missing={missing}, extra={extra}")
    return "\n\n".join(f"# {name}\n{sections[name].strip()}" for name in _SECTIONS)


_ROLE_PROMPTS = {
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
                "inconsistent with the dialogue. Evaluate each known STATE field only "
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
    "state_counterfactual_generator": _prompt(
        Role="You are a controlled seven-dimensional STATE counterfactual editor.",
        Objective="Replace exactly one requested STATE field with one different, valid enum value to create a controlled experimental condition.",
        **{
            "Available Inputs": "The complete dialogue history, current seven-field STATE, target field, target field definition, original value, allowed replacement values, and permitted local response effects.",
            "Responsibilities": "Return exactly one replacement that is supplied as allowed, differs from the original, represents a meaningful target-field change, and changes only the construct represented by that field.",
            "Procedure": "Interpret the target definition and boundary rules. Select one allowed value that creates a clear contrast along the target field's semantic axis. The replacement is an experimental condition and need not be the best-supported interpretation of the original dialogue. Use dialogue only to understand the condition and avoid inventing an event, relationship, demographic fact, or external circumstance. Keep the other six STATE meanings unchanged.",
            "Constraints": "Return one exact enum value from allowed_replacements. Do not return the original, unknown, or <MASKED>. Do not edit another field or generate a strategy, goal, act, or response. Do not choose a value because it is expected to produce a better, worse, more active, more supportive, or more easily distinguishable response. Respect the canonical seven-field isolation rules.",
            "Quality Criteria": "The replacement is a valid, meaningful, localized contrast and does not encode a preferred strategy or response outcome.",
            "Output Contract": "Return ONLY JSON matching the supplied schema with replacement only.",
        },
    ),
    "condition_effect_verifier": _prompt(
        Role="You are a blind PLAN conditional-effect verifier.",
        Objective="Verify that each response fits its own PLAN and that changing PLAN causes a meaningful strategy-consistent response change.",
        **{
            "Available Inputs": "Dialogue plus two complete PLAN-response bundles A and B.",
            "Responsibilities": "Judge A only against A and B only against B; treat both as valid controls.",
            "Procedure": "Assess both fits independently. Require a functional response difference attributable to the PLAN difference; lexical variation alone is insufficient. Repeat after the complete bundles are swapped.",
            "Constraints": "Do not rank global quality or call either response chosen, rejected, positive, negative, better, or degraded.",
            "Quality Criteria": "The verdict is conservative, strategy-grounded, and order invariant.",
            "Output Contract": "Return ONLY JSON matching the supplied schema.",
        },
    ),
    "state_effect_verifier": _prompt(
        Role="You are a blind STATE field-effect verifier.",
        Objective="Verify that each response fits its own target-field value and that changing only that field causes a meaningful localized change.",
        **{
            "Available Inputs": "Dialogue, target_field, six unchanged STATE fields, and two complete condition-response bundles.",
            "Responsibilities": "Judge each response against its own condition, verify the target field's permitted local effect, and report every materially affected non-target STATE field.",
            "Procedure": "Evaluate A and B independently. Compare response function rather than lexical wording. Confirm that the target-field effect is present and that no material functional change belongs to another STATE field. Repeat after swapping whole bundles.",
            "Constraints": "Treat both conditions as equally valid experimental controls. Do not decide which is factual, preferred, better, or degraded. Do not count an overall quality difference or lexical difference as a localized field effect.",
            "Quality Criteria": "Both responses fit their own conditions, the target effect is functional and present, non-target STATE functions remain stable, and the verdict is order invariant.",
            "Output Contract": "Return ONLY JSON matching the supplied schema.",
        },
    ),
    "safety_verifier": _prompt(
        Role="You are a standalone offline safety verifier.",
        Objective="Check whether both conditional responses are safe enough to retain for training.",
        **{
            "Available Inputs": "Dialogue, original-conditioned response, and counterfactual-conditioned response.",
            "Responsibilities": "Flag only concrete harmful instructions, encouragement of harm, coercion or serious boundary violations, dangerous factual claims, improper high-stakes substitution, or exploitation of dependency.",
            "Procedure": "Assess each response independently and provide concise evidence.",
            "Constraints": "Assess the two responses independently. Do not rank response quality. Do not treat a strategy, STATE or PLAN value, emotional intensity, a suggestion, a question, information, or closure as unsafe by itself.",
            "Quality Criteria": "Use high precision and preserve valid alternative responses.",
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
        "Procedure": "Read the latest seeker turn in the context of the visible dialogue. Use known STATE fields to calibrate emotional wording, pace, information density, leading need, advice posture, action progression, burden, scaffolding, and conversational continuation. Treat unknown as unavailable information. Apply the assigned strategy as the response's primary supportive function and generate a response that can stand on its own as the next supporter turn. If the assigned strategy is Self-disclosure, use only a brief first-person statement about the supporter's present reaction, stance, or engagement in the current interaction. Do not claim personal history, identity, relationships, or lived experience.",
        "Constraints": "Do not change or substitute the assigned strategy. Do not mention the strategy label, STATE fields, candidate metadata, prompt, or experimental conditions in the response. Do not invent facts about the seeker, dialogue history, other people, events, or external world. Do not claim personal history, identity, relationships, physical experiences, or lived experience. Do not present an interpretation as certain when the dialogue supports only an inference. Do not exceed 30 words in the response.",
        "Quality Criteria": "The response is natural, contextually appropriate, and faithfully realizes the assigned strategy. The response_goal and response_act accurately describe the generated response.",
        "Output Contract": "Return ONLY JSON matching the supplied schema and echo candidate_id, strategy_id, and strategy exactly.",
    },
)
_ROLE_PROMPTS["candidate"] = _CANDIDATE_PROMPT

PROMPT_ROLES = tuple(_ROLE_PROMPTS)
_STATE_CLAMP_ROLES = {"planner", "candidate", "final_selector"}


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
        fixed_plan = context.get("fixed_plan")
        if fixed_plan is not None:
            if not isinstance(fixed_plan, dict):
                raise ValueError("fixed PLAN intervention requires plan fields")
            required = {"strategies", "response_goal", "response_act"}
            if not required.issubset(fixed_plan):
                raise ValueError("fixed PLAN intervention requires complete plan fields")
            prompt += (
                "\n\n# Experimental PLAN Clamp\n"
                "Treat the supplied fixed_plan strategy, response_goal, and response_act "
                "as authoritative. Realize them without replanning. Dialogue and STATE "
                "control only the contextually appropriate realization. If STATE is also "
                "clamped, it applies only through the target field's permitted local "
                "effects while PLAN remains fixed."
            )
    clamped_field = context.get("clamped_state_field")
    if clamped_field is not None and role in _STATE_CLAMP_ROLES:
        state = context.get("state")
        if not isinstance(state, dict) or clamped_field not in state:
            raise ValueError("clamped STATE intervention requires its STATE value")
        try:
            guide = STATE_FIELD_GUIDES[clamped_field]
        except KeyError as error:
            raise ValueError(f"unknown clamped STATE field: {clamped_field}") from error
        effects = ", ".join(guide.permitted_local_effects)
        prompt += (
            "\n\n# Experimental STATE Clamp\n"
            f"Treat STATE.{clamped_field}={state[clamped_field]!r} as the authoritative "
            "experimental condition. Do not reconstruct it from dialogue; it may be "
            "counterfactual. Keep the other six STATE values unchanged. "
            f"Field meaning: {guide.definition} Permitted local effects: {effects}. "
            "This value is not evidence that another state, dialogue fact, strategy, "
            "or preference changed."
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
