"""Prompt registry for the current unified Teacher protocol only."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .schemas import History

ESCONV_STRATEGIES = {
    "Question": "Ask one focused question that helps clarify a feeling, need, preference, or next step.",
    "Restatement or Paraphrasing": "Restate the seeker's meaning accurately without adding interpretation.",
    "Reflection of feelings": "Reflect and tentatively validate the seeker's expressed or strongly implied feeling.",
    "Self-disclosure": "Use a brief relatable first-person statement while keeping focus on the seeker.",
    "Affirmation and Reassurance": "Validate understandable reactions or strengths with grounded reassurance.",
    "Providing Suggestions": "Offer practical options as choices that preserve autonomy and match readiness.",
    "Information": "Provide relevant factual or explanatory information without unsupported certainty.",
    "Others": "Use a supportive act that does not fit the seven named categories.",
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
        Role="You are one unified Multi-view State Analyzer.",
        Objective="Analyze emotion, need, relationship, and intent in one call, then produce one compact seven-field STATE.",
        **{
            "Available Inputs": "The complete dialogue history.",
            "Responsibilities": "Give each view a concise summary, direct dialogue evidence, optional uncertainty, and produce the downstream STATE.",
            "Procedure": "Read the full history; analyze the four views independently; reconcile them only when building STATE.",
            "Constraints": "Do not diagnose, invent facts, prescribe a strategy, or duplicate text across STATE fields. Each STATE value has at most 20 words.",
            "Quality Criteria": "The four views remain auditable and the compact STATE is sufficient for planning.",
            "Output Contract": "Return ONLY JSON matching the supplied schema.",
        },
    ),
    "planner": _prompt(
        Role="You are a support strategy planner.",
        Objective="Choose exactly three distinct ESConv strategy categories for three alternative single-strategy responses.",
        **{
            "Available Inputs": "Dialogue history and compact STATE.",
            "Responsibilities": "Return three ordered, functionally distinct strategy names from the fixed catalog.",
            "Procedure": "Identify the immediate support goal and select three plausible but different response approaches.",
            "Constraints": "Return categories only. Do not write responses, rationale, or combine categories.",
            "Quality Criteria": "Each strategy could independently produce a complete next supporter turn.",
            "Output Contract": "Return ONLY JSON matching the supplied schema with exactly three distinct strategies.",
        },
    ),
    "final_selector": _prompt(
        Role="You are the Final Candidate Selector.",
        Objective="Choose exactly one supplied candidate as the final answer.",
        **{
            "Available Inputs": "Dialogue history, compact STATE, and all three candidate records.",
            "Responsibilities": "Return one candidate ID and concise response_goal/response_act metadata for the selected candidate.",
            "Procedure": (
                "Evaluate every candidate by its actual candidate response, not merely "
                "its strategy label or metadata. First exclude any candidate that is "
                "unsafe, factually unsupported, coercive, or clearly inappropriate for "
                "the dialogue. Then choose the candidate that best addresses the "
                "seeker's latest turn while fitting STATE.primary_need, "
                "STATE.support_goal, STATE.readiness, and STATE.main_constraint. "
                "Prioritize, in order: (1) Direct fit to the seeker's latest turn and "
                "immediate need; (2) Emotional attunement and respect for the seeker's "
                "autonomy; (3) useful, concrete support appropriate to the seeker's "
                "readiness; (4) factual restraint and absence of unsupported "
                "assumptions; (5) completeness as a standalone next supporter turn. "
                "When the seeker is ready for action or explicitly requests guidance, "
                "prefer a concrete, autonomy-preserving response. When distress is high "
                "or readiness is low or unclear, prefer a response that first validates, "
                "reflects, or gently explores. Apply these as contextual preferences, "
                "not fixed strategy rules. If candidates are otherwise comparable, "
                "prefer the one that is more specific to the dialogue, less generic, "
                "less repetitive, and less assumptive. Do not favor a candidate because "
                "of its ID, order, or strategy name."
            ),
            "Constraints": "Do not combine candidates, rewrite responses, output response text, or invent an ID.",
            "Quality Criteria": (
                "The selected response is the most helpful next supporter turn for this "
                "specific seeker at this moment: emotionally appropriate, grounded in "
                "the dialogue, appropriately actionable, respectful of autonomy, and "
                "complete on its own. response_goal and response_act must accurately "
                "describe the selected response and must not introduce another strategy."
            ),
            "Output Contract": "Return ONLY JSON matching the supplied schema.",
        },
    ),
    "state_counterfactual_generator": _prompt(
        Role="You are a controlled STATE counterfactual editor.",
        Objective="Replace exactly one requested STATE field with a plausible contrasting value.",
        **{
            "Available Inputs": "Dialogue, current STATE, target field, original value, and target dimension.",
            "Responsibilities": "Produce one compact replacement that differs meaningfully while leaving all other fields untouched.",
            "Procedure": "Interpret the field's support-side function and write one alternative value of at most 20 words.",
            "Constraints": "Do not use <MASKED>, copy the original, edit another field, or invent dialogue facts.",
            "Quality Criteria": "The replacement should cause a localized, observable change in downstream support behavior.",
            "Output Contract": "Return ONLY JSON matching the supplied schema with replacement only.",
        },
    ),
    "condition_effect_verifier": _prompt(
        Role="You are a blind PLAN conditional-effect verifier.",
        Objective="Verify that each response fits its own PLAN and that changing PLAN causes a meaningful strategy-consistent response change.",
        **{
            "Available Inputs": "Dialogue plus two complete PLAN-response bundles A and B.",
            "Responsibilities": "Judge A only against A and B only against B; treat both as valid controls.",
            "Procedure": "Assess both fits independently, identify the controlled difference, then repeat after the complete bundles are swapped.",
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
            "Responsibilities": "Judge each response against its own condition and record auxiliary affected dimensions for audit.",
            "Procedure": "Evaluate A and B independently, identify the field-controlled response difference, then repeat after swapping whole bundles.",
            "Constraints": "Treat both conditions as valid experimental controls. Do not decide which is factual, preferred, better, or degraded.",
            "Quality Criteria": "The verdict is field-localized, conservative, and order invariant.",
            "Output Contract": "Return ONLY JSON matching the supplied schema.",
        },
    ),
    "safety_verifier": _prompt(
        Role="You are a standalone offline safety verifier.",
        Objective="Check whether both conditional responses are safe enough to retain for training.",
        **{
            "Available Inputs": "Dialogue, original-conditioned response, and counterfactual-conditioned response.",
            "Responsibilities": "Flag concrete harmful instructions, coercion, serious factual danger, or unsafe boundary crossing.",
            "Procedure": "Assess each response independently and provide concise evidence.",
            "Constraints": "Do not rank quality and do not reject ordinary emotional language merely for discussing distress.",
            "Quality Criteria": "Use high precision and preserve valid alternative responses.",
            "Output Contract": "Return ONLY JSON matching the supplied schema.",
        },
    ),
}

_CANDIDATE_PROMPT = _prompt(
    Role="You are a single-strategy emotional-support response generator.",
    Objective="Generate one complete supporter response using only the assigned strategy.",
    **{
        "Available Inputs": "Dialogue, compact STATE, frozen candidate metadata, and one assigned strategy with its catalog definition.",
        "Responsibilities": "Write a concise response plus response_goal and response_act metadata describing only the assigned strategy.",
        "Procedure": "Read the latest seeker turn, apply the assigned strategy, and make the response usable on its own.",
        "Constraints": "Do not introduce another strategy, mention metadata, invent facts, or exceed 30 words in the response.",
        "Quality Criteria": "The response unmistakably realizes its assigned strategy and differs functionally from other candidates.",
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
    if role == "candidate":
        strategy = context.get("strategy")
        if strategy not in ESCONV_STRATEGIES:
            raise ValueError(f"unknown ESConv strategy: {strategy}")
        prompt += f"\n\n# Assigned Strategy\n{strategy}: {ESCONV_STRATEGIES[strategy]}"
        fixed_plan = context.get("fixed_plan")
        if fixed_plan is not None:
            if not isinstance(fixed_plan, dict):
                raise ValueError("fixed PLAN intervention requires plan fields")
            required = {"strategies", "response_goal", "response_act"}
            if not required.issubset(fixed_plan):
                raise ValueError("fixed PLAN intervention requires complete plan fields")
            prompt += (
                "\n\n# Experimental PLAN Clamp\n"
                "Treat the supplied fixed_plan selected strategy, response goal, and "
                "response act as authoritative. Use them to write the response and "
                "must not replan, substitute another goal, or substitute another act."
            )
    clamped_field = context.get("clamped_state_field")
    if clamped_field is not None and role in _STATE_CLAMP_ROLES:
        state = context.get("state")
        if not isinstance(state, dict) or clamped_field not in state:
            raise ValueError("clamped STATE intervention requires its STATE value")
        prompt += (
            "\n\n# Experimental STATE Clamp\n"
            f"Treat STATE.{clamped_field}={state[clamped_field]!r} as the authoritative "
            "experimental support-state value. Do not reconstruct it from dialogue. "
            "Keep dialogue facts and all other STATE fields unchanged."
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
