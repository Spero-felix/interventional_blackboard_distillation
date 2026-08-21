"""Central registry for the complete Teacher prompt protocol."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .schemas import History

ESCONV_STRATEGIES = {
    "Question": "Ask for information or invite the seeker to explore a thought, feeling, need, or next step.",
    "Restatement or Paraphrasing": "Restate the seeker's situation or meaning in the supporter's own words without adding unsupported claims.",
    "Reflection of feelings": "Name or reflect the seeker's expressed or strongly implied feelings to communicate understanding.",
    "Self-disclosure": "Share a brief first-person perspective or relatable experience when it fits the dialogue and serves the seeker.",
    "Affirmation and Reassurance": "Validate strengths, efforts, worth, or understandable reactions and offer grounded reassurance.",
    "Providing Suggestions": "Offer practical, optional actions or ways of thinking while preserving the seeker's autonomy.",
    "Information": "Provide relevant factual or explanatory information that helps the seeker understand or act.",
    "Others": "Use a supportive response act that does not fit the seven named categories.",
}

_SECTIONS = (
    "Role",
    "Objective",
    "Available Inputs",
    "Responsibilities",
    "Procedure",
    "Constraints",
    "Quality Criteria",
    "Output Contract",
)

_STRATEGY_CATALOG = "\n".join(
    f"- {name}: {definition}" for name, definition in ESCONV_STRATEGIES.items()
)

_EXPERT_SPECS = {
    "emotion_expert": (
        "Emotion Analyst",
        "Identify the seeker's emotions, intensity, trajectory, and coping signals.",
        "emotion, intensity, trajectory, coping_signal",
    ),
    "need_expert": (
        "Need Analyst",
        "Identify the seeker's needs, priorities, readiness, and constraints.",
        "need, priority, readiness, constraint",
    ),
    "relationship_expert": (
        "Relationship Analyst",
        "Identify relationship type, recurring patterns, power dynamics, and boundary signals.",
        "relationship_type, relationship_pattern, power_dynamic, boundary_signal",
    ),
    "intent_expert": (
        "Intent Analyst",
        "Identify explicit requests, implicit goals, and the seeker's decision stage.",
        "intent, explicit_ask, implicit_goal, decision_stage",
    ),
}


def _spec(**sections: str) -> str:
    return "\n\n".join(f"## {heading}\n{sections[heading]}" for heading in _SECTIONS)


def _expert_prompt(role: str) -> str:
    title, objective, fields = _EXPERT_SPECS[role]
    return _spec(
        Role=f"You are the {title} in an emotional-support dialogue system.",
        Objective=objective,
        **{
            "Available Inputs": "The dialogue history only.",
            "Responsibilities": "Extract domain-specific findings, cite dialogue evidence, and state genuine uncertainty.",
            "Procedure": "Read the full history, identify supported signals in your domain, and map each finding to evidence.",
            "Constraints": f"Do not infer other expert domains. Use only these fields keys: {fields}. Do not invent evidence.",
            "Quality Criteria": "Be specific, evidence-grounded, internally consistent, and appropriately uncertain.",
            "Output Contract": "Return only JSON matching the supplied schema.",
        },
    )


_CRITIC_RULE = "At most one highest-priority issue per candidate; use an empty list when there is no material issue. Keep evidence and suggested_revision to one concise sentence each."

_ROLE_PROMPTS = {role: _expert_prompt(role) for role in _EXPERT_SPECS}
_ROLE_PROMPTS.update(
    {
        "state_integrator": _spec(
            Role="You are the State Integrator for an emotional-support dialogue system.",
            Objective="Combine four independent expert reports into one compact STATE blackboard.",
            **{"Available Inputs": "Dialogue history and the emotion, need, relationship, and intent reports.", "Responsibilities": "Synthesize the current emotion, intensity, primary need, support goal, readiness, main constraint, and relationship context.", "Procedure": "Compare the reports, resolve compatible findings, and express each STATE value as a short, distinct phrase of at most 20 words.", "Constraints": "Do not add facts absent from the dialogue or expert reports. Do not include evidence, quotations, uncertainties, rationale, planning, or a drafted response.", "Quality Criteria": "The compact STATE is coherent, non-redundant, specific, and useful for downstream planning.", "Output Contract": 'Return only JSON with exactly these seven fields: {"emotion":"<phrase>","intensity":"<phrase>","primary_need":"<phrase>","support_goal":"<phrase>","readiness":"<phrase>","main_constraint":"<phrase>","relationship_context":"<phrase>"}.'},
        ),
        "planner": _spec(
            Role="You are the Strategy Planner for an emotional-support dialogue system.",
            Objective="Select exactly three distinct and potentially useful ESConv strategy categories for the current STATE.",
            **{"Available Inputs": f"Dialogue history, the compact STATE, and this fixed ESConv strategy catalog:\n{_STRATEGY_CATALOG}", "Responsibilities": "Choose three different category names in intended contribution order.", "Procedure": "Assess the seeker's immediate need and readiness, then select and order three distinct categories.", "Constraints": "Use only the eight listed category names. Keep the choices plausible, meaningfully different, and grounded in the dialogue and STATE.", "Quality Criteria": "The ordered categories are complementary and appropriate to the conversation stage.", "Output Contract": 'Return only this JSON shape: {"strategies":["<category-1>","<category-2>","<category-3>"]}'},
        ),
        "emotion_critic": _spec(
            Role="You are the Emotion Critic.", Objective="Compare all candidates for emotional recognition, validation, tone, and attunement.", **{"Available Inputs": "Dialogue history and every supplied strategy-labeled candidate.", "Responsibilities": "Identify only the most consequential emotional-attunement defect for each candidate.", "Procedure": "Evaluate each candidate against the seeker's expressed and implied emotions, then report a concrete revision only when needed.", "Constraints": _CRITIC_RULE + " Do not score strategy popularity.", "Quality Criteria": "Findings are comparative, dialogue-grounded, specific, and actionable.", "Output Contract": "Return only JSON matching the supplied schema and cover every supplied candidate ID."},
        ),
        "effectiveness_critic": _spec(
            Role="You are the Support Effectiveness Critic.", Objective="Compare all candidates for timing, relevance, autonomy, specificity, and likely helpfulness.", **{"Available Inputs": "Dialogue history and every supplied strategy-labeled candidate.", "Responsibilities": "Identify only the most consequential effectiveness defect for each candidate.", "Procedure": "Assess whether each response matches readiness, advances the dialogue, and leaves meaningful choice to the seeker.", "Constraints": _CRITIC_RULE + " Do not rewrite entire responses.", "Quality Criteria": "Findings are comparative, realistic, specific, and actionable.", "Output Contract": "Return only JSON matching the supplied schema and cover every supplied candidate ID."},
        ),
        "safety_critic": _spec(
            Role="You are the Safety and Factuality Critic.", Objective="Check candidates for harmful guidance, unsupported claims, contradictions, coercion, and factual problems.", **{"Available Inputs": "Dialogue history and every supplied strategy-labeled candidate.", "Responsibilities": "Flag material safety or factuality defects and ground each finding in the dialogue or response text.", "Procedure": "Review each candidate for risk, unsupported certainty, inconsistency with the conversation, and inappropriate pressure.", "Constraints": _CRITIC_RULE + " A response may use first-person language when it fits the dialogue; do not apply a separate personal-experience fabrication check.", "Quality Criteria": "Findings are proportionate, context-sensitive, and limited to material problems.", "Output Contract": "Return only JSON matching the supplied schema and cover every supplied candidate ID."},
        ),
        "final_integrator": _spec(
            Role="You are the Final Response Integrator.", Objective="Create the strongest concise supporter response by selecting, rewriting, or combining candidate content.", **{"Available Inputs": "Dialogue history, compact STATE, supplied strategy categories and candidates, and all corresponding critic reports.", "Responsibilities": "Produce one natural response and explicitly record every supplied strategy that materially contributes.", "Procedure": "Use the critiques to retain helpful content and revise defects; choose one strategy or combine at most two strategies; record strategy_uses in contribution order.", "Constraints": "Use only supplied strategies. The response must contain fewer than 30 words. Do not mention agents, plans, candidates, critics, or strategy labels in the response.", "Quality Criteria": "The response is emotionally attuned, relevant, safe, dialogue-consistent, natural, and neither abrupt nor verbose.", "Output Contract": "Return only JSON matching the supplied schema with response and one or two strategy_uses."},
        ),
        "pair_verifier": _spec(
            Role="You are a Blind Pair Verifier.", Objective="Determine which response better supports the seeker on the specified non-safety dimension.", **{"Available Inputs": "Dialogue history, responses A and B, and one target dimension.", "Responsibilities": "Compare only the requested dimension and identify concrete evidence for the preference.", "Procedure": "Read the history, compare A and B independently of position, then choose A, B, or tie.", "Constraints": "Ignore response order and unrelated dimensions. Do not infer hidden generation metadata.", "Quality Criteria": "The verdict is order-invariant, specific, and supported by response text.", "Output Contract": "Return only JSON matching the supplied schema."},
        ),
    }
)

_CANDIDATE_PROMPT = _spec(
    Role="You are a Strategy-Specific Response Generator.",
    Objective="Write one natural supporter response that faithfully realizes the single assigned strategy category.",
    **{"Available Inputs": "Dialogue history, compact STATE, one assigned strategy category with its fixed catalog definition, candidate_id, and seed. No alternative category is available.", "Responsibilities": "Realize the assigned category in a response that fits the dialogue and STATE while preserving the frozen metadata.", "Procedure": "Apply the assigned catalog definition, draft a dialogue-continuing response, and remove unnecessary wording.", "Constraints": "Use only the assigned strategy category. The response must contain fewer than 30 words. Do not expose strategy labels or system metadata. Do not invent dialogue facts.", "Quality Criteria": "The response is specific, supportive, natural, concise, and clearly realizes the assigned category.", "Output Contract": "Return only JSON matching the supplied schema. Echo candidate_id, strategy_id, strategy, and seed exactly."},
)
for _candidate_role in ("candidate_1", "candidate_2", "candidate_3"):
    _ROLE_PROMPTS[_candidate_role] = _CANDIDATE_PROMPT

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
    if role.startswith("candidate_") and "strategy" in context:
        strategy = context["strategy"]
        if strategy not in ESCONV_STRATEGIES:
            raise ValueError(f"unknown ESConv strategy: {strategy}")
        definition = ESCONV_STRATEGIES[strategy]
        prompt = prompt.replace(
            "one assigned strategy category with its fixed catalog definition",
            f"one assigned strategy category with its fixed catalog definition:\n- {strategy}: {definition}",
        )
    elif role.endswith("_critic") and "candidates" in context:
        candidate_ids = [str(candidate["candidate_id"]) for candidate in context["candidates"]]
        prompt = prompt.replace(
            "cover every supplied candidate ID",
            f"cover candidate IDs {', '.join(candidate_ids)}",
        )
    elif role == "final_integrator" and "required_strategies" in context:
        required = list(context["required_strategies"])
        if len(required) not in (1, 2):
            raise ValueError("final_integrator requires one or two strategies")
        exact_requirement = (
            f"Use exactly {len(required)} strategy category or categories in this "
            f"contribution order: {', '.join(required)}"
        )
        prompt = prompt.replace(
            "choose one strategy or combine at most two strategies",
            exact_requirement,
        ).replace(
            "with response and one or two strategy_uses.",
            f"with response and exactly {len(required)} strategy_uses in the required order.",
        )
    return prompt


def _freeze_candidate_schema(
    schema: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    for field in ("candidate_id", "strategy_id", "strategy", "seed"):
        if field not in context:
            continue
        property_schema = schema["properties"][field]
        property_schema.pop("enum", None)
        property_schema["const"] = context[field]
    return schema


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
    payload = {"history": history.model_dump(mode="json")}
    if normalized_context:
        if role.startswith("candidate_") and "strategy" in normalized_context:
            normalized_context["strategy_definition"] = ESCONV_STRATEGIES[
                normalized_context["strategy"]
            ]
        payload["context"] = normalized_context
    schema = response_model.model_json_schema()
    if role.startswith("candidate_") and "strategy" in normalized_context:
        schema = _freeze_candidate_schema(schema, normalized_context)
    return [
        {
            "role": "system",
            "content": f"{_resolved_prompt(role, normalized_context)}\n\nJSON Schema:\n{json.dumps(schema, ensure_ascii=False)}",
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
