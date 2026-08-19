"""Prompt construction with explicit information boundaries."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .schemas import History

_ROLE_PURPOSES = {
    "emotion_expert": "Analyze only emotion, intensity, trajectory, and coping signals.",
    "need_expert": "Analyze only needs, priorities, readiness, and constraints.",
    "relationship_expert": "Analyze only relationship type, patterns, power, and boundary signals.",
    "intent_expert": "Analyze only explicit and implicit intent and decision stage.",
    "state_integrator": "Integrate the four independent reports into a STATE blackboard.",
    "planner": "Create a support PLAN with ordered response acts and avoid items.",
    "emotion_critic": "Compare candidates for emotional attunement.",
    "effectiveness_critic": "Compare candidates for timing, autonomy, and support effectiveness.",
    "safety_critic": "Audit candidate safety. This audit is Teacher-only.",
    "final_integrator": "Select, discard, and rewrite candidate content into one response.",
    "quality_gate": "Blindly assess the response quality and safety.",
    "repair": "Apply only the gate-requested repair.",
    "quality_gate_recheck": "Recheck the repaired response once.",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def build_messages(
    role: str,
    history: History,
    response_model: type[BaseModel],
    *,
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    purpose = _ROLE_PURPOSES.get(role, "Complete the assigned Teacher function.")
    payload = {"history": history.model_dump(mode="json")}
    if context:
        payload["context"] = _jsonable(context)
    schema = response_model.model_json_schema()
    return [
        {
            "role": "system",
            "content": f"{purpose} Do not invent evidence. Return JSON matching this schema: {json.dumps(schema, ensure_ascii=False)}",
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

