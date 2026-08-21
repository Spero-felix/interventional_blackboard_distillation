"""Explicit Student-data allowlists that prevent Teacher audit leakage."""

from __future__ import annotations

from typing import Any

from .schemas import InterventionRecord, MarginPair, TeacherTrace


def _ensure_trainable(trace: TeacherTrace) -> None:
    if trace.split == "test":
        raise ValueError("test split traces cannot be exported for Student training")
    if trace.split == "diagnostic_holdout":
        raise ValueError("diagnostic holdout traces cannot be exported for Student training")


def sft_row(trace: TeacherTrace) -> dict[str, Any]:
    _ensure_trainable(trace)
    return {
        "example_id": trace.example_id,
        "prompt": trace.history.as_prompt(),
        "response": trace.final_response,
        "strategy_uses": trace.final_answer.model_dump(mode="json")["strategy_uses"],
    }


def slot_row(trace: TeacherTrace) -> dict[str, Any]:
    _ensure_trainable(trace)
    return {
        "example_id": trace.example_id,
        "prompt": trace.history.as_prompt(),
        "state": trace.state.model_dump(mode="json"),
        "plan": {
            **trace.plan.model_dump(mode="json"),
            "strategy_uses": trace.final_answer.model_dump(mode="json")["strategy_uses"],
        },
    }


def intervention_row(record: InterventionRecord, prompt: str) -> dict[str, Any]:
    clamp = record.mutated_state if record.function == "STATE" else record.mutated_plan
    if clamp is None:
        raise ValueError("intervention record is missing its mutated clamp value")
    return {
        "example_id": record.example_id,
        "prompt": prompt,
        "function": record.function,
        "clamp": clamp.model_dump(mode="json"),
        "full_response": record.full_response,
        "counterfactual_response": record.counterfactual_response,
        "target_dimension": record.target_dimension,
    }


def margin_row(pair: MarginPair) -> dict[str, Any]:
    return {
        "example_id": pair.example_id,
        "prompt": pair.prompt,
        "chosen": pair.chosen,
        "rejected": pair.rejected,
        "chosen_strategy_uses": [
            item.model_dump(mode="json") for item in pair.chosen_strategy_uses
        ],
        "rejected_strategy_id": pair.rejected_strategy_id,
        "rejected_strategy": pair.rejected_strategy,
    }
