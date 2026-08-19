"""Explicit Student-data allowlists that prevent Teacher audit leakage."""

from __future__ import annotations

from typing import Any

from .schemas import InterventionRecord, MarginPair, TeacherTrace


def _ensure_trainable(trace: TeacherTrace) -> None:
    if trace.split == "test":
        raise ValueError("test split traces cannot be exported for Student training")


def sft_row(trace: TeacherTrace) -> dict[str, Any]:
    _ensure_trainable(trace)
    return {
        "example_id": trace.example_id,
        "prompt": trace.history.as_prompt(),
        "response": trace.final_response,
    }


def slot_row(trace: TeacherTrace) -> dict[str, Any]:
    _ensure_trainable(trace)
    return {
        "example_id": trace.example_id,
        "prompt": trace.history.as_prompt(),
        "state": trace.state.model_dump(mode="json"),
        "plan": trace.plan.model_dump(mode="json"),
    }


def intervention_row(record: InterventionRecord, prompt: str) -> dict[str, Any]:
    return {
        "example_id": record.example_id,
        "prompt": prompt,
        "function": record.function,
        "clamp": record.mutation.model_dump(mode="json"),
        "full_response": record.full_response,
        "intervened_response": record.ablated_response,
        "target_dimension": record.target_dimension,
    }


def margin_row(pair: MarginPair) -> dict[str, str]:
    return {
        "example_id": pair.example_id,
        "prompt": pair.prompt,
        "chosen": pair.chosen,
        "rejected": pair.rejected,
    }

