"""Student-data allowlists that prevent Teacher audit leakage."""

from __future__ import annotations

from typing import Any

from .schemas import InterventionRecord, TeacherTrace
from .visible_sft import serialize_visible_sft


def _ensure_trainable(trace: TeacherTrace) -> None:
    if trace.split in {"test", "diagnostic_holdout"}:
        raise ValueError(f"{trace.split} traces cannot be exported for Student training")


def sft_row(trace: TeacherTrace) -> dict[str, Any]:
    _ensure_trainable(trace)
    return {
        "example_id": trace.example_id,
        "prompt": trace.history.as_prompt(),
        "response": trace.final_response,
        "selected_strategy": trace.final_selection.selected_strategy,
    }


def slot_row(trace: TeacherTrace) -> dict[str, Any]:
    _ensure_trainable(trace)
    return {
        "example_id": trace.example_id,
        "prompt": trace.history.as_prompt(),
        "state": trace.state.model_dump(mode="json"),
        "plan": trace.final_selection.to_plan_selection().model_dump(mode="json"),
    }


def visible_sft_row(trace: TeacherTrace) -> dict[str, Any]:
    """Export a static visible STATE/strategy target for ordinary SFT."""

    _ensure_trainable(trace)
    return {
        "example_id": trace.example_id,
        "split": trace.split,
        "history": trace.history.model_dump(mode="json"),
        "response": serialize_visible_sft(trace),
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
        "affected_dimensions": list(record.affected_dimensions),
        "conditional_correspondence_verified": (
            record.conditional_correspondence_verified
        ),
    }
