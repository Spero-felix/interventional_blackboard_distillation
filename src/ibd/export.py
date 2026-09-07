"""Student-data allowlists that prevent Teacher audit leakage."""

from __future__ import annotations

from typing import Any

from .schemas import TeacherTrace
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
