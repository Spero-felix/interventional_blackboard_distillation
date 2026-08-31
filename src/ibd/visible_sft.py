"""Deterministic visible STATE/PLAN targets for ordinary SFT."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .schemas import History, STATE_ANCHOR_FIELDS, StrictModel, TeacherTrace


RESPONSE_MARKER = "[response]"
_STATE_MARKERS = tuple(f"[{field}]" for field in STATE_ANCHOR_FIELDS)
_SELECTED_STRATEGY_MARKER = "[selected_strategy]"
_RESERVED_PREFIX_MARKERS = (*_STATE_MARKERS, _SELECTED_STRATEGY_MARKER, RESPONSE_MARKER)
VISIBLE_SFT_MARKERS = (*_STATE_MARKERS, _SELECTED_STRATEGY_MARKER, RESPONSE_MARKER)


def _require_marker_safe(value: str, *, field: str) -> None:
    if any(marker in value for marker in _RESERVED_PREFIX_MARKERS):
        raise ValueError(f"{field} contains a reserved visible-SFT marker")


def serialize_visible_sft(trace: TeacherTrace) -> str:
    """Build the exact visible-SFT target from the canonical Teacher fields."""

    parts: list[str] = []
    for field in STATE_ANCHOR_FIELDS:
        value = getattr(trace.state, field)
        _require_marker_safe(value, field=field)
        parts.append(f"[{field}]{value}")
    strategy = trace.final_selection.selected_strategy
    _require_marker_safe(strategy, field="selected_strategy")
    parts.append(f"{_SELECTED_STRATEGY_MARKER}{strategy}")
    parts.append(f"{RESPONSE_MARKER}{trace.final_response}")
    return "".join(parts)


def parse_visible_sft_response(text: str) -> str:
    """Extract the user-visible reply from a visible-SFT completion."""

    _, marker, suffix = text.partition(RESPONSE_MARKER)
    if not marker:
        raise ValueError("visible-SFT completion is missing [response] marker")
    response = suffix.strip()
    if not response:
        raise ValueError("visible-SFT response must not be empty")
    return response


def validate_visible_sft_target(text: str) -> None:
    """Reject static dataset rows that do not follow the fixed marker order."""

    offset = 0
    for index, marker in enumerate(VISIBLE_SFT_MARKERS):
        if not text.startswith(marker, offset):
            raise ValueError(f"visible-SFT marker {marker} is missing or out of order")
        value_start = offset + len(marker)
        if index + 1 == len(VISIBLE_SFT_MARKERS):
            value_end = len(text)
        else:
            next_marker = VISIBLE_SFT_MARKERS[index + 1]
            value_end = text.find(next_marker, value_start)
            if value_end < 0:
                raise ValueError(
                    f"visible-SFT marker {next_marker} is missing or out of order"
                )
        if not text[value_start:value_end].strip():
            raise ValueError(f"visible-SFT value for {marker} must not be empty")
        offset = value_end


class VisibleSFTDatasetRecord(StrictModel):
    """Validated static row consumed by the ordinary SFT training command."""

    example_id: str = Field(min_length=1)
    split: Literal["train", "dev"]
    history: History
    response: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target(self) -> "VisibleSFTDatasetRecord":
        validate_visible_sft_target(self.response)
        return self
