"""Typed contracts for complete profile-driven conversation generation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .schemas import (
    CallRecord,
    DialogueTurn,
    History,
    StrictModel,
    TeacherTrace,
    UserContext,
)

DialogueMode = Literal[
    "opening",
    "exploration",
    "comforting",
    "action",
    "closing",
]
ConversationStatus = Literal["completed", "truncated"]
TerminationReason = Literal[
    "dialogue_manager_closing",
    "hard_max_without_closing",
]
ConversationSplit = Literal["train", "dev", "diagnostic_holdout"]


class ConversationGenerationConfig(StrictModel):
    min_rounds: int = Field(default=6, ge=1)
    soft_max_rounds: int = Field(default=16, ge=1)
    hard_max_rounds: int = Field(default=20, ge=1)
    split: ConversationSplit = "train"

    @model_validator(mode="after")
    def require_ordered_limits(self) -> "ConversationGenerationConfig":
        if not self.min_rounds <= self.soft_max_rounds <= self.hard_max_rounds:
            raise ValueError(
                "round limits must satisfy "
                "min_rounds <= soft_max_rounds <= hard_max_rounds"
            )
        return self


class DialogueManagementDecision(StrictModel):
    mode: DialogueMode
    transition_reason: str = Field(min_length=1)


class DialogueDecisionRecord(StrictModel):
    round_index: int = Field(ge=1)
    mode_before: DialogueMode
    decision: DialogueManagementDecision
    call_records: list[CallRecord]


class ConversationTurn(StrictModel):
    turn_index: int = Field(ge=1)
    round_index: int = Field(ge=1)
    role: Literal["seeker", "supporter"]
    content: str = Field(min_length=1)


class ConversationRoundAudit(StrictModel):
    round_index: int = Field(ge=1)
    seeker_turn_index: int = Field(ge=1)
    supporter_turn_index: int = Field(ge=1)
    dialogue_decision: DialogueDecisionRecord
    teacher_trace: TeacherTrace


class ConversationTermination(StrictModel):
    reason: TerminationReason
    round_count: int = Field(ge=1)
    final_mode: DialogueMode


def _validate_complete_rounds(
    turns: list[ConversationTurn],
    rounds: list[ConversationRoundAudit],
) -> None:
    expected_turn_indexes = list(range(1, len(turns) + 1))
    if [turn.turn_index for turn in turns] != expected_turn_indexes:
        raise ValueError("turn indexes must be contiguous from one")
    expected_round_indexes = list(range(1, len(rounds) + 1))
    if [round_audit.round_index for round_audit in rounds] != expected_round_indexes:
        raise ValueError("round indexes must be contiguous from one")
    if len(turns) != 2 * len(rounds):
        raise ValueError("conversation must contain one audit per seeker-supporter round")

    for turn_index, turn in enumerate(turns, start=1):
        expected_role = "seeker" if turn_index % 2 else "supporter"
        expected_round = (turn_index + 1) // 2
        if turn.role != expected_role:
            raise ValueError("conversation turns must alternate seeker and supporter")
        if turn.round_index != expected_round:
            raise ValueError("turn round indexes must identify their containing round")

    for index, audit in enumerate(rounds, start=1):
        seeker_turn_index = 2 * index - 1
        supporter_turn_index = 2 * index
        if (
            audit.seeker_turn_index != seeker_turn_index
            or audit.supporter_turn_index != supporter_turn_index
        ):
            raise ValueError("round audit turn indexes must bind its seeker and supporter")
        if audit.dialogue_decision.round_index != index:
            raise ValueError("dialogue decision round index must match its audit")

        public_prefix = History(
            turns=[
                DialogueTurn(role=turn.role, content=turn.content)
                for turn in turns[:seeker_turn_index]
            ]
        )
        if audit.teacher_trace.history != public_prefix:
            raise ValueError("teacher history must equal the public prefix through seeker")
        if audit.teacher_trace.final_response != turns[supporter_turn_index - 1].content:
            raise ValueError("teacher response must equal the bound supporter turn")


class ConversationTrace(StrictModel):
    protocol_version: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    seed: int
    split: ConversationSplit
    status: ConversationStatus
    turns: list[ConversationTurn] = Field(min_length=2)
    rounds: list[ConversationRoundAudit]
    termination: ConversationTermination
    seeker_call_records: list[CallRecord]

    @model_validator(mode="after")
    def validate_conversation(self) -> "ConversationTrace":
        _validate_complete_rounds(self.turns, self.rounds)
        if len(self.seeker_call_records) != len(self.rounds):
            raise ValueError("conversation must contain one seeker call record per round")
        if any(record.role != "seeker_simulator" for record in self.seeker_call_records):
            raise ValueError("seeker call records must use the seeker_simulator role")
        if self.termination.round_count != len(self.rounds):
            raise ValueError("termination round count must equal stored rounds")
        final_mode = self.rounds[-1].dialogue_decision.decision.mode
        if self.termination.final_mode != final_mode:
            raise ValueError("termination final mode must equal the final decision mode")
        if self.status == "completed":
            if final_mode != "closing":
                raise ValueError("completed conversation must end in closing")
            if self.termination.reason != "dialogue_manager_closing":
                raise ValueError("completed conversation requires manager-closing reason")
        else:
            if final_mode == "closing":
                raise ValueError("truncated conversation cannot end in closing")
            if self.termination.reason != "hard_max_without_closing":
                raise ValueError("truncated conversation requires hard-max reason")
        return self


class ConversationCheckpoint(StrictModel):
    protocol_version: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    seed: int
    generation_config: ConversationGenerationConfig
    turns: list[ConversationTurn]
    rounds: list[ConversationRoundAudit]
    last_context: UserContext
    previous_mode: DialogueMode
    next_round_index: int = Field(ge=1)
    seeker_call_records: list[CallRecord]

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "ConversationCheckpoint":
        _validate_complete_rounds(self.turns, self.rounds)
        if len(self.seeker_call_records) != len(self.rounds):
            raise ValueError("checkpoint must contain one seeker call record per round")
        if any(record.role != "seeker_simulator" for record in self.seeker_call_records):
            raise ValueError("seeker call records must use the seeker_simulator role")
        if self.next_round_index != len(self.rounds) + 1:
            raise ValueError("next round index must follow stored complete rounds")
        if self.rounds:
            if self.previous_mode != self.rounds[-1].dialogue_decision.decision.mode:
                raise ValueError("previous mode must equal the final stored decision mode")
            if self.last_context != self.rounds[-1].teacher_trace.context_after:
                raise ValueError("last context must equal the final committed context")
        elif self.previous_mode != "opening":
            raise ValueError("an empty checkpoint must begin in opening mode")
        return self


class ConversationFailure(StrictModel):
    conversation_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    seed: int
    failed_stage: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    error: str = Field(min_length=1)
    completed_rounds: int = Field(ge=0)
    checkpoint_path: str | None = None
