import copy

import pytest
from pydantic import ValidationError

from conftest import ScriptedBackend
from ibd.conversation_schemas import (
    ConversationGenerationConfig,
    ConversationTrace,
)
from ibd.schemas import (
    CallRecord,
    History,
    TeacherTrace,
    TurnObservation,
    TurnResponse,
)
from ibd.teacher import TeacherRunner


def _teacher_trace(app_config) -> TeacherTrace:
    history = History.model_validate(
        {"turns": [{"role": "seeker", "content": "最近总觉得被忽略。"}]}
    )
    return TeacherRunner(ScriptedBackend(), app_config).run("profile-1-seed-42:round:1", history)


def _conversation_payload(app_config, *, status="completed", mode="closing"):
    trace = _teacher_trace(app_config)
    turns = [
        {
            "turn_index": 1,
            "round_index": 1,
            "role": "seeker",
            "content": trace.history.turns[0].content,
        },
        {
            "turn_index": 2,
            "round_index": 1,
            "role": "supporter",
            "content": trace.final_response,
        },
    ]
    reason = (
        "dialogue_manager_closing"
        if status == "completed"
        else "hard_max_without_closing"
    )
    return {
        "protocol_version": "test-v1",
        "conversation_id": "profile-1-seed-42",
        "profile_id": "profile-1",
        "seed": 42,
        "split": "train",
        "status": status,
        "turns": turns,
        "rounds": [
            {
                "round_index": 1,
                "seeker_turn_index": 1,
                "supporter_turn_index": 2,
                "dialogue_decision": {
                    "round_index": 1,
                    "mode_before": "opening",
                    "decision": {
                        "mode": mode,
                        "transition_reason": "当前需要已经得到回应。",
                    },
                    "call_records": [],
                },
                "teacher_trace": trace.model_dump(mode="json"),
            }
        ],
        "termination": {
            "reason": reason,
            "round_count": 1,
            "final_mode": mode,
        },
        "seeker_call_records": [
            CallRecord(
                role="seeker_simulator",
                attempt=1,
                raw_text=turns[0]["content"],
                parsed={"text": turns[0]["content"]},
            ).model_dump(mode="json")
        ],
    }


@pytest.mark.parametrize(
    "limits",
    [
        {"min_rounds": 7, "soft_max_rounds": 6, "hard_max_rounds": 20},
        {"min_rounds": 6, "soft_max_rounds": 21, "hard_max_rounds": 20},
    ],
)
def test_generation_config_rejects_unordered_round_limits(limits):
    with pytest.raises(ValidationError, match="min_rounds <= soft_max_rounds <= hard_max_rounds"):
        ConversationGenerationConfig.model_validate(limits)


def test_completed_conversation_accepts_one_bound_round(app_config):
    trace = ConversationTrace.model_validate(_conversation_payload(app_config))

    assert [turn.role for turn in trace.turns] == ["seeker", "supporter"]
    assert trace.rounds[0].teacher_trace.history.turns == [
        trace.rounds[0].teacher_trace.history.turns[0]
    ]
    assert trace.rounds[0].teacher_trace.final_response == trace.turns[1].content
    assert trace.termination.final_mode == "closing"


def test_conversation_rejects_non_alternating_turns(app_config):
    payload = _conversation_payload(app_config)
    payload["turns"][1]["role"] = "seeker"

    with pytest.raises(ValidationError, match="alternate seeker and supporter"):
        ConversationTrace.model_validate(payload)


def test_conversation_rejects_missing_round_audit(app_config):
    payload = _conversation_payload(app_config)
    payload["rounds"] = []

    with pytest.raises(ValidationError, match="one audit per seeker-supporter round"):
        ConversationTrace.model_validate(payload)


def test_conversation_rejects_teacher_history_not_equal_to_public_prefix(app_config):
    payload = _conversation_payload(app_config)
    payload["rounds"][0]["teacher_trace"]["history"]["turns"][0]["content"] = "另一段话"

    with pytest.raises(ValidationError, match="teacher history must equal the public prefix"):
        ConversationTrace.model_validate(payload)


def test_completed_conversation_requires_closing_final_mode(app_config):
    payload = _conversation_payload(app_config, mode="exploration")

    with pytest.raises(ValidationError, match="completed conversation must end in closing"):
        ConversationTrace.model_validate(payload)


def test_truncated_conversation_rejects_closing_final_mode(app_config):
    payload = _conversation_payload(app_config, status="truncated", mode="closing")

    with pytest.raises(ValidationError, match="truncated conversation cannot end in closing"):
        ConversationTrace.model_validate(payload)


def test_teacher_trace_from_parts_preserves_provenance(app_config):
    original = _teacher_trace(app_config)
    observation = TurnObservation(
        example_id=original.example_id,
        history=original.history,
        context_before=original.context_before,
        context_patch=original.context_patch,
        context_after=original.context_after,
        context_merge_errors=original.context_merge_errors,
        state_analysis=original.state_analysis,
        call_records=original.call_records[:2],
    )
    response = TurnResponse(
        plan=original.plan,
        candidates=original.candidates,
        final_selection=original.final_selection,
        call_records=original.call_records[2:],
    )

    rebuilt = TeacherTrace.from_parts(observation, response, split="train")

    assert rebuilt == original


def test_conversation_rejects_non_contiguous_turn_indexes(app_config):
    payload = copy.deepcopy(_conversation_payload(app_config))
    payload["turns"][1]["turn_index"] = 3

    with pytest.raises(ValidationError, match="turn indexes must be contiguous"):
        ConversationTrace.model_validate(payload)


def test_conversation_rejects_missing_seeker_call_record(app_config):
    payload = _conversation_payload(app_config)
    payload["seeker_call_records"] = []

    with pytest.raises(ValidationError, match="one seeker call record per round"):
        ConversationTrace.model_validate(payload)
