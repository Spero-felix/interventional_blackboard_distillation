import json

import pytest

from conftest import ScriptedBackend
from ibd.backend import LLMResult
from ibd.conversation import (
    ConversationGenerator,
    ConversationStageError,
    flatten_teacher_traces,
)
from ibd.conversation_schemas import ConversationGenerationConfig
from ibd.dialogue_manager import DialogueManager
from ibd.schemas import DialogueTurn
from ibd.seeker import MappingProfileAdapter, SeekerSimulator
from ibd.teacher import TeacherRunner


class ConversationBackend(ScriptedBackend):
    def __init__(self, manager_modes, *, fail_role=None):
        super().__init__()
        self.manager_modes = list(manager_modes)
        self.fail_role = fail_role
        self.seeker_index = 0

    def complete(self, *, role, messages, model_config, json_mode=True, seed=None):
        if role == self.fail_role:
            raise RuntimeError(f"simulated {role} failure")
        if role == "seeker_simulator":
            self.seeker_index += 1
            self.calls.append(
                {
                    "role": role,
                    "messages": messages,
                    "model": model_config.model,
                    "json_mode": json_mode,
                    "seed": seed,
                }
            )
            return LLMResult(text=f"第{self.seeker_index}轮，我想继续说说。")
        if role == "dialogue_manager":
            mode = self.manager_modes.pop(0)
            self.calls.append(
                {
                    "role": role,
                    "messages": messages,
                    "model": model_config.model,
                    "json_mode": json_mode,
                    "seed": seed,
                }
            )
            return LLMResult(
                text=json.dumps(
                    {
                        "mode": mode,
                        "transition_reason": f"公开对话支持 {mode}。",
                    },
                    ensure_ascii=False,
                )
            )
        return super().complete(
            role=role,
            messages=messages,
            model_config=model_config,
            json_mode=json_mode,
            seed=seed,
        )


def _generator(backend, app_config, *, min_rounds=1, soft_max_rounds=3, hard_max_rounds=4):
    return ConversationGenerator(
        seeker=SeekerSimulator(backend, app_config),
        teacher_runner=TeacherRunner(backend, app_config),
        dialogue_manager=DialogueManager(backend, app_config),
        generation_config=ConversationGenerationConfig(
            min_rounds=min_rounds,
            soft_max_rounds=soft_max_rounds,
            hard_max_rounds=hard_max_rounds,
        ),
        protocol_version=app_config.protocol_version,
    )


def _profile():
    return MappingProfileAdapter().validate(
        {"ID": "profile-1", "Situation": "对明天的答辩感到担心"}
    )


def test_generator_runs_complete_rounds_until_manager_closing(app_config):
    backend = ConversationBackend(["exploration", "closing"])
    checkpoints = []

    trace = _generator(backend, app_config).generate(
        _profile(),
        base_seed=42,
        on_round_completed=checkpoints.append,
    )

    expected_round_roles = [
        "seeker_simulator",
        "context_updater",
        "multi_view_state_analyzer",
        "dialogue_manager",
        "planner",
        "candidate",
        "candidate",
        "candidate",
        "final_selector",
    ]
    assert [call["role"] for call in backend.calls] == expected_round_roles * 2
    assert trace.status == "completed"
    assert trace.termination.reason == "dialogue_manager_closing"
    assert [round_audit.dialogue_decision.mode_before for round_audit in trace.rounds] == [
        "opening",
        "exploration",
    ]
    assert [round_audit.dialogue_decision.decision.mode for round_audit in trace.rounds] == [
        "exploration",
        "closing",
    ]
    assert [turn.role for turn in trace.turns] == [
        "seeker",
        "supporter",
        "seeker",
        "supporter",
    ]
    assert len(checkpoints) == 2
    assert checkpoints[-1].next_round_index == 3
    assert len(trace.seeker_call_records) == 2


def test_hard_max_non_closing_generates_supporter_then_marks_truncated(app_config):
    backend = ConversationBackend(["exploration", "comforting"])

    trace = _generator(
        backend,
        app_config,
        min_rounds=1,
        soft_max_rounds=1,
        hard_max_rounds=2,
    ).generate(_profile(), base_seed=42)

    assert trace.status == "truncated"
    assert trace.termination.reason == "hard_max_without_closing"
    assert trace.termination.final_mode == "comforting"
    assert trace.turns[-1].role == "supporter"
    assert len(trace.rounds) == 2
    assert trace.rounds[-1].teacher_trace.final_response == trace.turns[-1].content


def test_closing_at_hard_max_is_completed(app_config):
    backend = ConversationBackend(["exploration", "closing"])

    trace = _generator(
        backend,
        app_config,
        min_rounds=1,
        soft_max_rounds=1,
        hard_max_rounds=2,
    ).generate(_profile(), base_seed=42)

    assert trace.status == "completed"
    assert trace.termination.reason == "dialogue_manager_closing"


def test_min_rounds_is_a_manager_hint_not_a_closing_gate(app_config):
    backend = ConversationBackend(["closing"])

    trace = _generator(
        backend,
        app_config,
        min_rounds=6,
        soft_max_rounds=16,
        hard_max_rounds=20,
    ).generate(_profile(), base_seed=42)

    assert trace.status == "completed"
    assert len(trace.rounds) == 1


def test_generator_passes_recent_selected_strategies_to_manager(app_config):
    backend = ConversationBackend(["exploration", "closing"])

    _generator(backend, app_config).generate(_profile(), base_seed=42)

    manager_calls = [call for call in backend.calls if call["role"] == "dialogue_manager"]
    first_context = json.loads(manager_calls[0]["messages"][1]["content"])["context"]
    second_context = json.loads(manager_calls[1]["messages"][1]["content"])["context"]
    assert first_context["recent_selected_strategies"] == []
    assert second_context["recent_selected_strategies"] == ["Question"]


def test_generator_binds_public_prefix_context_and_flattened_traces(app_config):
    trace = _generator(ConversationBackend(["exploration", "closing"]), app_config).generate(
        _profile(), base_seed=42
    )

    assert trace.rounds[1].teacher_trace.context_before == trace.rounds[0].teacher_trace.context_after
    assert [item.example_id for item in flatten_teacher_traces(trace)] == [
        "profile-1-seed-42:round:1",
        "profile-1-seed-42:round:2",
    ]
    assert trace.rounds[1].teacher_trace.history.turns == [
        DialogueTurn(role=turn.role, content=turn.content)
        for turn in trace.turns[:3]
    ]


@pytest.mark.parametrize(
    ("failed_role", "expected_stage"),
    [
        ("seeker_simulator", "seeker_simulator"),
        ("multi_view_state_analyzer", "teacher_observe"),
        ("dialogue_manager", "dialogue_manager"),
        ("planner", "teacher_respond"),
    ],
)
def test_generator_reports_model_failure_stage_without_checkpointing_partial_round(
    failed_role,
    expected_stage,
    app_config,
):
    backend = ConversationBackend(["exploration"], fail_role=failed_role)
    checkpoints = []

    with pytest.raises(ConversationStageError) as raised:
        _generator(backend, app_config).generate(
            _profile(),
            base_seed=42,
            on_round_completed=checkpoints.append,
        )

    assert raised.value.stage == expected_stage
    assert raised.value.completed_rounds == 0
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert checkpoints == []


def test_generator_failure_after_one_round_reports_only_completed_rounds(app_config):
    backend = ConversationBackend(["exploration", "closing"])
    original_complete = backend.complete

    def fail_second_seeker(**kwargs):
        if kwargs["role"] == "seeker_simulator" and backend.seeker_index == 1:
            raise RuntimeError("second seeker failed")
        return original_complete(**kwargs)

    backend.complete = fail_second_seeker
    checkpoints = []

    with pytest.raises(ConversationStageError) as raised:
        _generator(backend, app_config).generate(
            _profile(), base_seed=42, on_round_completed=checkpoints.append
        )

    assert raised.value.stage == "seeker_simulator"
    assert raised.value.completed_rounds == 1
    assert len(checkpoints) == 1


def test_generator_rejects_control_characters_in_profile_id(app_config):
    profile = MappingProfileAdapter().validate({"ID": "bad\nprofile"})

    with pytest.raises(ValueError, match="control characters"):
        _generator(ConversationBackend(["closing"]), app_config).generate(
            profile, base_seed=42
        )
