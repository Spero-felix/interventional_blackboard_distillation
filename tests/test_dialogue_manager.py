import json

import pytest

from conftest import ScriptedBackend
from ibd.backend import LLMResult
from ibd.conversation_schemas import ConversationGenerationConfig
from ibd.dialogue_manager import DialogueManager
from ibd.teacher import TeacherRunner


class ManagerBackend:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return LLMResult(text=outcome)
        return LLMResult(text=json.dumps(outcome, ensure_ascii=False))


def _observation(history, app_config):
    return TeacherRunner(ScriptedBackend(), app_config).observe("conversation-1:round:1", history)


def test_dialogue_manager_receives_only_public_management_inputs(history, app_config):
    backend = ManagerBackend(
        [{"mode": "comforting", "transition_reason": "用户当前更需要情感承接。"}]
    )
    observation = _observation(history, app_config)
    manager = DialogueManager(backend, app_config)

    record = manager.decide(
        observation,
        previous_mode="exploration",
        recent_selected_strategies=["Question", "Reflection of feelings"],
        round_index=5,
        generation_config=ConversationGenerationConfig(),
        seed=31,
        conversation_id="conversation-1",
    )

    payload = json.loads(backend.calls[0]["messages"][1]["content"])
    assert payload["history"] == history.model_dump(mode="json")
    assert payload["context"] == {
        "user_context": observation.context_after.model_dump(mode="json"),
        "state_analysis": observation.state_analysis.model_dump(mode="json"),
        "previous_mode": "exploration",
        "recent_selected_strategies": ["Question", "Reflection of feelings"],
        "round_index": 5,
        "min_rounds": 6,
        "soft_max_rounds": 16,
        "hard_max_rounds": 20,
    }
    assert "profile" not in json.dumps(payload, ensure_ascii=False).lower()
    assert "dialogue_goal" not in json.dumps(payload).lower()
    assert "should_close" not in json.dumps(payload).lower()
    assert record.mode_before == "exploration"
    assert record.decision.mode == "comforting"
    assert record.call_records[0].role == "dialogue_manager"


@pytest.mark.parametrize("mode", ["opening", "exploration", "comforting", "action", "closing"])
def test_dialogue_manager_accepts_any_mode_transition(mode, history, app_config):
    backend = ManagerBackend(
        [{"mode": mode, "transition_reason": f"选择 {mode} 符合当前公开对话。"}]
    )

    record = DialogueManager(backend, app_config).decide(
        _observation(history, app_config),
        previous_mode="action",
        recent_selected_strategies=[],
        round_index=2,
        generation_config=ConversationGenerationConfig(),
        seed=32,
        conversation_id="conversation-1",
    )

    assert record.decision.mode == mode


def test_dialogue_manager_retains_schema_retry_records(history, app_config):
    backend = ManagerBackend(
        [
            "{",
            {"mode": "exploration", "transition_reason": "仍有未澄清的信息。"},
        ]
    )

    record = DialogueManager(backend, app_config).decide(
        _observation(history, app_config),
        previous_mode="opening",
        recent_selected_strategies=[],
        round_index=1,
        generation_config=ConversationGenerationConfig(),
        seed=33,
        conversation_id="conversation-1",
    )

    assert record.decision.mode == "exploration"
    assert [call.attempt for call in record.call_records] == [1, 2]


def test_dialogue_manager_falls_back_only_after_schema_retries_exhaust(
    history,
    app_config,
):
    backend = ManagerBackend(["{", "{"])

    record = DialogueManager(backend, app_config).decide(
        _observation(history, app_config),
        previous_mode="comforting",
        recent_selected_strategies=[],
        round_index=3,
        generation_config=ConversationGenerationConfig(),
        seed=34,
        conversation_id="conversation-1",
    )

    assert record.decision.mode == "comforting"
    assert record.decision.transition_reason.startswith(
        "fallback_to_previous_mode_after_schema_failure:"
    )
    assert [call.attempt for call in record.call_records] == [1, 2]


def test_dialogue_manager_provider_errors_propagate(history, app_config):
    manager = DialogueManager(ManagerBackend([RuntimeError("provider down")]), app_config)

    with pytest.raises(RuntimeError, match="provider down"):
        manager.decide(
            _observation(history, app_config),
            previous_mode="opening",
            recent_selected_strategies=[],
            round_index=1,
            generation_config=ConversationGenerationConfig(),
            seed=35,
            conversation_id="conversation-1",
        )


def test_planner_alone_receives_dialogue_mode_as_a_weak_prior(history, app_config):
    backend = ScriptedBackend()
    runner = TeacherRunner(backend, app_config)
    observation = runner.observe("conversation-1:round:1", history)
    backend.calls.clear()

    runner.respond(observation, dialogue_mode="comforting")

    payloads = {
        role: [
            json.loads(call["messages"][1]["content"])
            for call in backend.calls
            if call["role"] == role
        ]
        for role in ("planner", "candidate", "final_selector")
    }
    assert payloads["planner"][0]["context"]["dialogue_mode"] == "comforting"
    assert "Dialogue mode describes the broad direction" in next(
        call["messages"][0]["content"]
        for call in backend.calls
        if call["role"] == "planner"
    )
    assert all(
        "dialogue_mode" not in payload["context"]
        for role in ("candidate", "final_selector")
        for payload in payloads[role]
    )
    assert "transition_reason" not in json.dumps(payloads)
