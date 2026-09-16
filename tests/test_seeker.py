import json

import pytest

from ibd.config import AppConfig, BackendConfig, ModelConfig
from ibd.conversation_schemas import ConversationTurn
from ibd.seeker import MappingProfileAdapter, SeekerSimulator, TextCaller


class RecordingTextBackend:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, **kwargs):
        from ibd.backend import LLMResult

        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResult(text=outcome, usage={"total_tokens": 7})


def _config(tmp_path=None):
    return AppConfig(
        backend=BackendConfig(cache_dir=tmp_path),
        default_model=ModelConfig(model="fake"),
        roles={
            "seeker_simulator": ModelConfig(
                model="seeker",
                temperature=0.7,
                max_tokens=256,
                provider_json_mode=False,
            )
        },
        schema_retries=1,
    )


@pytest.mark.parametrize(("raw_id", "expected"), [(7, "7"), (" user-7 ", "user-7")])
def test_mapping_profile_adapter_normalizes_only_the_required_id(raw_id, expected):
    raw = {"ID": raw_id, "Situation": "PRIVATE_PROFILE_SENTINEL"}

    profile = MappingProfileAdapter().validate(raw)

    assert profile.profile_id == expected
    assert json.loads(profile.render_for_seeker()) == raw
    assert not hasattr(profile, "situation")


@pytest.mark.parametrize("raw", [{}, {"ID": ""}, {"ID": "   "}, {"ID": None}])
def test_mapping_profile_adapter_rejects_missing_or_blank_id(raw):
    with pytest.raises(ValueError, match="non-empty ID"):
        MappingProfileAdapter().validate(raw)


def test_mapping_profile_adapter_copies_input_before_rendering():
    raw = {"ID": "p-1", "nested": {"concern": "original"}}
    profile = MappingProfileAdapter().validate(raw)
    raw["nested"]["concern"] = "mutated"

    assert json.loads(profile.render_for_seeker())["nested"]["concern"] == "original"


def test_text_caller_uses_plain_text_mode_and_strips_output():
    backend = RecordingTextBackend(["  我最近压力很大。  "])
    caller = TextCaller(backend, _config())

    text, record = caller.call(
        "seeker_simulator",
        [{"role": "user", "content": "speak"}],
        seed=19,
        example_id="p-1-seed-42",
        cache_variant="round-1",
    )

    assert text == "我最近压力很大。"
    assert record.raw_text == "  我最近压力很大。  "
    assert record.parsed == {"text": "我最近压力很大。"}
    assert record.attempt == 1
    assert backend.calls[0]["json_mode"] is False
    assert backend.calls[0]["seed"] == 19
    assert backend.calls[0]["model_config"].model == "seeker"


def test_text_caller_retries_empty_output_and_provider_error():
    empty_backend = RecordingTextBackend(["   ", "第二次成功"])
    text, record = TextCaller(empty_backend, _config()).call(
        "seeker_simulator",
        [],
        seed=1,
        example_id="empty",
    )
    assert text == "第二次成功"
    assert record.attempt == 2

    failure_backend = RecordingTextBackend([RuntimeError("provider down"), "恢复成功"])
    text, record = TextCaller(failure_backend, _config()).call(
        "seeker_simulator",
        [],
        seed=2,
        example_id="failure",
    )
    assert text == "恢复成功"
    assert record.attempt == 2


def test_text_caller_replays_persistent_cache(tmp_path):
    first_backend = RecordingTextBackend(["第一次回复"])
    first = TextCaller(first_backend, _config(tmp_path))
    first.call(
        "seeker_simulator",
        [],
        seed=3,
        example_id="cached-conversation",
        cache_variant="round-1",
    )
    replay_backend = RecordingTextBackend(["不应调用"])

    text, record = TextCaller(replay_backend, _config(tmp_path)).call(
        "seeker_simulator",
        [],
        seed=3,
        example_id="cached-conversation",
        cache_variant="round-1",
    )

    assert text == "第一次回复"
    assert record.cached is True
    assert replay_backend.calls == []


def test_seeker_prompt_contains_private_profile_but_no_round_budgets():
    backend = RecordingTextBackend(["我最近总在担心明天的答辩。"])
    simulator = SeekerSimulator(backend, _config())
    profile = MappingProfileAdapter().validate(
        {"ID": "p-1", "Situation": "PRIVATE_PROFILE_SENTINEL"}
    )

    result = simulator.generate(
        profile,
        [],
        "opening",
        round_index=1,
        seed=23,
        conversation_id="p-1-seed-42",
    )

    rendered_messages = json.dumps(backend.calls[0]["messages"], ensure_ascii=False)
    user_payload = json.loads(backend.calls[0]["messages"][1]["content"])
    assert result.utterance == "我最近总在担心明天的答辩。"
    assert result.call_record.role == "seeker_simulator"
    assert "PRIVATE_PROFILE_SENTINEL" in rendered_messages
    assert user_payload["mode"] == "opening"
    assert user_payload["round_index"] == 1
    for forbidden in (
        "min_rounds",
        "soft_max_rounds",
        "hard_max_rounds",
        "user_context",
        "state_analysis",
        "transition_reason",
        "candidates",
    ):
        assert forbidden not in rendered_messages


def test_seeker_prompt_receives_only_public_history_from_later_rounds():
    backend = RecordingTextBackend(["我还是有点犹豫。"])
    simulator = SeekerSimulator(backend, _config())
    profile = MappingProfileAdapter().validate({"ID": "p-1", "Situation": "秘密背景"})
    history = [
        ConversationTurn(
            turn_index=1,
            round_index=1,
            role="seeker",
            content="我最近压力很大。",
        ),
        ConversationTurn(
            turn_index=2,
            round_index=1,
            role="supporter",
            content="听起来这段时间很不容易。",
        ),
    ]

    simulator.generate(
        profile,
        history,
        "comforting",
        round_index=2,
        seed=24,
        conversation_id="p-1-seed-42",
    )

    payload = json.loads(backend.calls[0]["messages"][1]["content"])
    assert payload["history"] == [turn.model_dump(mode="json") for turn in history]
