import json

import pytest
from pydantic import BaseModel

from ibd.backend import (
    CallCache,
    LLMResult,
    StructuredCallError,
    StructuredCaller,
    _cache_digest,
)
from ibd.config import AppConfig, BackendConfig, ModelConfig


class Payload(BaseModel):
    value: str


class CountingBackend:
    def __init__(self):
        self.completions = 0

    def complete(self, **kwargs):
        self.completions += 1
        return LLMResult(
            text=json.dumps({"value": f"reply-{self.completions}"})
        )


class InvalidJsonBackend:
    def __init__(self):
        self.completions = 0

    def complete(self, **kwargs):
        self.completions += 1
        return LLMResult(text="{")


def test_cache_variant_separates_same_role_calls(tmp_path):
    backend = CountingBackend()
    config = AppConfig(
        backend=BackendConfig(cache_dir=tmp_path),
        default_model=ModelConfig(model="fake"),
    )
    caller = StructuredCaller(backend, config)
    messages = [{"role": "user", "content": "candidate"}]

    first, _ = caller.call(
        "candidate",
        messages,
        Payload,
        example_id="e-1",
        cache_variant="S1",
    )
    second, _ = caller.call(
        "candidate",
        messages,
        Payload,
        example_id="e-1",
        cache_variant="S2",
    )
    replay, records = caller.call(
        "candidate",
        messages,
        Payload,
        example_id="e-1",
        cache_variant="S1",
    )

    assert (first.value, second.value, replay.value) == (
        "reply-1",
        "reply-2",
        "reply-1",
    )
    assert records[0].cached is True
    assert backend.completions == 2


def test_missing_cache_variant_preserves_the_existing_cache_digest():
    caller = StructuredCaller(
        CountingBackend(),
        AppConfig(default_model=ModelConfig(model="fake")),
    )

    assert caller._cache_key("e-legacy", "planner") == _cache_digest(
        ("e-legacy", "planner", caller._cache_namespace)
    )


def test_call_cache_round_trips_raw_payload(tmp_path):
    config = AppConfig(
        backend=BackendConfig(cache_dir=tmp_path),
        default_model=ModelConfig(model="fake"),
    )
    cache = CallCache(config)
    key = cache.key("conversation-1", "seeker_simulator", "round-1")

    cache.write(key, {"text": "需要支持"})

    assert cache.read(key) == {"text": "需要支持"}
    assert (tmp_path / "calls" / f"{key}.json").is_file()


def test_exhausted_schema_retries_raise_records_and_persist_failure(tmp_path):
    backend = InvalidJsonBackend()
    config = AppConfig(
        backend=BackendConfig(cache_dir=tmp_path),
        default_model=ModelConfig(model="fake"),
        schema_retries=1,
    )
    caller = StructuredCaller(backend, config)

    with pytest.raises(StructuredCallError) as raised:
        caller.call(
            "dialogue_manager",
            [{"role": "user", "content": "choose a mode"}],
            Payload,
            example_id="conversation-1",
            cache_variant="round-1",
        )

    assert isinstance(raised.value, ValueError)
    assert raised.value.role == "dialogue_manager"
    assert [record.attempt for record in raised.value.records] == [1, 2]
    assert all(record.raw_text == "{" for record in raised.value.records)
    assert backend.completions == 2
    assert len(list((tmp_path / "failures").glob("*.json"))) == 1
