import json

from ibd.conversation_schemas import ConversationGenerationConfig
from ibd.schemas import UserContext
from ibd.storage import append_jsonl, read_jsonl
from ibd.storage import conversation_checkpoint_path, read_json, write_json_atomic


def test_append_jsonl_persists_separate_utf8_records(tmp_path):
    path = tmp_path / "traces.jsonl"
    append_jsonl(path, {"id": 1, "text": "需要支持"})
    append_jsonl(path, {"id": 2, "text": "关系紧张"})

    assert read_jsonl(path) == [
        {"id": 1, "text": "需要支持"},
        {"id": 2, "text": "关系紧张"},
    ]
    assert [json.loads(line) for line in path.read_text().splitlines()] == read_jsonl(path)


def test_atomic_json_round_trips_and_replaces_pydantic_payload(tmp_path):
    path = tmp_path / "nested" / "checkpoint.json"
    first = {
        "context": UserContext.empty(),
        "generation_config": ConversationGenerationConfig(),
        "text": "需要支持",
    }

    write_json_atomic(path, first)
    assert read_json(path) == {
        "context": UserContext.empty().model_dump(mode="json"),
        "generation_config": ConversationGenerationConfig().model_dump(mode="json"),
        "text": "需要支持",
    }

    write_json_atomic(path, {"replacement": True})
    assert read_json(path) == {"replacement": True}
    assert [item.name for item in path.parent.iterdir()] == ["checkpoint.json"]


def test_read_json_requires_one_object(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[]\n", encoding="utf-8")

    try:
        read_json(path)
    except ValueError as exc:
        assert str(exc) == "JSON checkpoint must contain one object"
    else:
        raise AssertionError("read_json accepted a non-object checkpoint")


def test_checkpoint_filename_is_a_digest_inside_requested_directory(tmp_path):
    path = conversation_checkpoint_path(tmp_path, "../../profile/with/slashes-seed-42")

    assert path.parent == tmp_path
    assert path.suffix == ".json"
    assert len(path.stem) == 64
    int(path.stem, 16)
