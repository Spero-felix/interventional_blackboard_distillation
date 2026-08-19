import json

from ibd.hashing import protocol_hash
from ibd.storage import append_jsonl, read_jsonl


def test_protocol_hash_is_canonical_across_mapping_order():
    left = {"roles": {"need": "n", "emotion": "e"}, "version": 1}
    right = {"version": 1, "roles": {"emotion": "e", "need": "n"}}

    assert protocol_hash(left) == protocol_hash(right)
    assert len(protocol_hash(left)) == 64


def test_append_jsonl_persists_separate_utf8_records(tmp_path):
    path = tmp_path / "traces.jsonl"
    append_jsonl(path, {"id": 1, "text": "需要支持"})
    append_jsonl(path, {"id": 2, "text": "关系紧张"})

    assert read_jsonl(path) == [
        {"id": 1, "text": "需要支持"},
        {"id": 2, "text": "关系紧张"},
    ]
    assert [json.loads(line) for line in path.read_text().splitlines()] == read_jsonl(path)

