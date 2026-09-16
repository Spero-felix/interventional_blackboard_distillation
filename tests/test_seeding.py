import builtins

import pytest

from ibd.seeding import SeedDeriver


def test_seed_derivation_is_stable_and_bounded():
    deriver = SeedDeriver(
        protocol_version="protocol-v1",
        conversation_id="profile-7-seed-42",
        base_seed=42,
        round_index=3,
    )

    first = deriver.for_call("planner")
    second = deriver.for_call("planner")

    assert first == second
    assert 0 <= first < 2**31


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("protocol_version", "protocol-v2"),
        ("conversation_id", "profile-8-seed-42"),
        ("base_seed", 43),
        ("round_index", 4),
    ],
)
def test_seed_derivation_changes_with_conversation_identity(field, changed):
    values = {
        "protocol_version": "protocol-v1",
        "conversation_id": "profile-7-seed-42",
        "base_seed": 42,
        "round_index": 3,
    }
    baseline = SeedDeriver(**values).for_call("planner")
    values[field] = changed

    assert SeedDeriver(**values).for_call("planner") != baseline


def test_candidate_cache_variants_receive_distinct_seeds():
    deriver = SeedDeriver("protocol-v1", "profile-7-seed-42", 42, 3)

    assert deriver.for_call("candidate", "S1") != deriver.for_call("candidate", "S2")


def test_seed_derivation_never_uses_python_hash(monkeypatch):
    def forbidden_hash(value):
        raise AssertionError(f"builtins.hash was called for {value!r}")

    monkeypatch.setattr(builtins, "hash", forbidden_hash)
    deriver = SeedDeriver("protocol-v1", "profile-7-seed-42", 42, 3)

    assert isinstance(deriver.for_call("dialogue_manager"), int)


@pytest.mark.parametrize("role", ["", "   "])
def test_seed_derivation_rejects_blank_roles(role):
    deriver = SeedDeriver("protocol-v1", "profile-7-seed-42", 42, 3)

    with pytest.raises(ValueError, match="role must be non-blank"):
        deriver.for_call(role)


def test_seed_deriver_rejects_non_positive_round_index():
    with pytest.raises(ValueError, match="round_index must be at least one"):
        SeedDeriver("protocol-v1", "profile-7-seed-42", 42, 0)
