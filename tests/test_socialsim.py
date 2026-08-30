import json

import pytest


def _dialogue(dialogue_id: int, *, pairs: int = 12) -> dict[str, object]:
    turns: list[dict[str, object]] = []
    for rank in range(1, pairs + 1):
        turns.extend(
            [
                {
                    "Turn": 2 * rank - 1,
                    "Seeker": f"seeker-{dialogue_id}-{rank}",
                },
                {
                    "Turn": 2 * rank,
                    "Supporter": f"support-{dialogue_id}-{rank}",
                    "Supporter Step by Step Reasoning": (
                        f"REASONING-{dialogue_id}-{rank}"
                    ),
                },
            ]
        )
    return {
        "ID": dialogue_id,
        "Dialogue": turns,
    }


def _profile(dialogue_id: int) -> dict[str, object]:
    return {
        "ID": dialogue_id,
        "Topic": "Study",
        "Situation": f"PROFILE_SECRET_{dialogue_id}",
    }


def test_socialsim_phase_sampling_has_exact_distribution_without_future_leakage():
    from ibd.socialsim import prepare_socialsim_examples

    dialogues = [_dialogue(index) for index in range(1, 101)]
    profiles = [_profile(index) for index in range(1, 101)]

    prepared = prepare_socialsim_examples(
        dialogues,
        profiles,
        seed=42,
        limit=100,
        split_sizes={"train": 60, "dev": 20, "diagnostic_holdout": 20},
    )

    assert prepared.manifest["phase_counts"] == {
        "early": 10,
        "middle": 80,
        "late": 10,
    }
    assert prepared.manifest["split_phase_counts"] == {
        "train": {"early": 6, "middle": 48, "late": 6},
        "dev": {"early": 2, "middle": 16, "late": 2},
        "diagnostic_holdout": {"early": 2, "middle": 16, "late": 2},
    }
    rank_ranges = {
        "early": range(1, 5),
        "middle": range(5, 9),
        "late": range(9, 13),
    }
    for examples in prepared.splits.values():
        for example in examples:
            assert example.target_turn is not None
            assert example.target_rank is not None
            assert example.eligible_target_count == 12
            assert example.conversation_phase is not None
            assert len(example.history.turns) == example.target_turn - 1
            assert example.history.turns[-1].role == "seeker"
            assert example.target_rank in rank_ranges[example.conversation_phase]
            serialized = json.dumps(
                example.history.model_dump(mode="json"), ensure_ascii=False
            )
            assert (
                f"support-{example.conversation_id}-{example.target_rank}"
                not in serialized
            )
            assert (
                f"REASONING-{example.conversation_id}-{example.target_rank}"
                not in serialized
            )


def test_socialsim_selection_is_stable_and_conversation_safe():
    from ibd.socialsim import prepare_socialsim_examples

    dialogues = [_dialogue(index) for index in range(1, 7)]
    profiles = [_profile(index) for index in range(1, 7)]
    split_sizes = {"train": 2, "dev": 1, "diagnostic_holdout": 1}

    first = prepare_socialsim_examples(
        dialogues,
        profiles,
        seed=42,
        limit=4,
        split_sizes=split_sizes,
    )
    second = prepare_socialsim_examples(
        list(reversed(dialogues)),
        list(reversed(profiles)),
        seed=42,
        limit=4,
        split_sizes=split_sizes,
    )

    assert first.manifest == second.manifest
    assert first.splits == second.splits
    assert first.manifest["counts"] == split_sizes
    assert set(first.splits) == {"train", "dev", "diagnostic_holdout"}

    examples = [item for split in first.splits.values() for item in split]
    assert len(examples) == 4
    assert len({item.conversation_id for item in examples}) == 4
    for item in examples:
        assert item.example_id == f"ssconv-{item.conversation_id}"
        assert item.history.turns[-1].role == "seeker"
        serialized = json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        assert "PROFILE_SECRET" not in serialized


@pytest.mark.parametrize(
    ("early_fraction", "late_fraction"),
    [
        (-0.1, 0.1),
        (0.1, float("nan")),
        (0.6, 0.4),
    ],
)
def test_socialsim_rejects_invalid_phase_fractions(
    early_fraction: float,
    late_fraction: float,
):
    from ibd.socialsim import prepare_socialsim_examples

    with pytest.raises(ValueError, match="phase fractions"):
        prepare_socialsim_examples(
            [_dialogue(1)],
            [_profile(1)],
            seed=42,
            limit=1,
            split_sizes={"train": 1, "dev": 0, "diagnostic_holdout": 0},
            early_fraction=early_fraction,
            late_fraction=late_fraction,
        )


def test_socialsim_excludes_conversations_without_three_phase_targets():
    from ibd.socialsim import prepare_socialsim_examples

    prepared = prepare_socialsim_examples(
        [_dialogue(1, pairs=3), _dialogue(2, pairs=2)],
        [_profile(1), _profile(2)],
        seed=42,
        limit=1,
        split_sizes={"train": 1, "dev": 0, "diagnostic_holdout": 0},
    )

    assert set(prepared.manifest["excluded"]) == {"2"}
    assert "fewer than three" in prepared.manifest["excluded"]["2"]


def test_socialsim_v1_allows_missing_phase_metadata_but_v2_rejects_it():
    from ibd.socialsim import PreparedSocialSim

    payload = {
        "protocol_version": "socialsim-qwen-conversation-v1",
        "seed": 42,
        "splits": {
            "train": [
                {
                    "example_id": "ssconv-1",
                    "conversation_id": "1",
                    "split": "train",
                    "history": {
                        "turns": [{"role": "seeker", "content": "help"}]
                    },
                }
            ],
            "dev": [],
            "diagnostic_holdout": [],
        },
        "manifest": {},
    }

    PreparedSocialSim.model_validate(payload)
    payload["protocol_version"] = "socialsim-qwen-conversation-v2"
    with pytest.raises(ValueError, match="phase metadata"):
        PreparedSocialSim.model_validate(payload)


def test_socialsim_requires_matching_unique_profile_ids():
    from ibd.socialsim import prepare_socialsim_examples

    dialogues = [_dialogue(1), _dialogue(2)]

    with pytest.raises(ValueError, match="profile IDs do not match"):
        prepare_socialsim_examples(
            dialogues,
            [_profile(1), _profile(3)],
            seed=42,
            limit=2,
            split_sizes={"train": 2, "dev": 0, "diagnostic_holdout": 0},
        )


def test_load_socialsim_files_does_not_record_source_hashes(tmp_path):
    from ibd.socialsim import load_socialsim_files

    dialogue_path = tmp_path / "dialogues.json"
    profile_path = tmp_path / "profiles.json"
    dialogue_path.write_text(json.dumps([_dialogue(1)]), encoding="utf-8")
    profile_path.write_text(json.dumps([_profile(1)]), encoding="utf-8")

    prepared = load_socialsim_files(
        dialogue_path,
        profile_path,
        seed=42,
        limit=1,
        split_sizes={"train": 1, "dev": 0, "diagnostic_holdout": 0},
    )

    assert not {"dialogue_sha256", "profile_sha256", "split_hash"} & set(
        prepared.manifest
    )


def test_prepare_socialsim_reports_conversation_progress(monkeypatch):
    import ibd.socialsim as socialsim

    calls = []

    def track(values, **options):
        calls.append(options)
        return values

    monkeypatch.setattr(socialsim, "track", track)

    socialsim.prepare_socialsim_examples(
        [_dialogue(1), _dialogue(2)],
        [_profile(1), _profile(2)],
        seed=42,
        limit=2,
        split_sizes={"train": 2, "dev": 0, "diagnostic_holdout": 0},
    )

    assert calls == [
        {"desc": "prepare SocialSim", "total": 2, "unit": "conversation"}
    ]
