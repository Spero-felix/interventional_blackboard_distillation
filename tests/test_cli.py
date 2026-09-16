import json

import pytest
import yaml

from conftest import ScriptedBackend
from ibd.schemas import ContextPatch, TeacherTrace, UserContext
from ibd.storage import read_jsonl, write_jsonl
from ibd.teacher import TeacherRunner


class RecordingBar:
    def __init__(self, values, call):
        self.values = list(values)
        self.call = call

    def __iter__(self):
        return iter(self.values)

    def set_postfix(self, **values):
        self.call["postfix"].append(values)


def _recording_track(calls):
    def track(values, **options):
        call = {**options, "postfix": []}
        calls.append(call)
        return RecordingBar(values, call)

    return track


def _write_config(path):
    path.write_text(
        yaml.safe_dump({"default_model": {"model": "fake-model"}}),
        encoding="utf-8",
    )


class BatchConversationBackend(ScriptedBackend):
    def __init__(self, *, fail_second_resume_seeker=False):
        super().__init__()
        self.active_profile = None
        self.fail_second_resume_seeker = fail_second_resume_seeker

    def complete(self, *, role, messages, model_config, json_mode=True, seed=None):
        from ibd.backend import LLMResult

        if role == "seeker_simulator":
            rendered = messages[1]["content"]
            for profile_id in ("p-complete", "p-truncated", "p-fail", "p-resume"):
                if profile_id in rendered:
                    self.active_profile = profile_id
                    break
            self.calls.append(
                {
                    "role": role,
                    "messages": messages,
                    "model": model_config.model,
                    "json_mode": json_mode,
                    "seed": seed,
                }
            )
            if self.active_profile == "p-fail":
                raise RuntimeError("simulated seeker provider failure")
            if (
                self.active_profile == "p-resume"
                and self.fail_second_resume_seeker
                and json.loads(messages[1]["content"])["round_index"] == 2
            ):
                raise RuntimeError("simulated second-round interruption")
            return LLMResult(text="我想谈谈最近的压力。")
        if role == "dialogue_manager":
            self.calls.append(
                {
                    "role": role,
                    "messages": messages,
                    "model": model_config.model,
                    "json_mode": json_mode,
                    "seed": seed,
                }
            )
            round_index = json.loads(messages[1]["content"])["context"]["round_index"]
            mode = (
                "closing"
                if self.active_profile == "p-complete"
                or (self.active_profile == "p-resume" and round_index == 2)
                else "exploration"
            )
            return LLMResult(
                text=json.dumps(
                    {"mode": mode, "transition_reason": f"选择 {mode}"},
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


def test_validate_trace_accepts_a_complete_teacher_trace(
    tmp_path, capsys, monkeypatch, history, app_config
):
    import ibd.cli as cli

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-valid", history)
    path = tmp_path / "trace.json"
    path.write_text(trace.model_dump_json(), encoding="utf-8")
    progress_calls = []
    monkeypatch.setattr(cli, "track", _recording_track(progress_calls))

    assert cli.main(["validate-trace", "--input", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "valid 1"
    assert progress_calls == [
        {"desc": "validate traces", "total": 1, "unit": "trace", "postfix": []}
    ]


def test_train_sft_control_accepts_only_control_training_inputs():
    import ibd.cli as cli

    args = cli._build_parser().parse_args(
        [
            "train-sft-control",
            "--config",
            "control.yaml",
            "--run-name",
            "sft-control-lr-8e-5",
            "--seed",
            "42",
            "--traces",
            "traces.jsonl",
        ]
    )

    assert args.command == "train-sft-control"
    assert args.run_dir == "runs"
    assert args.device == 0
    assert not hasattr(args, "anchors")
    assert not hasattr(args, "interventions")


def test_train_sft_control_accepts_a_static_visible_sft_dataset():
    import ibd.cli as cli

    args = cli._build_parser().parse_args(
        [
            "train-sft-control",
            "--config",
            "control.yaml",
            "--run-name",
            "visible-sft-lr-8e-5",
            "--seed",
            "42",
            "--dataset",
            "visible-sft.jsonl",
        ]
    )

    assert args.dataset == "visible-sft.jsonl"
    assert args.traces is None


def test_static_visible_sft_dataset_validates_format_and_preserves_splits(
    tmp_path, history
):
    from ibd.pipeline import read_visible_sft_dataset

    target = (
        "[dominant_emotion]sadness_loss[distress_level]high"
        "[primary_support_need]connection[advice_receptivity]open"
        "[action_intent]considering[action_capacity]limited"
        "[continuation_intent]engaged[selected_strategy]Question"
        "[response]What feels safest to say first?"
    )
    path = tmp_path / "visible-sft.jsonl"
    write_jsonl(
        path,
        [
            {
                "example_id": "train-1",
                "split": "train",
                "history": history.model_dump(mode="json"),
                "response": target,
            },
            {
                "example_id": "dev-1",
                "split": "dev",
                "history": history.model_dump(mode="json"),
                "response": target,
            },
        ],
    )

    rows = read_visible_sft_dataset(path)

    assert [row["split"] for row in rows] == ["train", "dev"]
    assert [row["response"] for row in rows] == [target, target]


def test_static_visible_sft_dataset_rejects_incomplete_prefix(tmp_path, history):
    from ibd.pipeline import read_visible_sft_dataset

    path = tmp_path / "invalid-visible-sft.jsonl"
    write_jsonl(
        path,
        [
            {
                "example_id": "broken-1",
                "split": "train",
                "history": history.model_dump(mode="json"),
                "response": "[emotion]sad[response]reply",
            }
        ],
    )

    with pytest.raises(ValueError, match="visible-SFT marker"):
        read_visible_sft_dataset(path)


def test_export_visible_sft_writes_static_training_rows(
    tmp_path, capsys, history, app_config
):
    import ibd.cli as cli

    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        "e-visible-cli", history, split="train"
    )
    input_path = tmp_path / "traces.jsonl"
    output_path = tmp_path / "visible-sft.jsonl"
    write_jsonl(input_path, [trace])

    assert cli.main(
        [
            "export-student",
            "--kind",
            "visible-sft",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    assert capsys.readouterr().out.strip() == "wrote 1"
    assert read_jsonl(output_path) == [
        {
            "example_id": "e-visible-cli",
            "split": "train",
            "history": history.model_dump(mode="json"),
                "response": (
                    f"[dominant_emotion]{trace.state.dominant_emotion}"
                    f"[distress_level]{trace.state.distress_level}"
                    f"[primary_support_need]{trace.state.primary_support_need}"
                    f"[advice_receptivity]{trace.state.advice_receptivity}"
                    f"[action_intent]{trace.state.action_intent}"
                    f"[action_capacity]{trace.state.action_capacity}"
                    f"[continuation_intent]{trace.state.continuation_intent}"
                f"[selected_strategy]{trace.final_selection.selected_strategy}"
                f"[response]{trace.final_response}"
            ),
        }
    ]


def test_run_teacher_writes_trace_through_configured_backend(
    tmp_path, monkeypatch, history
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "traces.jsonl"
    _write_config(config_path)
    input_path.write_text(
        json.dumps(
            {"example_id": "e-run", "history": history.model_dump(mode="json")},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: ScriptedBackend())
    progress_calls = []
    monkeypatch.setattr(cli, "track", _recording_track(progress_calls))

    assert cli.main(
        [
            "run-teacher",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    output = read_jsonl(output_path)[0]
    assert output["example_id"] == "e-run"
    assert output["context_before"] == UserContext.empty().model_dump(mode="json")
    assert progress_calls == [
        {
            "desc": "teacher examples",
            "total": 1,
            "unit": "example",
            "postfix": [{"example": "e-run"}],
        }
    ]


def test_run_teacher_accepts_explicit_context_before(
    tmp_path,
    monkeypatch,
    history,
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "traces.jsonl"
    _write_config(config_path)
    context_payload = UserContext.empty().model_dump(mode="json")
    context_payload["active_concerns"] = ["担心答辩"]
    input_path.write_text(
        json.dumps(
            {
                "example_id": "e-context-input",
                "history": history.model_dump(mode="json"),
                "context_before": context_payload,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: ScriptedBackend())

    assert cli.main(
        [
            "run-teacher",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    output = read_jsonl(output_path)[0]
    assert output["context_before"]["active_concerns"] == ["担心答辩"]
    assert output["context_after"]["active_concerns"] == ["担心答辩"]
    assert output["context_patch"] == ContextPatch.empty().model_dump(mode="json")
    assert output["context_merge_errors"] == []


def test_run_teacher_accepts_prepared_socialsim_artifact(
    tmp_path, monkeypatch, history
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "prepared.json"
    output_path = tmp_path / "traces.jsonl"
    _write_config(config_path)
    input_path.write_text(
        json.dumps(
            {
                "protocol_version": "socialsim-qwen-conversation-v1",
                "seed": 42,
                "splits": {
                    "train": [
                        {
                            "example_id": "ssconv-1",
                            "conversation_id": "1",
                            "split": "train",
                            "history": history.model_dump(mode="json"),
                        }
                    ],
                    "dev": [],
                    "diagnostic_holdout": [],
                },
                "manifest": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: ScriptedBackend())

    assert cli.main(
        [
            "run-teacher",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    assert read_jsonl(output_path)[0]["split"] == "train"


def test_run_teacher_resume_skips_existing_trace(
    tmp_path, monkeypatch, history, app_config
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "traces.jsonl"
    _write_config(config_path)
    input_path.write_text(
        json.dumps(
            {"example_id": "e-resume", "history": history.model_dump(mode="json")},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-resume", history)
    write_jsonl(output_path, [trace])
    backend = ScriptedBackend()
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: backend)

    assert cli.main(
        [
            "run-teacher",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--resume",
        ]
    ) == 0
    assert backend.calls == []
    assert [row["example_id"] for row in read_jsonl(output_path)] == ["e-resume"]


def test_run_teacher_rejects_existing_output_without_resume(tmp_path, history):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "traces.jsonl"
    _write_config(config_path)
    input_path.write_text(
        json.dumps(
            {"example_id": "e-existing", "history": history.model_dump(mode="json")},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        cli.main(
            [
                "run-teacher",
                "--config",
                str(config_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )


def test_run_teacher_continues_after_failure_and_writes_ledger(
    tmp_path, monkeypatch, history
):
    import ibd.cli as cli

    class PartiallyFailingRunner:
        def __init__(self, backend, config):
            self.delegate = TeacherRunner(ScriptedBackend(), config)

        def run(
            self,
            example_id,
            history,
            *,
            split="train",
            context_before=None,
        ):
            if example_id == "e-fail":
                raise RuntimeError("simulated provider failure")
            return self.delegate.run(
                example_id,
                history,
                split=split,
                context_before=context_before,
            )

    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "traces.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    _write_config(config_path)
    input_path.write_text(
        json.dumps(
            [
                {"example_id": "e-success", "history": history.model_dump(mode="json")},
                {"example_id": "e-fail", "history": history.model_dump(mode="json")},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: ScriptedBackend())
    monkeypatch.setattr(cli, "TeacherRunner", PartiallyFailingRunner)

    assert cli.main(
        [
            "run-teacher",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--continue-on-error",
            "--failures",
            str(failures_path),
        ]
    ) == 1
    assert [row["example_id"] for row in read_jsonl(output_path)] == ["e-success"]
    assert read_jsonl(failures_path) == [
        {
            "error": "simulated provider failure",
            "error_type": "RuntimeError",
            "example_id": "e-fail",
        }
    ]


def test_run_teacher_requires_failure_ledger_for_continuation(tmp_path, history):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "traces.jsonl"
    _write_config(config_path)
    input_path.write_text(
        json.dumps(
            {"example_id": "e-ledger", "history": history.model_dump(mode="json")},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="--continue-on-error requires --failures"):
        cli.main(
            [
                "run-teacher",
                "--config",
                str(config_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--continue-on-error",
            ]
        )


def test_export_student_writes_allowlisted_sft_rows(
    tmp_path, monkeypatch, history, app_config
):
    import ibd.cli as cli

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-export", history)
    input_path = tmp_path / "traces.jsonl"
    output_path = tmp_path / "sft.jsonl"
    write_jsonl(input_path, [trace])
    progress_calls = []
    monkeypatch.setattr(cli, "track", _recording_track(progress_calls))

    assert cli.main(
        [
            "export-student",
            "--kind",
            "sft",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    assert set(read_jsonl(output_path)[0]) == {
        "example_id",
        "prompt",
        "response",
        "selected_strategy",
    }
    assert progress_calls == [
        {"desc": "export sft", "total": 1, "unit": "record", "postfix": []}
    ]


def _socialsim_dialogue(dialogue_id):
    turns = []
    for rank in range(1, 7):
        turns.extend(
            [
                {
                    "Turn": 2 * rank - 1,
                    "Seeker": f"hello-{dialogue_id}-{rank}",
                },
                {
                    "Turn": 2 * rank,
                    "Supporter": f"reply-{dialogue_id}-{rank}",
                },
            ]
        )
    return {
        "ID": dialogue_id,
        "Dialogue": turns,
    }


def test_cli_registers_supported_qwen_pipeline_commands():
    from ibd.cli import _build_parser

    parser = _build_parser()
    subparsers = next(action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction")
    assert {"prepare-socialsim", "run-teacher", "precompute-anchors", "train", "train-pipeline", "generate"}.issubset(subparsers.choices)
    assert "build-interventions" not in subparsers.choices
    assert "evaluate" not in subparsers.choices

    anchor_args = parser.parse_args(
        [
            "precompute-anchors",
            "--config",
            "qwen.yaml",
            "--traces",
            "traces.jsonl",
            "--output",
            "anchors.safetensors",
        ]
    )
    assert not hasattr(anchor_args, "interventions")
    assert not hasattr(anchor_args, "global_seed")
    assert not hasattr(anchor_args, "diagnostic_state_field")

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "--config",
                "qwen.yaml",
                "--run-name",
                "run",
                "--checkpoint",
                "checkpoint",
                "--history",
                "history.json",
                "--clamp",
                "STATE",
            ]
        )


def test_prepare_socialsim_cli_writes_reproducible_artifact(tmp_path):
    from ibd.cli import main

    dialogues = [_socialsim_dialogue(index) for index in range(1, 31)]
    profiles = [{"ID": index, "Situation": f"SECRET-{index}"} for index in range(1, 31)]
    dialogue_path = tmp_path / "dialogues.json"
    profile_path = tmp_path / "profiles.json"
    output = tmp_path / "prepared.json"
    dialogue_path.write_text(json.dumps(dialogues), encoding="utf-8")
    profile_path.write_text(json.dumps(profiles), encoding="utf-8")

    assert (
        main(
            [
                "prepare-socialsim",
                "--dialogues",
                str(dialogue_path),
                "--profiles",
                str(profile_path),
                "--output",
                str(output),
                "--limit",
                "30",
                "--train-size",
                "10",
                "--dev-size",
                "10",
                "--holdout-size",
                "10",
                "--early-fraction",
                "0.1",
                "--late-fraction",
                "0.1",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol_version"] == "socialsim-qwen-conversation-v2"
    assert payload["manifest"]["counts"] == {
        "train": 10,
        "dev": 10,
        "diagnostic_holdout": 10,
    }
    assert payload["manifest"]["phase_counts"] == {
        "early": 3,
        "middle": 24,
        "late": 3,
    }
    assert payload["manifest"]["split_phase_counts"] == {
        split: {"early": 1, "middle": 8, "late": 1}
        for split in ("train", "dev", "diagnostic_holdout")
    }
    assert all(
        row["conversation_phase"]
        for rows in payload["splits"].values()
        for row in rows
    )
    assert "SECRET" not in output.read_text(encoding="utf-8")


def _conversation_cli_paths(tmp_path):
    return {
        "output": tmp_path / "conversations.jsonl",
        "truncated": tmp_path / "truncated-conversations.jsonl",
        "failures": tmp_path / "conversation-failures.jsonl",
        "checkpoints": tmp_path / "checkpoints",
    }


def _conversation_cli_args(config_path, profiles_path, paths, *extra):
    return [
        "generate-conversations",
        "--config",
        str(config_path),
        "--profiles",
        str(profiles_path),
        "--output",
        str(paths["output"]),
        "--truncated",
        str(paths["truncated"]),
        "--failures",
        str(paths["failures"]),
        "--checkpoint-dir",
        str(paths["checkpoints"]),
        "--seed",
        "42",
        *extra,
    ]


def test_generate_conversations_parser_exposes_dynamic_round_defaults():
    from ibd.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "generate-conversations",
            "--config",
            "config.yaml",
            "--profiles",
            "profiles.json",
            "--output",
            "conversations.jsonl",
            "--truncated",
            "truncated.jsonl",
            "--failures",
            "failures.jsonl",
            "--checkpoint-dir",
            "checkpoints",
            "--seed",
            "42",
        ]
    )

    assert args.min_rounds == 6
    assert args.soft_max_rounds == 16
    assert args.hard_max_rounds == 20
    assert args.split == "train"
    assert args.resume is False


def test_generate_conversations_routes_completed_truncated_and_failure_records(
    tmp_path,
    monkeypatch,
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    profiles_path = tmp_path / "profiles.json"
    paths = _conversation_cli_paths(tmp_path)
    _write_config(config_path)
    profiles_path.write_text(
        json.dumps(
            [
                {"ID": "p-complete", "Situation": "complete"},
                {"ID": "p-truncated", "Situation": "truncate"},
                {"ID": "p-fail", "Situation": "failure"},
            ]
        ),
        encoding="utf-8",
    )
    backend = BatchConversationBackend()
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: backend)

    result = cli.main(
        _conversation_cli_args(
            config_path,
            profiles_path,
            paths,
            "--min-rounds",
            "1",
            "--soft-max-rounds",
            "1",
            "--hard-max-rounds",
            "1",
        )
    )

    assert result == 1
    completed = read_jsonl(paths["output"])
    truncated = read_jsonl(paths["truncated"])
    failures = read_jsonl(paths["failures"])
    assert [(row["profile_id"], row["status"]) for row in completed] == [
        ("p-complete", "completed")
    ]
    assert [(row["profile_id"], row["status"]) for row in truncated] == [
        ("p-truncated", "truncated")
    ]
    assert failures == [
        {
            "checkpoint_path": None,
            "completed_rounds": 0,
            "conversation_id": "p-fail-seed-42",
            "error": "simulated seeker provider failure",
            "error_type": "RuntimeError",
            "failed_stage": "seeker_simulator",
            "profile_id": "p-fail",
            "seed": 42,
        }
    ]
    assert len(list(paths["checkpoints"].glob("*.json"))) == 2


def test_generate_conversations_resume_skips_finalized_outputs(tmp_path, monkeypatch):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    profiles_path = tmp_path / "profiles.json"
    paths = _conversation_cli_paths(tmp_path)
    _write_config(config_path)
    profiles_path.write_text(
        json.dumps(
            [
                {"ID": "p-complete", "Situation": "complete"},
                {"ID": "p-truncated", "Situation": "truncate"},
            ]
        ),
        encoding="utf-8",
    )
    first_backend = BatchConversationBackend()
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: first_backend)
    args = _conversation_cli_args(
        config_path,
        profiles_path,
        paths,
        "--min-rounds",
        "1",
        "--soft-max-rounds",
        "1",
        "--hard-max-rounds",
        "1",
    )
    assert cli.main(args) == 0

    replay_backend = BatchConversationBackend()
    def reject_backend_construction(config):
        raise AssertionError("finalized resume must not construct a backend")

    monkeypatch.setattr(cli, "OpenAIBackend", reject_backend_construction)
    assert cli.main([*args, "--resume"]) == 0

    assert replay_backend.calls == []
    assert len(read_jsonl(paths["output"])) == 1
    assert len(read_jsonl(paths["truncated"])) == 1


def test_generate_conversations_rejects_existing_outputs_without_resume(
    tmp_path,
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    profiles_path = tmp_path / "profiles.json"
    paths = _conversation_cli_paths(tmp_path)
    _write_config(config_path)
    profiles_path.write_text(json.dumps([{"ID": "p-complete"}]), encoding="utf-8")
    paths["output"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        cli.main(_conversation_cli_args(config_path, profiles_path, paths))


def test_generate_conversations_rejects_duplicate_normalized_profile_ids(
    tmp_path,
    monkeypatch,
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    profiles_path = tmp_path / "profiles.json"
    paths = _conversation_cli_paths(tmp_path)
    _write_config(config_path)
    profiles_path.write_text(
        json.dumps([{"ID": " duplicate "}, {"ID": "duplicate"}]),
        encoding="utf-8",
    )
    backend = BatchConversationBackend()
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: backend)

    with pytest.raises(ValueError, match="duplicate profile ID: duplicate"):
        cli.main(_conversation_cli_args(config_path, profiles_path, paths))
    assert backend.calls == []


def test_generate_conversations_records_adapter_failure_without_backend(
    tmp_path,
    monkeypatch,
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    profiles_path = tmp_path / "profiles.json"
    paths = _conversation_cli_paths(tmp_path)
    _write_config(config_path)
    profiles_path.write_text(json.dumps([{"Situation": "missing ID"}]), encoding="utf-8")

    def reject_backend_construction(config):
        raise AssertionError("adapter-only failure must not construct a backend")

    monkeypatch.setattr(cli, "OpenAIBackend", reject_backend_construction)

    assert cli.main(_conversation_cli_args(config_path, profiles_path, paths)) == 1
    assert read_jsonl(paths["failures"])[0]["failed_stage"] == "profile_adapter"


def test_generate_conversations_resume_loads_unfinished_checkpoint(
    tmp_path,
    monkeypatch,
):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    profiles_path = tmp_path / "profiles.json"
    paths = _conversation_cli_paths(tmp_path)
    _write_config(config_path)
    profiles_path.write_text(json.dumps([{"ID": "p-resume"}]), encoding="utf-8")
    first_backend = BatchConversationBackend(fail_second_resume_seeker=True)
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: first_backend)
    args = _conversation_cli_args(
        config_path,
        profiles_path,
        paths,
        "--min-rounds",
        "1",
        "--soft-max-rounds",
        "2",
        "--hard-max-rounds",
        "3",
    )

    assert cli.main(args) == 1
    assert len(list(paths["checkpoints"].glob("*.json"))) == 1

    resumed_backend = BatchConversationBackend()
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: resumed_backend)
    assert cli.main([*args, "--resume"]) == 0

    completed = read_jsonl(paths["output"])
    assert completed[0]["status"] == "completed"
    assert len(completed[0]["rounds"]) == 2
    seeker_rounds = [
        json.loads(call["messages"][1]["content"])["round_index"]
        for call in resumed_backend.calls
        if call["role"] == "seeker_simulator"
    ]
    assert seeker_rounds == [2]


def test_flatten_conversations_writes_teacher_trace_rows(tmp_path, monkeypatch):
    import ibd.cli as cli

    config_path = tmp_path / "config.yaml"
    profiles_path = tmp_path / "profiles.json"
    paths = _conversation_cli_paths(tmp_path)
    flattened_path = tmp_path / "teacher-traces.jsonl"
    _write_config(config_path)
    profiles_path.write_text(json.dumps([{"ID": "p-complete"}]), encoding="utf-8")
    monkeypatch.setattr(cli, "OpenAIBackend", lambda config: BatchConversationBackend())
    assert cli.main(
        _conversation_cli_args(
            config_path,
            profiles_path,
            paths,
            "--min-rounds",
            "1",
            "--soft-max-rounds",
            "1",
            "--hard-max-rounds",
            "1",
        )
    ) == 0

    assert cli.main(
        [
            "flatten-conversations",
            "--input",
            str(paths["output"]),
            "--output",
            str(flattened_path),
        ]
    ) == 0

    rows = read_jsonl(flattened_path)
    assert len(rows) == 1
    assert rows[0]["example_id"] == "p-complete-seed-42:round:1"
    TeacherTrace.model_validate(rows[0])
