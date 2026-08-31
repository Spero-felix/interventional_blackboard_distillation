import json

import pytest
import yaml

from conftest import ScriptedBackend
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
        "[emotion]sad[intensity]high[primary_need]connection"
        "[support_goal]clarity[readiness]ready[main_constraint]fear"
        "[relationship_context]partner[selected_strategy]Question"
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
                f"[emotion]{trace.state.emotion}"
                f"[intensity]{trace.state.intensity}"
                f"[primary_need]{trace.state.primary_need}"
                f"[support_goal]{trace.state.support_goal}"
                f"[readiness]{trace.state.readiness}"
                f"[main_constraint]{trace.state.main_constraint}"
                f"[relationship_context]{trace.state.relationship_context}"
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
    assert read_jsonl(output_path)[0]["example_id"] == "e-run"
    assert progress_calls == [
        {
            "desc": "teacher examples",
            "total": 1,
            "unit": "example",
            "postfix": [{"example": "e-run"}],
        }
    ]


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

        def run(self, example_id, history, *, split="train"):
            if example_id == "e-fail":
                raise RuntimeError("simulated provider failure")
            return self.delegate.run(example_id, history, split=split)

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


def test_cli_registers_complete_qwen_pipeline_command_surface():
    from ibd.cli import _build_parser

    parser = _build_parser()
    subparsers = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )

    assert {
        "prepare-socialsim",
        "run-teacher",
        "build-interventions",
        "precompute-anchors",
        "train",
        "train-pipeline",
        "evaluate",
        "generate",
    }.issubset(subparsers.choices)

    build_args = parser.parse_args(
        [
            "build-interventions",
            "--config",
            "teacher.yaml",
            "--input",
            "traces.jsonl",
            "--output",
            "interventions.jsonl",
            "--manifest",
            "manifest.json",
            "--global-seed",
            "73",
            "--state-plan-policy",
            "fixed-original",
        ]
    )
    assert build_args.global_seed == 73
    assert build_args.state_plan_policy == "fixed-original"

    anchor_args = parser.parse_args(
        [
            "precompute-anchors",
            "--config",
            "qwen.yaml",
            "--traces",
            "traces.jsonl",
            "--output",
            "anchors.safetensors",
            "--global-seed",
            "73",
            "--diagnostic-state-field",
            "primary_need",
            "--original-splits",
            "dev",
        ]
    )
    assert anchor_args.global_seed == 73
    assert anchor_args.diagnostic_state_field == "primary_need"
    assert anchor_args.original_splits == ["dev"]

    evaluate_args = parser.parse_args(
        [
            "evaluate",
            "--config",
            "qwen.yaml",
            "--run-name",
            "seed-42",
            "--checkpoint",
            "checkpoint",
            "--traces",
            "traces.jsonl",
            "--interventions",
            "interventions.jsonl",
            "--anchors",
            "anchors.safetensors",
            "--intervention-manifest",
            "manifest.json",
            "--output",
            "evaluation.json",
        ]
    )
    assert evaluate_args.interventions == "interventions.jsonl"
    assert evaluate_args.intervention_manifest == "manifest.json"

    generate_args = parser.parse_args(
        [
            "generate",
            "--config",
            "qwen.yaml",
            "--run-name",
            "seed-42",
            "--checkpoint",
            "checkpoint",
            "--history",
            "history.json",
            "--clamp",
            "STATE",
            "--anchors",
            "anchors.safetensors",
            "--example-id",
            "e-1",
        ]
    )
    assert generate_args.clamp == "STATE"

    train_args = parser.parse_args(
        [
            "train",
            "--stage",
            "B2",
            "--config",
            "qwen.yaml",
            "--run-name",
            "seed-42",
            "--seed",
            "42",
            "--traces",
            "traces.jsonl",
            "--interventions",
            "interventions.jsonl",
            "--anchors",
            "anchors.safetensors",
        ]
    )
    assert train_args.stage == "B2"


@pytest.mark.parametrize("command", ["build-interventions", "precompute-anchors"])
def test_dataset_build_commands_require_an_explicit_global_seed(command, capsys):
    from ibd.cli import _build_parser

    common = ["--config", "config.yaml"]
    if command == "build-interventions":
        argv = [
            command,
            *common,
            "--input",
            "traces.jsonl",
            "--output",
            "interventions.jsonl",
            "--margins-output",
            "margins.jsonl",
            "--manifest",
            "manifest.json",
        ]
    else:
        argv = [
            command,
            *common,
            "--traces",
            "traces.jsonl",
            "--output",
            "anchors.safetensors",
            "--diagnostic-state-field",
            "emotion",
        ]

    with pytest.raises(SystemExit):
        _build_parser().parse_args(argv)
    assert "--global-seed" in capsys.readouterr().err


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
