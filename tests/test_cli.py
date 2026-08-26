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
    return {
        "ID": dialogue_id,
        "Dialogue": [
            {"Turn": 1, "Seeker": f"hello-{dialogue_id}"},
            {"Turn": 2, "Supporter": f"reply-{dialogue_id}"},
        ],
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
        ]
    )
    assert build_args.global_seed == 73

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

    dialogues = [_socialsim_dialogue(index) for index in range(1, 5)]
    profiles = [{"ID": index, "Situation": f"SECRET-{index}"} for index in range(1, 5)]
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
                "4",
                "--train-size",
                "2",
                "--dev-size",
                "1",
                "--holdout-size",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["manifest"]["counts"] == {
        "train": 2,
        "dev": 1,
        "diagnostic_holdout": 1,
    }
    assert "SECRET" not in output.read_text(encoding="utf-8")
