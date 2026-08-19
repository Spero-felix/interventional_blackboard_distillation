import json
import re

import yaml

from conftest import ScriptedBackend
from ibd.storage import read_jsonl, write_jsonl
from ibd.teacher import TeacherRunner


def _write_config(path):
    path.write_text(
        yaml.safe_dump({"default_model": {"model": "fake-model"}}),
        encoding="utf-8",
    )


def test_protocol_hash_prints_one_lowercase_sha256_line(tmp_path, capsys):
    from ibd.cli import main

    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    assert main(["protocol-hash", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", output)


def test_validate_trace_accepts_a_complete_teacher_trace(
    tmp_path, capsys, history, app_config
):
    from ibd.cli import main

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-valid", history)
    path = tmp_path / "trace.json"
    path.write_text(trace.model_dump_json(), encoding="utf-8")

    assert main(["validate-trace", "--input", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "valid 1"


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


def test_export_student_writes_allowlisted_sft_rows(
    tmp_path, history, app_config
):
    from ibd.cli import main

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-export", history)
    input_path = tmp_path / "traces.jsonl"
    output_path = tmp_path / "sft.jsonl"
    write_jsonl(input_path, [trace])

    assert main(
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
    assert set(read_jsonl(output_path)[0]) == {"example_id", "prompt", "response"}

