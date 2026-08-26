from __future__ import annotations

import csv
import json


def _quality_config(tmp_path):
    from ibd.config import ModelConfig
    from ibd.quality import EvaluatedModel, QualityEvalConfig

    return QualityEvalConfig(
        models=[
            EvaluatedModel(model_id="teacher", source="teacher_trace"),
            EvaluatedModel(
                model_id="base",
                source="base_qwen",
                training_config=tmp_path / "qwen.yaml",
            ),
            EvaluatedModel(
                model_id="student",
                source="student_checkpoint",
                training_config=tmp_path / "qwen.yaml",
                checkpoint=tmp_path / "checkpoint",
                run_name="seed-42",
            ),
        ],
        judge=ModelConfig(model="fixed-judge"),
    )


def test_quality_report_keeps_splits_separate_and_computes_paired_results():
    from ibd.quality import (
        QUALITY_DIMENSIONS,
        QualityJudgment,
        QualityScores,
        build_quality_report,
        compute_overall,
    )

    def row(split: str, model_id: str, empathy: int) -> QualityJudgment:
        scores = QualityScores(
            empathy=empathy,
            relevance=4,
            coherence=4,
            effectiveness=4,
            non_coerciveness=4,
        )
        return QualityJudgment(
            example_id="e-1",
            split=split,
            model_id=model_id,
            scores=scores,
            overall=compute_overall(scores),
            dimension_reasons={name: "evidence" for name in QUALITY_DIMENSIONS},
            short_reason="summary",
            call_records=[],
        )

    judgments = [
        row("dev", "student", 5),
        row("dev", "base", 3),
        row("diagnostic_holdout", "student", 2),
        row("diagnostic_holdout", "base", 4),
    ]
    expected = {
        (item.split, item.example_id, item.model_id) for item in judgments
    }

    report = build_quality_report(judgments, expected, ["student", "base"])

    assert report["splits"]["dev"]["paired"]["student__vs__base"]["empathy"][
        "mean_delta"
    ] == 2.0
    assert report["splits"]["diagnostic_holdout"]["paired"][
        "student__vs__base"
    ]["empathy"]["mean_delta"] == -2.0
    assert "safety" not in str(report).lower()


def test_generation_reuses_teacher_and_separates_base_from_student_tokens(
    tmp_path, history, app_config
):
    from conftest import ScriptedBackend
    from ibd.quality_generation import generate_quality_responses
    from ibd.storage import read_jsonl
    from ibd.teacher import TeacherRunner

    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        "e-1", history, split="dev"
    )
    calls = []

    class FakeAdapter:
        def __init__(self, model_id, uses_structure_tokens):
            self.model_id = model_id
            self.uses_structure_tokens = uses_structure_tokens

        def generate(self, seen_history, settings):
            calls.append(
                (
                    self.model_id,
                    seen_history,
                    self.uses_structure_tokens,
                    settings,
                )
            )
            return f"{self.model_id} response"

        def close(self):
            return None

    generate_quality_responses(
        _quality_config(tmp_path),
        [trace],
        output_path=tmp_path / "responses.jsonl",
        manifest_path=tmp_path / "responses.manifest.json",
        failure_path=tmp_path / "generation_failures.jsonl",
        resume=False,
        continue_on_error=False,
        device=0,
        adapter_factories={
            "base_qwen": lambda spec, device: FakeAdapter(spec.model_id, False),
            "student_checkpoint": lambda spec, device: FakeAdapter(
                spec.model_id, True
            ),
        },
    )

    rows = read_jsonl(tmp_path / "responses.jsonl")
    assert {row["model_id"] for row in rows} == {"teacher", "base", "student"}
    teacher = next(row for row in rows if row["model_id"] == "teacher")
    assert teacher["response"] == trace.final_response
    assert teacher["generation_seed"] is None
    assert [(item[0], item[2]) for item in calls] == [
        ("base", False),
        ("student", True),
    ]


def test_quality_judge_is_anonymous_and_has_exactly_five_dimensions(
    tmp_path, history
):
    from ibd.backend import LLMResult
    from ibd.quality import QualityResponse
    from ibd.quality_judge import QualityJudge

    class QualityBackend:
        def __init__(self):
            self.calls = []

        def complete(
            self, *, role, messages, model_config, json_mode=True, seed=None
        ):
            self.calls.append(
                {
                    "role": role,
                    "messages": messages,
                    "model": model_config.model,
                    "seed": seed,
                }
            )
            scores = {
                "empathy": 4,
                "relevance": 4,
                "coherence": 4,
                "effectiveness": 4,
                "non_coerciveness": 4,
            }
            return LLMResult(
                text=json.dumps(
                    {
                        **scores,
                        "dimension_reasons": {
                            name: "The candidate uses context-specific support."
                            for name in scores
                        },
                        "short_reason": "A focused and supportive next response.",
                    }
                )
            )

    backend = QualityBackend()
    response = QualityResponse(
        example_id="e-judge",
        split="dev",
        model_id="student-secret",
        response_source="student_checkpoint",
        history=history,
        response="It makes sense that repeated avoidance leaves you unsure how to begin.",
        generation_seed=42,
        generation_config=_quality_config(tmp_path).generation,
    )

    judgment = QualityJudge(backend, _quality_config(tmp_path)).score(
        response, anonymous_id="candidate-0001"
    )

    prompt_text = str(backend.calls[0]["messages"])
    assert "student-secret" not in prompt_text
    assert "checkpoint" not in prompt_text.lower()
    assert "safety" not in prompt_text.lower()
    assert judgment.overall == 4.0
    assert set(judgment.scores.model_dump()) == {
        "empathy",
        "relevance",
        "coherence",
        "effectiveness",
        "non_coerciveness",
    }


def test_human_pair_export_is_blinded_deterministic_and_summarizable(
    tmp_path, history
):
    from ibd.quality import QualityResponse
    from ibd.quality_human import (
        export_human_pairs,
        summarize_human_annotations,
    )

    config = _quality_config(tmp_path)
    response_text = {
        "teacher": "Response one.",
        "base": "Response two.",
        "student": "Response three.",
    }
    responses = [
        QualityResponse(
            example_id=example_id,
            split="dev",
            model_id=model.model_id,
            response_source=model.source,
            history=history,
            response=response_text[model.model_id],
            generation_seed=None
            if model.source == "teacher_trace"
            else config.generation.seed,
            generation_config=config.generation,
            reused=model.source == "teacher_trace",
        )
        for example_id in ("e-1", "e-2")
        for model in config.models
    ]
    public_path = tmp_path / "pairs.csv"
    mapping_path = tmp_path / "mapping.jsonl"

    export_human_pairs(responses, public_path, mapping_path, seed=73)
    first = public_path.read_text(encoding="utf-8")
    export_human_pairs(responses, public_path, mapping_path, seed=73)

    assert public_path.read_text(encoding="utf-8") == first
    assert "student" not in first and "teacher" not in first and "base" not in first

    with public_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row, preference in zip(
        rows, ["A", "B", "tie", "A", "B", "tie"], strict=True
    ):
        row["preference"] = preference
    with public_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = summarize_human_annotations(public_path, mapping_path)

    assert sum(group["valid_votes"] for group in report["groups"].values()) == 6


def test_quality_cli_registers_the_four_workflow_commands():
    from ibd.cli import _build_parser

    parser = _build_parser()
    command_action = next(
        action for action in parser._actions if action.dest == "command"
    )

    assert {
        "quality-generate",
        "quality-judge",
        "quality-human-export",
        "quality-human-summarize",
    }.issubset(command_action.choices)
