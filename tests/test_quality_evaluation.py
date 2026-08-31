from __future__ import annotations

import csv
import json

import pytest


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


def test_quality_registry_accepts_a_standard_sft_checkpoint(tmp_path):
    from ibd.quality import EvaluatedModel

    model = EvaluatedModel(
        model_id="standard-sft",
        source="standard_sft_checkpoint",
        training_config=tmp_path / "sft-control.yaml",
        checkpoint=tmp_path / "stage-SFT-step-752",
        run_name="sft-control-lr-8e-5",
    )

    assert model.source == "standard_sft_checkpoint"


def test_quality_registry_accepts_a_visible_sft_checkpoint(tmp_path):
    from ibd.quality import EvaluatedModel

    model = EvaluatedModel(
        model_id="visible-sft",
        source="visible_sft_checkpoint",
        training_config=tmp_path / "visible-sft.yaml",
        checkpoint=tmp_path / "stage-SFT-step-752",
        run_name="visible-sft-lr-8e-5",
    )

    assert model.source == "visible_sft_checkpoint"


def test_standalone_quality_config_allows_one_student_without_teacher(tmp_path):
    from ibd.config import ModelConfig
    from ibd.quality import EvaluatedModel, QualityEvalConfig

    config = QualityEvalConfig(
        standalone=True,
        models=[
            EvaluatedModel(
                model_id="ibd-c",
                source="student_checkpoint",
                training_config=tmp_path / "training.yaml",
                checkpoint=tmp_path / "stage-C-step-906",
                run_name="first-plan-ab2000-lr-8e-5",
            )
        ],
        judge=ModelConfig(model="fixed-judge"),
    )

    assert config.standalone is True
    assert [model.model_id for model in config.models] == ["ibd-c"]


def test_non_standalone_quality_config_rejects_a_single_student(tmp_path):
    from pydantic import ValidationError

    from ibd.config import ModelConfig
    from ibd.quality import EvaluatedModel, QualityEvalConfig

    with pytest.raises(ValidationError, match="non-standalone"):
        QualityEvalConfig(
            models=[
                EvaluatedModel(
                    model_id="ibd-c",
                    source="student_checkpoint",
                    training_config=tmp_path / "training.yaml",
                    checkpoint=tmp_path / "stage-C-step-906",
                    run_name="first-plan-ab2000-lr-8e-5",
                )
            ],
            judge=ModelConfig(model="fixed-judge"),
        )


def test_first_plan_c_quality_config_uses_selected_stage_c_checkpoint():
    from ibd.quality import QualityEvalConfig

    config = QualityEvalConfig.from_yaml("configs/quality_eval_first_c.yaml")

    assert config.standalone is True
    assert [model.model_id for model in config.models] == ["ibd-c-8e-5"]
    model = config.models[0]
    assert model.training_config.is_file()
    assert model.checkpoint.is_dir()
    assert model.checkpoint.name == "stage-C-step-906"


def _write_judge_artifact(tmp_path, *, model_scores, judge_seed=4242):
    from ibd.quality import (
        QUALITY_DIMENSIONS,
        QualityJudgment,
        QualityManifest,
        QualityScores,
        atomic_write_json,
        compute_overall,
    )
    from ibd.storage import append_jsonl

    model_ids = list(model_scores)
    expected_keys = [
        ("diagnostic_holdout", "example-1", model_id) for model_id in model_ids
    ]
    manifest = QualityManifest(
        stage="judge",
        splits=["diagnostic_holdout"],
        model_ids=model_ids,
        expected_keys=expected_keys,
        settings={
            "judge": {"model": "fixed-judge", "temperature": 0.0},
            "judge_seed": judge_seed,
            "prompt_version": "quality-judge-v1",
        },
    )
    atomic_write_json(
        tmp_path / "judgments.manifest.json", manifest.model_dump(mode="json")
    )
    for model_id, score in model_scores.items():
        scores = QualityScores(**{dimension: score for dimension in QUALITY_DIMENSIONS})
        append_jsonl(
            tmp_path / "judgments.jsonl",
            QualityJudgment(
                example_id="example-1",
                split="diagnostic_holdout",
                model_id=model_id,
                scores=scores,
                overall=compute_overall(scores),
                dimension_reasons={
                    dimension: "Concrete evidence." for dimension in QUALITY_DIMENSIONS
                },
                short_reason="A focused response.",
                call_records=[],
            ),
        )


def test_merge_quality_judgments_writes_report_and_ranked_six_model_results(tmp_path):
    from ibd.quality_merge import merge_quality_judgments

    base_dir = tmp_path / "base"
    standalone_dir = tmp_path / "c-only"
    output_dir = tmp_path / "combined"
    _write_judge_artifact(
        base_dir,
        model_scores={
            "teacher": 2,
            "base": 4,
            "ibd-b-8e-5": 4,
            "standard-sft-8e-5": 3,
            "visible-sft-2000-lr-8e-5": 1,
        },
    )
    _write_judge_artifact(standalone_dir, model_scores={"ibd-c-8e-5": 5})

    ranking = merge_quality_judgments(base_dir, standalone_dir, output_dir)

    assert [entry["model_id"] for entry in ranking["ranking"]] == [
        "ibd-c-8e-5",
        "base",
        "ibd-b-8e-5",
        "standard-sft-8e-5",
        "teacher",
        "visible-sft-2000-lr-8e-5",
    ]
    assert [entry["rank"] for entry in ranking["ranking"]] == [1, 2, 3, 4, 5, 6]
    assert (output_dir / "judgments.jsonl").is_file()
    assert (output_dir / "judgments.manifest.json").is_file()
    assert (output_dir / "quality_report.json").is_file()
    assert (output_dir / "ranking.json").is_file()


def test_merge_quality_judgments_rejects_incompatible_judge_settings(tmp_path):
    from ibd.quality_merge import merge_quality_judgments

    base_dir = tmp_path / "base"
    standalone_dir = tmp_path / "c-only"
    _write_judge_artifact(base_dir, model_scores={"teacher": 3, "base": 3})
    _write_judge_artifact(
        standalone_dir, model_scores={"ibd-c-8e-5": 3}, judge_seed=7
    )

    with pytest.raises(ValueError, match="Judge settings"):
        merge_quality_judgments(base_dir, standalone_dir, tmp_path / "combined")


def test_visible_sft_quality_generation_extracts_only_response_suffix(
    tmp_path, history, app_config
):
    from conftest import ScriptedBackend
    from ibd.quality import EvaluatedModel, QualityEvalConfig
    from ibd.quality_generation import generate_quality_responses
    from ibd.storage import read_jsonl
    from ibd.teacher import TeacherRunner

    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        "e-visible-quality", history, split="dev"
    )
    config = QualityEvalConfig(
        models=[
            EvaluatedModel(model_id="teacher", source="teacher_trace"),
            EvaluatedModel(
                model_id="visible-sft",
                source="visible_sft_checkpoint",
                training_config=tmp_path / "visible-sft.yaml",
                checkpoint=tmp_path / "checkpoint",
                run_name="visible-sft-lr-8e-5",
            ),
        ],
        judge=_quality_config(tmp_path).judge,
    )

    class FakeVisibleAdapter:
        def generate(self, seen_history, settings):
            assert seen_history == history
            return "[emotion]sad[selected_strategy]Question[response]I hear you."

        def close(self):
            return None

    generate_quality_responses(
        config,
        [trace],
        output_path=tmp_path / "responses.jsonl",
        manifest_path=tmp_path / "responses.manifest.json",
        failure_path=tmp_path / "generation_failures.jsonl",
        resume=False,
        continue_on_error=False,
        device=0,
        adapter_factories={
            "visible_sft_checkpoint": lambda spec, device: FakeVisibleAdapter(),
        },
    )

    visible_row = next(
        row for row in read_jsonl(tmp_path / "responses.jsonl")
        if row["model_id"] == "visible-sft"
    )
    assert visible_row["response"] == "I hear you."
    assert visible_row["raw_response"] == (
        "[emotion]sad[selected_strategy]Question[response]I hear you."
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


def test_quality_generation_reports_progress_for_each_model(
    tmp_path, monkeypatch, history, app_config
):
    from conftest import ScriptedBackend
    import ibd.quality_generation as quality_generation
    from ibd.teacher import TeacherRunner

    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        "e-progress", history, split="dev"
    )
    progress_calls = []

    class RecordingBar:
        def __init__(self, values, call):
            self.values = list(values)
            self.call = call

        def __iter__(self):
            return iter(self.values)

        def set_postfix(self, **values):
            self.call["postfix"].append(values)

    def recording_track(values, **options):
        call = {**options, "postfix": []}
        progress_calls.append(call)
        return RecordingBar(values, call)

    class FakeAdapter:
        def __init__(self, model_id):
            self.model_id = model_id

        def generate(self, seen_history, settings):
            assert seen_history == history
            return f"{self.model_id} response"

        def close(self):
            return None

    monkeypatch.setattr(quality_generation, "track", recording_track, raising=False)

    quality_generation.generate_quality_responses(
        _quality_config(tmp_path),
        [trace],
        output_path=tmp_path / "responses.jsonl",
        manifest_path=tmp_path / "responses.manifest.json",
        failure_path=tmp_path / "generation_failures.jsonl",
        resume=False,
        continue_on_error=False,
        device=0,
        adapter_factories={
            "base_qwen": lambda spec, device: FakeAdapter(spec.model_id),
            "student_checkpoint": lambda spec, device: FakeAdapter(spec.model_id),
        },
    )

    assert progress_calls == [
        {
            "desc": "quality generate teacher",
            "total": 1,
            "unit": "response",
            "postfix": [{"example": "e-progress"}],
        },
        {
            "desc": "quality generate base",
            "total": 1,
            "unit": "response",
            "postfix": [{"example": "e-progress"}],
        },
        {
            "desc": "quality generate student",
            "total": 1,
            "unit": "response",
            "postfix": [{"example": "e-progress"}],
        },
    ]


def test_quality_judge_reports_progress_for_pending_responses_on_resume(
    tmp_path, monkeypatch, history
):
    import ibd.quality_judge as quality_judge
    from ibd.quality import (
        QUALITY_DIMENSIONS,
        QualityJudgment,
        QualityResponse,
        QualityScores,
        compute_overall,
    )

    config = _quality_config(tmp_path)
    progress_calls = []
    score_calls = []

    class RecordingBar:
        def __init__(self, values, call):
            self.values = list(values)
            self.call = call

        def __iter__(self):
            return iter(self.values)

        def set_postfix(self, **values):
            self.call["postfix"].append(values)

    def recording_track(values, **options):
        call = {**options, "postfix": []}
        progress_calls.append(call)
        return RecordingBar(values, call)

    class FakeJudge:
        def __init__(self, backend, seen_config):
            assert seen_config == config

        def score(self, response, *, anonymous_id):
            score_calls.append((response.model_id, anonymous_id))
            scores = QualityScores(
                empathy=4,
                relevance=4,
                coherence=4,
                effectiveness=4,
                non_coerciveness=4,
            )
            return QualityJudgment(
                example_id=response.example_id,
                split=response.split,
                model_id=response.model_id,
                scores=scores,
                overall=compute_overall(scores),
                dimension_reasons={
                    dimension: "Concrete evidence."
                    for dimension in QUALITY_DIMENSIONS
                },
                short_reason="A focused response.",
                call_records=[],
            )

    def response_for(model):
        is_teacher = model.source == "teacher_trace"
        return QualityResponse(
            example_id="e-progress",
            split="dev",
            model_id=model.model_id,
            response_source=model.source,
            history=history,
            response=f"{model.model_id} response",
            generation_seed=None if is_teacher else config.generation.seed,
            generation_config=config.generation,
            reused=is_teacher,
        )

    monkeypatch.setattr(quality_judge, "track", recording_track, raising=False)
    monkeypatch.setattr(quality_judge, "QualityJudge", FakeJudge)
    paths = {
        "output_path": tmp_path / "judgments.jsonl",
        "manifest_path": tmp_path / "judgments.manifest.json",
        "failure_path": tmp_path / "judge_failures.jsonl",
        "report_path": tmp_path / "quality_report.json",
    }

    teacher = next(model for model in config.models if model.model_id == "teacher")
    quality_judge.judge_quality_responses(
        config,
        [response_for(teacher)],
        resume=False,
        continue_on_error=False,
        backend=object(),
        **paths,
    )
    progress_calls.clear()
    score_calls.clear()

    quality_judge.judge_quality_responses(
        config,
        [response_for(model) for model in config.models],
        resume=True,
        continue_on_error=False,
        backend=object(),
        **paths,
    )

    assert progress_calls == [
        {
            "desc": "quality judge",
            "total": 2,
            "unit": "response",
            "postfix": [
                {"example": "e-progress", "model": "base"},
                {"example": "e-progress", "model": "student"},
            ],
        }
    ]
    assert score_calls == [
        ("base", "candidate-000001"),
        ("student", "candidate-000002"),
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


def test_quality_cli_registers_quality_merge_command():
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
        "quality-merge",
    }.issubset(command_action.choices)

    args = parser.parse_args(
        [
            "quality-merge",
            "--base-dir",
            "base",
            "--standalone-dir",
            "c-only",
            "--output-dir",
            "combined",
        ]
    )
    assert args.base_dir == "base"
    assert args.standalone_dir == "c-only"
    assert args.output_dir == "combined"
