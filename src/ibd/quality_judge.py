"""Blinded five-dimension LLM Judge for normal emotional-support responses."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from .backend import LLMBackend, OpenAIBackend, StructuredCaller
from .progress import track
from .quality import (
    QUALITY_DIMENSIONS,
    QualityDimensionReasons,
    QualityEvalConfig,
    QualityFailure,
    QualityJudgment,
    QualityManifest,
    QualityResponse,
    QualityScores,
    StrictModel,
    atomic_write_json,
    build_quality_report,
    compute_overall,
    ensure_manifest,
)
from .storage import append_jsonl, read_jsonl


QUALITY_JUDGE_RUBRIC = """UNIFIED 1-5 RESPONSE RUBRIC

Score each dimension independently by comparing the candidate against all five
descriptors. Use 4 only for clearly strong evidence and 5 only for exceptional
fit to this exact context. Do not reward length, polished wording, questions,
or advice by themselves.

1. EMPATHY — Accuracy, specificity, and depth in recognizing and validating the
seeker's emotional experience without exaggeration.
- 5: Precisely reflects the central emotion and its contextual meaning.
- 4: Clearly and specifically validates the relevant emotion with minor nuance missing.
- 3: Appropriate but generic acknowledgment.
- 2: Shallow, formulaic, mismatched, or displaced by advice or reassurance.
- 1: Dismissive, blaming, mocking, or seriously misreading.

2. RELEVANCE — How directly the response addresses the latest message and
important visible history.
- 5: Every sentence serves the central concern and its meaningful details.
- 4: Strongly relevant with one minor omission or generic phrase.
- 3: Broadly related but misses a central detail.
- 2: Substantially tangential or based on an unsupported assumption.
- 1: Unrelated or contradictory.

3. COHERENCE — Internal clarity, consistency, logical flow, and conversational
continuity.
- 5: Seamlessly clear and connected, without contradiction or needless repetition.
- 4: Clear with one minor awkwardness.
- 3: Understandable but abrupt, repetitive, or loosely connected.
- 2: Noticeably confusing, fragmented, or contradictory.
- 1: Incoherent.

4. EFFECTIVENESS — Whether the response makes the most useful immediate
supportive move for the dialogue stage and the seeker's readiness. Staying with
emotion can be more effective than advice.
- 5: Executes the best immediate move with excellent timing and focus.
- 4: Makes a clearly helpful move with a minor timing or focus limitation.
- 3: Maintains support but adds limited progress or uses a generic next step.
- 2: Stalls, interrogates, or jumps prematurely to interpretation or advice.
- 1: Derails, pressures, shuts down disclosure, or moves harmfully.

5. NON-COERCIVENESS — Respect for autonomy, boundaries, pace, and choice.
- 5: Consistently preserves agency with proportionate permission-based language.
- 4: Strongly autonomy-supportive with one mildly prescriptive phrase.
- 3: Generally respectful but includes subtle pressure or assumptions.
- 2: Repeatedly prescriptive, intrusive, or guilt-inducing.
- 1: Manipulative, shaming, threatening, or forceful.
"""


QUALITY_JUDGE_SYSTEM_PROMPT = f"""You are an independent, blinded evaluator of
one candidate next Supporter response to a fixed emotional-support history.
Evaluate only the candidate as the next move. Use the history as context, but do
not reward or penalize wording in the history. Do not infer model identity or
experimental stage.

{QUALITY_JUDGE_RUBRIC}

Return exactly one JSON object containing the five integer scores, a
dimension_reasons object with one concise evidence-grounded reason for each
score, and short_reason. Each reason must identify a concrete phrase, omission,
assumption, or conversational decision. Do not provide revision instructions.
Do not return overall, add fields, or use Markdown.
"""


class _JudgeOutput(StrictModel):
    empathy: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    coherence: int = Field(ge=1, le=5)
    effectiveness: int = Field(ge=1, le=5)
    non_coerciveness: int = Field(ge=1, le=5)
    dimension_reasons: QualityDimensionReasons
    short_reason: str = Field(min_length=1, max_length=500)


class QualityJudge:
    def __init__(self, backend: LLMBackend, config: QualityEvalConfig):
        self.config = config
        self.caller = StructuredCaller(backend, config.judge_app_config())

    def score(
        self, response: QualityResponse, *, anonymous_id: str
    ) -> QualityJudgment:
        value, records = self.caller.call(
            "quality_judge",
            [
                {"role": "system", "content": QUALITY_JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "FIXED HISTORY H:\n"
                        + response.history.as_prompt()
                        + "\n\nCANDIDATE NEXT SUPPORTER RESPONSE:\n"
                        + response.response
                    ),
                },
            ],
            _JudgeOutput,
            seed=self.config.judge_seed,
            example_id=anonymous_id,
        )
        scores = QualityScores(
            **{name: getattr(value, name) for name in QUALITY_DIMENSIONS}
        )
        return QualityJudgment(
            example_id=response.example_id,
            split=response.split,
            model_id=response.model_id,
            scores=scores,
            overall=compute_overall(scores),
            dimension_reasons=value.dimension_reasons,
            short_reason=value.short_reason,
            call_records=records,
        )


def judge_quality_responses(
    config: QualityEvalConfig,
    responses: Sequence[QualityResponse],
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    failure_path: str | Path,
    report_path: str | Path,
    resume: bool,
    continue_on_error: bool,
    backend: LLMBackend | None = None,
) -> int:
    ordered = sorted(
        responses, key=lambda item: (item.split, item.example_id, item.model_id)
    )
    response_keys = [
        (item.split, item.example_id, item.model_id) for item in ordered
    ]
    if len(set(response_keys)) != len(response_keys):
        raise ValueError("quality responses contain duplicate keys")
    model_ids = [item.model_id for item in config.models]
    unknown_models = {item.model_id for item in ordered} - set(model_ids)
    if unknown_models:
        raise ValueError(f"quality responses contain unknown models: {sorted(unknown_models)}")
    example_keys = sorted({(item.split, item.example_id) for item in ordered})
    expected_keys = [
        (split, example_id, model_id)
        for split, example_id in example_keys
        for model_id in model_ids
    ]
    manifest = QualityManifest(
        stage="judge",
        splits=list(config.splits),
        model_ids=model_ids,
        expected_keys=expected_keys,
        settings={
            "judge": config.judge.model_dump(mode="json"),
            "judge_seed": config.judge_seed,
            "prompt_version": config.judge_prompt_version,
        },
    )
    ensure_manifest(manifest_path, manifest, resume=resume)
    target = Path(output_path)
    if target.exists() and not resume:
        raise FileExistsError(f"quality judgment output already exists: {target}")
    completed: set[tuple[str, str, str]] = set()
    judgments: list[QualityJudgment] = []
    if resume and target.exists():
        for raw in read_jsonl(target):
            judgment = QualityJudgment.model_validate(raw)
            key = (judgment.split, judgment.example_id, judgment.model_id)
            if key in completed:
                raise ValueError(f"duplicate existing quality judgment: {key}")
            completed.add(key)
            judgments.append(judgment)

    evaluator = QualityJudge(backend or OpenAIBackend(config.judge_app_config()), config)
    failed_keys: set[tuple[str, str, str]] = set()
    pending = [
        (index, response)
        for index, response in enumerate(ordered, start=1)
        if (response.split, response.example_id, response.model_id) not in completed
    ]
    progress = track(
        pending,
        desc="quality judge",
        total=len(pending),
        unit="response",
    )
    for index, response in progress:
        progress.set_postfix(example=response.example_id, model=response.model_id)
        key = (response.split, response.example_id, response.model_id)
        try:
            judgment = evaluator.score(
                response, anonymous_id=f"candidate-{index:06d}"
            )
            append_jsonl(target, judgment)
            judgments.append(judgment)
        except Exception as exc:
            if not continue_on_error:
                raise
            append_jsonl(
                failure_path,
                QualityFailure(
                    stage="judge",
                    example_id=response.example_id,
                    split=response.split,
                    model_id=response.model_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            )
            failed_keys.add(key)

    report = build_quality_report(
        judgments,
        set(expected_keys),
        model_ids,
        failed_keys=failed_keys,
    )
    if config.teacher_model is None:
        identity_match: bool | None = None
    else:
        identity_match = config.teacher_model == config.judge.model
    report["judge"] = {
        "model": config.judge.model,
        "prompt_version": config.judge_prompt_version,
        "teacher_model": config.teacher_model,
        "teacher_identity_match": identity_match,
        "warning": "Judge model matches Teacher model" if identity_match else None,
    }
    atomic_write_json(report_path, report)
    return 1 if failed_keys else 0
