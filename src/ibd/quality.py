"""Contracts and pure aggregation for generation-quality evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Mapping, Sequence

import yaml
from pydantic import Field, model_validator

from .config import AppConfig, BackendConfig, ModelConfig
from .schemas import CallRecord, History, StrictModel


QUALITY_PROTOCOL_VERSION = "generation-quality-v1"
QUALITY_JUDGE_PROMPT_VERSION = "quality-judge-v1"
QUALITY_DIMENSIONS = (
    "empathy",
    "relevance",
    "coherence",
    "effectiveness",
    "non_coerciveness",
)

QualitySplit = Literal["dev", "diagnostic_holdout"]
ResponseSource = Literal["teacher_trace", "base_qwen", "student_checkpoint"]


class GenerationSettings(StrictModel):
    max_new_tokens: int = Field(default=256, gt=0)
    do_sample: Literal[False] = False
    seed: int = 42


class EvaluatedModel(StrictModel):
    model_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    source: ResponseSource
    training_config: Path | None = None
    checkpoint: Path | None = None
    run_name: str | None = None

    @model_validator(mode="after")
    def validate_source_fields(self) -> "EvaluatedModel":
        if self.source == "teacher_trace":
            if any((self.training_config, self.checkpoint, self.run_name)):
                raise ValueError("teacher_trace must not configure local model paths")
        elif self.source == "base_qwen":
            if self.training_config is None:
                raise ValueError("base_qwen requires training_config")
            if self.checkpoint is not None or self.run_name is not None:
                raise ValueError("base_qwen must use the original model without a checkpoint")
        elif not all((self.training_config, self.checkpoint, self.run_name)):
            raise ValueError(
                "student_checkpoint requires training_config, checkpoint, and run_name"
            )
        return self


class QualityEvalConfig(StrictModel):
    protocol_version: Literal["generation-quality-v1"] = QUALITY_PROTOCOL_VERSION
    splits: tuple[QualitySplit, ...] = ("dev", "diagnostic_holdout")
    models: list[EvaluatedModel] = Field(min_length=2)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    judge: ModelConfig
    judge_seed: int = 4242
    judge_prompt_version: Literal["quality-judge-v1"] = QUALITY_JUDGE_PROMPT_VERSION
    human_seed: int = 2026
    schema_retries: int = Field(default=1, ge=0, le=1)
    teacher_model: str | None = None

    @model_validator(mode="after")
    def validate_registry(self) -> "QualityEvalConfig":
        model_ids = [item.model_id for item in self.models]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("quality model_id values must be unique")
        if sum(item.source == "teacher_trace" for item in self.models) != 1:
            raise ValueError("generation-quality-v1 requires exactly one teacher_trace")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("quality splits must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QualityEvalConfig":
        config_path = Path(path).resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = cls.model_validate(payload)
        root = config_path.parent
        resolved = []
        for item in config.models:
            updates: dict[str, Path] = {}
            for name in ("training_config", "checkpoint"):
                value = getattr(item, name)
                if value is not None and not value.is_absolute():
                    updates[name] = (root / value).resolve()
            resolved.append(item.model_copy(update=updates))
        backend = config.backend
        if backend.cache_dir is not None and not backend.cache_dir.is_absolute():
            backend = backend.model_copy(
                update={"cache_dir": (root / backend.cache_dir).resolve()}
            )
        return config.model_copy(update={"models": resolved, "backend": backend})

    def judge_app_config(self) -> AppConfig:
        # Judgment artifacts provide durable resume. Disabling role-call caching
        # prevents a changed response from reusing an older anonymous ordinal.
        backend = self.backend.model_copy(update={"cache_dir": None})
        return AppConfig(
            protocol_version=self.judge_prompt_version,
            backend=backend,
            default_model=self.judge,
            roles={"quality_judge": self.judge},
            schema_retries=self.schema_retries,
        )


class QualityResponse(StrictModel):
    protocol_version: Literal["generation-quality-v1"] = QUALITY_PROTOCOL_VERSION
    example_id: str = Field(min_length=1)
    split: QualitySplit
    model_id: str = Field(min_length=1)
    response_source: ResponseSource
    history: History
    response: str = Field(min_length=1)
    generation_seed: int | None = None
    generation_config: GenerationSettings
    reused: bool = False

    @model_validator(mode="after")
    def validate_teacher_provenance(self) -> "QualityResponse":
        if self.response_source == "teacher_trace":
            if self.generation_seed is not None or not self.reused:
                raise ValueError("teacher_trace responses must be marked reused without a seed")
        elif self.reused or self.generation_seed is None:
            raise ValueError("locally generated responses require a seed and reused=false")
        return self


class QualityScores(StrictModel):
    empathy: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    coherence: int = Field(ge=1, le=5)
    effectiveness: int = Field(ge=1, le=5)
    non_coerciveness: int = Field(ge=1, le=5)


class QualityDimensionReasons(StrictModel):
    empathy: str = Field(min_length=1)
    relevance: str = Field(min_length=1)
    coherence: str = Field(min_length=1)
    effectiveness: str = Field(min_length=1)
    non_coerciveness: str = Field(min_length=1)


def compute_overall(scores: QualityScores) -> float:
    return round(mean(float(getattr(scores, name)) for name in QUALITY_DIMENSIONS), 4)


class QualityJudgment(StrictModel):
    protocol_version: Literal["generation-quality-v1"] = QUALITY_PROTOCOL_VERSION
    example_id: str = Field(min_length=1)
    split: QualitySplit
    model_id: str = Field(min_length=1)
    scores: QualityScores
    overall: float = Field(ge=1.0, le=5.0)
    dimension_reasons: QualityDimensionReasons
    short_reason: str = Field(min_length=1, max_length=500)
    call_records: list[CallRecord]

    @model_validator(mode="after")
    def validate_overall(self) -> "QualityJudgment":
        expected = compute_overall(self.scores)
        if self.overall != expected:
            raise ValueError(f"overall must equal the five-dimension mean {expected}")
        return self


class QualityFailure(StrictModel):
    stage: Literal["generation", "judge"]
    example_id: str
    split: QualitySplit
    model_id: str
    error_type: str
    error: str


class QualityManifest(StrictModel):
    protocol_version: Literal["generation-quality-v1"] = QUALITY_PROTOCOL_VERSION
    stage: Literal["generation", "judge"]
    splits: list[QualitySplit]
    model_ids: list[str]
    expected_keys: list[tuple[QualitySplit, str, str]]
    settings: dict[str, Any]


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_manifest(path: str | Path, expected: QualityManifest, *, resume: bool) -> None:
    target = Path(path)
    if target.exists():
        if not resume:
            raise FileExistsError(f"manifest already exists: {target}; pass --resume")
        current = QualityManifest.model_validate_json(target.read_text(encoding="utf-8"))
        if current != expected:
            raise ValueError("existing quality manifest does not match requested protocol")
        return
    if resume:
        raise FileNotFoundError(f"cannot resume without manifest: {target}")
    atomic_write_json(target, expected.model_dump(mode="json"))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _score_value(judgment: QualityJudgment, dimension: str) -> float:
    return judgment.overall if dimension == "overall" else float(
        getattr(judgment.scores, dimension)
    )


def build_quality_report(
    judgments: Sequence[QualityJudgment],
    expected_keys: set[tuple[str, str, str]],
    model_ids: Sequence[str],
    failed_keys: set[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    failed_keys = failed_keys or set()
    by_key: dict[tuple[str, str, str], QualityJudgment] = {}
    for judgment in judgments:
        key = (judgment.split, judgment.example_id, judgment.model_id)
        if key in by_key:
            raise ValueError(f"duplicate quality judgment: {key}")
        by_key[key] = judgment

    splits = sorted({key[0] for key in expected_keys} | {item.split for item in judgments})
    report: dict[str, Any] = {
        "protocol_version": QUALITY_PROTOCOL_VERSION,
        "dimensions": list(QUALITY_DIMENSIONS),
        "splits": {},
    }
    all_dimensions = (*QUALITY_DIMENSIONS, "overall")
    for split in splits:
        split_expected = {key for key in expected_keys if key[0] == split}
        model_summary: dict[str, Any] = {}
        for model_id in model_ids:
            expected = {key for key in split_expected if key[2] == model_id}
            available = [
                by_key[key] for key in sorted(expected) if key in by_key
            ]
            failed = len(expected & failed_keys)
            missing = len(expected) - len(available) - failed
            model_summary[model_id] = {
                "expected": len(expected),
                "successful": len(available),
                "failed": failed,
                "missing": missing,
                "coverage_rate": round(len(available) / len(expected), 4)
                if expected
                else None,
                "means": {
                    dimension: _mean_or_none(
                        [_score_value(item, dimension) for item in available]
                    )
                    for dimension in all_dimensions
                },
            }

        paired: dict[str, Any] = {}
        for left, right in combinations(model_ids, 2):
            left_rows = {
                item.example_id: item
                for item in judgments
                if item.split == split and item.model_id == left
            }
            right_rows = {
                item.example_id: item
                for item in judgments
                if item.split == split and item.model_id == right
            }
            common = sorted(set(left_rows) & set(right_rows))
            dimension_summary: dict[str, Any] = {}
            for dimension in all_dimensions:
                deltas = [
                    _score_value(left_rows[example_id], dimension)
                    - _score_value(right_rows[example_id], dimension)
                    for example_id in common
                ]
                wins = sum(delta > 0 for delta in deltas)
                ties = sum(delta == 0 for delta in deltas)
                losses = sum(delta < 0 for delta in deltas)
                count = len(deltas)
                dimension_summary[dimension] = {
                    "pairs": count,
                    "mean_delta": _mean_or_none(deltas),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": round(wins / count, 4) if count else None,
                    "tie_rate": round(ties / count, 4) if count else None,
                    "loss_rate": round(losses / count, 4) if count else None,
                }
            paired[f"{left}__vs__{right}"] = dimension_summary

        report["splits"][split] = {"models": model_summary, "paired": paired}
    return report
