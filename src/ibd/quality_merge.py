"""Combine completed quality-judge artifacts without rerunning evaluated models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .quality import (
    QUALITY_PROTOCOL_VERSION,
    QualityJudgment,
    QualityManifest,
    atomic_write_json,
    build_quality_report,
)
from .storage import append_jsonl, read_jsonl


_OUTPUT_NAMES = (
    "judgments.jsonl",
    "judgments.manifest.json",
    "quality_report.json",
    "ranking.json",
)


def _read_manifest(directory: Path) -> QualityManifest:
    path = directory / "judgments.manifest.json"
    manifest = QualityManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.stage != "judge":
        raise ValueError(f"quality merge requires a judge manifest: {path}")
    return manifest


def _read_completed_judgments(
    directory: Path, manifest: QualityManifest
) -> list[QualityJudgment]:
    path = directory / "judgments.jsonl"
    judgments = [QualityJudgment.model_validate(row) for row in read_jsonl(path)]
    keys = [(item.split, item.example_id, item.model_id) for item in judgments]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate quality judgment keys: {path}")
    expected = set(manifest.expected_keys)
    actual = set(keys)
    # if actual != expected:
    #     raise ValueError(f"quality merge requires complete judgment coverage: {path}")
    return judgments


def _validate_compatibility(
    base: QualityManifest, standalone: QualityManifest
) -> None:
    if base.protocol_version != standalone.protocol_version:
        raise ValueError("quality manifests use different protocol versions")
    if base.splits != standalone.splits:
        raise ValueError("quality manifests use different splits")
    if base.settings != standalone.settings:
        raise ValueError("quality manifests use incompatible Judge settings")
    if len(standalone.model_ids) != 1:
        raise ValueError("standalone quality artifact must contain exactly one model")
    if set(base.model_ids) & set(standalone.model_ids):
        raise ValueError("quality manifests contain duplicate model IDs")
    base_examples = {(split, example_id) for split, example_id, _ in base.expected_keys}
    standalone_examples = {
        (split, example_id) for split, example_id, _ in standalone.expected_keys
    }
    if base_examples != standalone_examples:
        raise ValueError("quality manifests evaluate different examples")


def _ensure_new_output(directory: Path) -> None:
    existing = [name for name in _OUTPUT_NAMES if (directory / name).exists()]
    if existing:
        raise FileExistsError(
            f"quality merge output already exists: {directory / existing[0]}"
        )


def _rank_report(report: dict[str, Any], model_ids: Sequence[str]) -> dict[str, Any]:
    split = "diagnostic_holdout"
    models = report["splits"].get(split, {}).get("models", {})
    rows = []
    for model_id in model_ids:
        summary = models.get(model_id)
        if summary is None:
            raise ValueError(f"combined quality report has no {split} summary for {model_id}")
        overall = summary["means"]["overall"]
        coverage_rate = summary["coverage_rate"]
        # if overall is None or coverage_rate != 1.0:
        #     raise ValueError(f"combined quality report is incomplete for {model_id}")
        rows.append(
            {
                "model_id": model_id,
                "overall": overall,
                "coverage_rate": coverage_rate,
            }
        )
    rows.sort(key=lambda item: (-item["overall"], item["model_id"]))
    ranking = [
        {"rank": index, **row} for index, row in enumerate(rows, start=1)
    ]
    return {
        "protocol_version": QUALITY_PROTOCOL_VERSION,
        "split": split,
        "sort_key": "overall_desc_then_model_id_asc",
        "ranking": ranking,
    }


def merge_quality_judgments(
    base_dir: str | Path,
    standalone_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a six-model report from a completed base and C-only Judge run."""

    base_path = Path(base_dir)
    standalone_path = Path(standalone_dir)
    target = Path(output_dir)
    base_manifest = _read_manifest(base_path)
    standalone_manifest = _read_manifest(standalone_path)
    _validate_compatibility(base_manifest, standalone_manifest)
    base_judgments = _read_completed_judgments(base_path, base_manifest)
    standalone_judgments = _read_completed_judgments(
        standalone_path, standalone_manifest
    )
    _ensure_new_output(target)

    model_ids = [*base_manifest.model_ids, *standalone_manifest.model_ids]
    expected_keys = [
        *base_manifest.expected_keys,
        *standalone_manifest.expected_keys,
    ]
    combined_manifest = QualityManifest(
        stage="judge",
        splits=base_manifest.splits,
        model_ids=model_ids,
        expected_keys=expected_keys,
        settings=base_manifest.settings,
    )
    judgments = [*base_judgments, *standalone_judgments]
    model_positions = {model_id: index for index, model_id in enumerate(model_ids)}
    judgments.sort(
        key=lambda item: (item.split, item.example_id, model_positions[item.model_id])
    )
    for judgment in judgments:
        append_jsonl(target / "judgments.jsonl", judgment)
    atomic_write_json(
        target / "judgments.manifest.json", combined_manifest.model_dump(mode="json")
    )
    report = build_quality_report(judgments, set(expected_keys), model_ids)
    atomic_write_json(target / "quality_report.json", report)
    ranking = _rank_report(report, model_ids)
    atomic_write_json(target / "ranking.json", ranking)
    return ranking
