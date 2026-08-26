"""Overall retention, functional fidelity, and Stage C controllability diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from .schemas import STATE_ANCHOR_FIELDS

Matrix = Mapping[str, Mapping[str, float]]


def aggregate_causal_metrics(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate symmetric clamp observations without merging STATE and PLAN."""

    result: dict[str, dict[str, float | int | None]] = {}
    for function in ("STATE", "PLAN"):
        rows = [row for row in observations if row.get("function") == function]
        if not rows:
            result[function] = {
                "pairs": 0,
                "original_clamp_preference_accuracy": None,
                "counterfactual_clamp_preference_accuracy": None,
                "flip_consistency": None,
                "non_target_slot_invariance_rate": None,
            }
            continue
        original_correct = [
            float(row["original_clamp_original_score"])
            > float(row["original_clamp_counterfactual_score"])
            for row in rows
        ]
        counterfactual_correct = [
            float(row["counterfactual_clamp_counterfactual_score"])
            > float(row["counterfactual_clamp_original_score"])
            for row in rows
        ]
        count = len(rows)
        result[function] = {
            "pairs": count,
            "original_clamp_preference_accuracy": sum(original_correct) / count,
            "counterfactual_clamp_preference_accuracy": (
                sum(counterfactual_correct) / count
            ),
            "flip_consistency": sum(
                left and right
                for left, right in zip(
                    original_correct, counterfactual_correct, strict=True
                )
            )
            / count,
            "non_target_slot_invariance_rate": sum(
                bool(row["non_target_slot_invariant"]) for row in rows
            )
            / count,
        }
    invalid = {row.get("function") for row in observations} - {"STATE", "PLAN"}
    if invalid:
        raise ValueError("causal observations must use STATE or PLAN")
    return result


def aggregate_intervention_audit(
    audit_rows: Sequence[Mapping[str, object]],
    *,
    planner_strategies: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, object]]:
    """Summarize Stage C attempts and single-field/candidate coverage."""

    result: dict[str, dict[str, object]] = {}
    for function in ("STATE", "PLAN"):
        rows = [row for row in audit_rows if row.get("function") == function]
        eligibility = Counter(str(row.get("eligibility")) for row in rows)
        status = Counter(str(row.get("status")) for row in rows)
        reasons = Counter(
            str(row["exclusion_reason"])
            for row in rows
            if row.get("exclusion_reason") is not None
        )
        common: dict[str, object] = {
            "attempted": len(rows),
            "eligibility": {
                "eligible": eligibility["eligible"],
                "ineligible": eligibility["ineligible"],
            },
            "status": {
                "retained": status["retained"],
                "excluded": status["excluded"],
            },
            "exclusion_reasons": dict(sorted(reasons.items())),
        }
        if function == "STATE":
            coverage = Counter(
                str(row["state_field"])
                for row in rows
                if row.get("state_field") is not None
            )
            common["field_coverage"] = {
                field: coverage[field] for field in STATE_ANCHOR_FIELDS
            }
        else:
            existing_candidate_count = 0
            for row in rows:
                planned = set(planner_strategies.get(str(row["example_id"]), ()))
                counterfactual = row.get("counterfactual_plan_categories") or ()
                if counterfactual and all(
                    str(category) in planned for category in counterfactual
                ):
                    existing_candidate_count += 1
            common["existing_candidate_reuse_rate"] = (
                existing_candidate_count / len(rows) if rows else None
            )
        result[function] = common
    invalid = {row.get("function") for row in audit_rows} - {"STATE", "PLAN"}
    if invalid:
        raise ValueError("intervention audit rows must use STATE or PLAN")
    return result


def retention(
    *,
    base: float,
    teacher: float,
    student: float,
    minimum_teacher_gain: float = 0.20,
) -> float:
    teacher_gain = teacher - base
    if teacher_gain < minimum_teacher_gain:
        raise ValueError("Teacher gain is below the preregistered minimum")
    return (student - base) / teacher_gain


def _validate_rows(matrix: Matrix) -> None:
    if set(matrix) != {"STATE", "PLAN"}:
        raise ValueError("functional fidelity matrices must contain exactly STATE and PLAN rows")


def functional_fidelity_matrix(
    *,
    full_scores: Mapping[str, float],
    ablated_scores: Matrix,
) -> dict[str, dict[str, float]]:
    _validate_rows(ablated_scores)
    dimensions = set(full_scores)
    if not dimensions:
        raise ValueError("at least one evaluation dimension is required")
    result: dict[str, dict[str, float]] = {}
    for function in ("STATE", "PLAN"):
        if set(ablated_scores[function]) != dimensions:
            raise ValueError("full and ablated score dimensions must match")
        result[function] = {
            dimension: float(full_scores[dimension] - ablated_scores[function][dimension])
            for dimension in sorted(dimensions)
        }
    return result


def _aligned_values(teacher: Matrix, student: Matrix) -> tuple[list[float], list[float]]:
    _validate_rows(teacher)
    _validate_rows(student)
    teacher_values: list[float] = []
    student_values: list[float] = []
    for function in ("STATE", "PLAN"):
        if set(teacher[function]) != set(student[function]):
            raise ValueError("Teacher and Student matrix dimensions must match")
        for dimension in sorted(teacher[function]):
            teacher_values.append(float(teacher[function][dimension]))
            student_values.append(float(student[function][dimension]))
    if len(teacher_values) < 2:
        raise ValueError("matrix alignment requires at least two cells")
    return teacher_values, student_values


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position]] = average
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for a constant matrix")
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denominator


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def matrix_alignment(teacher: Matrix, student: Matrix) -> dict[str, float]:
    teacher_values, student_values = _aligned_values(teacher, student)
    sign_agreement = sum(
        _sign(left) == _sign(right)
        for left, right in zip(teacher_values, student_values, strict=True)
    ) / len(teacher_values)
    teacher_l1 = sum(abs(value) for value in teacher_values)
    if teacher_l1 == 0:
        raise ValueError("normalized L1 is undefined for an all-zero Teacher matrix")
    normalized_l1 = sum(
        abs(left - right)
        for left, right in zip(teacher_values, student_values, strict=True)
    ) / teacher_l1
    spearman = _pearson(_average_ranks(teacher_values), _average_ranks(student_values))
    return {
        "sign_agreement": sign_agreement,
        "spearman": spearman,
        "normalized_l1": normalized_l1,
    }
