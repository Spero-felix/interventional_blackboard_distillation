"""Overall retention, functional fidelity, and alignment diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from .schemas import STATE_ANCHOR_FIELDS

Matrix = Mapping[str, Mapping[str, float]]




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
