"""Deterministic merge logic for seeker-provided user context."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import CONTEXT_FIELDS, ContextPatch, UserContext


@dataclass(frozen=True)
class ContextMergeResult:
    context: UserContext
    errors: tuple[str, ...]


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def apply_context_patch(
    previous: UserContext,
    patch: ContextPatch,
) -> ContextMergeResult:
    merged = previous.model_dump()
    errors: list[str] = []

    for field in CONTEXT_FIELDS:
        values = list(getattr(previous, field))
        additions = list(getattr(patch.add, field))
        replacements = list(getattr(patch.replace, field))
        removals = list(getattr(patch.remove, field))
        removal_set = set(removals)

        replace_remove_conflicts = {
            pair.old for pair in replacements if pair.old in removal_set
        }
        add_remove_conflicts = {
            value for value in additions if value in removal_set
        }

        for pair in replacements:
            if pair.old in replace_remove_conflicts:
                errors.append(f"{field}: replace/remove conflict: {pair.old}")
                continue
            if pair.old not in values:
                errors.append(f"{field}: replace.old not found: {pair.old}")
                continue
            values[values.index(pair.old)] = pair.new

        for value in removals:
            if value in replace_remove_conflicts:
                continue
            if value in add_remove_conflicts:
                errors.append(f"{field}: add/remove conflict: {value}")
                continue
            if value in values:
                values.remove(value)

        for value in additions:
            if value in add_remove_conflicts:
                continue
            if value not in values:
                values.append(value)

        merged[field] = _deduplicate(values)

    return ContextMergeResult(
        context=UserContext.model_validate(merged),
        errors=tuple(errors),
    )
