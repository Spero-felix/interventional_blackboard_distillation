#!/usr/bin/env python3
"""Create an auditable, balanced derivative of Teacher traces.

This is deliberately a data-operations script rather than an ``ibd`` CLI
feature.  It reuses the project's validated contracts and generation backend
while keeping balancing policy and outputs outside ``src/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from ibd.backend import OpenAIBackend
from ibd.config import AppConfig
from ibd.schemas import Candidate, FinalSelection, StrategyName, TeacherTrace
from ibd.teacher import TeacherRunner


TARGET_TOTALS: dict[StrategyName, int] = {
    "Reflection of feelings": 592,
    "Question": 429,
    "Providing Suggestions": 379,
    "Affirmation and Reassurance": 324,
    "Restatement or Paraphrasing": 297,
    "Others": 279,
    "Information": 123,
    "Self-disclosure": 77,
}

TRANSITION_QUOTAS: dict[str, dict[StrategyName, int]] = {
    "train": {
        "Information": 83,
        "Restatement or Paraphrasing": 208,
        "Self-disclosure": 62,
    },
    "dev": {
        "Information": 9,
        "Restatement or Paraphrasing": 20,
        "Self-disclosure": 6,
    },
    "diagnostic_holdout": {
        "Information": 13,
        "Restatement or Paraphrasing": 27,
        "Self-disclosure": 9,
    },
}

TARGET_STRATEGIES: tuple[StrategyName, ...] = (
    "Information",
    "Restatement or Paraphrasing",
    "Self-disclosure",
)


@dataclass(frozen=True)
class Assignment:
    """One Reflection-to-target transition in the derivative dataset."""

    example_id: str
    split: str
    target_strategy: StrategyName
    source: Literal["reuse", "generate"]
    candidate_id: str


def _rank(seed: int, target_strategy: StrategyName, example_id: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{target_strategy}\0{example_id}".encode("utf-8")
    ).hexdigest()


def _stratum(trace: TeacherTrace, prepared_row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(prepared_row.get("conversation_phase", "unknown")),
        trace.state.dominant_emotion,
        trace.state.distress_level,
    )


def _reusable_candidate(trace: TeacherTrace, strategy: StrategyName):
    matches = [
        candidate
        for candidate in trace.candidates
        if candidate.candidate_id != trace.final_selection.selected_candidate_id
        and candidate.strategy == strategy
    ]
    if len(matches) > 1:
        raise ValueError(
            f"example {trace.example_id} has multiple reusable {strategy!r} candidates"
        )
    return matches[0] if matches else None


def _allocate_by_stratum(
    traces: Sequence[TeacherTrace],
    prepared: Mapping[str, Mapping[str, Any]],
    quota: int,
    *,
    seed: int,
    target_strategy: StrategyName,
) -> list[TeacherTrace]:
    """Choose a deterministic, largest-remainder sample over source strata."""
    if quota > len(traces):
        raise ValueError(
            f"short quota for {target_strategy}: need {quota}, have {len(traces)}"
        )
    groups: dict[tuple[str, str, str], list[TeacherTrace]] = defaultdict(list)
    for trace in traces:
        groups[_stratum(trace, prepared[trace.example_id])].append(trace)
    total = len(traces)
    allocation: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[int, tuple[str, str, str]]] = []
    for key, rows in groups.items():
        numerator = quota * len(rows)
        allocation[key] = numerator // total
        remainders.append((numerator % total, key))
    remaining = quota - sum(allocation.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocation[key] += 1

    selected: list[TeacherTrace] = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda trace: _rank(seed, target_strategy, trace.example_id),
        )
        selected.extend(ordered[: allocation[key]])
    return selected


def plan_assignments(
    traces: Sequence[TeacherTrace],
    prepared: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> list[Assignment]:
    """Plan all quota-exact, non-overlapping Reflection transitions."""
    by_id: dict[str, TeacherTrace] = {}
    for trace in traces:
        if trace.example_id in by_id:
            raise ValueError(f"duplicate example_id {trace.example_id}")
        if trace.example_id not in prepared:
            raise ValueError(f"prepared metadata missing {trace.example_id}")
        if prepared[trace.example_id].get("split") != trace.split:
            raise ValueError(f"prepared split mismatch for {trace.example_id}")
        by_id[trace.example_id] = trace

    assignments: list[Assignment] = []
    assigned_ids: set[str] = set()
    for split, quotas in TRANSITION_QUOTAS.items():
        source_rows = [
            trace
            for trace in traces
            if trace.split == split
            and trace.final_selection.selected_strategy == "Reflection of feelings"
        ]
        for strategy_index, strategy in enumerate(TARGET_STRATEGIES):
            quota = quotas[strategy]
            available = [trace for trace in source_rows if trace.example_id not in assigned_ids]
            reusable_rows = [
                trace for trace in available if _reusable_candidate(trace, strategy) is not None
            ]
            reuse_selected = _allocate_by_stratum(
                reusable_rows,
                prepared,
                min(quota, len(reusable_rows)),
                seed=seed,
                target_strategy=strategy,
            ) if reusable_rows else []
            reused_ids = {item.example_id for item in reuse_selected}
            remaining = [trace for trace in available if trace.example_id not in reused_ids]
            needed_generation = quota - len(reuse_selected)
            later_strategies = TARGET_STRATEGIES[strategy_index + 1 :]
            generation_pool = [
                trace
                for trace in remaining
                if not any(_reusable_candidate(trace, later) is not None for later in later_strategies)
            ]
            generated_selected = _allocate_by_stratum(
                generation_pool,
                prepared,
                min(needed_generation, len(generation_pool)),
                seed=seed,
                target_strategy=strategy,
            ) if needed_generation and generation_pool else []
            generated_ids = {item.example_id for item in generated_selected}
            if len(generated_selected) < needed_generation:
                generated_selected.extend(
                    _allocate_by_stratum(
                        [trace for trace in remaining if trace.example_id not in generated_ids],
                        prepared,
                        needed_generation - len(generated_selected),
                        seed=seed,
                        target_strategy=strategy,
                    )
                )
            selected = [*reuse_selected, *generated_selected]
            for trace in selected:
                if trace.example_id in assigned_ids:
                    raise ValueError(f"duplicate assignment {trace.example_id}")
                reusable = _reusable_candidate(trace, strategy)
                selected_candidate = next(
                    candidate
                    for candidate in trace.candidates
                    if candidate.candidate_id
                    == trace.final_selection.selected_candidate_id
                )
                planned_target = next(
                    (candidate for candidate in trace.candidates if candidate.strategy == strategy),
                    None,
                )
                assignments.append(
                    Assignment(
                        example_id=trace.example_id,
                        split=split,
                        target_strategy=strategy,
                        source="reuse" if reusable is not None else "generate",
                        candidate_id=(
                            reusable.candidate_id
                            if reusable is not None
                            else (
                                planned_target.candidate_id
                                if planned_target is not None
                                else selected_candidate.candidate_id
                            )
                        ),
                    )
                )
                assigned_ids.add(trace.example_id)

    expected = sum(sum(quotas.values()) for quotas in TRANSITION_QUOTAS.values())
    if len(assignments) != expected:
        raise ValueError(f"planned {len(assignments)} assignments; expected {expected}")
    return assignments


_FIRST_PERSON = re.compile(r"\b(i|i'm|i’ve|i've|me|my|mine)\b", re.IGNORECASE)
_DIRECTIVE = re.compile(r"\b(you should|try|consider)\b", re.IGNORECASE)
_SELF_DISCLOSURE_RISK = re.compile(
    r"\b(therapist|doctor|clinician|diagnos\w*|medication|prescri\w*|"
    r"self-harm|suicide|abuse|crime|trauma|race|religion|sexuality)\b",
    re.IGNORECASE,
)


def _latest_seeker_tokens(trace: TeacherTrace) -> set[str]:
    latest = trace.history.turns[-1].content.lower()
    return set(re.findall(r"\b[\w']+\b", latest))


def _candidate_rejection_reason(
    trace: TeacherTrace, strategy: StrategyName, candidate: Candidate
) -> str | None:
    response = candidate.response
    if strategy == "Information" and _FIRST_PERSON.search(response):
        return "first-person"
    if strategy == "Restatement or Paraphrasing":
        if _DIRECTIVE.search(response):
            return "directive"
        response_tokens = set(re.findall(r"\b[\w']+\b", response.lower()))
        if not response_tokens.intersection(_latest_seeker_tokens(trace)):
            return "no-overlap"
    if strategy == "Self-disclosure" and _SELF_DISCLOSURE_RISK.search(response):
        return "self-disclosure"
    return None


def generate_target_candidate(
    runner: Any,
    trace: TeacherTrace,
    strategy: StrategyName,
    candidate_id: str,
) -> tuple[Candidate, list[Any]]:
    """Use the established Teacher candidate call without extending its API."""
    if candidate_id not in {"1", "2", "3"}:
        raise ValueError(f"unsupported candidate_id {candidate_id!r}")
    records: list[Any] = []
    candidate = runner._generate_candidate(
        example_id=trace.example_id,
        history=trace.history,
        state=trace.state,
        user_context=trace.context_after,
        strategy=strategy,
        local_index=int(candidate_id),
        records=records,
    )
    return candidate, records


def _review_row(
    trace: TeacherTrace,
    assignment: Assignment,
    *,
    old_candidate: Candidate,
    new_candidate: Candidate,
    review_flags: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "example_id": trace.example_id,
        "split": trace.split,
        "conversation_phase": "unknown",
        "source": assignment.source,
        "target_strategy": assignment.target_strategy,
        "candidate_id": assignment.candidate_id,
        "old_candidate": old_candidate.model_dump(mode="json"),
        "new_candidate": new_candidate.model_dump(mode="json"),
        "state": trace.state.model_dump(mode="json"),
        "review_flags": list(review_flags),
        "review_status": "pending",
    }


def _validated_trace(payload: dict[str, Any]) -> TeacherTrace:
    return TeacherTrace.model_validate(payload)


def _record_payload(record: Any) -> dict[str, Any]:
    return (
        record.model_dump(mode="json")
        if hasattr(record, "model_dump")
        else dict(record)
    )


def apply_assignments(
    traces: Sequence[TeacherTrace],
    assignments: Sequence[Assignment],
    runner: Any,
    *,
    max_generation_attempts: int = 3,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[TeacherTrace], list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild canonical traces for reuse/generation assignments.

    Heuristic strategy-boundary hits are retained as review flags.  They do
    not discard rows from the pending-human-review derivative.
    """
    if max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be at least one")
    by_id = {trace.example_id: trace for trace in traces}
    if len(by_id) != len(traces):
        raise ValueError("duplicate trace IDs")
    by_assignment = {assignment.example_id: assignment for assignment in assignments}
    if len(by_assignment) != len(assignments):
        raise ValueError("duplicate assignments")

    derived: list[TeacherTrace] = []
    review_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for trace in traces:
        assignment = by_assignment.get(trace.example_id)
        if assignment is None:
            derived.append(trace)
            continue
        if assignment.split != trace.split:
            raise ValueError(f"assignment split mismatch for {trace.example_id}")
        if trace.final_selection.selected_strategy != "Reflection of feelings":
            raise ValueError(f"source trace is not Reflection: {trace.example_id}")
        selected = next(
            candidate
            for candidate in trace.candidates
            if candidate.candidate_id == trace.final_selection.selected_candidate_id
        )

        if assignment.source == "reuse":
            candidate = next(
                (
                    item
                    for item in trace.candidates
                    if item.candidate_id == assignment.candidate_id
                ),
                None,
            )
            if candidate is None or candidate.strategy != assignment.target_strategy:
                raise ValueError(f"invalid reuse assignment for {trace.example_id}")
            if candidate.candidate_id == selected.candidate_id:
                raise ValueError(f"ineligible reuse assignment for {trace.example_id}")
            payload = trace.model_dump(mode="json")
            payload["final_selection"] = FinalSelection.from_candidate(candidate).model_dump(
                mode="json"
            )
            payload["final_response"] = candidate.response
            balanced = _validated_trace(payload)
            review_rows.append(
                _review_row(
                    trace,
                    assignment,
                    old_candidate=selected,
                    new_candidate=candidate,
                )
            )
            derived.append(balanced)
            if progress_callback is not None:
                progress_callback(trace.example_id)
            continue

        if assignment.source != "generate":
            raise ValueError(f"unknown assignment source {assignment.source!r}")
        accepted, generated_records = generate_target_candidate(
            runner, trace, assignment.target_strategy, assignment.candidate_id
        )
        reason = _candidate_rejection_reason(trace, assignment.target_strategy, accepted)
        review_flags = [reason] if reason is not None else []
        if reason is not None:
            failures.append(
                {
                    "example_id": trace.example_id,
                    "target_strategy": assignment.target_strategy,
                    "candidate_id": assignment.candidate_id,
                    "attempt": 1,
                    "reason": reason,
                    "response": accepted.response,
                }
            )
        replaced_candidate = next(
            (
                candidate
                for candidate in trace.candidates
                if candidate.candidate_id == assignment.candidate_id
            ),
            None,
        )
        if replaced_candidate is None:
            raise ValueError(f"candidate slot is missing for {trace.example_id}")
        slot = int(assignment.candidate_id) - 1
        if slot < 0 or slot >= len(trace.plan.strategies):
            raise ValueError(f"candidate slot is outside plan for {trace.example_id}")
        payload = trace.model_dump(mode="json")
        if assignment.target_strategy in trace.plan.strategies:
            if replaced_candidate.strategy != assignment.target_strategy:
                raise ValueError(f"target slot mismatch for {trace.example_id}")
        else:
            if assignment.candidate_id != selected.candidate_id:
                raise ValueError(
                    f"generated candidate must replace selected Reflection slot for {trace.example_id}"
                )
            if selected.strategy != "Reflection of feelings":
                raise ValueError(f"selected slot is not Reflection for {trace.example_id}")
            payload["plan"]["strategies"][slot] = assignment.target_strategy
        for index, candidate in enumerate(trace.candidates):
            if candidate.candidate_id == assignment.candidate_id:
                payload["candidates"][index] = accepted.model_dump(mode="json")
                break
        else:
            raise ValueError(f"candidate slot is missing for {trace.example_id}")
        payload["call_records"].extend(_record_payload(record) for record in generated_records)
        payload["final_selection"] = FinalSelection.from_candidate(accepted).model_dump(
            mode="json"
        )
        payload["final_response"] = accepted.response
        balanced = _validated_trace(payload)
        review_rows.append(
            _review_row(
                trace,
                assignment,
                old_candidate=replaced_candidate,
                new_candidate=accepted,
                review_flags=review_flags,
            )
        )
        derived.append(balanced)
        if progress_callback is not None:
            progress_callback(trace.example_id)
    return derived, review_rows, failures


def read_validated_traces(path: Path) -> list[TeacherTrace]:
    """Read JSONL with line-local errors and strict trace validation."""
    traces: list[TeacherTrace] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                traces.append(TeacherTrace.model_validate(json.loads(line)))
            except Exception as error:
                raise ValueError(f"invalid trace at {path}:{line_number}: {error}") from error
    return traces


def load_prepared_examples(path: Path) -> dict[str, dict[str, Any]]:
    """Flatten prepared split rows into metadata keyed by example ID."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_rows = payload["splits"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid prepared data {path}: {error}") from error
    if not isinstance(split_rows, dict):
        raise ValueError(f"prepared splits must be an object: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for split, rows in split_rows.items():
        if not isinstance(rows, list):
            raise ValueError(f"prepared split {split!r} is not a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("example_id"), str):
                raise ValueError(f"prepared split {split!r} has an invalid row")
            example_id = row["example_id"]
            if example_id in by_id:
                raise ValueError(f"duplicate prepared example_id {example_id}")
            if row.get("split") != split:
                raise ValueError(f"prepared split mismatch for {example_id}")
            by_id[example_id] = row
    return by_id


def _ordered_counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(defaultdict(int, ((value, values.count(value)) for value in set(values))).items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_derivative(
    original: Sequence[TeacherTrace],
    derived: Sequence[TeacherTrace],
    assignments: Sequence[Assignment],
    *,
    expected_total: int = 2500,
    expected_strategy_totals: Mapping[str, int] = TARGET_TOTALS,
) -> None:
    """Ensure the derivative is source-preserving, canonical, and quota exact."""
    if len(original) != len(derived):
        raise ValueError(f"trace count changed: {len(original)} -> {len(derived)}")
    original_by_id = {trace.example_id: trace for trace in original}
    derived_by_id = {trace.example_id: trace for trace in derived}
    if len(original_by_id) != len(original):
        raise ValueError("source contains duplicate IDs")
    if len(derived_by_id) != len(derived):
        raise ValueError("derived contains duplicate IDs")
    if original_by_id.keys() != derived_by_id.keys():
        raise ValueError("example IDs changed")
    assignments_by_id = {item.example_id: item for item in assignments}
    if len(assignments_by_id) != len(assignments):
        raise ValueError("duplicate assignments")

    for example_id, source in original_by_id.items():
        target = derived_by_id[example_id]
        if source.split != target.split:
            raise ValueError(f"split changed for {example_id}")
        if source.history != target.history:
            raise ValueError(f"history changed for {example_id}")
        if source.state != target.state:
            raise ValueError(f"state changed for {example_id}")
        if source.state_analysis != target.state_analysis:
            raise ValueError(f"state analysis changed for {example_id}")
        TeacherTrace.model_validate(target.model_dump(mode="json"))
        assignment = assignments_by_id.get(example_id)
        if assignment is None:
            if source.model_dump(mode="json") != target.model_dump(mode="json"):
                raise ValueError(f"unassigned trace changed for {example_id}")
            continue
        if assignment.split != source.split:
            raise ValueError(f"assignment split mismatch for {example_id}")
        if source.final_selection.selected_strategy != "Reflection of feelings":
            raise ValueError(f"source is not Reflection for {example_id}")
        if target.final_selection.selected_strategy != assignment.target_strategy:
            raise ValueError(f"target strategy mismatch for {example_id}")

    if len(derived) != expected_total:
        raise ValueError(f"expected {expected_total} traces; found {len(derived)}")
    selected_counts = _ordered_counts(
        [trace.final_selection.selected_strategy for trace in derived]
    )
    if selected_counts != dict(sorted(expected_strategy_totals.items())):
        raise ValueError(
            f"selected strategy totals differ: {selected_counts} != {expected_strategy_totals}"
        )
    transition_counts = {
        (split, strategy): quota
        for split, quotas in TRANSITION_QUOTAS.items()
        for strategy, quota in quotas.items()
    }
    observed_transitions = _ordered_counts(
        [f"{item.split}\0{item.target_strategy}" for item in assignments]
    )
    expected_transitions = dict(
        sorted(
            {
                f"{split}\0{strategy}": quota
                for (split, strategy), quota in transition_counts.items()
            }.items()
        )
    )
    if observed_transitions != expected_transitions:
        raise ValueError("assignment transition quotas differ")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_artifacts_atomically(
    output_dir: Path,
    traces: Sequence[TeacherTrace],
    manifest: dict[str, Any],
    review_rows: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
) -> None:
    """Publish a complete new directory and never replace an existing one."""
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_jsonl(temporary_dir / "teacher-traces.jsonl", traces)
        _write_json(temporary_dir / "manifest.json", manifest)
        _write_jsonl(temporary_dir / "review_queue.jsonl", review_rows)
        _write_jsonl(temporary_dir / "failures.jsonl", failures)
        _write_json(
            temporary_dir / "analysis_metrics.json",
            {
                "trace_rows": len(traces),
                "selected_strategy_counts": manifest.get("observed_selected_strategy_counts", {}),
                "review_rows": len(review_rows),
                "failures": len(failures),
            },
        )
        directory_fd = os.open(temporary_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_dir.exists():
            raise FileExistsError(f"output already exists: {output_dir}")
        os.rename(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _review_queue(
    review_rows: Sequence[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    generated = [row for row in review_rows if row["source"] == "generate"]
    reusable = [row for row in review_rows if row["source"] == "reuse"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reusable:
        groups[(row["split"], row["target_strategy"])].append(row)
    sampled: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        count = max(1, round(len(rows) * 0.2))
        sampled.extend(
            sorted(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{seed}\0review\0{row['example_id']}".encode("utf-8")
                ).hexdigest(),
            )[:count]
        )
    return [*generated, *sampled]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a balanced derivative of 2,500 Teacher traces."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=20260903, type=int)
    parser.add_argument("--max-generation-attempts", default=3, type=int)
    return parser


def prepare_generation_config(config: AppConfig) -> AppConfig:
    """Disable slot-keyed source cache for strategy-changing candidate calls."""
    return config.model_copy(
        update={"backend": config.backend.model_copy(update={"cache_dir": None})}
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    input_hash = _sha256(args.input)
    prepared_hash = _sha256(args.prepared)
    original = read_validated_traces(args.input)
    prepared = load_prepared_examples(args.prepared)
    assignments = plan_assignments(original, prepared, seed=args.seed)
    config = prepare_generation_config(AppConfig.from_yaml(args.config))
    runner = TeacherRunner(OpenAIBackend(config), config)
    try:
        from tqdm import tqdm
    except ImportError:
        derived, review_rows, failures = apply_assignments(
            original,
            assignments,
            runner,
            max_generation_attempts=args.max_generation_attempts,
        )
    else:
        with tqdm(total=len(assignments), desc="balanced traces", unit="trace") as progress:
            derived, review_rows, failures = apply_assignments(
                original,
                assignments,
                runner,
                max_generation_attempts=args.max_generation_attempts,
                progress_callback=lambda _example_id: progress.update(1),
            )
    if input_hash != _sha256(args.input) or prepared_hash != _sha256(args.prepared):
        raise RuntimeError("source input changed during processing")
    validate_derivative(original, derived, assignments)
    selected_counts = _ordered_counts(
        [trace.final_selection.selected_strategy for trace in derived]
    )
    manifest = {
        "source_trace_path": str(args.input),
        "source_trace_sha256": input_hash,
        "prepared_path": str(args.prepared),
        "prepared_sha256": prepared_hash,
        "seed": args.seed,
        "target_selected_strategy_counts": TARGET_TOTALS,
        "observed_selected_strategy_counts": selected_counts,
        "transition_quotas": TRANSITION_QUOTAS,
        "changed_ids": [assignment.example_id for assignment in assignments],
        "reuse_count": sum(item.source == "reuse" for item in assignments),
        "generation_count": sum(item.source == "generate" for item in assignments),
        "failures_count": len(failures),
        "generation_flag_count": len(failures),
        "review_status": "pending_human_review",
    }
    queue = _review_queue(review_rows, args.seed)
    write_artifacts_atomically(args.output_dir, derived, manifest, queue, failures)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "selected_strategy_counts": selected_counts,
                "reuse_count": manifest["reuse_count"],
                "generation_count": manifest["generation_count"],
                "generation_flag_count": manifest["generation_flag_count"],
                "review_status": manifest["review_status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
