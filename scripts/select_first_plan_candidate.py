#!/usr/bin/env python3
"""Build canonical Teacher traces by selecting the first planned candidate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ibd.schemas import FinalSelection, TeacherTrace


EXPECTED_TOTAL = 2000
EXPECTED_SPLITS = {
    "train": 1500,
    "dev": 200,
    "diagnostic_holdout": 300,
}


@dataclass(frozen=True)
class ConversionSummary:
    total: int
    splits: dict[str, int]
    selected_before: dict[str, int]
    selected_after: dict[str, int]


def _ordered_counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def select_s1(trace: TeacherTrace) -> TeacherTrace:
    """Return a fully validated copy whose final response comes from S1."""
    matches = [candidate for candidate in trace.candidates if candidate.strategy_id == "S1"]
    if len(matches) != 1:
        raise ValueError(
            f"example {trace.example_id} must contain exactly one S1 candidate; "
            f"found {len(matches)}"
        )
    candidate = matches[0]
    if trace.plan.strategies[0] != candidate.strategy:
        raise ValueError(
            f"example {trace.example_id} S1 candidate does not match plan[0]"
        )
    payload = trace.model_dump(mode="json")
    payload["final_selection"] = FinalSelection.from_candidate(candidate).model_dump(
        mode="json"
    )
    payload["final_response"] = candidate.response
    return TeacherTrace.model_validate(payload)


def convert_dataset(
    traces: Sequence[TeacherTrace],
) -> tuple[list[TeacherTrace], ConversionSummary]:
    """Validate and convert the fixed 2,000-row experiment without reordering."""
    if len(traces) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} traces; found {len(traces)}")

    seen: set[str] = set()
    for trace in traces:
        if trace.example_id in seen:
            raise ValueError(f"duplicate example_id {trace.example_id}")
        seen.add(trace.example_id)

    splits = _ordered_counts([trace.split for trace in traces])
    if splits != dict(sorted(EXPECTED_SPLITS.items())):
        raise ValueError(
            f"unexpected split counts: expected {EXPECTED_SPLITS}; found {splits}"
        )

    selected_before = _ordered_counts(
        [trace.final_selection.selected_strategy_id for trace in traces]
    )
    converted = [select_s1(trace) for trace in traces]
    selected_after = _ordered_counts(
        [trace.final_selection.selected_strategy_id for trace in converted]
    )
    if selected_after != {"S1": EXPECTED_TOTAL}:
        raise ValueError(f"conversion did not select S1 for every trace: {selected_after}")
    return converted, ConversionSummary(
        total=len(converted),
        splits=splits,
        selected_before=selected_before,
        selected_after=selected_after,
    )


def read_traces(path: Path) -> list[TeacherTrace]:
    """Read JSONL with line-local diagnostics and full schema validation."""
    traces: list[TeacherTrace] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                traces.append(TeacherTrace.model_validate(payload))
            except Exception as error:
                raise ValueError(f"invalid trace at {path}:{line_number}: {error}") from error
    return traces


def write_jsonl_atomically(path: Path, traces: Sequence[TeacherTrace]) -> None:
    """Publish complete JSONL without ever replacing an existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for trace in traces:
                handle.write(
                    json.dumps(
                        trace.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the S1 candidate in every 2,000-row Teacher trace."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input
    output_path = args.output
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    converted, summary = convert_dataset(read_traces(input_path))
    write_jsonl_atomically(output_path, converted)
    print(
        json.dumps(
            {
                "total": summary.total,
                "splits": summary.splits,
                "selected_before": summary.selected_before,
                "selected_after": summary.selected_after,
                "output": str(output_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
