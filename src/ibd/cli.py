"""Command-line entry points for protocol, trace, and export workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .backend import OpenAIBackend
from .config import AppConfig
from .export import intervention_row, margin_row, sft_row, slot_row
from .hashing import protocol_hash
from .schemas import History, InterventionRecord, MarginPair, TeacherTrace
from .storage import read_jsonl, write_jsonl
from .teacher import TeacherRunner


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return read_jsonl(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("input must be a JSON object, JSON array, or JSONL records")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ibd")
    commands = parser.add_subparsers(dest="command", required=True)

    hash_parser = commands.add_parser("protocol-hash")
    hash_parser.add_argument("--config", required=True)

    validate_parser = commands.add_parser("validate-trace")
    validate_parser.add_argument("--input", required=True)

    run_parser = commands.add_parser("run-teacher")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", required=True)

    export_parser = commands.add_parser("export-student")
    export_parser.add_argument("--kind", choices=("sft", "slot", "intervention", "margin"), required=True)
    export_parser.add_argument("--input", required=True)
    export_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "protocol-hash":
        config = AppConfig.from_yaml(args.config)
        print(protocol_hash(config))
        return 0

    records = _load_records(args.input)
    if args.command == "validate-trace":
        for record in records:
            TeacherTrace.model_validate(record)
        print(f"valid {len(records)}")
        return 0

    if args.command == "run-teacher":
        config = AppConfig.from_yaml(args.config)
        runner = TeacherRunner(OpenAIBackend(config), config)
        traces = []
        for record in records:
            traces.append(
                runner.run(
                    str(record["example_id"]),
                    History.model_validate(record["history"]),
                    split=record.get("split", "train"),
                )
            )
        write_jsonl(args.output, traces)
        print(f"wrote {len(traces)}")
        return 0

    rows: list[dict[str, Any]] = []
    for record in records:
        if args.kind == "sft":
            rows.append(sft_row(TeacherTrace.model_validate(record)))
        elif args.kind == "slot":
            rows.append(slot_row(TeacherTrace.model_validate(record)))
        elif args.kind == "margin":
            rows.append(margin_row(MarginPair.model_validate(record)))
        else:
            prompt = str(record["prompt"])
            payload = {key: value for key, value in record.items() if key != "prompt"}
            rows.append(intervention_row(InterventionRecord.model_validate(payload), prompt))
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

