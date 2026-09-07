"""Command-line entry points for protocol, trace, and export workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .backend import OpenAIBackend
from .config import AppConfig
from .export import sft_row, slot_row, visible_sft_row
from .progress import track
from .schemas import (
    History,
    TeacherTrace,
)
from .storage import append_jsonl, read_jsonl, write_jsonl
from .teacher import TeacherRunner


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return read_jsonl(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if payload.get("protocol_version") in {
            "socialsim-qwen-conversation-v1",
            "socialsim-qwen-conversation-v2",
        }:
            from .socialsim import PreparedSocialSim

            prepared = PreparedSocialSim.model_validate(payload)
            return [
                example.model_dump(mode="json")
                for split in ("train", "dev", "diagnostic_holdout")
                for example in prepared.splits[split]
            ]
        return [payload]
    raise ValueError("input must be a JSON object, JSON array, or JSONL records")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ibd")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser("validate-trace")
    validate_parser.add_argument("--input", required=True)

    run_parser = commands.add_parser("run-teacher")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--continue-on-error", action="store_true")
    run_parser.add_argument("--failures")

    export_parser = commands.add_parser("export-student")
    export_parser.add_argument(
        "--kind",
        choices=("sft", "slot", "visible-sft"),
        required=True,
    )
    export_parser.add_argument("--input", required=True)
    export_parser.add_argument("--output", required=True)

    prepare_parser = commands.add_parser("prepare-socialsim")
    prepare_parser.add_argument(
        "--dialogues",
        default="/home/wangnianxiang/supervisor/data/raw/SocialSim_SSConv_ESC_dataset_full_3229.json",
    )
    prepare_parser.add_argument(
        "--profiles",
        default="/home/wangnianxiang/supervisor/data/raw/SocialSim_UserProfile_full_3229.json",
    )
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--limit", type=int, default=128)
    prepare_parser.add_argument("--train-size", type=int, default=96)
    prepare_parser.add_argument("--dev-size", type=int, default=16)
    prepare_parser.add_argument("--holdout-size", type=int, default=16)
    prepare_parser.add_argument("--early-fraction", type=float, default=0.1)
    prepare_parser.add_argument("--late-fraction", type=float, default=0.1)

    anchor_parser = commands.add_parser("precompute-anchors")
    anchor_parser.add_argument("--config", required=True)
    anchor_parser.add_argument("--traces", required=True)
    anchor_parser.add_argument("--output", required=True)
    anchor_parser.add_argument("--batch-size", type=int, default=1)
    anchor_parser.add_argument("--device", type=int, default=0)
    anchor_parser.add_argument(
        "--original-splits",
        nargs="+",
        choices=("train", "dev", "diagnostic_holdout"),
        default=["train"],
    )

    def add_training_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--config", required=True)
        command_parser.add_argument("--run-name", required=True)
        command_parser.add_argument("--seed", type=int, required=True)
        command_parser.add_argument("--traces", required=True)
        command_parser.add_argument("--anchors", required=True)
        command_parser.add_argument("--run-dir", default="runs")
        command_parser.add_argument("--resume")
        command_parser.add_argument("--device", type=int, default=0)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--stage", choices=("A", "B"), required=True)
    add_training_arguments(train_parser)

    pipeline_parser = commands.add_parser("train-pipeline")
    add_training_arguments(pipeline_parser)

    control_parser = commands.add_parser("train-sft-control")
    control_parser.add_argument("--config", required=True)
    control_parser.add_argument("--run-name", required=True)
    control_parser.add_argument("--seed", type=int, required=True)
    control_input = control_parser.add_mutually_exclusive_group(required=True)
    control_input.add_argument("--traces")
    control_input.add_argument("--dataset")
    control_parser.add_argument("--run-dir", default="runs")
    control_parser.add_argument("--resume")
    control_parser.add_argument("--device", type=int, default=0)

    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--run-name", required=True)
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--history", required=True)
    generate_parser.add_argument("--output")
    generate_parser.add_argument("--max-new-tokens", type=int, default=256)
    generate_parser.add_argument("--device", type=int, default=0)

    from .quality_cli import register_quality_commands

    register_quality_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from .quality_cli import QUALITY_COMMANDS, run_quality_command

    if args.command in QUALITY_COMMANDS:
        return run_quality_command(args)
    if args.command == "prepare-socialsim":
        from .socialsim import load_socialsim_files

        prepared = load_socialsim_files(
            args.dialogues,
            args.profiles,
            seed=args.seed,
            limit=args.limit,
            split_sizes={
                "train": args.train_size,
                "dev": args.dev_size,
                "diagnostic_holdout": args.holdout_size,
            },
            early_fraction=args.early_fraction,
            late_fraction=args.late_fraction,
        )
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                prepared.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.limit} SocialSim conversations")
        return 0

    if args.command in {
        "precompute-anchors",
        "train",
        "train-pipeline",
        "train-sft-control",
        "generate",
    }:
        from .pipeline import run_pipeline_command

        return run_pipeline_command(args)

    records = _load_records(args.input)
    if args.command == "validate-trace":
        for record in track(
            records,
            desc="validate traces",
            total=len(records),
            unit="trace",
        ):
            TeacherTrace.model_validate(record)
        print(f"valid {len(records)}")
        return 0

    if args.command == "run-teacher":
        output_path = Path(args.output)
        if args.continue_on_error and args.failures is None:
            raise ValueError("--continue-on-error requires --failures")
        if output_path.exists() and not args.resume:
            raise FileExistsError(
                f"output already exists: {output_path}; pass --resume to continue"
            )

        completed_ids: set[str] = set()
        if args.resume:
            if not output_path.is_file():
                raise FileNotFoundError(
                    f"cannot resume because output does not exist: {output_path}"
                )
            for stored_trace in read_jsonl(output_path):
                trace = TeacherTrace.model_validate(stored_trace)
                completed_ids.add(trace.example_id)

        config = AppConfig.from_yaml(args.config)
        runner = TeacherRunner(OpenAIBackend(config), config)
        pending_records = [
            record
            for record in records
            if str(record["example_id"]) not in completed_ids
        ]
        written = 0
        failures = 0
        progress = track(
            pending_records,
            desc="teacher examples",
            total=len(pending_records),
            unit="example",
        )
        for record in progress:
            progress.set_postfix(example=str(record["example_id"]))
            try:
                trace = runner.run(
                    str(record["example_id"]),
                    History.model_validate(record["history"]),
                    split=record.get("split", "train"),
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                failure = {
                    "example_id": str(record["example_id"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if "split" in record:
                    failure["split"] = record["split"]
                append_jsonl(args.failures, failure)
                failures += 1
                continue
            append_jsonl(output_path, trace)
            written += 1
        print(f"wrote {written}")
        if failures:
            print(f"failed {failures}; see {args.failures}")
            return 1
        return 0

    rows: list[dict[str, Any]] = []
    for record in track(
        records,
        desc=f"export {args.kind}",
        total=len(records),
        unit="record",
    ):
        if args.kind == "sft":
            rows.append(sft_row(TeacherTrace.model_validate(record)))
        elif args.kind == "slot":
            rows.append(slot_row(TeacherTrace.model_validate(record)))
        elif args.kind == "visible-sft":
            rows.append(visible_sft_row(TeacherTrace.model_validate(record)))
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
