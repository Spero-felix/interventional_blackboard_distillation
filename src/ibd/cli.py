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
from .progress import track
from .schemas import (
    History,
    InterventionRecord,
    MarginPair,
    STATE_ANCHOR_FIELDS,
    TeacherTrace,
)
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
        if payload.get("protocol_version") == "socialsim-qwen-conversation-v1":
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

    intervention_parser = commands.add_parser("build-interventions")
    intervention_parser.add_argument("--config", required=True)
    intervention_parser.add_argument("--input", required=True)
    intervention_parser.add_argument("--output", required=True)
    intervention_parser.add_argument("--margins-output", required=True)
    intervention_parser.add_argument("--manifest", required=True)
    intervention_parser.add_argument("--global-seed", type=int, required=True)

    anchor_parser = commands.add_parser("precompute-anchors")
    anchor_parser.add_argument("--config", required=True)
    anchor_parser.add_argument("--traces", required=True)
    anchor_parser.add_argument("--interventions")
    anchor_parser.add_argument("--output", required=True)
    anchor_parser.add_argument("--batch-size", type=int, default=1)
    anchor_parser.add_argument("--device", type=int, default=0)
    anchor_parser.add_argument("--global-seed", type=int, required=True)
    anchor_parser.add_argument(
        "--diagnostic-state-field",
        choices=STATE_ANCHOR_FIELDS,
        required=True,
    )

    def add_training_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--config", required=True)
        command_parser.add_argument("--run-name", required=True)
        command_parser.add_argument("--seed", type=int, required=True)
        command_parser.add_argument("--traces", required=True)
        command_parser.add_argument("--interventions", required=True)
        command_parser.add_argument("--margins", required=True)
        command_parser.add_argument("--anchors", required=True)
        command_parser.add_argument("--run-dir", default="runs")
        command_parser.add_argument("--resume")
        command_parser.add_argument("--device", type=int, default=0)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--stage", choices=("A", "B", "C", "D"), required=True)
    add_training_arguments(train_parser)

    pipeline_parser = commands.add_parser("train-pipeline")
    add_training_arguments(pipeline_parser)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--run-name", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--traces", required=True)
    evaluate_parser.add_argument("--interventions", required=True)
    evaluate_parser.add_argument("--margins", required=True)
    evaluate_parser.add_argument("--anchors", required=True)
    evaluate_parser.add_argument("--intervention-manifest", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--device", type=int, default=0)

    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--run-name", required=True)
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--history", required=True)
    generate_parser.add_argument("--clamp", choices=("STATE", "PLAN"))
    generate_parser.add_argument("--anchors")
    generate_parser.add_argument("--example-id")
    generate_parser.add_argument("--output")
    generate_parser.add_argument("--max-new-tokens", type=int, default=256)
    generate_parser.add_argument("--device", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "protocol-hash":
        config = AppConfig.from_yaml(args.config)
        print(protocol_hash(config))
        return 0

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
        "build-interventions",
        "precompute-anchors",
        "train",
        "train-pipeline",
        "evaluate",
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
        config = AppConfig.from_yaml(args.config)
        runner = TeacherRunner(OpenAIBackend(config), config)
        traces = []
        progress = track(
            records,
            desc="teacher examples",
            total=len(records),
            unit="example",
        )
        for record in progress:
            progress.set_postfix(example=str(record["example_id"]))
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
