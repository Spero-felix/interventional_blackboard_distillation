"""Command-line entry points for protocol, trace, and export workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .backend import OpenAIBackend
from .config import AppConfig
from .conversation import (
    ConversationGenerator,
    ConversationStageError,
    flatten_teacher_traces,
)
from .conversation_schemas import (
    ConversationCheckpoint,
    ConversationFailure,
    ConversationGenerationConfig,
    ConversationTrace,
)
from .dialogue_manager import DialogueManager
from .export import sft_row, slot_row, visible_sft_row
from .progress import track
from .schemas import (
    History,
    TeacherTrace,
    UserContext,
)
from .seeker import MappingProfileAdapter, SeekerSimulator
from .storage import (
    append_jsonl,
    conversation_checkpoint_path,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl,
)
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

    conversation_parser = commands.add_parser("generate-conversations")
    conversation_parser.add_argument("--config", required=True)
    conversation_parser.add_argument("--profiles", required=True)
    conversation_parser.add_argument("--output", required=True)
    conversation_parser.add_argument("--truncated", required=True)
    conversation_parser.add_argument("--failures", required=True)
    conversation_parser.add_argument("--checkpoint-dir", required=True)
    conversation_parser.add_argument("--seed", type=int, required=True)
    conversation_parser.add_argument(
        "--split",
        choices=("train", "dev", "diagnostic_holdout"),
        default="train",
    )
    conversation_parser.add_argument("--min-rounds", type=int, default=6)
    conversation_parser.add_argument("--soft-max-rounds", type=int, default=16)
    conversation_parser.add_argument("--hard-max-rounds", type=int, default=20)
    conversation_parser.add_argument("--resume", action="store_true")

    flatten_parser = commands.add_parser("flatten-conversations")
    flatten_parser.add_argument("--input", required=True)
    flatten_parser.add_argument("--output", required=True)

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

    if args.command == "generate-conversations":
        return _generate_conversations(args)

    if args.command == "flatten-conversations":
        rows = []
        for record in _load_records(args.input):
            trace = ConversationTrace.model_validate(record)
            rows.extend(flatten_teacher_traces(trace))
        write_jsonl(args.output, rows)
        print(f"wrote {len(rows)}")
        return 0

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
                context_before = (
                    UserContext.model_validate(record["context_before"])
                    if "context_before" in record
                    else None
                )
                trace = runner.run(
                    str(record["example_id"]),
                    History.model_validate(record["history"]),
                    split=record.get("split", "train"),
                    context_before=context_before,
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


def _stored_conversation_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        ConversationTrace.model_validate(record).conversation_id
        for record in read_jsonl(path)
    }


def _generate_conversations(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    truncated_path = Path(args.truncated)
    failures_path = Path(args.failures)
    artifact_paths = (output_path, truncated_path, failures_path)
    if not args.resume:
        existing = [path for path in artifact_paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"output already exists: {existing[0]}; pass --resume to continue"
            )

    raw_profiles = _load_records(args.profiles)
    adapter = MappingProfileAdapter()
    adapted_profiles = []
    adapter_failures: list[ConversationFailure] = []
    seen_profile_ids: set[str] = set()
    for index, raw_profile in enumerate(raw_profiles):
        fallback_id = f"input-index-{index}"
        try:
            profile = adapter.validate(raw_profile)
        except Exception as exc:
            adapter_failures.append(
                ConversationFailure(
                    conversation_id=f"{fallback_id}-seed-{args.seed}",
                    profile_id=fallback_id,
                    seed=args.seed,
                    failed_stage="profile_adapter",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    completed_rounds=0,
                )
            )
            continue
        if profile.profile_id in seen_profile_ids:
            raise ValueError(f"duplicate profile ID: {profile.profile_id}")
        seen_profile_ids.add(profile.profile_id)
        adapted_profiles.append(profile)

    for failure in adapter_failures:
        append_jsonl(failures_path, failure)

    generation_config = ConversationGenerationConfig(
        min_rounds=args.min_rounds,
        soft_max_rounds=args.soft_max_rounds,
        hard_max_rounds=args.hard_max_rounds,
        split=args.split,
    )
    config = AppConfig.from_yaml(args.config)
    finalized_ids = (
        _stored_conversation_ids(output_path)
        | _stored_conversation_ids(truncated_path)
        if args.resume
        else set()
    )

    completed_count = 0
    truncated_count = 0
    failure_count = len(adapter_failures)
    pending_profiles = [
        profile
        for profile in adapted_profiles
        if f"{profile.profile_id}-seed-{args.seed}" not in finalized_ids
    ]
    if not pending_profiles:
        print(
            f"completed 0; truncated 0; failed {failure_count}"
        )
        return 1 if failure_count else 0

    backend = OpenAIBackend(config)
    generator = ConversationGenerator(
        seeker=SeekerSimulator(backend, config),
        teacher_runner=TeacherRunner(backend, config),
        dialogue_manager=DialogueManager(backend, config),
        generation_config=generation_config,
        protocol_version=config.protocol_version,
    )
    progress = track(
        pending_profiles,
        desc="generate conversations",
        total=len(pending_profiles),
        unit="profile",
    )
    for profile in progress:
        conversation_id = f"{profile.profile_id}-seed-{args.seed}"
        progress.set_postfix(profile=profile.profile_id)
        checkpoint_path = conversation_checkpoint_path(
            args.checkpoint_dir,
            conversation_id,
        )
        checkpoint = None
        if args.resume and checkpoint_path.is_file():
            checkpoint = ConversationCheckpoint.model_validate(
                read_json(checkpoint_path)
            )
        try:
            trace = generator.generate(
                profile,
                base_seed=args.seed,
                checkpoint=checkpoint,
                on_round_completed=lambda record, path=checkpoint_path: (
                    write_json_atomic(path, record)
                ),
            )
        except Exception as exc:
            if isinstance(exc, ConversationStageError):
                failed_stage = exc.stage
                completed_rounds = exc.completed_rounds
                root_error = exc.__cause__ if exc.__cause__ is not None else exc
            else:
                failed_stage = "conversation_generator"
                completed_rounds = len(checkpoint.rounds) if checkpoint else 0
                root_error = exc
            append_jsonl(
                failures_path,
                ConversationFailure(
                    conversation_id=conversation_id,
                    profile_id=profile.profile_id,
                    seed=args.seed,
                    failed_stage=failed_stage,
                    error_type=type(root_error).__name__,
                    error=str(root_error),
                    completed_rounds=completed_rounds,
                    checkpoint_path=(
                        str(checkpoint_path) if checkpoint_path.is_file() else None
                    ),
                ),
            )
            failure_count += 1
            continue

        if trace.status == "completed":
            append_jsonl(output_path, trace)
            completed_count += 1
        else:
            append_jsonl(truncated_path, trace)
            truncated_count += 1

    print(
        f"completed {completed_count}; truncated {truncated_count}; "
        f"failed {failure_count}"
    )
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
