"""CLI boundary for the independent generation-quality workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .quality import QualityEvalConfig, QualityResponse, atomic_write_json
from .schemas import TeacherTrace
from .storage import read_jsonl


QUALITY_COMMANDS = {
    "quality-generate",
    "quality-judge",
    "quality-human-export",
    "quality-human-summarize",
    "quality-merge",
}


def register_quality_commands(commands: argparse._SubParsersAction) -> None:
    generate = commands.add_parser("quality-generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--traces", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--continue-on-error", action="store_true")
    generate.add_argument("--device", type=int, default=0)

    judge = commands.add_parser("quality-judge")
    judge.add_argument("--config", required=True)
    judge.add_argument("--responses", required=True)
    judge.add_argument("--output-dir", required=True)
    judge.add_argument("--resume", action="store_true")
    judge.add_argument("--continue-on-error", action="store_true")

    human_export = commands.add_parser("quality-human-export")
    human_export.add_argument("--config", required=True)
    human_export.add_argument("--responses", required=True)
    human_export.add_argument("--output-dir", required=True)

    human_summary = commands.add_parser("quality-human-summarize")
    human_summary.add_argument("--annotations", required=True)
    human_summary.add_argument("--mapping", required=True)
    human_summary.add_argument("--output", required=True)

    merge = commands.add_parser("quality-merge")
    merge.add_argument("--base-dir", required=True)
    merge.add_argument("--standalone-dir", required=True)
    merge.add_argument("--output-dir", required=True)


def _models(path: str | Path, model_type):
    return [model_type.model_validate(row) for row in read_jsonl(path)]


def run_quality_command(args: argparse.Namespace) -> int:
    if args.command == "quality-human-summarize":
        from .quality_human import summarize_human_annotations

        report = summarize_human_annotations(args.annotations, args.mapping)
        atomic_write_json(args.output, report)
        print(f"wrote {args.output}")
        return 0
    if args.command == "quality-merge":
        from .quality_merge import merge_quality_judgments

        merge_quality_judgments(args.base_dir, args.standalone_dir, args.output_dir)
        print(f"wrote {Path(args.output_dir) / 'ranking.json'}")
        return 0

    config = QualityEvalConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "quality-generate":
        from .quality_generation import generate_quality_responses

        return generate_quality_responses(
            config,
            _models(args.traces, TeacherTrace),
            output_path=output_dir / "responses.jsonl",
            manifest_path=output_dir / "responses.manifest.json",
            failure_path=output_dir / "generation_failures.jsonl",
            resume=args.resume,
            continue_on_error=args.continue_on_error,
            device=args.device,
        )
    responses = _models(args.responses, QualityResponse)
    if args.command == "quality-judge":
        from .quality_judge import judge_quality_responses

        return judge_quality_responses(
            config,
            responses,
            output_path=output_dir / "judgments.jsonl",
            manifest_path=output_dir / "judgments.manifest.json",
            failure_path=output_dir / "judge_failures.jsonl",
            report_path=output_dir / "quality_report.json",
            resume=args.resume,
            continue_on_error=args.continue_on_error,
        )
    if args.command == "quality-human-export":
        from .quality_human import export_human_pairs

        summary = export_human_pairs(
            responses,
            output_dir / "pairs.csv",
            output_dir / "private_mapping.jsonl",
            seed=config.human_seed,
        )
        print(f"wrote {summary['pairs']} blinded pairs")
        return 0
    raise ValueError(f"unknown quality command: {args.command}")
