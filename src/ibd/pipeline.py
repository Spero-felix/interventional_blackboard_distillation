"""Artifact construction and CLI orchestration for local Qwen training."""

from __future__ import annotations

import json
import math
import random
import shutil
from argparse import Namespace
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from .anchors import (
    AnchorArtifact,
    AnchorEncoder,
    plan_anchor_payload,
    serialize_anchor_payload,
    state_anchor_payload,
    validate_state_token_budget,
)
from .checkpointing import CheckpointManager, CheckpointMetadata
from .progress import track
from .qwen import (
    QwenTrainingConfig,
    StandardSFTControlConfig,
    load_frozen_qwen_for_anchors,
    load_qwen_qlora,
    load_qwen_standard_sft,
    restore_qwen_for_inference,
    validate_local_qwen_directory,
)
from .schemas import (
    History,
    TeacherTrace,
)
from .storage import read_jsonl, write_jsonl
from .student_data import (
    QwenStageCollator,
    StandardSFTCollator,
    encode_generation_prompt,
)
from .trainer import StandardSFTTrainer, StageTrainer, build_paged_adamw_8bit
from .visible_sft import VisibleSFTDatasetRecord


StageName = Literal["A", "B"]
_STAGES: tuple[StageName, ...] = ("A", "B")


def _unique_by_id(items: Sequence[Any], *, label: str) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    for item in items:
        example_id = str(item.example_id)
        if example_id in by_id:
            raise ValueError(f"duplicate {label} example_id {example_id}")
        by_id[example_id] = item
    return by_id




def build_stage_rows(
    stage: StageName,
    traces: Sequence[TeacherTrace],
    *,
    splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed_splits = splits or {"train"}
    trace_by_id = _unique_by_id(traces, label="Teacher trace")
    selected_traces = {
        example_id: trace
        for example_id, trace in trace_by_id.items()
        if trace.split in allowed_splits
    }
    if stage in {"A", "B"}:
        return [
            {
                "example_id": trace.example_id,
                "history": trace.history.model_dump(mode="json"),
                "response": trace.final_response,
                "selected_strategy": trace.final_selection.selected_strategy,
            }
            for trace in selected_traces.values()
        ]
    raise ValueError(f"unknown training stage {stage}")


def read_visible_sft_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read static visible-SFT rows without permitting split or format drift."""

    rows = [VisibleSFTDatasetRecord.model_validate(row) for row in read_jsonl(path)]
    seen_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.example_id in seen_ids:
            raise ValueError(f"duplicate visible-SFT example_id {row.example_id}")
        seen_ids.add(row.example_id)
        result.append(row.model_dump(mode="json"))
    return result


def _encode_in_batches(
    encoder: AnchorEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
    description: str,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("anchor batch_size must be positive")
    offsets = range(0, len(texts), batch_size)
    pieces = [
        encoder.encode(texts[offset : offset + batch_size])
        for offset in track(
            offsets,
            desc=description,
            total=len(offsets),
            unit="batch",
        )
    ]
    if not pieces:
        raise ValueError("anchor text collection is empty")
    return torch.cat(pieces, dim=0)


def build_anchor_artifact(
    traces: Sequence[TeacherTrace],
    encoder: AnchorEncoder,
    *,
    metadata: Mapping[str, Any],
    batch_size: int = 1,
    original_splits: set[str] | None = None,
) -> AnchorArtifact:
    all_trace_by_id = _unique_by_id(traces, label="Teacher trace")
    selected_splits = {"train"} if original_splits is None else original_splits
    allowed_splits = {"train", "dev", "diagnostic_holdout"}
    invalid_splits = selected_splits - allowed_splits
    if invalid_splits:
        raise ValueError(f"unknown original anchor splits: {sorted(invalid_splits)}")
    if not selected_splits:
        raise ValueError("original anchor splits must not be empty")
    trace_by_id = {
        example_id: trace
        for example_id, trace in all_trace_by_id.items()
        if trace.split in selected_splits
    }
    ordered_ids = list(trace_by_id)
    state_texts = [
        serialize_anchor_payload(state_anchor_payload(trace_by_id[item].state))
        for item in ordered_ids
    ]
    plan_texts = [
        serialize_anchor_payload(
            plan_anchor_payload(
                trace_by_id[item].final_selection.to_plan_selection()
            )
        )
        for item in ordered_ids
    ]

    for text in state_texts:
        validate_state_token_budget(text, encoder.tokenizer)
    state = _encode_in_batches(
        encoder,
        state_texts,
        batch_size=batch_size,
        description="encode STATE anchors",
    )
    plan = _encode_in_batches(
        encoder,
        plan_texts,
        batch_size=batch_size,
        description="encode PLAN anchors",
    )
    return AnchorArtifact(
        state=state,
        plan=plan,
        example_to_row={example_id: index for index, example_id in enumerate(ordered_ids)},
        metadata=dict(metadata),
    )


def _read_models(path: str | Path, model_type: type[Any]) -> list[Any]:
    return [model_type.model_validate(record) for record in read_jsonl(path)]


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )







def _command_precompute_anchors(args: Namespace) -> int:
    config = QwenTrainingConfig.from_yaml(args.config)
    traces = _read_models(args.traces, TeacherTrace)
    loaded = load_frozen_qwen_for_anchors(config, device=args.device)
    encoder = AnchorEncoder(
        loaded.model,
        loaded.tokenizer,
        slot_layer=config.slot_layer,
    )
    artifact = build_anchor_artifact(
        traces,
        encoder,
        metadata={},
        batch_size=args.batch_size,
        original_splits=set(args.original_splits),
    )
    artifact.save(args.output)
    print(f"wrote {len(artifact.example_to_row)} anchor rows")
    return 0


def _checkpoint_payload(path: str | Path) -> dict[str, Any]:
    return json.loads((Path(path) / "checkpoint.json").read_text(encoding="utf-8"))


def _stage_collator(stage: StageName, loaded: Any, config: QwenTrainingConfig):
    common = {
        "max_length": config.max_length,
        "token_ids": loaded.token_ids,
    }
    if stage in {"A", "B"}:
        return QwenStageCollator(loaded.tokenizer, **common)
    raise ValueError(f"unknown training stage {stage}")


def _requested_stages(
    args: Namespace,
    config: QwenTrainingConfig,
) -> list[StageName]:
    if args.command == "train":
        if config.epochs.for_stage(args.stage) == 0:
            raise ValueError(f"Stage {args.stage} is disabled by epochs configuration")
        return [args.stage]
    enabled = _configured_stages(config)
    if not args.resume:
        return enabled
    source_metadata = _checkpoint_payload(args.resume)["metadata"]
    source_stage = source_metadata["stage"]
    source_epoch = source_metadata["epoch"]
    if source_stage not in enabled:
        raise ValueError(f"resume checkpoint Stage {source_stage} is disabled by configuration")
    source_index = enabled.index(source_stage)
    if source_epoch < config.epochs.for_stage(source_stage):
        remaining = enabled[source_index:]
    else:
        remaining = enabled[source_index + 1 :]
    if not remaining:
        raise ValueError("pipeline resume checkpoint already completed configured stages")
    return remaining


def _configured_stages(config: QwenTrainingConfig) -> list[StageName]:
    return [stage for stage in _STAGES if config.epochs.for_stage(stage) > 0]


def _allow_fresh_stage_start(
    stages: Sequence[StageName],
    config: QwenTrainingConfig,
) -> bool:
    """Allow the explicit Base → B ablation, but no other fresh late start."""

    return list(stages) == ["B"] and config.epochs.for_stage("A") == 0


def _stage_epoch(
    stage: StageName,
    restored: CheckpointMetadata | None,
) -> int:
    if restored is not None and restored.stage == stage:
        return restored.epoch + 1
    return 1


def _training_batch_total(
    loader_length: int,
    max_optimizer_steps: int | None,
    gradient_accumulation_steps: int,
) -> int:
    if max_optimizer_steps is None:
        return loader_length
    return min(
        loader_length,
        max_optimizer_steps * gradient_accumulation_steps,
    )


def _optimizer_steps_per_epoch(
    loader_length: int,
    max_optimizer_steps: int | None,
    gradient_accumulation_steps: int,
) -> int:
    microbatches = _training_batch_total(
        loader_length,
        max_optimizer_steps,
        gradient_accumulation_steps,
    )
    return math.ceil(microbatches / gradient_accumulation_steps)


def _scheduler_factor(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= total_steps:
        return 0.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _configured_scheduler_total_steps(config: QwenTrainingConfig) -> int | None:
    return config.scheduler_total_steps


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    config: QwenTrainingConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    warmup_steps = int(total_steps * config.warmup_ratio)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduler_factor(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )


def _standard_sft_scheduler_total_steps(config: StandardSFTControlConfig) -> int:
    return config.scheduler_total_steps


def _stage_metric_name(stage: StageName) -> str:
    if stage == "A":
        return "dev_sft_loss"
    if stage == "B":
        return "dev_replay_loss"
    raise ValueError(f"Stage {stage} has no configured dev selection metric")


def _copy_training_config(
    source: str | Path,
    run_path: Path,
    *,
    stage: StageName,
) -> None:
    target = run_path / "training_config.yaml"
    source_path = Path(source)
    if not target.exists():
        shutil.copy2(source_path, target)
        return
    if target.read_bytes() == source_path.read_bytes():
        return
    stage_target = run_path / f"training_config-stage-{stage}.yaml"
    if stage_target.exists() and stage_target.read_bytes() != source_path.read_bytes():
        raise ValueError(f"Stage {stage} training config differs from its saved config")
    if not stage_target.exists():
        shutil.copy2(source_path, stage_target)


def _should_restore_scheduler(source_stage: StageName, target_stage: StageName) -> bool:
    return source_stage == target_stage


def _stage_trainer_kwargs(config: QwenTrainingConfig) -> dict[str, Any]:
    return {
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "temperature": config.anchor_temperature,
        "cosine_weight": config.cosine_weight,
        "state_weight": config.state_weight,
        "plan_weight": config.plan_weight,
        "replay_weight": config.replay_weight,
        "max_grad_norm": config.max_grad_norm,
    }


def _read_epoch_metrics(checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint) / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _command_train(args: Namespace) -> int:
    config = QwenTrainingConfig.from_yaml(args.config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    loaded = load_qwen_qlora(config, device=args.device)
    anchors = AnchorArtifact.load(args.anchors)
    traces = _read_models(args.traces, TeacherTrace)
    stages = _requested_stages(args, config)
    scheduler_stages = _configured_stages(config) if args.command == "train-pipeline" else stages
    scheduler_rows = {
        stage: build_stage_rows(stage, traces)
        for stage in scheduler_stages
    }
    train_rows = {stage: scheduler_rows[stage] for stage in stages}
    dev_rows = {
        stage: build_stage_rows(
            stage,
            traces,
            splits={"dev"},
        )
        for stage in stages
        if config.evaluate_dev
    }
    for stage, rows in dev_rows.items():
        if not rows:
            raise ValueError(f"Stage {stage} dev dataset is empty")
    if "B" in dev_rows:
        missing = [
            row["example_id"]
            for row in dev_rows["B"]
            if row["example_id"] not in anchors.example_to_row
        ]
        if missing:
            raise ValueError(
                "Stage B dev evaluation requires dev anchors; regenerate anchors "
                "with --original-splits train dev"
            )

    optimizer = build_paged_adamw_8bit(
        loaded.model,
        learning_rate=config.learning_rate,
    )
    calculated_scheduler_steps = sum(
        _optimizer_steps_per_epoch(
            len(scheduler_rows[stage]),
            config.max_optimizer_steps_per_stage,
            config.gradient_accumulation_steps,
        )
        * config.epochs.for_stage(stage)
        for stage in scheduler_stages
    )
    total_scheduler_steps = (
        _configured_scheduler_total_steps(config) or calculated_scheduler_steps
    )
    scheduler = _build_scheduler(
        optimizer,
        config=config,
        total_steps=total_scheduler_steps,
    )
    scaler = None
    if config.bnb_4bit_compute_dtype == "float16":
        scaler = torch.amp.GradScaler("cuda")
    trainer = StageTrainer(
        loaded.slot_model,
        optimizer,
        anchors=anchors,
        scheduler=scheduler,
        scaler=scaler,
        **_stage_trainer_kwargs(config),
    )
    run_path = Path(args.run_dir) / args.run_name
    manager = CheckpointManager(
        run_path,
        run_name=args.run_name,
        seed=args.seed,
    )
    _copy_training_config(args.config, run_path, stage=stages[0])
    restored_metadata: CheckpointMetadata | None = None
    if args.resume:
        source_stage = _checkpoint_payload(args.resume)["metadata"]["stage"]
        restore_scheduler = _should_restore_scheduler(source_stage, stages[0])
        restored_metadata = manager.load(
            args.resume,
            target_stage=stages[0],
            model=loaded.model,
            optimizer=optimizer,
            scheduler=scheduler if restore_scheduler else None,
            scaler=scaler,
            expected={
                "slot_layer": config.slot_layer,
                "special_token_ids": loaded.token_ids,
            },
        )
        trainer.optimizer_steps = restored_metadata.global_step
        trainer.micro_steps = (
            restored_metadata.global_step * config.gradient_accumulation_steps
        )
        if not restore_scheduler:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = config.learning_rate
    elif not _allow_fresh_stage_start(stages, config):
        raise ValueError(f"Stage {stages[0]} requires --resume from its legal predecessor")

    summaries: list[dict[str, Any]] = []
    for stage in stages:
        rows = train_rows[stage]
        first_epoch = _stage_epoch(stage, restored_metadata)
        best_metric: float | None = None
        bad_epochs = 0
        if restored_metadata is not None and restored_metadata.stage == stage:
            resumed_metrics = _read_epoch_metrics(args.resume)
            best_metric = resumed_metrics.get("best_metric")
            bad_epochs = int(resumed_metrics.get("bad_epochs", 0))
        for epoch in range(first_epoch, config.epochs.for_stage(stage) + 1):
            generator = torch.Generator().manual_seed(
                args.seed + sum(map(ord, stage)) + epoch * 1_000_003
            )
            loader = DataLoader(
                rows,
                batch_size=config.micro_batch_size,
                shuffle=True,
                generator=generator,
                collate_fn=_stage_collator(stage, loaded, config),
            )
            batches: Iterable[Mapping[str, Any]] = loader
            batch_total = _training_batch_total(
                len(loader),
                config.max_optimizer_steps_per_stage,
                config.gradient_accumulation_steps,
            )
            if config.max_optimizer_steps_per_stage is not None:
                batches = islice(
                    loader,
                    config.max_optimizer_steps_per_stage
                    * config.gradient_accumulation_steps,
                )
            result = trainer.train_stage(stage, batches, total=batch_total)
            dev_metrics: dict[str, float] | None = None
            selection_metric: float | None = None
            metric_name: str | None = None
            if stage in dev_rows:
                dev_loader = DataLoader(
                    dev_rows[stage],
                    batch_size=config.micro_batch_size,
                    shuffle=False,
                    collate_fn=_stage_collator(stage, loaded, config),
                )
                evaluation = trainer.evaluate_stage(
                    stage,
                    dev_loader,
                    total=len(dev_loader),
                )
                dev_metrics = {
                    "total_loss": evaluation.total_loss,
                    "sft_loss": evaluation.sft_loss,
                }
                if evaluation.replay_loss is not None:
                    dev_metrics["replay_loss"] = evaluation.replay_loss
                metric_name = _stage_metric_name(stage)
                selection_metric = {
                    "A": evaluation.sft_loss,
                    "B": evaluation.replay_loss,
                }[stage]
                if selection_metric is None:
                    raise RuntimeError(f"missing selection metric for Stage {stage}")
                improved = best_metric is None or selection_metric < best_metric
                if improved:
                    best_metric = selection_metric
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            metadata = CheckpointMetadata(
                stage=stage,
                run_name=args.run_name,
                seed=args.seed,
                epoch=epoch,
                global_step=trainer.optimizer_steps,
                slot_layer=config.slot_layer,
                special_token_ids=loaded.token_ids,
            )
            checkpoint = manager.save(
                metadata,
                model=loaded.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            metrics: dict[str, Any] = {
                "stage": stage,
                "epoch": epoch,
                "train": {
                    "examples": len(rows),
                    "last_loss": result.losses[-1],
                    "mean_loss": sum(result.losses) / len(result.losses),
                    "optimizer_steps": result.optimizer_steps,
                    "max_gradient_norm": max(result.gradient_norms),
                },
                "dev": dev_metrics,
                "selection_metric": metric_name,
                "selection_value": selection_metric,
                "best_metric": best_metric,
                "bad_epochs": bad_epochs,
            }
            _write_json(checkpoint / "metrics.json", metrics)
            if selection_metric is not None and improved:
                _write_json(
                    run_path / f"best-{stage}.json",
                    {
                        "stage": stage,
                        "metric": metric_name,
                        "value": selection_metric,
                        "epoch": epoch,
                        "checkpoint": str(checkpoint),
                    },
                )
            summaries.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "examples": len(rows),
                    "optimizer_steps": result.optimizer_steps,
                    "last_loss": result.losses[-1],
                    "dev": dev_metrics,
                    "checkpoint": str(checkpoint),
                }
            )
            if (
                selection_metric is not None
                and config.early_stopping_patience is not None
                and bad_epochs >= config.early_stopping_patience
            ):
                break
        restored_metadata = None
    print(json.dumps(summaries, ensure_ascii=False))
    return 0


def _standard_sft_epoch_seed(seed: int, epoch: int) -> int:
    if epoch <= 3:
        return seed + ord("A") + epoch * 1_000_003
    return seed + ord("B") + (epoch - 3) * 1_000_003


def _command_train_sft_control(args: Namespace) -> int:
    config = StandardSFTControlConfig.from_yaml(args.config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    loaded = load_qwen_standard_sft(config, device=args.device)
    if args.dataset:
        rows = read_visible_sft_dataset(args.dataset)
        train_rows = [row for row in rows if row["split"] == "train"]
        dev_rows = [row for row in rows if row["split"] == "dev"]
    else:
        traces = _read_models(args.traces, TeacherTrace)
        train_rows = build_stage_rows("A", traces)
        dev_rows = build_stage_rows("A", traces, splits={"dev"})
    if not train_rows:
        raise ValueError("standard SFT train dataset is empty")
    if not dev_rows:
        raise ValueError("standard SFT dev dataset is empty")
    optimizer = build_paged_adamw_8bit(loaded.model, learning_rate=config.learning_rate)
    scheduler = _build_scheduler(
        optimizer,
        config=config,
        total_steps=_standard_sft_scheduler_total_steps(config),
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if config.bnb_4bit_compute_dtype == "float16"
        else None
    )
    trainer = StandardSFTTrainer(
        loaded.model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
    )
    run_path = Path(args.run_dir) / args.run_name
    manager = CheckpointManager(run_path, run_name=args.run_name, seed=args.seed)
    _copy_training_config(args.config, run_path, stage="SFT")
    first_epoch = 1
    best_metric: float | None = None
    bad_epochs = 0
    if args.resume:
        restored = manager.load(
            args.resume,
            target_stage="SFT",
            model=loaded.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected={"slot_layer": None, "special_token_ids": None},
        )
        trainer.optimizer_steps = restored.global_step
        trainer.micro_steps = restored.global_step * config.gradient_accumulation_steps
        first_epoch = restored.epoch + 1
        resumed_metrics = _read_epoch_metrics(args.resume)
        best_metric = resumed_metrics.get("best_metric")
        bad_epochs = int(resumed_metrics.get("bad_epochs", 0))
    summaries: list[dict[str, Any]] = []
    for epoch in range(first_epoch, config.sft_epochs + 1):
        loader = DataLoader(
            train_rows,
            batch_size=config.micro_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(
                _standard_sft_epoch_seed(args.seed, epoch)
            ),
            collate_fn=StandardSFTCollator(loaded.tokenizer, max_length=config.max_length),
        )
        batches: Iterable[Mapping[str, Any]] = loader
        batch_total = _training_batch_total(
            len(loader), config.max_optimizer_steps_per_stage, config.gradient_accumulation_steps
        )
        if config.max_optimizer_steps_per_stage is not None:
            batches = islice(
                loader,
                config.max_optimizer_steps_per_stage * config.gradient_accumulation_steps,
            )
        result = trainer.train_epoch(batches, total=batch_total)
        dev_loader = DataLoader(
            dev_rows,
            batch_size=config.micro_batch_size,
            shuffle=False,
            collate_fn=StandardSFTCollator(loaded.tokenizer, max_length=config.max_length),
        )
        evaluation = trainer.evaluate(dev_loader, total=len(dev_loader))
        selection_value = evaluation.sft_loss
        improved = best_metric is None or selection_value < best_metric
        if improved:
            best_metric = selection_value
            bad_epochs = 0
        else:
            bad_epochs += 1
        metadata = CheckpointMetadata(
            checkpoint_version="qwen-standard-sft-control-v1",
            stage="SFT",
            run_name=args.run_name,
            seed=args.seed,
            epoch=epoch,
            global_step=trainer.optimizer_steps,
        )
        checkpoint = manager.save(
            metadata,
            model=loaded.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        metrics = {
            "stage": "SFT",
            "epoch": epoch,
            "train": {
                "examples": len(train_rows),
                "last_loss": result.losses[-1],
                "mean_loss": sum(result.losses) / len(result.losses),
                "optimizer_steps": result.optimizer_steps,
                "max_gradient_norm": max(result.gradient_norms),
            },
            "dev": {"sft_loss": selection_value, "total_loss": selection_value},
            "selection_metric": "dev_sft_loss",
            "selection_value": selection_value,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
        }
        _write_json(checkpoint / "metrics.json", metrics)
        if improved:
            _write_json(
                run_path / "best-SFT.json",
                {
                    "stage": "SFT",
                    "metric": "dev_sft_loss",
                    "value": selection_value,
                    "epoch": epoch,
                    "checkpoint": str(checkpoint),
                },
            )
        summaries.append({"stage": "SFT", "epoch": epoch, "checkpoint": str(checkpoint)})
        if (
            config.early_stopping_patience is not None
            and bad_epochs >= config.early_stopping_patience
        ):
            break
    print(json.dumps(summaries, ensure_ascii=False))
    return 0





def _restore_for_inference(
    args: Namespace,
):
    config = QwenTrainingConfig.from_yaml(args.config)
    loaded = restore_qwen_for_inference(
        config,
        checkpoint=args.checkpoint,
        run_name=args.run_name,
        device=args.device,
    )
    return config, loaded



def _command_generate(args: Namespace) -> int:
    config, loaded = _restore_for_inference(args)
    payload = json.loads(Path(args.history).read_text(encoding="utf-8"))
    history = History.model_validate(payload.get("history", payload))
    prompt = encode_generation_prompt(
        loaded.tokenizer,
        history,
        token_ids=loaded.token_ids,
        max_length=config.max_length,
    )
    device = next(loaded.model.parameters()).device
    prompt = {
        key: (
            {name: value.to(device) for name, value in item.items()}
            if key == "slot_positions"
            else item.to(device)
        )
        for key, item in prompt.items()
    }
    generated = loaded.slot_model.generate(
        input_ids=prompt["input_ids"],
        attention_mask=prompt["attention_mask"],
        slot_positions=prompt["slot_positions"],
        clamps=None,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    response_ids = generated[0, prompt["input_ids"].shape[1] :]
    response = loaded.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    result = {"condition": "normal", "response": response}
    if args.output:
        _write_json(args.output, result)
    print(response)
    return 0


def run_pipeline_command(args: Namespace) -> int:
    if args.command == "precompute-anchors":
        return _command_precompute_anchors(args)
    if args.command in {"train", "train-pipeline"}:
        return _command_train(args)
    if args.command == "train-sft-control":
        return _command_train_sft_control(args)
    if args.command == "generate":
        return _command_generate(args)
    raise ValueError(f"unknown pipeline command {args.command}")
