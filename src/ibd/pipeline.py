"""Artifact construction and CLI orchestration for local Qwen training."""

from __future__ import annotations

import hashlib
import json
import random
from argparse import Namespace
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from .anchors import (
    AnchorArtifact,
    AnchorEncoder,
    anchor_protocol_metadata,
    plan_anchor_payload,
    serialize_anchor_payload,
    state_anchor_payload,
    validate_anchor_protocol_metadata,
    validate_state_token_budget,
)
from .backend import OpenAIBackend, StructuredCaller
from .checkpointing import CheckpointManager, CheckpointMetadata
from .config import ANCHOR_PROTOCOL_VERSION, AppConfig
from .evaluation import (
    aggregate_causal_metrics,
    aggregate_intervention_audit,
    mean_rank_gap,
    pair_accuracy,
)
from .hashing import protocol_hash
from .interventions import (
    EffectVerification,
    InterventionBuilder,
    InterventionExcluded,
    MarginPairBuilder,
    PairVerdict,
    StateField,
    mask_state_field,
    select_counterfactual_plan,
)
from .prompting import build_messages
from .progress import track
from .qwen import (
    QwenTrainingConfig,
    load_frozen_qwen_for_anchors,
    load_qwen_qlora,
    tokenizer_manifest_hash,
    validate_local_qwen_directory,
)
from .schemas import (
    History,
    InterventionRecord,
    MASKED_STATE_VALUE,
    MarginPair,
    PlanSelection,
    SafetyCritiqueReport,
    STATE_ANCHOR_FIELDS,
    TeacherTrace,
)
from .storage import read_jsonl, write_jsonl
from .student_data import (
    QwenPairCollator,
    QwenStageCollator,
    encode_generation_prompt,
)
from .teacher import TeacherRunner
from .trainer import StageTrainer, build_paged_adamw_8bit
from .training import length_normalized_score, response_token_log_probs


StageName = Literal["A", "B", "C", "D"]
_STAGES: tuple[StageName, ...] = ("A", "B", "C", "D")


def _intervention_shape_exclusion(
    trace: TeacherTrace,
    function: str,
    *,
    state_field: StateField | None = None,
) -> str | None:
    if function == "STATE":
        if (
            state_field is None
            or not hasattr(trace.state, state_field)
            or any(
                getattr(trace.state, field) == MASKED_STATE_VALUE
                for field in STATE_ANCHOR_FIELDS
            )
        ):
            return "state_not_single_field"
        return None
    if function == "PLAN":
        used = PlanSelection.from_final_answer(trace.final_answer)
        if len(used.strategies) not in {1, 2}:
            return "plan_cardinality"
        if not set(used.strategies).issubset(trace.plan.strategies):
            return "plan_overlap"
    return None


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
    interventions: Sequence[InterventionRecord] = (),
    margins: Sequence[MarginPair] = (),
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
                "strategy_uses": trace.final_answer.model_dump(mode="json")[
                    "strategy_uses"
                ],
            }
            for trace in selected_traces.values()
        ]
    if stage == "C":
        rows: list[dict[str, Any]] = []
        for intervention in _unique_by_id(
            interventions, label="intervention"
        ).values():
            trace = selected_traces.get(intervention.example_id)
            if trace is None:
                continue
            if intervention.full_response != trace.final_response:
                raise ValueError(
                    f"intervention full_response mismatch for {intervention.example_id}"
                )
            if not intervention.localized_degradation or not intervention.bidirectional_verified:
                raise ValueError(
                    f"intervention verification flags are false for {intervention.example_id}"
                )
            rows.append(
                {
                    "example_id": intervention.example_id,
                    "history": trace.history.model_dump(mode="json"),
                    "function": intervention.function,
                    "full_response": intervention.full_response,
                    "counterfactual_response": intervention.counterfactual_response,
                }
            )
        return rows
    if stage == "D":
        rows = []
        for margin in _unique_by_id(margins, label="margin pair").values():
            trace = selected_traces.get(margin.example_id)
            if trace is None:
                continue
            if margin.chosen != trace.final_response:
                raise ValueError(f"margin chosen mismatch for {margin.example_id}")
            if margin.prompt != trace.history.as_prompt():
                raise ValueError(f"margin prompt mismatch for {margin.example_id}")
            if not margin.order_swap_verified or not margin.safety_filter_passed:
                raise ValueError(f"margin verification flags are false for {margin.example_id}")
            rows.append(
                {
                    "example_id": margin.example_id,
                    "history": trace.history.model_dump(mode="json"),
                    "chosen": margin.chosen,
                    "rejected": margin.rejected,
                    "chosen_strategy_uses": [
                        item.model_dump(mode="json")
                        for item in margin.chosen_strategy_uses
                    ],
                    "rejected_strategy_id": margin.rejected_strategy_id,
                    "rejected_strategy": margin.rejected_strategy,
                }
            )
        return rows
    raise ValueError(f"unknown training stage {stage}")


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
    interventions: Sequence[InterventionRecord],
    encoder: AnchorEncoder,
    *,
    metadata: Mapping[str, Any],
    batch_size: int = 1,
    diagnostic_state_field: StateField | None = None,
    global_seed: int | None = None,
) -> AnchorArtifact:
    all_trace_by_id = _unique_by_id(traces, label="Teacher trace")
    intervention_by_id = _unique_by_id(interventions, label="intervention")
    trace_by_id = {
        example_id: trace
        for example_id, trace in all_trace_by_id.items()
        if trace.split == "train"
    }
    ordered_ids = list(trace_by_id)
    state_texts = [
        serialize_anchor_payload(state_anchor_payload(trace_by_id[item].state))
        for item in ordered_ids
    ]
    plan_texts = [
        serialize_anchor_payload(
            plan_anchor_payload(
                PlanSelection.from_final_answer(trace_by_id[item].final_answer)
            )
        )
        for item in ordered_ids
    ]

    state_mutations = [
        (item.example_id, item.mutated_state)
        for item in intervention_by_id.values()
        if item.example_id in all_trace_by_id
        and all_trace_by_id[item.example_id].split in {"train", "dev"}
        and item.function == "STATE"
    ]
    plan_mutations = [
        (item.example_id, item.mutated_plan)
        for item in intervention_by_id.values()
        if item.example_id in all_trace_by_id
        and all_trace_by_id[item.example_id].split in {"train", "dev"}
        and item.function == "PLAN"
    ]
    diagnostic_ids: list[str] = []
    for trace in all_trace_by_id.values():
        if trace.split != "diagnostic_holdout":
            continue
        if diagnostic_state_field is None or global_seed is None:
            raise ValueError(
                "diagnostic intervention clamps require state field and global seed"
            )
        if _intervention_shape_exclusion(
            trace,
            "STATE",
            state_field=diagnostic_state_field,
        ) is not None:
            continue
        if _intervention_shape_exclusion(trace, "PLAN") is not None:
            continue
        diagnostic_state, _ = mask_state_field(
            trace.state,
            diagnostic_state_field,
        )
        diagnostic_plan, _ = select_counterfactual_plan(
            planned=trace.plan,
            used=PlanSelection.from_final_answer(trace.final_answer),
            example_id=trace.example_id,
            global_seed=global_seed,
        )
        state_mutations.append((trace.example_id, diagnostic_state))
        plan_mutations.append((trace.example_id, diagnostic_plan))
        diagnostic_ids.append(trace.example_id)
    mutated_state_texts = [
        serialize_anchor_payload(state_anchor_payload(value))
        for _, value in state_mutations
    ]
    mutated_plan_texts = [
        serialize_anchor_payload(plan_anchor_payload(value))
        for _, value in plan_mutations
    ]
    for text in [*state_texts, *mutated_state_texts]:
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
    mutated_state = (
        _encode_in_batches(
            encoder,
            mutated_state_texts,
            batch_size=batch_size,
            description="encode mutated STATE anchors",
        )
        if state_mutations
        else None
    )
    mutated_plan = (
        _encode_in_batches(
            encoder,
            mutated_plan_texts,
            batch_size=batch_size,
            description="encode mutated PLAN anchors",
        )
        if plan_mutations
        else None
    )
    artifact_metadata = {**dict(metadata), **anchor_protocol_metadata()}
    artifact_metadata["diagnostic_clamp_ids"] = diagnostic_ids
    return AnchorArtifact(
        state=state,
        plan=plan,
        example_to_row={example_id: index for index, example_id in enumerate(ordered_ids)},
        metadata=artifact_metadata,
        mutated_state=mutated_state,
        mutated_plan=mutated_plan,
        mutated_state_to_row={
            example_id: index for index, (example_id, _) in enumerate(state_mutations)
        },
        mutated_plan_to_row={
            example_id: index for index, (example_id, _) in enumerate(plan_mutations)
        },
    )


def _read_models(path: str | Path, model_type: type[Any]) -> list[Any]:
    return [model_type.model_validate(record) for record in read_jsonl(path)]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_anchor_metadata(
    anchors: AnchorArtifact,
    expected: Mapping[str, Any],
) -> None:
    validate_anchor_protocol_metadata(anchors.metadata)
    for key, value in expected.items():
        if key not in anchors.metadata:
            raise ValueError(f"anchor metadata is missing {key}")
        if anchors.metadata[key] != value:
            raise ValueError(
                f"anchor {key} mismatch: artifact={anchors.metadata[key]!r}, "
                f"expected={value!r}"
            )


def _teacher_protocol_hash(traces: Sequence[TeacherTrace]) -> str:
    hashes = {trace.teacher_protocol_hash for trace in traces}
    if len(hashes) != 1:
        raise ValueError("anchor inputs must use exactly one Teacher protocol")
    return next(iter(hashes))


def _anchor_runtime_metadata(
    config: QwenTrainingConfig,
    loaded: Any,
    *,
    traces: Sequence[TeacherTrace] | None = None,
) -> dict[str, Any]:
    model_manifest = validate_local_qwen_directory(config.model_path)
    expected = {
        "model_hash": model_manifest["model_manifest_hash"],
        "tokenizer_hash": tokenizer_manifest_hash(config.model_path),
        "special_token_ids": loaded.token_ids,
        "teacher_schema_hash": protocol_hash(TeacherTrace.model_json_schema()),
        "protocol_hash": config.anchor_config_hash,
        "slot_layer": config.slot_layer,
        **anchor_protocol_metadata(),
    }
    if traces is not None:
        expected["split_hash"] = protocol_hash(
            {trace.example_id: trace.split for trace in traces}
        )
        expected["teacher_protocol_hash"] = _teacher_protocol_hash(traces)
    return expected


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_intervention_effect(
    caller: StructuredCaller,
    trace: TeacherTrace,
    full_response: str,
    intervened_response: str,
    dimension: str,
) -> EffectVerification:
    first, _ = caller.call(
        "pair_verifier",
        build_messages(
            "pair_verifier",
            trace.history,
            PairVerdict,
            context={"A": full_response, "B": intervened_response, "dimension": dimension},
        ),
        PairVerdict,
        example_id=f"{trace.example_id}:effect:{dimension}:ab",
    )
    second, _ = caller.call(
        "pair_verifier",
        build_messages(
            "pair_verifier",
            trace.history,
            PairVerdict,
            context={"A": intervened_response, "B": full_response, "dimension": dimension},
        ),
        PairVerdict,
        example_id=f"{trace.example_id}:effect:{dimension}:ba",
    )
    first_preference = {
        "A": "full",
        "B": "counterfactual",
        "tie": "tie",
    }[first.preferred]
    second_preference = {
        "A": "counterfactual",
        "B": "full",
        "tie": "tie",
    }[second.preferred]
    if first_preference != second_preference:
        return EffectVerification(
            passed=False,
            reason="bidirectional_disagreement",
        )
    if (
        first_preference != "full"
        or first.defect_dimension != dimension
        or second.defect_dimension != dimension
    ):
        return EffectVerification(passed=False, reason="no_localized_effect")
    return EffectVerification(passed=True)


def _verify_intervention_safety(
    caller: StructuredCaller,
    trace: TeacherTrace,
    full_response: str,
    counterfactual_response: str,
) -> bool:
    candidate_ids = {"full", "counterfactual"}
    report, _ = caller.call(
        "safety_critic",
        build_messages(
            "safety_critic",
            trace.history,
            SafetyCritiqueReport,
            context={
                "state": trace.state,
                "candidates": [
                    {
                        "candidate_id": "full",
                        "strategy_id": "S1",
                        "strategy": "Others",
                        "response": full_response,
                        "seed": 0,
                    },
                    {
                        "candidate_id": "counterfactual",
                        "strategy_id": "S2",
                        "strategy": "Others",
                        "response": counterfactual_response,
                        "seed": 1,
                    },
                ],
            },
        ),
        SafetyCritiqueReport,
        example_id=f"{trace.example_id}:intervention:safety",
    )
    return (
        report.critic == "safety"
        and set(report.candidate_issues) == candidate_ids
        and not any(report.candidate_issues.values())
    )


def _command_build_interventions(args: Namespace) -> int:
    config = AppConfig.from_yaml(args.config)
    backend = OpenAIBackend(config)
    runner = TeacherRunner(backend, config)
    caller = StructuredCaller(backend, config)
    traces = _read_models(args.input, TeacherTrace)
    ordered = sorted(traces, key=lambda trace: trace.example_id)
    retained: list[InterventionRecord] = []
    intervention_excluded: dict[str, str] = {}
    intervention_audit: list[dict[str, Any]] = []
    attempted = {"STATE": 0, "PLAN": 0}
    state_attempt_index = 0
    intervention_traces = track(
        ordered,
        desc="build interventions",
        total=len(ordered),
        unit="example",
    )
    for index, trace in enumerate(intervention_traces):
        function = "STATE" if index % 2 == 0 else "PLAN"
        attempted[function] += 1
        state_field: StateField | None = None
        original_plan_categories: list[str] | None = None
        counterfactual_plan_categories: list[str] | None = None
        if function == "STATE":
            state_field = STATE_ANCHOR_FIELDS[
                state_attempt_index % len(STATE_ANCHOR_FIELDS)
            ]
            state_attempt_index += 1
        else:
            original_plan = PlanSelection.from_final_answer(trace.final_answer)
            counterfactual_plan, _ = select_counterfactual_plan(
                planned=trace.plan,
                used=original_plan,
                example_id=trace.example_id,
                global_seed=args.global_seed,
            )
            original_plan_categories = list(original_plan.strategies)
            counterfactual_plan_categories = list(counterfactual_plan.strategies)
        audit_entry: dict[str, Any] = {
            "example_id": trace.example_id,
            "eligibility": "eligible",
            "status": "excluded",
            "function": function,
            "state_field": state_field,
            "original_plan_categories": original_plan_categories,
            "counterfactual_plan_categories": counterfactual_plan_categories,
            "global_seed": args.global_seed,
            "protocol_version": ANCHOR_PROTOCOL_VERSION,
            "original_anchor_hash": None,
            "counterfactual_anchor_hash": None,
            "effect_verification": {
                "passed": False,
                "localized_degradation": False,
                "bidirectional_verified": False,
            },
            "exclusion_reason": None,
        }
        shape_exclusion = _intervention_shape_exclusion(
            trace,
            function,
            state_field=state_field,
        )
        if shape_exclusion is not None:
            intervention_excluded[trace.example_id] = shape_exclusion
            audit_entry["eligibility"] = "ineligible"
            audit_entry["exclusion_reason"] = shape_exclusion
            intervention_audit.append(audit_entry)
            continue
        builder = InterventionBuilder(
            runner,
            lambda full, changed, dimension, current=trace: _verify_intervention_effect(
                caller,
                current,
                full,
                changed,
                dimension,
            ),
            lambda full, changed, current=trace: _verify_intervention_safety(
                caller,
                current,
                full,
                changed,
            ),
        )
        try:
            record = builder.build(
                trace,
                function,
                state_field=state_field,
                global_seed=args.global_seed,
            )
        except InterventionExcluded as error:
            intervention_excluded[trace.example_id] = error.reason
            audit_entry["exclusion_reason"] = error.reason
            intervention_audit.append(audit_entry)
        else:
            retained.append(record)
            if function == "STATE":
                original_payload = state_anchor_payload(trace.state)
                counterfactual_payload = state_anchor_payload(record.mutated_state)
            else:
                original_payload = plan_anchor_payload(
                    PlanSelection.from_final_answer(trace.final_answer)
                )
                counterfactual_payload = plan_anchor_payload(record.mutated_plan)
            audit_entry.update(
                {
                    "status": "retained",
                    "original_anchor_hash": hashlib.sha256(
                        serialize_anchor_payload(original_payload).encode("utf-8")
                    ).hexdigest(),
                    "counterfactual_anchor_hash": hashlib.sha256(
                        serialize_anchor_payload(counterfactual_payload).encode("utf-8")
                    ).hexdigest(),
                    "effect_verification": {
                        "passed": True,
                        "localized_degradation": record.localized_degradation,
                        "bidirectional_verified": record.bidirectional_verified,
                    },
                }
            )
            intervention_audit.append(audit_entry)
    margin_builder = MarginPairBuilder(backend, config)
    margins: list[MarginPair] = []
    margin_excluded: dict[str, str] = {}
    for trace in track(
        ordered,
        desc="build margin pairs",
        total=len(ordered),
        unit="example",
    ):
        pair = margin_builder.build(trace)
        if pair is None:
            margin_excluded[trace.example_id] = (
                "no safe localized near-negative or order-stable preference"
            )
        else:
            margins.append(pair)
    write_jsonl(args.output, retained)
    write_jsonl(args.margins_output, margins)

    def counts_by_split(items: Sequence[Any]) -> dict[str, int]:
        trace_splits = {trace.example_id: trace.split for trace in traces}
        return {
            split: sum(trace_splits[item.example_id] == split for item in items)
            for split in ("train", "dev", "diagnostic_holdout")
        }

    _write_json(
        args.manifest,
        {
            "protocol_version": "teacher-interventions-v1",
            "stage_ab": {
                "retained": counts_by_split(traces),
                "excluded": {},
            },
            "stage_c": {
                "attempted": attempted,
                "retained": len(retained),
                "retained_by_split": counts_by_split(retained),
                "excluded": intervention_excluded,
                "audit": intervention_audit,
            },
            "stage_d": {
                "attempted": len(ordered),
                "retained": len(margins),
                "retained_by_split": counts_by_split(margins),
                "excluded": margin_excluded,
            },
            "input_sha256": _file_sha256(args.input),
        },
    )
    print(f"wrote {len(retained)} interventions and {len(margins)} margin pairs")
    return 0


def _command_precompute_anchors(args: Namespace) -> int:
    config = QwenTrainingConfig.from_yaml(args.config)
    traces = _read_models(args.traces, TeacherTrace)
    interventions = (
        _read_models(args.interventions, InterventionRecord)
        if args.interventions
        else []
    )
    loaded = load_frozen_qwen_for_anchors(config, device=args.device)
    encoder = AnchorEncoder(
        loaded.model,
        loaded.tokenizer,
        slot_layer=config.slot_layer,
    )
    split_hash = protocol_hash(
        {trace.example_id: trace.split for trace in traces}
    )
    metadata = {
        **_anchor_runtime_metadata(config, loaded, traces=traces),
        "split_hash": split_hash,
        "trace_source_hash": _file_sha256(args.traces),
        "intervention_source_hash": (
            _file_sha256(args.interventions) if args.interventions else None
        ),
    }
    artifact = build_anchor_artifact(
        traces,
        interventions,
        encoder,
        metadata=metadata,
        batch_size=args.batch_size,
        diagnostic_state_field=args.diagnostic_state_field,
        global_seed=args.global_seed,
    )
    artifact.save(args.output)
    print(f"wrote {len(artifact.example_to_row)} anchor rows")
    return 0


def _checkpoint_payload(path: str | Path) -> dict[str, Any]:
    return json.loads((Path(path) / "checkpoint.json").read_text(encoding="utf-8"))


def _data_manifest_hash(args: Namespace) -> str:
    return protocol_hash(
        {
            "traces": _file_sha256(args.traces),
            "interventions": _file_sha256(args.interventions),
            "margins": _file_sha256(args.margins),
        }
    )


def _stage_collator(stage: StageName, loaded: Any, config: QwenTrainingConfig):
    common = {
        "max_length": config.max_length,
        "token_ids": loaded.token_ids,
    }
    if stage in {"A", "B"}:
        return QwenStageCollator(loaded.tokenizer, **common)
    if stage == "C":
        return QwenPairCollator(
            loaded.tokenizer,
            left_key="full_response",
            right_key="counterfactual_response",
            **common,
        )
    return QwenPairCollator(
        loaded.tokenizer,
        left_key="chosen",
        right_key="rejected",
        **common,
    )


def _requested_stages(args: Namespace) -> list[StageName]:
    if args.command == "train":
        return [args.stage]
    if not args.resume:
        return list(_STAGES)
    source_stage = _checkpoint_payload(args.resume)["metadata"]["stage"]
    source_index = _STAGES.index(source_stage)
    remaining = list(_STAGES[source_index + 1 :])
    if not remaining:
        raise ValueError("pipeline resume checkpoint already completed Stage D")
    return remaining


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


def _command_train(args: Namespace) -> int:
    config = QwenTrainingConfig.from_yaml(args.config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    loaded = load_qwen_qlora(config, device=args.device)
    anchors = AnchorArtifact.load(args.anchors)
    traces = _read_models(args.traces, TeacherTrace)
    _validate_anchor_metadata(
        anchors,
        {
            **_anchor_runtime_metadata(config, loaded, traces=traces),
            "trace_source_hash": _file_sha256(args.traces),
            "intervention_source_hash": _file_sha256(args.interventions),
        },
    )
    interventions = _read_models(args.interventions, InterventionRecord)
    margins = _read_models(args.margins, MarginPair)
    optimizer = build_paged_adamw_8bit(
        loaded.model,
        learning_rate=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = None
    if config.bnb_4bit_compute_dtype == "float16":
        scaler = torch.amp.GradScaler("cuda")
    trainer = StageTrainer(
        loaded.slot_model,
        optimizer,
        anchors=anchors,
        scheduler=scheduler,
        scaler=scaler,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        temperature=config.anchor_temperature,
        cosine_weight=config.cosine_weight,
        state_weight=config.state_weight,
        plan_weight=config.plan_weight,
        margin=config.margin,
        replay_weight=config.replay_weight,
    )
    run_path = Path(args.run_dir) / args.run_name
    manager = CheckpointManager(
        run_path,
        run_name=args.run_name,
        seed=args.seed,
        config_hash=config.config_hash,
    )
    stages = _requested_stages(args)
    parent_hash: str | None = None
    restored_metadata: CheckpointMetadata | None = None
    if args.resume:
        resume_payload = _checkpoint_payload(args.resume)
        parent_hash = resume_payload["checkpoint_hash"]
        restored_metadata = manager.load(
            args.resume,
            target_stage=stages[0],
            model=loaded.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected={
                "protocol_hash": anchors.metadata["protocol_hash"],
                "seed": args.seed,
                "config_hash": config.config_hash,
                "model_hash": anchors.metadata["model_hash"],
                "tokenizer_hash": anchors.metadata["tokenizer_hash"],
                "data_manifest_hash": _data_manifest_hash(args),
                "split_hash": anchors.metadata["split_hash"],
                "anchor_hash": _file_sha256(args.anchors),
                "slot_layer": config.slot_layer,
                "special_token_ids": loaded.token_ids,
            },
        )
        trainer.optimizer_steps = restored_metadata.global_step
        trainer.micro_steps = (
            restored_metadata.global_step * config.gradient_accumulation_steps
        )
    elif stages[0] != "A":
        raise ValueError(f"Stage {stages[0]} requires --resume from its legal predecessor")

    summaries: list[dict[str, Any]] = []
    data_hash = _data_manifest_hash(args)
    anchor_hash = _file_sha256(args.anchors)
    for stage in stages:
        epoch = _stage_epoch(stage, restored_metadata)
        rows = build_stage_rows(
            stage,
            traces,
            interventions=interventions,
            margins=margins,
        )
        generator = torch.Generator().manual_seed(
            args.seed + ord(stage) + epoch * 1_000_003
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
        metadata = CheckpointMetadata(
            stage=stage,
            run_name=args.run_name,
            seed=args.seed,
            epoch=epoch,
            global_step=trainer.optimizer_steps,
            protocol_hash=anchors.metadata["protocol_hash"],
            config_hash=config.config_hash,
            model_hash=anchors.metadata["model_hash"],
            tokenizer_hash=anchors.metadata["tokenizer_hash"],
            data_manifest_hash=data_hash,
            split_hash=anchors.metadata["split_hash"],
            anchor_hash=anchor_hash,
            slot_layer=config.slot_layer,
            special_token_ids=loaded.token_ids,
            parent_checkpoint_hash=parent_hash,
        )
        checkpoint = manager.save(
            metadata,
            model=loaded.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=loaded.tokenizer,
        )
        payload = _checkpoint_payload(checkpoint)
        parent_hash = payload["checkpoint_hash"]
        summaries.append(
            {
                "stage": stage,
                "examples": len(rows),
                "optimizer_steps": result.optimizer_steps,
                "last_loss": result.losses[-1],
                "max_gradient_norm": max(result.gradient_norms),
                "total_parameters": loaded.total_parameters,
                "trainable_parameters": loaded.trainable_parameters,
                "lora_parameters": loaded.lora_parameters,
                "token_row_parameters": loaded.token_row_parameters,
                "checkpoint": str(checkpoint),
                "checkpoint_hash": parent_hash,
            }
        )
    print(json.dumps(summaries, ensure_ascii=False))
    return 0


def _anchor_clamps(
    anchors: AnchorArtifact,
    function: str,
    example_ids: Sequence[str],
    *,
    repeats: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if function == "STATE":
        bank = anchors.mutated_state
        mapping = anchors.mutated_state_to_row
    elif function == "PLAN":
        bank = anchors.mutated_plan
        mapping = anchors.mutated_plan_to_row
    else:
        raise ValueError("clamp function must be STATE or PLAN")
    if bank is None:
        raise ValueError(f"anchor artifact has no mutated {function} bank")
    try:
        vectors = torch.stack([bank[mapping[example_id]] for example_id in example_ids])
    except KeyError as error:
        raise ValueError(f"no mutated {function} anchor for {error.args[0]}") from error
    return {function: vectors.to(device).repeat(repeats, 1)}


def _score_encoded(
    loaded: Any,
    batch: Mapping[str, Any],
    *,
    clamps: Mapping[str, torch.Tensor] | None = None,
    capture_slots: bool = False,
) -> Any:
    device = next(loaded.model.parameters()).device
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key != "labels"
    }
    slot_positions = {
        key: value.to(device) for key, value in batch["slot_positions"].items()
    }
    with torch.inference_mode():
        output = loaded.slot_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            slot_positions=slot_positions,
            clamps=clamps,
            use_cache=False,
        )
    token_scores, mask = response_token_log_probs(
        output.logits,
        batch["labels"].to(device),
    )
    scores = length_normalized_score(token_scores, mask).cpu()
    if not capture_slots:
        return scores
    slots = type(output.slots)(
        STATE=output.slots.STATE.detach().cpu(),
        PLAN=output.slots.PLAN.detach().cpu(),
    )
    return scores, slots


def _causal_anchor_clamp(
    anchors: AnchorArtifact,
    function: str,
    example_id: str,
    *,
    counterfactual: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if function not in {"STATE", "PLAN"}:
        raise ValueError("clamp function must be STATE or PLAN")
    if counterfactual:
        bank = anchors.mutated_state if function == "STATE" else anchors.mutated_plan
        mapping = (
            anchors.mutated_state_to_row
            if function == "STATE"
            else anchors.mutated_plan_to_row
        )
        label = "counterfactual"
    else:
        bank = anchors.state if function == "STATE" else anchors.plan
        mapping = anchors.example_to_row
        label = "original"
    if bank is None:
        raise ValueError(f"anchor artifact has no {label} {function} bank")
    try:
        vector = bank[mapping[example_id]]
    except KeyError as error:
        raise ValueError(
            f"no {label} {function} anchor for {error.args[0]}"
        ) from error
    return {function: vector.unsqueeze(0).to(device)}


def _restore_for_inference(
    args: Namespace,
    *,
    expected: Mapping[str, Any] | None = None,
):
    config = QwenTrainingConfig.from_yaml(args.config)
    loaded = load_qwen_qlora(config, device=args.device)
    model_manifest = validate_local_qwen_directory(config.model_path)
    payload = _checkpoint_payload(args.checkpoint)
    manager = CheckpointManager(
        Path(args.checkpoint).parent,
        run_name=args.run_name,
        seed=int(payload["metadata"]["seed"]),
        config_hash=config.config_hash,
    )
    manager.load(
        args.checkpoint,
        target_stage=payload["metadata"]["stage"],
        model=loaded.model,
        expected={
            "config_hash": config.config_hash,
            "model_hash": model_manifest["model_manifest_hash"],
            "tokenizer_hash": tokenizer_manifest_hash(config.model_path),
            "slot_layer": config.slot_layer,
            "special_token_ids": loaded.token_ids,
            **dict(expected or {}),
        },
        restore_rng=False,
    )
    loaded.slot_model.eval()
    return config, loaded


def _command_evaluate(args: Namespace) -> int:
    traces = _read_models(args.traces, TeacherTrace)
    interventions = _read_models(args.interventions, InterventionRecord)
    margins = _read_models(args.margins, MarginPair)
    anchors = AnchorArtifact.load(args.anchors)
    config, loaded = _restore_for_inference(
        args,
        expected={
            "protocol_hash": anchors.metadata["protocol_hash"],
            "data_manifest_hash": _data_manifest_hash(args),
            "split_hash": anchors.metadata["split_hash"],
            "anchor_hash": _file_sha256(args.anchors),
        },
    )
    _validate_anchor_metadata(
        anchors,
        {
            **_anchor_runtime_metadata(config, loaded, traces=traces),
            "trace_source_hash": _file_sha256(args.traces),
            "intervention_source_hash": _file_sha256(args.interventions),
        },
    )
    rows = build_stage_rows("D", traces, margins=margins, splits={"dev"})
    if not rows:
        raise ValueError("evaluation has no dev margin pairs")
    collator = QwenPairCollator(
        loaded.tokenizer,
        max_length=config.max_length,
        token_ids=loaded.token_ids,
        left_key="chosen",
        right_key="rejected",
    )
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    for row in track(
        rows,
        desc="evaluate margins",
        total=len(rows),
        unit="example",
    ):
        batch = collator([row])
        chosen_scores.append(float(_score_encoded(loaded, batch["chosen"])[0]))
        rejected_scores.append(float(_score_encoded(loaded, batch["rejected"])[0]))
    functional_rows = build_stage_rows(
        "C",
        traces,
        interventions=interventions,
        splits={"dev"},
    )
    functional_collator = QwenPairCollator(
        loaded.tokenizer,
        max_length=config.max_length,
        token_ids=loaded.token_ids,
        left_key="full_response",
        right_key="counterfactual_response",
    )
    causal_observations: list[dict[str, Any]] = []
    slot_atol = 1e-5
    slot_rtol = 1e-5
    for row in track(
        functional_rows,
        desc="evaluate interventions",
        total=len(functional_rows),
        unit="example",
    ):
        batch = functional_collator([row])
        device = next(loaded.model.parameters()).device
        original_clamp = _causal_anchor_clamp(
            anchors,
            row["function"],
            row["example_id"],
            counterfactual=False,
            device=device,
        )
        counterfactual_clamp = _causal_anchor_clamp(
            anchors,
            row["function"],
            row["example_id"],
            counterfactual=True,
            device=device,
        )
        original_original, original_original_slots = _score_encoded(
            loaded,
            batch["chosen"],
            clamps=original_clamp,
            capture_slots=True,
        )
        original_counterfactual, original_counterfactual_slots = _score_encoded(
            loaded,
            batch["rejected"],
            clamps=original_clamp,
            capture_slots=True,
        )
        counterfactual_original, counterfactual_original_slots = _score_encoded(
            loaded,
            batch["chosen"],
            clamps=counterfactual_clamp,
            capture_slots=True,
        )
        counterfactual_counterfactual, counterfactual_counterfactual_slots = (
            _score_encoded(
                loaded,
                batch["rejected"],
                clamps=counterfactual_clamp,
                capture_slots=True,
            )
        )
        non_target = "PLAN" if row["function"] == "STATE" else "STATE"
        original_candidate_invariant = torch.allclose(
            getattr(original_original_slots, non_target),
            getattr(counterfactual_original_slots, non_target),
            atol=slot_atol,
            rtol=slot_rtol,
        )
        counterfactual_candidate_invariant = torch.allclose(
            getattr(original_counterfactual_slots, non_target),
            getattr(counterfactual_counterfactual_slots, non_target),
            atol=slot_atol,
            rtol=slot_rtol,
        )
        causal_observations.append(
            {
                "example_id": row["example_id"],
                "function": row["function"],
                "original_clamp_original_score": float(original_original[0]),
                "original_clamp_counterfactual_score": float(
                    original_counterfactual[0]
                ),
                "counterfactual_clamp_original_score": float(
                    counterfactual_original[0]
                ),
                "counterfactual_clamp_counterfactual_score": float(
                    counterfactual_counterfactual[0]
                ),
                "non_target_slot_invariant": bool(
                    original_candidate_invariant
                    and counterfactual_candidate_invariant
                ),
            }
        )
    manifest = json.loads(
        Path(args.intervention_manifest).read_text(encoding="utf-8")
    )
    try:
        audit_rows = manifest["stage_c"]["audit"]
    except (KeyError, TypeError) as error:
        raise ValueError("intervention manifest must contain stage_c.audit") from error
    if not isinstance(audit_rows, list):
        raise ValueError("intervention manifest stage_c.audit must be a list")
    planner_strategies = {
        trace.example_id: list(trace.plan.strategies) for trace in traces
    }
    report = {
        "split": "dev",
        "pairs": len(rows),
        "pair_accuracy": pair_accuracy(chosen_scores, rejected_scores),
        "mean_rank_gap": mean_rank_gap(chosen_scores, rejected_scores),
        "chosen_scores": chosen_scores,
        "rejected_scores": rejected_scores,
        "causal": aggregate_causal_metrics(causal_observations),
        "causal_observations": causal_observations,
        "intervention_audit": aggregate_intervention_audit(
            audit_rows,
            planner_strategies=planner_strategies,
        ),
        "non_target_slot_allclose": {"atol": slot_atol, "rtol": slot_rtol},
    }
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _command_generate(args: Namespace) -> int:
    clamp_args = (args.clamp, args.anchors, args.example_id)
    if any(clamp_args) and not all(clamp_args):
        raise ValueError("--clamp, --anchors, and --example-id must be provided together")
    anchors = AnchorArtifact.load(args.anchors) if args.clamp else None
    config, loaded = _restore_for_inference(
        args,
        expected=(
            {
                "protocol_hash": anchors.metadata["protocol_hash"],
                "split_hash": anchors.metadata["split_hash"],
                "anchor_hash": _file_sha256(args.anchors),
            }
            if anchors is not None
            else None
        ),
    )
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
    clamps = None
    if args.clamp:
        assert anchors is not None
        _validate_anchor_metadata(
            anchors,
            _anchor_runtime_metadata(config, loaded),
        )
        clamps = _anchor_clamps(
            anchors,
            args.clamp,
            [args.example_id],
            repeats=1,
            device=device,
        )
    generated = loaded.slot_model.generate(
        input_ids=prompt["input_ids"],
        attention_mask=prompt["attention_mask"],
        slot_positions=prompt["slot_positions"],
        clamps=clamps,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    response_ids = generated[0, prompt["input_ids"].shape[1] :]
    response = loaded.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    result = {"condition": args.clamp or "normal", "response": response}
    if args.output:
        _write_json(args.output, result)
    print(response)
    return 0


def run_pipeline_command(args: Namespace) -> int:
    if args.command == "build-interventions":
        return _command_build_interventions(args)
    if args.command == "precompute-anchors":
        return _command_precompute_anchors(args)
    if args.command in {"train", "train-pipeline"}:
        return _command_train(args)
    if args.command == "evaluate":
        return _command_evaluate(args)
    if args.command == "generate":
        return _command_generate(args)
    raise ValueError(f"unknown pipeline command {args.command}")
