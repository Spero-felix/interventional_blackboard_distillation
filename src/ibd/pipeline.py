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
    masked_state_anchor_payload,
    plan_anchor_payload,
    serialize_anchor_payload,
    state_anchor_payload,
    validate_state_token_budget,
)
from .backend import OpenAIBackend, StructuredCaller
from .checkpointing import CheckpointManager, CheckpointMetadata
from .config import AppConfig
from .evaluation import aggregate_causal_metrics, aggregate_intervention_audit
from .interventions import (
    ConditionalEffectVerdict,
    EffectVerification,
    InterventionBuilder,
    InterventionExcluded,
    StateEffectVerdict,
    select_counterfactual_plan,
)
from .prompting import build_messages
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
    InterventionRecord,
    MASKED_STATE_VALUE,
    PlanSelection,
    SafetyVerdict,
    STATE_ANCHOR_FIELDS,
    StateBlackboard,
    StateField,
    TeacherTrace,
)
from .storage import read_jsonl, write_jsonl
from .student_data import (
    QwenPairCollator,
    QwenStageCollator,
    StandardSFTCollator,
    encode_generation_prompt,
)
from .state_guides import STATE_FIELD_GUIDES
from .teacher import TeacherRunner
from .trainer import StandardSFTTrainer, StageTrainer, build_paged_adamw_8bit
from .training import length_normalized_score, response_token_log_probs
from .visible_sft import VisibleSFTDatasetRecord


StageName = Literal["A", "B", "B2", "C"]
_STAGES: tuple[StageName, ...] = ("A", "B", "C")


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
        if len(trace.candidates) < 2:
            return "plan_no_alternative"
        used = trace.final_selection.to_plan_selection()
        if used.strategies[0] not in trace.plan.strategies:
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
    if stage in {"B2", "C"}:
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
            if (
                stage == "B2"
                and intervention.conditioning_contract != "single_variable_v1"
            ):
                raise ValueError(
                    f"B2 intervention {intervention.example_id} requires "
                    "single_variable_v1 conditioning"
                )
            if (
                not intervention.localized_effect
                or not intervention.conditional_correspondence_verified
                or not intervention.bidirectional_verified
            ):
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
    interventions: Sequence[InterventionRecord],
    encoder: AnchorEncoder,
    *,
    metadata: Mapping[str, Any],
    batch_size: int = 1,
    diagnostic_state_field: StateField | None = None,
    global_seed: int | None = None,
    original_splits: set[str] | None = None,
) -> AnchorArtifact:
    all_trace_by_id = _unique_by_id(traces, label="Teacher trace")
    intervention_by_id = _unique_by_id(interventions, label="intervention")
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
    diagnostic_state_payloads: list[tuple[str, dict[str, str]]] = []
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
        diagnostic_state = masked_state_anchor_payload(
            trace.state, diagnostic_state_field
        )
        diagnostic_plan, _ = select_counterfactual_plan(
            candidates=trace.candidates,
            selected_candidate_id=trace.final_selection.selected_candidate_id,
            example_id=trace.example_id,
            global_seed=global_seed,
        )
        diagnostic_state_payloads.append((trace.example_id, diagnostic_state))
        plan_mutations.append((trace.example_id, diagnostic_plan))
        diagnostic_ids.append(trace.example_id)
    state_mutation_payloads = [
        (example_id, state_anchor_payload(value))
        for example_id, value in state_mutations
    ] + diagnostic_state_payloads
    mutated_state_texts = [
        serialize_anchor_payload(payload)
        for _, payload in state_mutation_payloads
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
        if state_mutation_payloads
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
    artifact_metadata = dict(metadata)
    artifact_metadata["diagnostic_clamp_ids"] = diagnostic_ids
    return AnchorArtifact(
        state=state,
        plan=plan,
        example_to_row={example_id: index for index, example_id in enumerate(ordered_ids)},
        metadata=artifact_metadata,
        mutated_state=mutated_state,
        mutated_plan=mutated_plan,
        mutated_state_to_row={
            example_id: index
            for index, (example_id, _) in enumerate(state_mutation_payloads)
        },
        mutated_plan_to_row={
            example_id: index for index, (example_id, _) in enumerate(plan_mutations)
        },
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


def _verify_intervention_effect(
    caller: StructuredCaller,
    trace: TeacherTrace,
    original_plan: PlanSelection,
    counterfactual_plan: PlanSelection,
    original_response: str,
    counterfactual_response: str,
) -> EffectVerification:
    first, _ = caller.call(
        "condition_effect_verifier",
        build_messages(
            "condition_effect_verifier",
            trace.history,
            ConditionalEffectVerdict,
            context={
                "condition_A": original_plan,
                "response_A": original_response,
                "condition_B": counterfactual_plan,
                "response_B": counterfactual_response,
                "function": "PLAN",
            },
        ),
        ConditionalEffectVerdict,
        example_id=f"{trace.example_id}:effect:PLAN:ab",
    )
    second, _ = caller.call(
        "condition_effect_verifier",
        build_messages(
            "condition_effect_verifier",
            trace.history,
            ConditionalEffectVerdict,
            context={
                "condition_A": counterfactual_plan,
                "response_A": counterfactual_response,
                "condition_B": original_plan,
                "response_B": original_response,
                "function": "PLAN",
            },
        ),
        ConditionalEffectVerdict,
        example_id=f"{trace.example_id}:effect:PLAN:ba",
    )
    if (
        first.condition_a_fit != second.condition_b_fit
        or first.condition_b_fit != second.condition_a_fit
        or first.control_effect_present != second.control_effect_present
    ):
        return EffectVerification(
            passed=False,
            reason="bidirectional_disagreement",
        )
    if (
        not first.condition_a_fit
        or not first.condition_b_fit
        or not first.control_effect_present
    ):
        return EffectVerification(passed=False, reason="no_localized_effect")
    return EffectVerification(passed=True)


def _combine_state_effect_verdicts(
    first: StateEffectVerdict,
    second: StateEffectVerdict,
) -> EffectVerification:
    first_fields = set(first.affected_non_target_fields)
    second_fields = set(second.affected_non_target_fields)
    if (
        first.condition_a_fit != second.condition_b_fit
        or first.condition_b_fit != second.condition_a_fit
        or first.target_effect_present != second.target_effect_present
        or first.localized_effect != second.localized_effect
        or first_fields != second_fields
    ):
        return EffectVerification(
            passed=False,
            reason="bidirectional_disagreement",
        )
    if (
        not first.condition_a_fit
        or not first.condition_b_fit
        or not first.target_effect_present
    ):
        return EffectVerification(passed=False, reason="no_localized_effect")
    if not first.localized_effect or first_fields:
        return EffectVerification(passed=False, reason="non_localized_effect")
    return EffectVerification(passed=True)


def _verify_state_intervention_effect(
    caller: StructuredCaller,
    trace: TeacherTrace,
    original_state: StateBlackboard,
    mutated_state: StateBlackboard,
    target_field: StateField,
    full_response: str,
    counterfactual_response: str,
) -> EffectVerification:
    original_payload = original_state.model_dump(mode="json")
    mutated_payload = mutated_state.model_dump(mode="json")
    unchanged_state = {
        field: value
        for field, value in original_payload.items()
        if field != target_field
    }
    condition_original = {
        "target_value": original_payload[target_field],
        "response": full_response,
    }
    condition_counterfactual = {
        "target_value": mutated_payload[target_field],
        "response": counterfactual_response,
    }
    target_guide = STATE_FIELD_GUIDES[target_field]
    verification_context = {
        "target_field": target_field,
        "target_field_definition": target_guide.definition,
        "permitted_local_effects": list(target_guide.permitted_local_effects),
        "prohibited_local_effects": list(target_guide.prohibited_local_effects),
        "unchanged_state": unchanged_state,
    }
    first, _ = caller.call(
        "state_effect_verifier",
        build_messages(
            "state_effect_verifier",
            trace.history,
            StateEffectVerdict,
            context={
                **verification_context,
                "condition_A": condition_original,
                "condition_B": condition_counterfactual,
            },
        ),
        StateEffectVerdict,
        example_id=f"{trace.example_id}:effect:STATE:{target_field}:ab",
    )
    second, _ = caller.call(
        "state_effect_verifier",
        build_messages(
            "state_effect_verifier",
            trace.history,
            StateEffectVerdict,
            context={
                **verification_context,
                "condition_A": condition_counterfactual,
                "condition_B": condition_original,
            },
        ),
        StateEffectVerdict,
        example_id=f"{trace.example_id}:effect:STATE:{target_field}:ba",
    )
    return _combine_state_effect_verdicts(first, second)


def _verify_intervention_safety(
    caller: StructuredCaller,
    trace: TeacherTrace,
    full_response: str,
    counterfactual_response: str,
) -> bool:
    report, _ = caller.call(
        "safety_verifier",
        build_messages(
            "safety_verifier",
            trace.history,
            SafetyVerdict,
            context={
                "original_response": full_response,
                "counterfactual_response": counterfactual_response,
            },
        ),
        SafetyVerdict,
        example_id=f"{trace.example_id}:intervention:safety",
    )
    return report.original_safe and report.counterfactual_safe


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
    state_plan_policy = getattr(args, "state_plan_policy", "rerun").replace(
        "-", "_"
    )
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
            original_plan = trace.final_selection.to_plan_selection()
            original_plan_categories = list(original_plan.strategies)
        audit_entry: dict[str, Any] = {
            "example_id": trace.example_id,
            "eligibility": "eligible",
            "status": "excluded",
            "function": function,
            "state_field": state_field,
            "original_plan_categories": original_plan_categories,
            "counterfactual_plan_categories": counterfactual_plan_categories,
            "global_seed": args.global_seed,
            "effect_verification": {
                "passed": False,
                "localized_effect": False,
                "conditional_correspondence_verified": False,
                "bidirectional_verified": False,
                "affected_non_target_fields": [],
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
        if function == "PLAN":
            counterfactual_plan, _ = select_counterfactual_plan(
                candidates=trace.candidates,
                selected_candidate_id=trace.final_selection.selected_candidate_id,
                example_id=trace.example_id,
                global_seed=args.global_seed,
            )
            counterfactual_plan_categories = list(counterfactual_plan.strategies)
            audit_entry["counterfactual_plan_categories"] = (
                counterfactual_plan_categories
            )
        builder = InterventionBuilder(
            runner,
            verify_safety=(
                lambda full, changed, current=trace: _verify_intervention_safety(
                    caller,
                    current,
                    full,
                    changed,
                )
            ),
            verify_state_effect=(
                lambda original, mutated, field, full, changed, current=trace:
                _verify_state_intervention_effect(
                    caller,
                    current,
                    original,
                    mutated,
                    field,
                    full,
                    changed,
                )
            ),
            verify_plan_effect=(
                lambda original_plan, counterfactual_plan, original, changed,
                current=trace: _verify_intervention_effect(
                    caller,
                    current,
                    original_plan,
                    counterfactual_plan,
                    original,
                    changed,
                )
            ),
        )
        try:
            record = builder.build(
                trace,
                function,
                state_field=state_field,
                global_seed=args.global_seed,
                state_plan_policy=state_plan_policy,
            )
        except InterventionExcluded as error:
            intervention_excluded[trace.example_id] = error.reason
            audit_entry["exclusion_reason"] = error.reason
            intervention_audit.append(audit_entry)
        else:
            retained.append(record)
            audit_entry.update(
                {
                    "status": "retained",
                    "effect_verification": {
                        "passed": True,
                        "localized_effect": record.localized_effect,
                        "conditional_correspondence_verified": (
                            record.conditional_correspondence_verified
                        ),
                        "bidirectional_verified": record.bidirectional_verified,
                        "affected_non_target_fields": list(
                            record.affected_non_target_fields
                        ),
                    },
                }
            )
            intervention_audit.append(audit_entry)
    write_jsonl(args.output, retained)

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
            "state_plan_policy": state_plan_policy,
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
        },
    )
    print(f"wrote {len(retained)} interventions")
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
    artifact = build_anchor_artifact(
        traces,
        interventions,
        encoder,
        metadata={},
        batch_size=args.batch_size,
        diagnostic_state_field=args.diagnostic_state_field,
        global_seed=args.global_seed,
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
    if stage in {"B2", "C"}:
        return QwenPairCollator(
            loaded.tokenizer,
            left_key="full_response",
            right_key="counterfactual_response",
            **common,
        )
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
    if stage in {"B2", "C"}:
        return "dev_total_loss"
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
        "margin": config.margin,
        "replay_weight": config.replay_weight,
        "b2_natural_weight": config.b2.natural_weight,
        "b2_conditioned_weight": config.b2.conditioned_weight,
        "b2_alignment_weight": config.b2.alignment_weight,
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
    interventions = _read_models(args.interventions, InterventionRecord)
    stages = _requested_stages(args, config)
    scheduler_stages = _configured_stages(config) if args.command == "train-pipeline" else stages
    scheduler_rows = {
        stage: build_stage_rows(stage, traces, interventions=interventions)
        for stage in scheduler_stages
    }
    train_rows = {stage: scheduler_rows[stage] for stage in stages}
    dev_rows = {
        stage: build_stage_rows(
            stage,
            traces,
            interventions=interventions,
            splits={"dev"},
        )
        for stage in stages
        if config.evaluate_dev and stage in {"A", "B", "B2", "C"}
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
    for conditioned_stage in ("B2", "C"):
        if conditioned_stage not in dev_rows:
            continue
        missing_original = [
            row["example_id"]
            for row in dev_rows[conditioned_stage]
            if row["example_id"] not in anchors.example_to_row
        ]
        missing_mutated = [
            row["example_id"]
            for row in dev_rows[conditioned_stage]
            if (
                row["function"] == "STATE"
                and row["example_id"] not in anchors.mutated_state_to_row
            )
            or (
                row["function"] == "PLAN"
                and row["example_id"] not in anchors.mutated_plan_to_row
            )
        ]
        if missing_original or missing_mutated:
            raise ValueError(
                f"Stage {conditioned_stage} dev evaluation requires original and mutated dev anchors; "
                "regenerate anchors from the intervention file with "
                "--original-splits train dev"
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
    elif stages[0] != "A":
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
                if evaluation.natural_ce is not None:
                    dev_metrics.update(
                        {
                            "natural_ce": evaluation.natural_ce,
                            "alignment_loss": evaluation.alignment_loss,
                            "original_conditioned_ce": evaluation.original_conditioned_ce,
                            "counterfactual_conditioned_ce": (
                                evaluation.counterfactual_conditioned_ce
                            ),
                            "conditioned_ce": evaluation.conditioned_ce,
                        }
                    )
                metric_name = _stage_metric_name(stage)
                selection_metric = {
                    "A": evaluation.sft_loss,
                    "B": evaluation.replay_loss,
                    "B2": evaluation.total_loss,
                    "C": evaluation.total_loss,
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
):
    config = QwenTrainingConfig.from_yaml(args.config)
    loaded = restore_qwen_for_inference(
        config,
        checkpoint=args.checkpoint,
        run_name=args.run_name,
        device=args.device,
    )
    return config, loaded


def _command_evaluate(args: Namespace) -> int:
    traces = _read_models(args.traces, TeacherTrace)
    interventions = _read_models(args.interventions, InterventionRecord)
    anchors = AnchorArtifact.load(args.anchors)
    config, loaded = _restore_for_inference(args)
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
            batch["original"],
            clamps=original_clamp,
            capture_slots=True,
        )
        original_counterfactual, original_counterfactual_slots = _score_encoded(
            loaded,
            batch["counterfactual"],
            clamps=original_clamp,
            capture_slots=True,
        )
        counterfactual_original, counterfactual_original_slots = _score_encoded(
            loaded,
            batch["original"],
            clamps=counterfactual_clamp,
            capture_slots=True,
        )
        counterfactual_counterfactual, counterfactual_counterfactual_slots = (
            _score_encoded(
                loaded,
                batch["counterfactual"],
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
        "pairs": len(functional_rows),
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
    clamps = None
    if args.clamp:
        assert anchors is not None
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
    if args.command == "train-sft-control":
        return _command_train_sft_control(args)
    if args.command == "evaluate":
        return _command_evaluate(args)
    if args.command == "generate":
        return _command_generate(args)
    raise ValueError(f"unknown pipeline command {args.command}")
    masked_state_anchor_payload,
