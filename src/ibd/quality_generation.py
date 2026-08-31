"""Normal-response generation for Teacher, Base, and trained Student models."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from .quality import (
    EvaluatedModel,
    GenerationSettings,
    QualityEvalConfig,
    QualityFailure,
    QualityManifest,
    QualityResponse,
    ensure_manifest,
)
from .progress import track
from .qwen import (
    QwenTrainingConfig,
    StandardSFTControlConfig,
    load_qwen_base_for_generation,
    restore_qwen_for_inference,
    restore_qwen_standard_sft_for_inference,
)
from .schemas import History, TeacherTrace
from .storage import append_jsonl, read_jsonl
from .student_data import encode_base_generation_prompt, encode_generation_prompt
from .visible_sft import parse_visible_sft_response


class ResponseAdapter(Protocol):
    def generate(self, history: History, settings: GenerationSettings) -> str: ...
    def close(self) -> None: ...


class _BaseAdapter:
    def __init__(self, spec: EvaluatedModel, device: int):
        assert spec.training_config is not None
        self.config = QwenTrainingConfig.from_yaml(spec.training_config)
        self.loaded = load_qwen_base_for_generation(self.config, device=device)

    def generate(self, history: History, settings: GenerationSettings) -> str:
        prompt = encode_base_generation_prompt(
            self.loaded.tokenizer,
            history,
            max_length=self.config.max_length,
        )
        device = next(self.loaded.model.parameters()).device
        prompt = {name: value.to(device) for name, value in prompt.items()}
        generated = self.loaded.model.generate(
            **prompt,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        response_ids = generated[0, prompt["input_ids"].shape[1] :]
        return self.loaded.tokenizer.decode(
            response_ids, skip_special_tokens=True
        ).strip()

    def close(self) -> None:
        del self.loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _StudentAdapter:
    def __init__(self, spec: EvaluatedModel, device: int):
        assert spec.training_config is not None
        assert spec.checkpoint is not None
        assert spec.run_name is not None
        self.config = QwenTrainingConfig.from_yaml(spec.training_config)
        self.loaded = restore_qwen_for_inference(
            self.config,
            checkpoint=spec.checkpoint,
            run_name=spec.run_name,
            device=device,
        )

    def generate(self, history: History, settings: GenerationSettings) -> str:
        prompt = encode_generation_prompt(
            self.loaded.tokenizer,
            history,
            token_ids=self.loaded.token_ids,
            max_length=self.config.max_length,
        )
        device = next(self.loaded.model.parameters()).device
        prompt = {
            key: (
                {name: value.to(device) for name, value in item.items()}
                if key == "slot_positions"
                else item.to(device)
            )
            for key, item in prompt.items()
        }
        generated = self.loaded.slot_model.generate(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            slot_positions=prompt["slot_positions"],
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        response_ids = generated[0, prompt["input_ids"].shape[1] :]
        return self.loaded.tokenizer.decode(
            response_ids, skip_special_tokens=True
        ).strip()

    def close(self) -> None:
        del self.loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _StandardSFTAdapter:
    def __init__(self, spec: EvaluatedModel, device: int):
        assert spec.training_config is not None
        assert spec.checkpoint is not None
        assert spec.run_name is not None
        self.config = StandardSFTControlConfig.from_yaml(spec.training_config)
        self.loaded = restore_qwen_standard_sft_for_inference(
            self.config,
            checkpoint=spec.checkpoint,
            run_name=spec.run_name,
            device=device,
        )

    def generate(self, history: History, settings: GenerationSettings) -> str:
        prompt = encode_base_generation_prompt(
            self.loaded.tokenizer,
            history,
            max_length=self.config.max_length,
        )
        device = next(self.loaded.model.parameters()).device
        prompt = {name: value.to(device) for name, value in prompt.items()}
        generated = self.loaded.model.generate(
            **prompt,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        response_ids = generated[0, prompt["input_ids"].shape[1] :]
        return self.loaded.tokenizer.decode(
            response_ids, skip_special_tokens=True
        ).strip()

    def close(self) -> None:
        del self.loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


AdapterFactory = Callable[[EvaluatedModel, int], ResponseAdapter]


def _default_factories() -> dict[str, AdapterFactory]:
    return {
        "base_qwen": _BaseAdapter,
        "student_checkpoint": _StudentAdapter,
        "standard_sft_checkpoint": _StandardSFTAdapter,
        "visible_sft_checkpoint": _StandardSFTAdapter,
    }


def _selected_traces(
    config: QualityEvalConfig, traces: Sequence[TeacherTrace]
) -> list[TeacherTrace]:
    selected = [trace for trace in traces if trace.split in config.splits]
    keys: set[tuple[str, str]] = set()
    for trace in selected:
        key = (trace.split, trace.example_id)
        if key in keys:
            raise ValueError(f"duplicate quality trace: {key}")
        keys.add(key)
    return sorted(selected, key=lambda item: (item.split, item.example_id))


def generate_quality_responses(
    config: QualityEvalConfig,
    traces: Sequence[TeacherTrace],
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    failure_path: str | Path,
    resume: bool,
    continue_on_error: bool,
    device: int,
    adapter_factories: Mapping[str, AdapterFactory] | None = None,
) -> int:
    selected = _selected_traces(config, traces)
    expected_keys = [
        (trace.split, trace.example_id, spec.model_id)
        for trace in selected
        for spec in config.models
    ]
    manifest = QualityManifest(
        stage="generation",
        splits=list(config.splits),
        model_ids=[item.model_id for item in config.models],
        expected_keys=expected_keys,
        settings={
            "generation": config.generation.model_dump(mode="json"),
            "models": [item.model_dump(mode="json") for item in config.models],
        },
    )
    ensure_manifest(manifest_path, manifest, resume=resume)
    target = Path(output_path)
    if target.exists() and not resume:
        raise FileExistsError(f"quality response output already exists: {target}")
    completed: set[tuple[str, str, str]] = set()
    if resume and target.exists():
        for raw in read_jsonl(target):
            row = QualityResponse.model_validate(raw)
            key = (row.split, row.example_id, row.model_id)
            if key in completed:
                raise ValueError(f"duplicate existing quality response: {key}")
            completed.add(key)

    factories = {**_default_factories(), **dict(adapter_factories or {})}
    failures = 0
    for spec in config.models:
        pending = [
            trace
            for trace in selected
            if (trace.split, trace.example_id, spec.model_id) not in completed
        ]
        if not pending:
            continue
        adapter = None
        if spec.source != "teacher_trace":
            try:
                adapter = factories[spec.source](spec, device)
            except Exception as exc:
                if not continue_on_error:
                    raise
                for trace in pending:
                    append_jsonl(
                        failure_path,
                        QualityFailure(
                            stage="generation",
                            example_id=trace.example_id,
                            split=trace.split,
                            model_id=spec.model_id,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        ),
                    )
                    failures += 1
                continue
        try:
            progress = track(
                pending,
                desc=f"quality generate {spec.model_id}",
                total=len(pending),
                unit="response",
            )
            for trace in progress:
                progress.set_postfix(example=trace.example_id)
                try:
                    if spec.source == "teacher_trace":
                        response = trace.final_response
                        raw_response = None
                        seed = None
                        reused = True
                    else:
                        assert adapter is not None
                        generated = adapter.generate(trace.history, config.generation).strip()
                        raw_response = (
                            generated
                            if spec.source == "visible_sft_checkpoint"
                            else None
                        )
                        response = (
                            parse_visible_sft_response(generated)
                            if raw_response is not None
                            else generated
                        )
                        if not response:
                            raise ValueError("generated response is empty")
                        seed = config.generation.seed
                        reused = False
                    append_jsonl(
                        target,
                        QualityResponse(
                            example_id=trace.example_id,
                            split=trace.split,
                            model_id=spec.model_id,
                            response_source=spec.source,
                            history=trace.history,
                            response=response,
                            raw_response=raw_response,
                            generation_seed=seed,
                            generation_config=config.generation,
                            reused=reused,
                        ),
                    )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    append_jsonl(
                        failure_path,
                        QualityFailure(
                            stage="generation",
                            example_id=trace.example_id,
                            split=trace.split,
                            model_id=spec.model_id,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        ),
                    )
                    failures += 1
        finally:
            if adapter is not None:
                adapter.close()
    return 1 if failures else 0
