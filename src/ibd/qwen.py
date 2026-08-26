"""Frozen Qwen2.5-7B QLoRA configuration and local model loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model import QwenSlotCausalLM
from .student_data import IBD_PLAN_TOKEN, IBD_STATE_TOKEN, add_ibd_tokens


LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class QwenTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: Path = Path("/home/wangnianxiang/models/Qwen2.5-7B-Instruct")
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    lora_r: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=64, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    lora_targets: tuple[str, ...] = LORA_TARGETS
    max_length: int = Field(default=2048, gt=0)
    micro_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    gradient_checkpointing: bool = True
    optimizer: Literal["paged_adamw_8bit"] = "paged_adamw_8bit"
    learning_rate: float = Field(default=2e-4, gt=0.0)
    slot_layer: int = 13
    anchor_temperature: float = Field(default=0.07, gt=0.0)
    cosine_weight: float = Field(default=1.0, ge=0.0)
    state_weight: float = Field(default=1.0, ge=0.0)
    plan_weight: float = Field(default=1.0, ge=0.0)
    margin: float = Field(default=0.5, ge=0.0)
    replay_weight: float = Field(default=0.1, ge=0.0)
    max_optimizer_steps_per_stage: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_frozen_protocol(self) -> "QwenTrainingConfig":
        if not self.load_in_4bit or not self.bnb_4bit_use_double_quant:
            raise ValueError("3090 protocol requires 4-bit double quantization")
        if self.lora_targets != LORA_TARGETS:
            raise ValueError("QLoRA target modules must match the frozen seven-module set")
        if not self.gradient_checkpointing:
            raise ValueError("3090 protocol requires gradient checkpointing")
        if self.micro_batch_size != 1:
            raise ValueError("3090 functional-pair protocol requires micro_batch_size=1")
        if self.slot_layer != 13:
            raise ValueError("production slot_layer must be 13")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QwenTrainingConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)

@dataclass(frozen=True)
class LoadedQwen:
    tokenizer: Any
    model: torch.nn.Module
    slot_model: QwenSlotCausalLM
    token_ids: dict[str, int]
    trainable_parameters: int
    total_parameters: int
    lora_parameters: int
    token_row_parameters: int
    unexpected_trainables: list[str]


@dataclass(frozen=True)
class FrozenQwen:
    tokenizer: Any
    model: torch.nn.Module
    token_ids: dict[str, int]


def validate_local_qwen_directory(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    required = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"local Qwen directory is incomplete: {', '.join(missing)}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if config.get("architectures") != ["Qwen2ForCausalLM"]:
        mismatches["architectures"] = (
            config.get("architectures"),
            ["Qwen2ForCausalLM"],
        )
    shards = sorted(root.glob("model-*-of-00004.safetensors"))
    if len(shards) != 4:
        raise ValueError("local Qwen directory must contain exactly four weight shards")
    if mismatches:
        raise ValueError(f"local Qwen architecture mismatch: {mismatches}")
    return {
        "path": str(root),
        "architecture": "Qwen2ForCausalLM",
        **expected,
        "weight_shards": [shard.name for shard in shards],
    }


def _validate_loaded_architecture(model: torch.nn.Module) -> None:
    config = model.config
    expected = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
    }
    mismatches = {
        name: (getattr(config, name, None), expected_value)
        for name, expected_value in expected.items()
        if getattr(config, name, None) != expected_value
    }
    if mismatches:
        raise ValueError(f"loaded model does not match Qwen2.5-7B protocol: {mismatches}")


def _parameter_accounting(
    model: torch.nn.Module,
) -> tuple[int, int, int, int, list[str]]:
    total = 0
    trainable = 0
    lora = 0
    token_rows = 0
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if not parameter.requires_grad:
            continue
        trainable += parameter.numel()
        if "lora_" in name:
            lora += parameter.numel()
        elif "trainable_tokens" in name:
            token_rows += parameter.numel()
        else:
            unexpected.append(name)
    return trainable, total, lora, token_rows, unexpected


def load_qwen_qlora(
    config: QwenTrainingConfig,
    *,
    device: int = 0,
    tokenizer_loader: Callable[..., Any] | None = None,
    model_loader: Callable[..., torch.nn.Module] | None = None,
    quantization_config_factory: Callable[..., Any] | None = None,
    prepare_model_fn: Callable[..., torch.nn.Module] | None = None,
    peft_model_factory: Callable[..., torch.nn.Module] | None = None,
) -> LoadedQwen:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
    model_loader = model_loader or AutoModelForCausalLM.from_pretrained
    quantization_config_factory = quantization_config_factory or BitsAndBytesConfig
    prepare_model_fn = prepare_model_fn or prepare_model_for_kbit_training
    peft_model_factory = peft_model_factory or get_peft_model

    model_path = str(config.model_path)
    tokenizer = tokenizer_loader(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    token_ids = add_ibd_tokens(tokenizer)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.bnb_4bit_compute_dtype]
    quantization_config = quantization_config_factory(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = model_loader(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map={"": device},
        torch_dtype=compute_dtype,
    )
    _validate_loaded_architecture(model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    checkpointing_kwargs = {"use_reentrant": False}
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpointing_kwargs
        )
    model = prepare_model_fn(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
        trainable_token_indices={
            "embed_tokens": [token_ids["STATE"], token_ids["PLAN"]],
            "lm_head": [token_ids["STATE"], token_ids["PLAN"]],
        },
    )
    model = peft_model_factory(model, lora_config)
    trainable, total, lora, token_rows, unexpected = _parameter_accounting(model)
    if not trainable:
        raise ValueError("QLoRA model has no trainable parameters")
    if unexpected:
        raise ValueError(f"unexpected trainable parameters: {unexpected}")
    slot_model = QwenSlotCausalLM(
        model,
        state_token_id=token_ids["STATE"],
        plan_token_id=token_ids["PLAN"],
        slot_layer=config.slot_layer,
    )
    return LoadedQwen(
        tokenizer=tokenizer,
        model=model,
        slot_model=slot_model,
        token_ids=token_ids,
        trainable_parameters=trainable,
        total_parameters=total,
        lora_parameters=lora,
        token_row_parameters=token_rows,
        unexpected_trainables=unexpected,
    )


def load_frozen_qwen_for_anchors(
    config: QwenTrainingConfig,
    *,
    device: int = 0,
) -> FrozenQwen:
    """Load original local BF16 Qwen without quantization or PEFT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    token_ids = add_ibd_tokens(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    _validate_loaded_architecture(model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return FrozenQwen(tokenizer=tokenizer, model=model, token_ids=token_ids)
