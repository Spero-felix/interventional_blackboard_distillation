"""Frozen Qwen anchor encoding and safetensors persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase

from .config import STATE_TOKEN_LIMIT
from .model import QwenSlotCausalLM
from .schemas import (
    MASKED_STATE_VALUE,
    PlanSelection,
    STATE_ANCHOR_FIELDS,
    StateBlackboard,
    StateField,
)


_METADATA_KEY = "ibd_anchor_metadata"
_EXAMPLE_ROWS_KEY = "ibd_example_to_row"
_MUTATED_STATE_ROWS_KEY = "ibd_mutated_state_to_row"
_MUTATED_PLAN_ROWS_KEY = "ibd_mutated_plan_to_row"
def state_anchor_payload(state: StateBlackboard) -> dict[str, str]:
    return {field: getattr(state, field) for field in STATE_ANCHOR_FIELDS}


def masked_state_anchor_payload(
    state: StateBlackboard, field: StateField
) -> dict[str, str]:
    """Mask one field only in a temporary diagnostic anchor payload."""

    payload = state_anchor_payload(state)
    payload[field] = MASKED_STATE_VALUE
    return payload


def plan_anchor_payload(selection: PlanSelection) -> dict[str, str]:
    return {
        "strategy": selection.strategies[0],
        "response_goal": selection.response_goal,
        "response_act": selection.response_act,
    }


def serialize_anchor_payload(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_state_token_budget(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count > STATE_TOKEN_LIMIT:
        raise ValueError(
            f"compact STATE uses {token_count} tokens; limit is {STATE_TOKEN_LIMIT}"
        )


@dataclass
class AnchorArtifact:
    state: torch.Tensor
    plan: torch.Tensor
    example_to_row: dict[str, int]
    metadata: dict[str, Any]
    mutated_state: torch.Tensor | None = None
    mutated_plan: torch.Tensor | None = None
    mutated_state_to_row: dict[str, int] = field(default_factory=dict)
    mutated_plan_to_row: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = self._frozen(self.state)
        self.plan = self._frozen(self.plan)
        self.mutated_state = self._frozen_optional(self.mutated_state)
        self.mutated_plan = self._frozen_optional(self.mutated_plan)
        if self.state.ndim != 2 or self.plan.ndim != 2:
            raise ValueError("STATE and PLAN anchor banks must be rank two")
        if self.state.shape[0] == 0 or self.state.shape[0] != self.plan.shape[0]:
            raise ValueError("STATE and PLAN banks must have the same non-zero row count")
        if self.state.shape[1] != self.plan.shape[1]:
            raise ValueError("STATE and PLAN anchors must share hidden size")
        self._validate_mapping(self.example_to_row, self.state.shape[0], "example")
        self._validate_optional(
            self.mutated_state,
            self.mutated_state_to_row,
            "mutated STATE",
        )
        self._validate_optional(
            self.mutated_plan,
            self.mutated_plan_to_row,
            "mutated PLAN",
        )
        for name, tensor in self._tensors().items():
            norms = tensor.float().norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=2e-3, rtol=2e-3):
                raise ValueError(f"{name} anchor rows must be L2 normalized")

    @staticmethod
    def _frozen(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu").contiguous()

    @classmethod
    def _frozen_optional(cls, tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else cls._frozen(tensor)

    @staticmethod
    def _validate_mapping(mapping: Mapping[str, int], rows: int, label: str) -> None:
        if len(mapping) != rows or set(mapping.values()) != set(range(rows)):
            raise ValueError(f"{label} row mapping must cover every row exactly once")

    @classmethod
    def _validate_optional(
        cls,
        tensor: torch.Tensor | None,
        mapping: Mapping[str, int],
        label: str,
    ) -> None:
        if tensor is None:
            if mapping:
                raise ValueError(f"{label} mapping requires a tensor bank")
            return
        if tensor.ndim != 2 or tensor.shape[0] == 0:
            raise ValueError(f"{label} anchor bank must be non-empty and rank two")
        cls._validate_mapping(mapping, tensor.shape[0], label)

    def _tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"state": self.state, "plan": self.plan}
        if self.mutated_state is not None:
            tensors["mutated_state"] = self.mutated_state
        if self.mutated_plan is not None:
            tensors["mutated_plan"] = self.mutated_plan
        return tensors

    def positive_rows(self, example_ids: Sequence[str]) -> torch.Tensor:
        try:
            rows = [self.example_to_row[example_id] for example_id in example_ids]
        except KeyError as error:
            raise ValueError(f"anchor bank has no row for {error.args[0]}") from error
        return torch.tensor(rows, dtype=torch.long)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            _METADATA_KEY: json.dumps(self.metadata, ensure_ascii=False, sort_keys=True),
            _EXAMPLE_ROWS_KEY: json.dumps(self.example_to_row, sort_keys=True),
            _MUTATED_STATE_ROWS_KEY: json.dumps(self.mutated_state_to_row, sort_keys=True),
            _MUTATED_PLAN_ROWS_KEY: json.dumps(self.mutated_plan_to_row, sort_keys=True),
        }
        save_file(self._tensors(), target, metadata=metadata)

    @classmethod
    def load(cls, path: str | Path) -> "AnchorArtifact":
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
        required = {_METADATA_KEY, _EXAMPLE_ROWS_KEY}
        if not required.issubset(metadata):
            raise ValueError("anchor artifact metadata is incomplete")
        artifact_metadata = json.loads(metadata[_METADATA_KEY])
        return cls(
            state=tensors["state"],
            plan=tensors["plan"],
            example_to_row=json.loads(metadata[_EXAMPLE_ROWS_KEY]),
            metadata=artifact_metadata,
            mutated_state=tensors.get("mutated_state"),
            mutated_plan=tensors.get("mutated_plan"),
            mutated_state_to_row=json.loads(metadata.get(_MUTATED_STATE_ROWS_KEY, "{}")),
            mutated_plan_to_row=json.loads(metadata.get(_MUTATED_PLAN_ROWS_KEY, "{}")),
        )


class AnchorEncoder:
    """Encode text with a frozen Qwen and capture last non-padding layer states."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, *, slot_layer: int = 13):
        layers = QwenSlotCausalLM._decoder_layers(model)
        if not 0 <= slot_layer < len(layers):
            raise ValueError(f"slot_layer {slot_layer} is outside {len(layers)} decoder layers")
        self.model = model
        self.tokenizer = tokenizer
        self.slot_layer = slot_layer
        self.layer = layers[slot_layer]

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("anchor encoding requires at least one text")
        batch = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("anchor tokenizer must return attention_mask")
        captured: list[torch.Tensor] = []

        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden)

        handle = self.layer.register_forward_hook(hook)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                self.model(**model_inputs, use_cache=False)
        finally:
            handle.remove()
            self.model.train(was_training)
        if len(captured) != 1:
            raise RuntimeError("anchor layer must execute exactly once")
        hidden = captured[0]
        sequence_index = torch.arange(hidden.shape[1], device=hidden.device)
        last_positions = (attention_mask * sequence_index.unsqueeze(0)).argmax(dim=-1)
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        vectors = F.normalize(hidden[batch_index, last_positions].float(), dim=-1)
        return vectors.to(dtype=torch.float16, device="cpu").contiguous()
