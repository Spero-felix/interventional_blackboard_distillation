"""Two-slot adapters, including a layer-level Qwen capture/clamp wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


class LatentBlackboardCausalLM(nn.Module):
    """Inject exactly two learned latent vectors at caller-supplied positions."""

    SLOT_NAMES = ("STATE", "PLAN")

    def __init__(self, base_model: nn.Module, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.latent_slots = nn.ParameterDict(
            {
                "STATE": nn.Parameter(torch.zeros(hidden_size)),
                "PLAN": nn.Parameter(torch.zeros(hidden_size)),
            }
        )
        nn.init.normal_(self.latent_slots["STATE"], mean=0.0, std=0.02)
        nn.init.normal_(self.latent_slots["PLAN"], mean=0.0, std=0.02)

    def inject_slots(
        self,
        inputs_embeds: torch.Tensor,
        slot_positions: dict[str, torch.Tensor],
        *,
        clamps: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        clamps = clamps or {}
        invalid = (set(slot_positions) | set(clamps)) - set(self.SLOT_NAMES)
        if invalid:
            raise ValueError("latent slot name must be STATE or PLAN")
        missing = set(self.SLOT_NAMES) - set(slot_positions)
        if missing:
            raise ValueError("slot_positions must provide STATE and PLAN")
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.hidden_size:
            raise ValueError("inputs_embeds must have shape [batch, sequence, hidden_size]")

        output = inputs_embeds.clone()
        batch_size = output.shape[0]
        batch_index = torch.arange(batch_size, device=output.device)
        for name in self.SLOT_NAMES:
            position = slot_positions[name].to(device=output.device)
            if position.shape != (batch_size,):
                raise ValueError(f"{name} positions must have shape [batch]")
            vector = clamps.get(name, self.latent_slots[name])
            vector = vector.to(device=output.device, dtype=output.dtype)
            if vector.ndim == 1:
                vector = vector.unsqueeze(0).expand(batch_size, -1)
            if vector.shape != (batch_size, self.hidden_size):
                raise ValueError(f"{name} clamp must have shape [hidden] or [batch, hidden]")
            output[batch_index, position] = vector
        return output

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        slot_positions: dict[str, torch.Tensor],
        clamps: dict[str, torch.Tensor] | None = None,
        **model_kwargs: Any,
    ):
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        embeddings = self.inject_slots(embeddings, slot_positions, clamps=clamps)
        return self.base_model(inputs_embeds=embeddings, **model_kwargs)


@dataclass(frozen=True)
class SlotActivations:
    STATE: torch.Tensor
    PLAN: torch.Tensor


@dataclass(frozen=True)
class SlotForwardOutput:
    """Preserve the Transformers output while exposing effective slot vectors."""

    model_output: Any
    slots: SlotActivations

    @property
    def logits(self) -> torch.Tensor:
        return self.model_output.logits

    @property
    def loss(self) -> torch.Tensor | None:
        return getattr(self.model_output, "loss", None)

    @property
    def past_key_values(self) -> Any:
        return getattr(self.model_output, "past_key_values", None)


@dataclass
class _SlotHookContext:
    positions: dict[str, torch.Tensor]
    clamps: dict[str, torch.Tensor]
    captured: dict[str, torch.Tensor]


class QwenSlotCausalLM(nn.Module):
    """Capture or replace two token states at one Qwen decoder layer."""

    SLOT_NAMES = ("STATE", "PLAN")

    def __init__(
        self,
        base_model: nn.Module,
        *,
        state_token_id: int,
        plan_token_id: int,
        slot_layer: int = 13,
    ):
        super().__init__()
        if state_token_id == plan_token_id:
            raise ValueError("STATE and PLAN token IDs must be distinct")
        self.base_model = base_model
        self.state_token_id = int(state_token_id)
        self.plan_token_id = int(plan_token_id)
        self.slot_layer = int(slot_layer)
        layers = self._decoder_layers(base_model)
        if not 0 <= self.slot_layer < len(layers):
            raise ValueError(
                f"slot_layer {self.slot_layer} is outside {len(layers)} decoder layers"
            )
        self._active_slot_context: _SlotHookContext | None = None
        self._slot_hook_handle = layers[self.slot_layer].register_forward_hook(
            self._layer_output_hook
        )

    @staticmethod
    def _decoder_layers(model: nn.Module) -> nn.ModuleList:
        candidates = [
            getattr(model, "model", None),
            getattr(getattr(model, "base_model", None), "model", None),
        ]
        nested = getattr(getattr(model, "base_model", None), "model", None)
        candidates.append(getattr(nested, "model", None))
        for candidate in candidates:
            layers = getattr(candidate, "layers", None)
            if layers is not None:
                return layers
        raise ValueError("base_model does not expose Qwen decoder layers")

    def _slot_positions(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        positions: dict[str, torch.Tensor] = {}
        for name, token_id in (
            ("STATE", self.state_token_id),
            ("PLAN", self.plan_token_id),
        ):
            matches = input_ids.eq(token_id)
            counts = matches.sum(dim=-1)
            if not torch.all(counts.eq(1)):
                raise ValueError(f"every sequence must contain exactly one {name} token")
            positions[name] = matches.to(dtype=torch.int64).argmax(dim=-1)
        return positions

    def _normalize_clamps(
        self,
        clamps: Mapping[str, torch.Tensor] | None,
        *,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        normalized = dict(clamps or {})
        invalid = set(normalized) - set(self.SLOT_NAMES)
        if invalid:
            raise ValueError("clamp name must be STATE or PLAN")
        hidden_size = int(self.base_model.config.hidden_size)
        for name, value in normalized.items():
            if value.ndim == 1:
                value = value.unsqueeze(0).expand(batch_size, -1)
            if value.shape != (batch_size, hidden_size):
                raise ValueError(
                    f"{name} clamp must have shape [hidden] or [batch, hidden]"
                )
            normalized[name] = value
        return normalized

    @staticmethod
    def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    def _layer_output_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        context = self._active_slot_context
        if context is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("Qwen decoder layer must return [batch, sequence, hidden]")
        max_position = max(int(value.max().item()) for value in context.positions.values())
        if hidden.shape[1] <= max_position:
            return output

        effective = hidden
        if context.clamps:
            effective = hidden.clone()
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            for name, clamp in context.clamps.items():
                position = context.positions[name].to(device=hidden.device)
                effective[batch_index, position] = clamp.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        for name in self.SLOT_NAMES:
            position = context.positions[name].to(device=hidden.device)
            context.captured[name] = effective[batch_index, position]
        return self._replace_hidden(output, effective) if effective is not hidden else output

    def _activate(
        self,
        input_ids: torch.Tensor,
        clamps: Mapping[str, torch.Tensor] | None,
        slot_positions: Mapping[str, torch.Tensor] | None,
    ) -> _SlotHookContext:
        positions = (
            {name: value for name, value in slot_positions.items()}
            if slot_positions is not None
            else self._slot_positions(input_ids)
        )
        if set(positions) != set(self.SLOT_NAMES):
            raise ValueError("slot_positions must provide STATE and PLAN")
        context = _SlotHookContext(
            positions=positions,
            clamps=self._normalize_clamps(clamps, batch_size=input_ids.shape[0]),
            captured={},
        )
        self._active_slot_context = context
        return context

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        slot_positions: Mapping[str, torch.Tensor] | None = None,
        clamps: Mapping[str, torch.Tensor] | None = None,
        **model_kwargs: Any,
    ) -> SlotForwardOutput:
        context = self._activate(input_ids, clamps, slot_positions)
        try:
            model_output = self.base_model(input_ids=input_ids, **model_kwargs)
            if set(context.captured) != set(self.SLOT_NAMES):
                raise RuntimeError("slot layer did not observe the prompt prefill")
            return SlotForwardOutput(
                model_output=model_output,
                slots=SlotActivations(
                    STATE=context.captured["STATE"],
                    PLAN=context.captured["PLAN"],
                ),
            )
        finally:
            if not (self.training and torch.is_grad_enabled()):
                self._active_slot_context = None

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        slot_positions: Mapping[str, torch.Tensor] | None = None,
        clamps: Mapping[str, torch.Tensor] | None = None,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        self._activate(input_ids, clamps, slot_positions)
        try:
            return self.base_model.generate(input_ids=input_ids, **generation_kwargs)
        finally:
            self._active_slot_context = None
