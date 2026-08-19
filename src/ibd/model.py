"""A minimal two-slot adapter for an arbitrary causal language model."""

from __future__ import annotations

from typing import Any

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

