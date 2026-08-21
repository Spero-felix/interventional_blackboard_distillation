"""Single-process Stage A-D optimization for the Qwen slot wrapper."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .anchors import AnchorArtifact, validate_anchor_protocol_metadata
from .model import QwenSlotCausalLM, SlotForwardOutput
from .progress import track
from .training import (
    length_normalized_score,
    margin_alignment_loss,
    response_token_log_probs,
    stage_b_loss,
    stage_d_loss,
    symmetric_margin_loss,
)


StageName = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class StageResult:
    stage: StageName
    losses: list[float]
    optimizer_steps: int
    gradient_norms: list[float]


def build_paged_adamw_8bit(
    model: torch.nn.Module,
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    import bitsandbytes as bnb

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("cannot optimize a model with no trainable parameters")
    return bnb.optim.PagedAdamW8bit(parameters, lr=learning_rate)


class StageTrainer:
    def __init__(
        self,
        model: QwenSlotCausalLM,
        optimizer: torch.optim.Optimizer,
        *,
        anchors: AnchorArtifact | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        gradient_accumulation_steps: int = 1,
        temperature: float = 0.07,
        cosine_weight: float = 1.0,
        state_weight: float = 1.0,
        plan_weight: float = 1.0,
        margin: float = 0.5,
        replay_weight: float = 0.1,
        max_grad_norm: float = 1.0,
    ):
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.model = model
        self.optimizer = optimizer
        self.anchors = anchors
        self.scheduler = scheduler
        self.scaler = scaler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.temperature = temperature
        self.cosine_weight = cosine_weight
        self.state_weight = state_weight
        self.plan_weight = plan_weight
        self.margin = margin
        self.replay_weight = replay_weight
        self.max_grad_norm = max_grad_norm
        self.micro_steps = 0
        self.optimizer_steps = 0
        self.gradient_norms: list[float] = []
        self.optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        return {
            key: (
                {name: value.to(device) for name, value in item.items()}
                if key == "slot_positions"
                else item.to(device)
                if isinstance(item, torch.Tensor)
                else item
            )
            for key, item in batch.items()
        }

    def _forward(self, batch: Mapping[str, Any], *, clamps=None) -> SlotForwardOutput:
        device = next(self.model.parameters()).device
        prepared = self._device_batch(batch, device)
        return self.model(
            input_ids=prepared["input_ids"],
            attention_mask=prepared.get("attention_mask"),
            slot_positions=prepared["slot_positions"],
            clamps=clamps,
            use_cache=False,
        )

    @staticmethod
    def _scores(output: SlotForwardOutput, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(output.logits.device)
        token_scores, mask = response_token_log_probs(output.logits, labels)
        return length_normalized_score(token_scores, mask)

    def _require_anchors(self) -> AnchorArtifact:
        if self.anchors is None:
            raise ValueError("Stage B-D require a frozen anchor artifact")
        validate_anchor_protocol_metadata(self.anchors.metadata)
        return self.anchors

    def _replay_loss(
        self,
        output: SlotForwardOutput,
        scores: torch.Tensor,
        example_ids: list[str],
        *,
        count: int | None = None,
    ) -> torch.Tensor:
        anchors = self._require_anchors()
        count = count or len(example_ids)
        rows = anchors.positive_rows(example_ids[:count]).to(scores.device)
        return stage_b_loss(
            response_loss=-scores[:count].mean(),
            state_slots=output.slots.STATE[:count],
            plan_slots=output.slots.PLAN[:count],
            state_bank=anchors.state.to(scores.device),
            plan_bank=anchors.plan.to(scores.device),
            positive_rows=rows,
            temperature=self.temperature,
            cosine_weight=self.cosine_weight,
            state_weight=self.state_weight,
            plan_weight=self.plan_weight,
        )

    def _backward(self, loss: torch.Tensor, *, finish_microstep: bool) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is NaN or Inf")
        scaled_loss = loss / self.gradient_accumulation_steps
        if self.scaler is None:
            scaled_loss.backward()
        else:
            self.scaler.scale(scaled_loss).backward()
        if not finish_microstep:
            return
        self.micro_steps += 1
        if self.micro_steps % self.gradient_accumulation_steps == 0:
            self._optimizer_step()

    def _optimizer_step(self, *, remainder_scale: float = 1.0) -> None:
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not parameters:
            raise FloatingPointError("optimizer step has no gradients")
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if remainder_scale != 1.0:
            for parameter in parameters:
                parameter.grad.mul_(remainder_scale)
        if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise FloatingPointError("gradient is NaN or Inf")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("gradient norm is NaN or Inf")
        self.gradient_norms.append(float(gradient_norm.detach().cpu()))
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_steps += 1

    def finish_accumulation(self) -> None:
        remainder = self.micro_steps % self.gradient_accumulation_steps
        if remainder:
            self._optimizer_step(
                remainder_scale=self.gradient_accumulation_steps / remainder
            )
            self.micro_steps += self.gradient_accumulation_steps - remainder

    def _pad_pair_tensor(
        self,
        tensor: torch.Tensor,
        *,
        target_length: int,
        fill: int,
    ) -> torch.Tensor:
        if tensor.shape[1] == target_length:
            return tensor
        padding = torch.full(
            (tensor.shape[0], target_length - tensor.shape[1]),
            fill,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat((tensor, padding), dim=1)

    def _merge_pair(
        self,
        chosen: Mapping[str, Any],
        rejected: Mapping[str, Any],
    ) -> tuple[dict[str, Any], int]:
        chosen_count = chosen["input_ids"].shape[0]
        if rejected["input_ids"].shape[0] != chosen_count:
            raise ValueError("chosen and rejected batches must have equal batch size")
        target_length = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
        pad_id = int(getattr(self.model.base_model.config, "pad_token_id", 0) or 0)
        merged = {
            "input_ids": torch.cat(
                (
                    self._pad_pair_tensor(chosen["input_ids"], target_length=target_length, fill=pad_id),
                    self._pad_pair_tensor(rejected["input_ids"], target_length=target_length, fill=pad_id),
                )
            ),
            "attention_mask": torch.cat(
                (
                    self._pad_pair_tensor(chosen["attention_mask"], target_length=target_length, fill=0),
                    self._pad_pair_tensor(rejected["attention_mask"], target_length=target_length, fill=0),
                )
            ),
            "labels": torch.cat(
                (
                    self._pad_pair_tensor(chosen["labels"], target_length=target_length, fill=-100),
                    self._pad_pair_tensor(rejected["labels"], target_length=target_length, fill=-100),
                )
            ),
            "slot_positions": {
                name: torch.cat(
                    (chosen["slot_positions"][name], rejected["slot_positions"][name])
                )
                for name in ("STATE", "PLAN")
            },
            "example_ids": chosen["example_ids"] + rejected["example_ids"],
        }
        return merged, chosen_count

    def _intervention_clamp(
        self,
        function: str,
        example_ids: list[str],
        *,
        mutated: bool,
        repeats: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        anchors = self._require_anchors()
        if function == "STATE":
            bank = anchors.mutated_state if mutated else anchors.state
            mapping = anchors.mutated_state_to_row if mutated else anchors.example_to_row
        elif function == "PLAN":
            bank = anchors.mutated_plan if mutated else anchors.plan
            mapping = anchors.mutated_plan_to_row if mutated else anchors.example_to_row
        else:
            raise ValueError("intervention function must be STATE or PLAN")
        if bank is None:
            raise ValueError(f"anchor artifact has no mutated {function} bank")
        if not example_ids:
            raise ValueError("intervention clamp requires at least one example ID")
        if repeats <= 0:
            raise ValueError("intervention clamp repeats must be positive")
        try:
            vectors = torch.stack([bank[mapping[example_id]] for example_id in example_ids])
        except KeyError as error:
            anchor_kind = "mutated" if mutated else "original"
            raise ValueError(
                f"no {anchor_kind} {function} anchor for {error.args[0]}"
            ) from error
        return {function: vectors.to(device).repeat(repeats, 1)}

    def train_batch(self, stage: StageName, batch: Mapping[str, Any]) -> float:
        self.model.train()
        if stage in {"B", "C", "D"}:
            self._require_anchors()
        if stage == "A":
            output = self._forward(batch)
            loss = -self._scores(output, batch["labels"]).mean()
            self._backward(loss, finish_microstep=True)
            return float(loss.detach())
        if stage == "B":
            output = self._forward(batch)
            scores = self._scores(output, batch["labels"])
            loss = self._replay_loss(output, scores, batch["example_ids"])
            self._backward(loss, finish_microstep=True)
            return float(loss.detach())
        if stage == "C":
            functions = list(batch["functions"])
            if len(set(functions)) != 1:
                raise ValueError("Stage C microbatch must contain one intervention function")
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            merged, chosen_count = self._merge_pair(chosen, rejected)
            example_ids = list(batch["example_ids"])
            if len(functions) != chosen_count or len(example_ids) != chosen_count:
                raise ValueError(
                    "Stage C functions and example IDs must match the microbatch size"
                )
            if (
                list(chosen["example_ids"]) != example_ids
                or list(rejected["example_ids"]) != example_ids
            ):
                raise ValueError(
                    "Stage C chosen, counterfactual, and intervention example IDs must match"
                )
            normal = self._forward(merged)
            normal_scores = self._scores(normal, merged["labels"])
            normal_chosen = normal_scores[:chosen_count]
            replay = self._replay_loss(
                normal,
                normal_scores,
                chosen["example_ids"],
                count=chosen_count,
            )

            device = next(self.model.parameters()).device
            original_clamps = self._intervention_clamp(
                functions[0],
                example_ids,
                mutated=False,
                repeats=2,
                device=device,
            )
            original_clamp = self._forward(merged, clamps=original_clamps)
            original_scores = self._scores(original_clamp, merged["labels"])

            counterfactual_clamps = self._intervention_clamp(
                functions[0],
                example_ids,
                mutated=True,
                repeats=2,
                device=device,
            )
            counterfactual_clamp = self._forward(
                merged,
                clamps=counterfactual_clamps,
            )
            counterfactual_scores = self._scores(
                counterfactual_clamp,
                merged["labels"],
            )
            preference = symmetric_margin_loss(
                original_clamp_original_score=original_scores[:chosen_count],
                original_clamp_counterfactual_score=original_scores[chosen_count:],
                counterfactual_clamp_original_score=counterfactual_scores[:chosen_count],
                counterfactual_clamp_counterfactual_score=counterfactual_scores[
                    chosen_count:
                ],
                margin=self.margin,
            )
            loss = -normal_chosen.mean() + self.replay_weight * replay + preference
            self._backward(loss, finish_microstep=True)
            return float(loss.detach())
        if stage == "D":
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            merged, chosen_count = self._merge_pair(chosen, rejected)
            output = self._forward(merged)
            scores = self._scores(output, merged["labels"])
            chosen_scores = scores[:chosen_count]
            rejected_scores = scores[chosen_count:]
            replay = self._replay_loss(
                output,
                scores,
                chosen["example_ids"],
                count=chosen_count,
            )
            loss = stage_d_loss(
                chosen_sft=-chosen_scores.mean(),
                chosen_scores=chosen_scores,
                rejected_scores=rejected_scores,
                replay_loss=replay,
                margin=self.margin,
                replay_weight=self.replay_weight,
            )
            self._backward(loss, finish_microstep=True)
            return float(loss.detach())
        raise ValueError(f"unknown training stage {stage}")

    def train_stage(
        self,
        stage: StageName,
        batches: Iterable[Mapping[str, Any]],
        *,
        total: int | None = None,
    ) -> StageResult:
        start_steps = self.optimizer_steps
        start_norms = len(self.gradient_norms)
        progress = track(
            batches,
            desc=f"train stage {stage}",
            total=total,
            unit="batch",
        )
        losses: list[float] = []
        for batch in progress:
            loss = self.train_batch(stage, batch)
            losses.append(loss)
            progress.set_postfix(loss=f"{loss:.4f}", step=self.optimizer_steps)
        if not losses:
            raise ValueError(f"Stage {stage} dataset is empty")
        steps_before_finish = self.optimizer_steps
        self.finish_accumulation()
        if self.optimizer_steps != steps_before_finish:
            progress.set_postfix(
                loss=f"{losses[-1]:.4f}",
                step=self.optimizer_steps,
            )
        return StageResult(
            stage=stage,
            losses=losses,
            optimizer_steps=self.optimizer_steps - start_steps,
            gradient_norms=self.gradient_norms[start_norms:],
        )
