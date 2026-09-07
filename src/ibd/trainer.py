"""Single-process Stage A/B optimization for the Qwen slot wrapper."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .anchors import AnchorArtifact
from .model import QwenSlotCausalLM, SlotForwardOutput
from .progress import track
from .training import (
    length_normalized_score,
    response_token_log_probs,
    slot_alignment_loss,
    stage_b_loss,
)


StageName = Literal["A", "B"]


@dataclass(frozen=True)
class StageResult:
    stage: StageName
    losses: list[float]
    optimizer_steps: int
    gradient_norms: list[float]


@dataclass(frozen=True)
class StageEvaluation:
    stage: StageName
    examples: int
    total_loss: float
    sft_loss: float
    replay_loss: float | None


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
            raise ValueError("Stage B requires a frozen anchor artifact")
        return self.anchors

    def _alignment_loss(
        self,
        output: SlotForwardOutput,
        example_ids: list[str],
        *,
        count: int | None = None,
    ) -> torch.Tensor:
        anchors = self._require_anchors()
        count = count or len(example_ids)
        rows = anchors.positive_rows(example_ids[:count]).to(output.logits.device)
        return slot_alignment_loss(
            state_slots=output.slots.STATE[:count],
            plan_slots=output.slots.PLAN[:count],
            state_bank=anchors.state.to(output.logits.device),
            plan_bank=anchors.plan.to(output.logits.device),
            positive_rows=rows,
            temperature=self.temperature,
            cosine_weight=self.cosine_weight,
            state_weight=self.state_weight,
            plan_weight=self.plan_weight,
        )

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

    def train_batch(self, stage: StageName, batch: Mapping[str, Any]) -> float:
        self.model.train()
        if stage == "B":
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

    def evaluate_stage(
        self,
        stage: StageName,
        batches: Iterable[Mapping[str, Any]],
        *,
        total: int | None = None,
    ) -> StageEvaluation:
        if stage not in {"A", "B"}:
            raise ValueError(f"unknown training stage {stage}")
        self.model.eval()
        examples = 0
        sft_total = 0.0
        replay_total = 0.0
        total_loss_total = 0.0
        progress = track(batches, desc=f"evaluate stage {stage}", total=total, unit="batch")
        with torch.inference_mode():
            for batch in progress:
                output = self._forward(batch)
                scores = self._scores(output, batch["labels"])
                count = scores.shape[0]
                sft_loss = -scores.mean()
                examples += count
                sft_total += float(sft_loss) * count
                if stage == "B":
                    replay_loss = self._replay_loss(output, scores, list(batch["example_ids"]))
                    replay_total += float(replay_loss) * count
                    total_loss_total += float(replay_loss) * count
        if examples == 0:
            raise ValueError(f"Stage {stage} dev dataset is empty")
        mean_sft = sft_total / examples
        mean_replay = replay_total / examples if stage == "B" else None
        return StageEvaluation(
            stage=stage,
            examples=examples,
            total_loss=total_loss_total / examples if stage == "B" else mean_sft,
            sft_loss=mean_sft,
            replay_loss=mean_replay,
        )


@dataclass(frozen=True)
class StandardSFTResult:
    losses: list[float]
    optimizer_steps: int
    gradient_norms: list[float]


@dataclass(frozen=True)
class StandardSFTEvaluation:
    examples: int
    sft_loss: float


class StandardSFTTrainer:
    """Response-only SFT optimizer for the base Qwen model without slots."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any = None,
        scaler: Any = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.micro_steps = 0
        self.optimizer_steps = 0
        self.gradient_norms: list[float] = []
        self.optimizer.zero_grad(set_to_none=True)

    def _forward(self, batch: Mapping[str, Any]):
        device = next(self.model.parameters()).device
        return self.model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
        )

    @staticmethod
    def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        token_scores, mask = response_token_log_probs(logits, labels.to(logits.device))
        return -length_normalized_score(token_scores, mask).mean()

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

    def train_epoch(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        total: int | None = None,
    ) -> StandardSFTResult:
        self.model.train()
        start_steps = self.optimizer_steps
        start_norms = len(self.gradient_norms)
        losses: list[float] = []
        progress = track(batches, desc="train standard SFT", total=total, unit="batch")
        for batch in progress:
            loss = self._loss(self._forward(batch).logits, batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss is NaN or Inf")
            scaled_loss = loss / self.gradient_accumulation_steps
            if self.scaler is None:
                scaled_loss.backward()
            else:
                self.scaler.scale(scaled_loss).backward()
            self.micro_steps += 1
            if self.micro_steps % self.gradient_accumulation_steps == 0:
                self._optimizer_step()
            losses.append(float(loss.detach()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}", step=self.optimizer_steps)
        if not losses:
            raise ValueError("standard SFT dataset is empty")
        remainder = self.micro_steps % self.gradient_accumulation_steps
        if remainder:
            self._optimizer_step(
                remainder_scale=self.gradient_accumulation_steps / remainder
            )
            self.micro_steps += self.gradient_accumulation_steps - remainder
        return StandardSFTResult(
            losses=losses,
            optimizer_steps=self.optimizer_steps - start_steps,
            gradient_norms=self.gradient_norms[start_norms:],
        )

    def evaluate(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        total: int | None = None,
    ) -> StandardSFTEvaluation:
        self.model.eval()
        examples = 0
        loss_total = 0.0
        progress = track(batches, desc="evaluate standard SFT", total=total, unit="batch")
        with torch.inference_mode():
            for batch in progress:
                loss = self._loss(self._forward(batch).logits, batch["labels"])
                count = batch["input_ids"].shape[0]
                examples += count
                loss_total += float(loss) * count
        if not examples:
            raise ValueError("standard SFT dev dataset is empty")
        return StandardSFTEvaluation(examples=examples, sft_loss=loss_total / examples)
