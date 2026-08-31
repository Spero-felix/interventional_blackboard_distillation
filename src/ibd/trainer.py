"""Single-process Stage A-C optimization for the Qwen slot wrapper."""

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


def _stage_c_natural_sft_scores(
    original_scores: torch.Tensor,
    counterfactual_scores: torch.Tensor,
) -> torch.Tensor:
    if original_scores.shape != counterfactual_scores.shape:
        raise ValueError("Stage C response score shapes must match")
    # The natural, unclamped path always learns the factual Teacher answer.
    # Counterfactual responses are valid conditional targets only under their clamp.
    return original_scores


StageName = Literal["A", "B", "B2", "C"]


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
    natural_ce: float | None = None
    alignment_loss: float | None = None
    original_conditioned_ce: float | None = None
    counterfactual_conditioned_ce: float | None = None
    conditioned_ce: float | None = None


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
        b2_natural_weight: float = 1.0,
        b2_conditioned_weight: float = 1.0,
        b2_alignment_weight: float = 0.1,
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
        self.b2_natural_weight = b2_natural_weight
        self.b2_conditioned_weight = b2_conditioned_weight
        self.b2_alignment_weight = b2_alignment_weight
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
            raise ValueError("Stage B/B2/C require a frozen anchor artifact")
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
        original: Mapping[str, Any],
        counterfactual: Mapping[str, Any],
    ) -> tuple[dict[str, Any], int]:
        original_count = original["input_ids"].shape[0]
        if counterfactual["input_ids"].shape[0] != original_count:
            raise ValueError("response batches must have equal batch size")
        target_length = max(
            original["input_ids"].shape[1], counterfactual["input_ids"].shape[1]
        )
        pad_id = int(getattr(self.model.base_model.config, "pad_token_id", 0) or 0)
        merged = {
            "input_ids": torch.cat(
                (
                    self._pad_pair_tensor(original["input_ids"], target_length=target_length, fill=pad_id),
                    self._pad_pair_tensor(counterfactual["input_ids"], target_length=target_length, fill=pad_id),
                )
            ),
            "attention_mask": torch.cat(
                (
                    self._pad_pair_tensor(original["attention_mask"], target_length=target_length, fill=0),
                    self._pad_pair_tensor(counterfactual["attention_mask"], target_length=target_length, fill=0),
                )
            ),
            "labels": torch.cat(
                (
                    self._pad_pair_tensor(original["labels"], target_length=target_length, fill=-100),
                    self._pad_pair_tensor(counterfactual["labels"], target_length=target_length, fill=-100),
                )
            ),
            "slot_positions": {
                name: torch.cat(
                    (original["slot_positions"][name], counterfactual["slot_positions"][name])
                )
                for name in ("STATE", "PLAN")
            },
            "example_ids": original["example_ids"] + counterfactual["example_ids"],
        }
        return merged, original_count

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

    def _conditioned_clamps(
        self,
        function: str,
        example_ids: list[str],
        *,
        counterfactual: bool,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if function not in {"STATE", "PLAN"}:
            raise ValueError("intervention function must be STATE or PLAN")
        return {
            **self._intervention_clamp(
                "STATE",
                example_ids,
                mutated=counterfactual and function == "STATE",
                repeats=1,
                device=device,
            ),
            **self._intervention_clamp(
                "PLAN",
                example_ids,
                mutated=counterfactual and function == "PLAN",
                repeats=1,
                device=device,
            ),
        }

    def train_batch(self, stage: StageName, batch: Mapping[str, Any]) -> float:
        self.model.train()
        if stage in {"B", "B2", "C"}:
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
        if stage == "B2":
            functions = list(batch["functions"])
            if len(set(functions)) != 1:
                raise ValueError("Stage B2 microbatch must contain one intervention function")
            original = batch["original"]
            counterfactual = batch["counterfactual"]
            example_ids = list(batch["example_ids"])
            count = original["input_ids"].shape[0]
            if len(functions) != count or len(example_ids) != count:
                raise ValueError(
                    "Stage B2 functions and example IDs must match the microbatch size"
                )
            if (
                list(original["example_ids"]) != example_ids
                or list(counterfactual["example_ids"]) != example_ids
            ):
                raise ValueError(
                    "Stage B2 original, counterfactual, and intervention example IDs must match"
                )

            normal = self._forward(original)
            natural_ce = -self._scores(normal, original["labels"]).mean()
            alignment = self._alignment_loss(normal, example_ids)
            natural_component = (
                self.b2_natural_weight * natural_ce
                + self.b2_alignment_weight * alignment
            )
            natural_value = natural_component.detach()
            self._backward(natural_component, finish_microstep=False)
            del normal, natural_ce, alignment, natural_component

            device = next(self.model.parameters()).device
            original_conditioned = self._forward(
                original,
                clamps=self._conditioned_clamps(
                    functions[0],
                    example_ids,
                    counterfactual=False,
                    device=device,
                ),
            )
            original_ce = -self._scores(
                original_conditioned, original["labels"]
            ).mean()
            original_component = 0.5 * self.b2_conditioned_weight * original_ce
            original_value = original_component.detach()
            self._backward(original_component, finish_microstep=False)
            del original_conditioned, original_ce, original_component

            counterfactual_conditioned = self._forward(
                counterfactual,
                clamps=self._conditioned_clamps(
                    functions[0],
                    example_ids,
                    counterfactual=True,
                    device=device,
                ),
            )
            counterfactual_ce = -self._scores(
                counterfactual_conditioned, counterfactual["labels"]
            ).mean()
            counterfactual_component = (
                0.5 * self.b2_conditioned_weight * counterfactual_ce
            )
            counterfactual_value = counterfactual_component.detach()
            self._backward(counterfactual_component, finish_microstep=True)
            return float(natural_value + original_value + counterfactual_value)
        if stage == "C":
            functions = list(batch["functions"])
            if len(set(functions)) != 1:
                raise ValueError("Stage C microbatch must contain one intervention function")
            original = batch["original"]
            counterfactual = batch["counterfactual"]
            merged, original_count = self._merge_pair(original, counterfactual)
            example_ids = list(batch["example_ids"])
            if len(functions) != original_count or len(example_ids) != original_count:
                raise ValueError(
                    "Stage C functions and example IDs must match the microbatch size"
                )
            if (
                list(original["example_ids"]) != example_ids
                or list(counterfactual["example_ids"]) != example_ids
            ):
                raise ValueError(
                    "Stage C original, counterfactual, and intervention example IDs must match"
                )
            normal = self._forward(merged)
            normal_scores = self._scores(normal, merged["labels"])
            normal_original = _stage_c_natural_sft_scores(
                normal_scores[:original_count],
                normal_scores[original_count:],
            )
            replay = self._replay_loss(
                normal,
                normal_scores,
                original["example_ids"],
                count=original_count,
            )
            normal_loss = -normal_original.mean() + self.replay_weight * replay
            normal_loss_value = normal_loss.detach()
            self._backward(normal_loss, finish_microstep=False)
            del normal, normal_scores, normal_original, replay, normal_loss

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
            original_preference = torch.relu(
                self.margin
                - (
                    original_scores[:original_count]
                    - original_scores[original_count:]
                )
            ).mean()
            original_preference_value = original_preference.detach()
            self._backward(original_preference, finish_microstep=False)
            del original_clamp, original_scores, original_preference

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
            counterfactual_preference = torch.relu(
                self.margin
                - (
                    counterfactual_scores[original_count:]
                    - counterfactual_scores[:original_count]
                )
            ).mean()
            counterfactual_preference_value = counterfactual_preference.detach()
            self._backward(counterfactual_preference, finish_microstep=True)
            loss = (
                normal_loss_value
                + original_preference_value
                + counterfactual_preference_value
            )
            return float(loss)
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
        if stage not in {"A", "B", "B2", "C"}:
            raise ValueError(f"unknown training stage {stage}")
        self.model.eval()
        examples = 0
        sft_total = 0.0
        replay_total = 0.0
        total_loss_total = 0.0
        alignment_total = 0.0
        original_conditioned_total = 0.0
        counterfactual_conditioned_total = 0.0
        progress = track(
            batches,
            desc=f"evaluate stage {stage}",
            total=total,
            unit="batch",
        )
        with torch.inference_mode():
            for batch in progress:
                if stage == "B2":
                    functions = list(batch["functions"])
                    if len(set(functions)) != 1:
                        raise ValueError("Stage B2 microbatch must contain one intervention function")
                    original = batch["original"]
                    counterfactual = batch["counterfactual"]
                    example_ids = list(batch["example_ids"])
                    count = original["input_ids"].shape[0]
                    if len(functions) != count or len(example_ids) != count:
                        raise ValueError(
                            "Stage B2 functions and example IDs must match the microbatch size"
                        )
                    if (
                        list(original["example_ids"]) != example_ids
                        or list(counterfactual["example_ids"]) != example_ids
                    ):
                        raise ValueError(
                            "Stage B2 original, counterfactual, and intervention example IDs must match"
                        )
                    normal = self._forward(original)
                    natural_ce = -self._scores(normal, original["labels"]).mean()
                    alignment = self._alignment_loss(normal, example_ids)
                    device = next(self.model.parameters()).device
                    original_conditioned = self._forward(
                        original,
                        clamps=self._conditioned_clamps(
                            functions[0],
                            example_ids,
                            counterfactual=False,
                            device=device,
                        ),
                    )
                    original_ce = -self._scores(
                        original_conditioned, original["labels"]
                    ).mean()
                    counterfactual_conditioned = self._forward(
                        counterfactual,
                        clamps=self._conditioned_clamps(
                            functions[0],
                            example_ids,
                            counterfactual=True,
                            device=device,
                        ),
                    )
                    counterfactual_ce = -self._scores(
                        counterfactual_conditioned, counterfactual["labels"]
                    ).mean()
                    conditioned_ce = 0.5 * (original_ce + counterfactual_ce)
                    total_loss = (
                        self.b2_natural_weight * natural_ce
                        + self.b2_alignment_weight * alignment
                        + self.b2_conditioned_weight * conditioned_ce
                    )
                    examples += count
                    sft_total += float(natural_ce) * count
                    alignment_total += float(alignment) * count
                    original_conditioned_total += float(original_ce) * count
                    counterfactual_conditioned_total += float(counterfactual_ce) * count
                    total_loss_total += float(total_loss) * count
                    continue
                if stage == "C":
                    functions = list(batch["functions"])
                    if len(set(functions)) != 1:
                        raise ValueError("Stage C microbatch must contain one intervention function")
                    original = batch["original"]
                    counterfactual = batch["counterfactual"]
                    merged, original_count = self._merge_pair(original, counterfactual)
                    example_ids = list(batch["example_ids"])
                    if len(functions) != original_count or len(example_ids) != original_count:
                        raise ValueError(
                            "Stage C functions and example IDs must match the microbatch size"
                        )
                    if (
                        list(original["example_ids"]) != example_ids
                        or list(counterfactual["example_ids"]) != example_ids
                    ):
                        raise ValueError(
                            "Stage C original, counterfactual, and intervention example IDs must match"
                        )
                    normal = self._forward(merged)
                    normal_scores = self._scores(normal, merged["labels"])
                    normal_sft_loss = -_stage_c_natural_sft_scores(
                        normal_scores[:original_count],
                        normal_scores[original_count:],
                    ).mean()
                    replay_loss = self._replay_loss(
                        normal,
                        normal_scores,
                        example_ids,
                        count=original_count,
                    )
                    device = next(self.model.parameters()).device
                    original_clamp = self._forward(
                        merged,
                        clamps=self._intervention_clamp(
                            functions[0],
                            example_ids,
                            mutated=False,
                            repeats=2,
                            device=device,
                        ),
                    )
                    original_scores = self._scores(original_clamp, merged["labels"])
                    original_preference = torch.relu(
                        self.margin
                        - (
                            original_scores[:original_count]
                            - original_scores[original_count:]
                        )
                    ).mean()
                    counterfactual_clamp = self._forward(
                        merged,
                        clamps=self._intervention_clamp(
                            functions[0],
                            example_ids,
                            mutated=True,
                            repeats=2,
                            device=device,
                        ),
                    )
                    counterfactual_scores = self._scores(
                        counterfactual_clamp,
                        merged["labels"],
                    )
                    counterfactual_preference = torch.relu(
                        self.margin
                        - (
                            counterfactual_scores[original_count:]
                            - counterfactual_scores[:original_count]
                        )
                    ).mean()
                    total_loss = (
                        normal_sft_loss
                        + self.replay_weight * replay_loss
                        + original_preference
                        + counterfactual_preference
                    )
                    examples += original_count
                    sft_total += float(normal_sft_loss) * original_count
                    replay_total += float(replay_loss) * original_count
                    total_loss_total += float(total_loss) * original_count
                    continue
                output = self._forward(batch)
                scores = self._scores(output, batch["labels"])
                count = scores.shape[0]
                sft_loss = -scores.mean()
                examples += count
                sft_total += float(sft_loss) * count
                if stage == "B":
                    replay_loss = self._replay_loss(
                        output,
                        scores,
                        list(batch["example_ids"]),
                    )
                    replay_total += float(replay_loss) * count
                    total_loss_total += float(replay_loss) * count
        if examples == 0:
            raise ValueError(f"Stage {stage} dev dataset is empty")
        mean_sft = sft_total / examples
        mean_replay = replay_total / examples if stage in {"B", "C"} else None
        mean_total = (
            total_loss_total / examples
            if stage in {"B", "B2", "C"}
            else mean_sft
        )
        mean_alignment = alignment_total / examples if stage == "B2" else None
        mean_original_conditioned = (
            original_conditioned_total / examples if stage == "B2" else None
        )
        mean_counterfactual_conditioned = (
            counterfactual_conditioned_total / examples if stage == "B2" else None
        )
        return StageEvaluation(
            stage=stage,
            examples=examples,
            total_loss=mean_total,
            sft_loss=mean_sft,
            replay_loss=mean_replay,
            natural_ce=mean_sft if stage == "B2" else None,
            alignment_loss=mean_alignment,
            original_conditioned_ce=mean_original_conditioned,
            counterfactual_conditioned_ce=mean_counterfactual_conditioned,
            conditioned_ce=(
                0.5 * (mean_original_conditioned + mean_counterfactual_conditioned)
                if stage == "B2"
                else None
            ),
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
