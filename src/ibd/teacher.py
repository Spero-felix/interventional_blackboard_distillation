"""Four-expert Teacher controller with deterministic call accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .backend import LLMBackend, LLMResult, StructuredCaller
from .config import AppConfig, ModelConfig
from .prompting import build_messages
from .schemas import (
    CallRecord,
    Candidate,
    CritiqueReport,
    EffectivenessCritiqueReport,
    EmotionCritiqueReport,
    ExpertName,
    ExpertOutput,
    FinalAnswer,
    History,
    NonSafetyDimension,
    PlanSelection,
    SafetyCritiqueReport,
    StateBlackboard,
    StateCounterfactual,
    StrictModel,
    StrategyName,
    StrategyPlanSet,
    TeacherTrace,
)

NORMAL_ROLES = (
    "emotion_expert",
    "need_expert",
    "relationship_expert",
    "intent_expert",
    "state_integrator",
    "planner",
    "candidate_1",
    "candidate_2",
    "candidate_3",
    "emotion_critic",
    "effectiveness_critic",
    "safety_critic",
    "final_integrator",
)


@dataclass(frozen=True)
class _PlainTextJSONBackend:
    backend: LLMBackend
    fixed_fields: dict[str, Any]

    def complete(
        self,
        *,
        role: str,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        json_mode: bool = True,
        seed: int | None = None,
    ) -> LLMResult:
        result = self.backend.complete(
            role=role,
            messages=messages,
            model_config=model_config,
            json_mode=json_mode,
            seed=seed,
        )
        try:
            payload = json.loads(result.text)
        except json.JSONDecodeError:
            # Preserve malformed JSON for StructuredCaller retries. Plain text
            # is the only provider output that this compatibility layer wraps.
            if result.text.lstrip().startswith(("{", "[")):
                return result
            payload = None
        if isinstance(payload, dict):
            return result
        response = payload if isinstance(payload, str) else result.text
        wrapped = {**self.fixed_fields, "response": response}
        return LLMResult(
            text=json.dumps(wrapped, ensure_ascii=False),
            usage=result.usage,
        )


@dataclass(frozen=True)
class DownstreamResult:
    final_answer: FinalAnswer
    plan: StrategyPlanSet
    candidates: list[Candidate]
    critiques: list[CritiqueReport]
    call_records: list[CallRecord]

    @property
    def response(self) -> str:
        return self.final_answer.response


@dataclass(frozen=True)
class SelectedDownstreamResult:
    selection: PlanSelection
    candidates: list[Candidate]
    critiques: list[CritiqueReport]
    final_answer: FinalAnswer
    call_records: list[CallRecord]


class TeacherRunner:
    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.config = config
        self.caller = StructuredCaller(backend, config)

    def _call(
        self,
        role: str,
        history: History,
        response_model: type[StrictModel],
        records: list[CallRecord],
        *,
        context: dict[str, Any] | None = None,
        seed: int | None = None,
        example_id: str | None = None,
    ):
        caller = self.caller
        if (
            not self.config.for_role(role).provider_json_mode
            and response_model is Candidate
        ):
            required = {"candidate_id", "strategy_id", "strategy", "seed"}
            if context is None or not required.issubset(context):
                raise ValueError(f"missing frozen candidate metadata for {role}")
            fixed_fields = {
                "candidate_id": context["candidate_id"],
                "strategy_id": context["strategy_id"],
                "strategy": context["strategy"],
                "seed": context["seed"],
            }
            caller = StructuredCaller(
                _PlainTextJSONBackend(self.caller.backend, fixed_fields),
                self.config,
            )
        elif response_model is FinalAnswer and context is not None:
            required_strategies = context.get("required_strategies")
            if required_strategies is None:
                plan = context.get("plan")
                required_strategies = [plan.strategy_for_id("S1")] if plan else []
            strategy_uses = [
                {
                    "strategy_id": f"S{index}",
                    "strategy": strategy,
                    "contribution": "Contributes to the final supporter response.",
                }
                for index, strategy in enumerate(required_strategies, start=1)
            ][:2]
            if strategy_uses:
                caller = StructuredCaller(
                    _PlainTextJSONBackend(
                        self.caller.backend,
                        {"strategy_uses": strategy_uses},
                    ),
                    self.config,
                )
        parsed, new_records = caller.call(
            role,
            build_messages(role, history, response_model, context=context),
            response_model,
            seed=seed,
            example_id=example_id,
        )
        records.extend(new_records)
        return parsed

    def run(
        self,
        example_id: str,
        history: History,
        *,
        split: Literal["train", "dev", "test", "diagnostic_holdout"] = "train",
    ) -> TeacherTrace:
        records: list[CallRecord] = []
        expert_outputs: dict[ExpertName, ExpertOutput] = {}
        for expert in ("emotion", "need", "relationship", "intent"):
            output = self._call(
                f"{expert}_expert", history, ExpertOutput, records, example_id=example_id
            )
            if output.expert != expert:
                raise ValueError(f"expert metadata mismatch for {expert}_expert")
            expert_outputs[expert] = output

        state = self._call(
            "state_integrator",
            history,
            StateBlackboard,
            records,
            context={"expert_outputs": expert_outputs},
            example_id=example_id,
        )
        downstream = self._run_from_state(history, state, records, example_id=example_id)

        return TeacherTrace(
            example_id=example_id,
            teacher_protocol_hash=self.caller.protocol_hash,
            split=split,
            history=history,
            expert_outputs=expert_outputs,
            state=state,
            plan=downstream.plan,
            candidates=downstream.candidates,
            critiques=downstream.critiques,
            final_answer=downstream.final_answer,
            final_response=downstream.response,
            call_records=records,
        )

    def generate_state_counterfactual(
        self,
        history: History,
        state: StateBlackboard,
        target_field: str,
        target_dimension: NonSafetyDimension,
        *,
        example_id: str,
    ) -> str:
        records: list[CallRecord] = []
        result = self._call(
            "state_counterfactual_generator",
            history,
            StateCounterfactual,
            records,
            context={
                "state": state,
                "target_field": target_field,
                "target_dimension": target_dimension,
                "original_value": getattr(state, target_field),
            },
            example_id=example_id,
        )
        return result.replacement

    def _run_from_state(
        self,
        history: History,
        state: StateBlackboard,
        records: list[CallRecord],
        *,
        example_id: str | None = None,
    ) -> DownstreamResult:
        plan = self._call(
            "planner",
            history,
            StrategyPlanSet,
            records,
            context={"state": state},
            example_id=example_id,
        )
        return self._run_from_plan(history, state, plan, records, example_id=example_id)

    def _run_from_plan(
        self,
        history: History,
        state: StateBlackboard,
        plan: StrategyPlanSet,
        records: list[CallRecord],
        *,
        example_id: str | None = None,
        frozen_candidates: list[Candidate] | None = None,
        regenerate_strategy_ids: set[str] | None = None,
    ) -> DownstreamResult:
        candidates = self._run_candidates(
            history=history,
            state=state,
            plan=plan,
            records=records,
            example_id=example_id,
            frozen_candidates=frozen_candidates,
            regenerate_strategy_ids=regenerate_strategy_ids,
        )
        critiques = self._run_critics(
            history=history,
            state=state,
            candidates=candidates,
            records=records,
            example_id=example_id,
        )
        final = self._run_final_integrator(
            history=history,
            state=state,
            candidates=candidates,
            critiques=critiques,
            records=records,
            example_id=example_id,
            plan=plan,
        )

        # strategy_id is the model's actual selection.
        # strategy name is redundant controller-owned metadata and should
        # be canonicalized from the planner.
        strategy_ids = [item.strategy_id for item in final.strategy_uses]

        if len(strategy_ids) != len(set(strategy_ids)):
            raise ValueError("final_integrator reused the same strategy_id")

        canonical_strategy_uses = [
            item.model_copy(
                update={
                    "strategy": plan.strategy_for_id(item.strategy_id),
                }
            )
            for item in final.strategy_uses
        ]

        # Reconstruct through Pydantic so all FinalAnswer validators run again.
        final = FinalAnswer(
            response=final.response,
            strategy_uses=canonical_strategy_uses,
        )

        # Keep this as a final sanity check.
        if any(
            plan.strategy_for_id(item.strategy_id) != item.strategy
            for item in final.strategy_uses
        ):
            raise ValueError("final_integrator used a strategy not produced by planner")
        return DownstreamResult(final, plan, candidates, critiques, records)

    def _run_candidates(
        self,
        *,
        history: History,
        state: StateBlackboard,
        plan: StrategyPlanSet,
        records: list[CallRecord],
        example_id: str | None,
        frozen_candidates: Sequence[Candidate] | None = None,
        regenerate_strategy_ids: set[str] | None = None,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        frozen_by_strategy = {
            candidate.strategy: candidate for candidate in (frozen_candidates or [])
        }
        for index, seed in enumerate(self.config.candidate_seeds, start=1):
            candidate_id = str(index)
            strategy_id = f"S{index}"
            strategy = plan.strategy_for_id(strategy_id)
            candidate = frozen_by_strategy.get(strategy)
            if candidate is None or (
                regenerate_strategy_ids is not None
                and strategy_id in regenerate_strategy_ids
            ):
                candidate = self._generate_candidate(
                    example_id=example_id,
                    history=history,
                    state=state,
                    strategy=strategy,
                    local_index=index,
                    records=records,
                )
            else:
                candidate = candidate.model_copy(
                    update={
                        "candidate_id": candidate_id,
                        "strategy_id": strategy_id,
                        "seed": seed,
                    }
                )
            if (
                candidate.candidate_id != candidate_id
                or candidate.seed != seed
                or candidate.strategy_id != strategy_id
                or candidate.strategy != strategy
            ):
                raise ValueError(f"candidate metadata mismatch for candidate_{index}")
            candidates.append(candidate)
        return candidates

    def _generate_candidate(
        self,
        *,
        example_id: str | None,
        history: History,
        state: StateBlackboard,
        strategy: StrategyName,
        local_index: int,
        records: list[CallRecord],
    ) -> Candidate:
        seed = self.config.candidate_seeds[local_index - 1]

        candidate = self._call(
            f"candidate_{local_index}",
            history,
            Candidate,
            records,
            context={
                "state": state,
                "candidate_id": str(local_index),
                "strategy_id": f"S{local_index}",
                "strategy": strategy,
                "seed": seed,
            },
            seed=seed,
            example_id=example_id,
        )

        # Candidate metadata is controller-owned, not model-generated.
        return candidate.model_copy(
            update={
                "candidate_id": str(local_index),
                "strategy_id": f"S{local_index}",
                "strategy": strategy,
                "seed": seed,
            }
        )

    def _run_critics(
        self,
        *,
        history: History,
        state: StateBlackboard,
        candidates: list[Candidate],
        records: list[CallRecord],
        example_id: str | None,
    ) -> list[CritiqueReport]:
        critiques: list[CritiqueReport] = []
        critic_models = (
            ("emotion_critic", EmotionCritiqueReport),
            ("effectiveness_critic", EffectivenessCritiqueReport),
            ("safety_critic", SafetyCritiqueReport),
        )
        for role, response_model in critic_models:
            critique = self._call(
                role,
                history,
                response_model,
                records,
                context={"state": state, "candidates": candidates},
                example_id=example_id,
            )
            if critique.critic != role.removesuffix("_critic"):
                raise ValueError(f"critic metadata mismatch for {role}")
            expected_candidate_ids = {candidate.candidate_id for candidate in candidates}
            if role == "safety_critic" and not critique.candidate_issues:
                critique = critique.model_copy(
                    update={
                        "candidate_issues": {
                            candidate_id: [] for candidate_id in expected_candidate_ids
                        }
                    }
                )
            if set(critique.candidate_issues) != expected_candidate_ids:
                raise ValueError(f"critic candidate coverage mismatch for {role}")
            critiques.append(critique)
        return critiques

    def _run_final_integrator(
        self,
        *,
        history: History,
        state: StateBlackboard,
        candidates: list[Candidate],
        critiques: list[CritiqueReport],
        records: list[CallRecord],
        example_id: str | None,
        plan: StrategyPlanSet | None = None,
        required_strategies: Sequence[StrategyName] | None = None,
    ) -> FinalAnswer:
        context: dict[str, Any] = {
            "state": state,
            "candidates": candidates,
            "critiques": critiques,
        }
        if plan is not None:
            context["plan"] = plan
        if required_strategies is not None:
            context["required_strategies"] = list(required_strategies)
        return self._call(
            "final_integrator",
            history,
            FinalAnswer,
            records,
            context=context,
            example_id=example_id,
        )

    def _reuse_or_generate_candidate(
        self,
        *,
        example_id: str,
        history: History,
        state: StateBlackboard,
        strategy: StrategyName,
        local_index: int,
        reusable: Candidate | None,
        records: list[CallRecord],
    ) -> Candidate:
        if reusable is None:
            return self._generate_candidate(
                example_id=example_id,
                history=history,
                state=state,
                strategy=strategy,
                local_index=local_index,
                records=records,
            )
        return reusable.model_copy(
            update={
                "candidate_id": str(local_index),
                "strategy_id": f"S{local_index}",
                "seed": self.config.candidate_seeds[local_index - 1],
            }
        )

    @staticmethod
    def _validate_selected_strategies(
        final_answer: FinalAnswer, selection: PlanSelection
    ) -> None:
        actual = {item.strategy for item in final_answer.strategy_uses}
        expected = set(selection.strategies)
        if (
            len(final_answer.strategy_uses) != len(selection.strategies)
            or actual != expected
        ):
            raise ValueError(
                "final_integrator did not use exactly the selected strategies"
            )

    def run_selected_strategies(
        self,
        *,
        example_id: str,
        history: History,
        state: StateBlackboard,
        selection: PlanSelection,
        reusable_candidates: Sequence[Candidate],
    ) -> SelectedDownstreamResult:
        records: list[CallRecord] = []
        reusable_by_strategy = {
            candidate.strategy: candidate for candidate in reusable_candidates
        }
        candidates = [
            self._reuse_or_generate_candidate(
                example_id=example_id,
                history=history,
                state=state,
                strategy=strategy,
                local_index=index,
                reusable=reusable_by_strategy.get(strategy),
                records=records,
            )
            for index, strategy in enumerate(selection.strategies, start=1)
        ]
        critiques = self._run_critics(
            history=history,
            state=state,
            candidates=candidates,
            records=records,
            example_id=example_id,
        )
        final_answer = self._run_final_integrator(
            history=history,
            state=state,
            candidates=candidates,
            critiques=critiques,
            records=records,
            example_id=example_id,
            required_strategies=selection.strategies,
        )
        self._validate_selected_strategies(final_answer, selection)
        return SelectedDownstreamResult(
            selection=selection,
            candidates=candidates,
            critiques=critiques,
            final_answer=final_answer,
            call_records=records,
        )

    def rerun_downstream(
        self,
        history: History,
        state: StateBlackboard,
        plan: StrategyPlanSet | None = None,
        *,
        example_id: str | None = None,
    ) -> DownstreamResult:
        """Rerun only nodes downstream of a clamped STATE or PLAN."""
        records: list[CallRecord] = []
        if plan is None:
            return self._run_from_state(history, state, records, example_id=example_id)
        return self._run_from_plan(history, state, plan, records, example_id=example_id)

    def rerun_plan_intervention(
        self,
        history: History,
        state: StateBlackboard,
        plan: StrategyPlanSet,
        original_candidates: list[Candidate],
        changed_strategy_id: str,
        *,
        example_id: str | None = None,
    ) -> DownstreamResult:
        """Regenerate one affected candidate, then rerun critics and final integration."""
        records: list[CallRecord] = []
        return self._run_from_plan(
            history,
            state,
            plan,
            records,
            example_id=example_id,
            frozen_candidates=original_candidates,
            regenerate_strategy_ids={changed_strategy_id},
        )
