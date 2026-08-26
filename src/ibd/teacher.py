"""Unified multi-view Teacher with deterministic single-candidate selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .backend import LLMBackend, LLMResult, StructuredCaller
from .config import AppConfig, ModelConfig
from .prompting import build_messages
from .schemas import (
    CallRecord,
    Candidate,
    FinalSelection,
    FinalSelectionDecision,
    History,
    MultiViewStateAnalysis,
    NonSafetyDimension,
    StateBlackboard,
    StateCounterfactual,
    StrictModel,
    StrategyName,
    StrategyPlanSet,
    TeacherTrace,
)

NORMAL_ROLES = (
    "multi_view_state_analyzer",
    "planner",
    "candidate_1",
    "candidate_2",
    "candidate_3",
    "final_selector",
)


@dataclass(frozen=True)
class _PlainTextCandidateBackend:
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
            payload = None
        if isinstance(payload, dict):
            return result
        if result.text.lstrip().startswith(("{", "[")):
            return result
        response = payload if isinstance(payload, str) else result.text
        return LLMResult(
            text=json.dumps(
                {
                    **self.fixed_fields,
                    "response": response,
                    "response_goal": "Support the seeker's immediate goal",
                    "response_act": "Apply the assigned strategy",
                },
                ensure_ascii=False,
            ),
            usage=result.usage,
        )


@dataclass(frozen=True)
class DownstreamResult:
    final_selection: FinalSelection
    plan: StrategyPlanSet
    candidates: list[Candidate]
    call_records: list[CallRecord]

    @property
    def response(self) -> str:
        return self.final_selection.response


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
        if not self.config.for_role(role).provider_json_mode and response_model is Candidate:
            required = {"candidate_id", "strategy_id", "strategy", "seed"}
            if context is None or not required.issubset(context):
                raise ValueError(f"missing frozen candidate metadata for {role}")
            caller = StructuredCaller(
                _PlainTextCandidateBackend(
                    self.caller.backend,
                    {field: context[field] for field in required},
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
        analysis = self._call(
            "multi_view_state_analyzer",
            history,
            MultiViewStateAnalysis,
            records,
            example_id=example_id,
        )
        downstream = self._run_from_state(
            history, analysis.state, records, example_id=example_id
        )
        return TeacherTrace(
            example_id=example_id,
            split=split,
            history=history,
            state_analysis=analysis,
            state=analysis.state,
            plan=downstream.plan,
            candidates=downstream.candidates,
            final_selection=downstream.final_selection,
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
        clamped_state_field: str | None = None,
    ) -> DownstreamResult:
        context: dict[str, Any] = {"state": state}
        if clamped_state_field is not None:
            context["clamped_state_field"] = clamped_state_field
        plan = self._call(
            "planner",
            history,
            StrategyPlanSet,
            records,
            context=context,
            example_id=example_id,
        )
        candidates = [
            self._generate_candidate(
                example_id=example_id,
                history=history,
                state=state,
                strategy=plan.strategy_for_id(f"S{index}"),
                local_index=index,
                records=records,
                clamped_state_field=clamped_state_field,
            )
            for index in range(1, 4)
        ]
        final_selection = self._run_final_selector(
            history=history,
            state=state,
            candidates=candidates,
            records=records,
            example_id=example_id,
            clamped_state_field=clamped_state_field,
        )
        return DownstreamResult(final_selection, plan, candidates, records)

    def _generate_candidate(
        self,
        *,
        example_id: str | None,
        history: History,
        state: StateBlackboard,
        strategy: StrategyName,
        local_index: int,
        records: list[CallRecord],
        clamped_state_field: str | None = None,
    ) -> Candidate:
        seed = self.config.candidate_seeds[local_index - 1]
        context: dict[str, Any] = {
            "state": state,
            "candidate_id": str(local_index),
            "strategy_id": f"S{local_index}",
            "strategy": strategy,
            "seed": seed,
        }
        if clamped_state_field is not None:
            context["clamped_state_field"] = clamped_state_field
        generated = self._call(
            f"candidate_{local_index}",
            history,
            Candidate,
            records,
            context=context,
            seed=seed,
            example_id=example_id,
        )
        return generated.model_copy(
            update={
                "candidate_id": str(local_index),
                "strategy_id": f"S{local_index}",
                "strategy": strategy,
                "seed": seed,
            }
        )

    def _run_final_selector(
        self,
        *,
        history: History,
        state: StateBlackboard,
        candidates: list[Candidate],
        records: list[CallRecord],
        example_id: str | None,
        clamped_state_field: str | None = None,
    ) -> FinalSelection:
        context: dict[str, Any] = {"state": state, "candidates": candidates}
        if clamped_state_field is not None:
            context["clamped_state_field"] = clamped_state_field
        decision = self._call(
            "final_selector",
            history,
            FinalSelectionDecision,
            records,
            context=context,
            example_id=example_id,
        )
        selected = [
            candidate
            for candidate in candidates
            if candidate.candidate_id == decision.selected_candidate_id
        ]
        if len(selected) != 1:
            raise ValueError("final_selector must identify exactly one supplied candidate")
        return FinalSelection.from_candidate(selected[0], decision)

    def rerun_downstream(
        self,
        history: History,
        state: StateBlackboard,
        *,
        example_id: str | None = None,
        clamped_state_field: str | None = None,
    ) -> DownstreamResult:
        records: list[CallRecord] = []
        return self._run_from_state(
            history,
            state,
            records,
            example_id=example_id,
            clamped_state_field=clamped_state_field,
        )
