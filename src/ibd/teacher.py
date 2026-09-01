"""Unified multi-view Teacher with deterministic single-candidate selection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import create_model

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
    PlanSelection,
    StateField,
    StateBlackboard,
    StrictModel,
    StrategyName,
    StrategyPlanSet,
    TeacherTrace,
)
from .state_guides import counterfactual_context

NORMAL_ROLES = (
    "multi_view_state_analyzer",
    "planner",
    "candidate",
    "candidate",
    "candidate",
    "final_selector",
)

_JSON_FENCE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    re.IGNORECASE | re.DOTALL,
)


def _single_fenced_payload(text: str) -> object | None:
    matches = _JSON_FENCE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return json.loads(matches[0])
    except json.JSONDecodeError:
        return None


@dataclass
class _PlainTextCandidateBackend:
    backend: LLMBackend
    fixed_fields: dict[str, Any]
    plain_text_attempts: int = 0
    fallback_used: bool = False

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
        fenced_payload = _single_fenced_payload(result.text)
        if fenced_payload is not None:
            payload = fenced_payload
            result = LLMResult(
                text=json.dumps(payload, ensure_ascii=False),
                usage=result.usage,
            )
        if isinstance(payload, dict):
            return result
        if "```" in result.text or result.text.lstrip().startswith(("{", "[")):
            return result
        self.plain_text_attempts += 1
        if self.plain_text_attempts == 1:
            return result
        response = payload if isinstance(payload, str) else result.text
        self.fallback_used = True
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
        cache_variant: str | None = None,
    ):
        caller = self.caller
        if not self.config.for_role(role).provider_json_mode and response_model is Candidate:
            required = {"candidate_id", "strategy_id", "strategy"}
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
            cache_variant=cache_variant,
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
        target_field: StateField,
        *,
        example_id: str,
    ) -> str:
        records: list[CallRecord] = []
        target_context = counterfactual_context(
            target_field, getattr(state, target_field)
        )
        allowed_replacements = tuple(target_context["allowed_replacements"])
        replacement_type = Literal.__getitem__(allowed_replacements)
        response_model = create_model(
            f"{target_field.title().replace('_', '')}Counterfactual",
            __base__=StrictModel,
            replacement=(replacement_type, ...),
        )
        result = self._call(
            "state_counterfactual_generator",
            history,
            response_model,
            records,
            context={
                "state": state,
                **target_context,
            },
            example_id=example_id,
        )
        replacement = result.replacement
        if replacement not in target_context["allowed_replacements"]:
            raise ValueError(
                f"STATE counterfactual replacement {replacement!r} is not allowed "
                f"for {target_field}"
            )
        return replacement

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
                strategy=strategy,
                local_index=index,
                records=records,
                clamped_state_field=clamped_state_field,
            )
            for index, strategy in enumerate(plan.strategies, start=1)
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
        fixed_plan: PlanSelection | None = None,
    ) -> Candidate:
        strategy_id = f"S{local_index}"
        context: dict[str, Any] = {
            "state": state,
            "candidate_id": str(local_index),
            "strategy_id": strategy_id,
            "strategy": strategy,
        }
        if clamped_state_field is not None:
            context["clamped_state_field"] = clamped_state_field
        if fixed_plan is not None:
            if fixed_plan.strategies[0] != strategy:
                raise ValueError("fixed plan strategy must match assigned candidate strategy")
            context["fixed_plan"] = fixed_plan
        generated = self._call(
            "candidate",
            history,
            Candidate,
            records,
            context=context,
            example_id=example_id,
            cache_variant=strategy_id,
        )
        return generated.model_copy(
            update={
                "candidate_id": str(local_index),
                "strategy_id": strategy_id,
                "strategy": strategy,
                "seed": None,
            }
        )

    def _selector_candidates(
        self,
        candidates: list[Candidate],
        example_id: str | None,
    ) -> list[Candidate]:
        identity = example_id or json.dumps(
            [candidate.model_dump(mode="json") for candidate in candidates],
            ensure_ascii=False,
            sort_keys=True,
        )
        return sorted(
            candidates,
            key=lambda candidate: hashlib.sha256(
                (
                    f"{self.config.protocol_version}\0{identity}\0"
                    f"{candidate.candidate_id}"
                ).encode("utf-8")
            ).digest(),
        )

    def generate_response_under_fixed_plan(
        self,
        history: History,
        state: StateBlackboard,
        plan: PlanSelection,
        *,
        clamped_state_field: str,
        example_id: str,
    ) -> str:
        """Generate one response after changing STATE while holding PLAN fixed."""

        records: list[CallRecord] = []
        candidate = self._generate_candidate(
            example_id=example_id,
            history=history,
            state=state,
            strategy=plan.strategies[0],
            local_index=1,
            records=records,
            clamped_state_field=clamped_state_field,
            fixed_plan=plan,
        )
        return candidate.response

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
        context: dict[str, Any] = {
            "state": state,
            "candidates": self._selector_candidates(candidates, example_id),
        }
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
