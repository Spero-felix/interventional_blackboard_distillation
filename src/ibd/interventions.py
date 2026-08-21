"""Minimal STATE/PLAN interventions and critique-grounded Stage D pairs."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from .backend import LLMBackend, StructuredCaller
from .config import ANCHOR_PROTOCOL_VERSION, STRATEGY_CATALOG, AppConfig
from .hashing import protocol_hash
from .prompting import build_messages
from .schemas import (
    Candidate,
    InterventionRecord,
    MarginPair,
    MASKED_STATE_VALUE,
    Mutation,
    NonSafetyDimension,
    PlanSelection,
    STATE_ANCHOR_FIELDS,
    StateBlackboard,
    StrictModel,
    StrategyPlanSet,
    TeacherTrace,
)
from .teacher import TeacherRunner

_NON_SAFETY_DIMENSIONS = {
    "emotion",
    "need",
    "relationship",
    "intent",
    "specificity",
    "timing",
    "effectiveness",
    "autonomy",
    "factuality",
    "non_template",
}


class PairVerdict(StrictModel):
    preferred: Literal["A", "B", "tie"]
    defect_dimension: NonSafetyDimension
    evidence: str = Field(min_length=1)


ExclusionReason = Literal[
    "state_not_single_field",
    "plan_cardinality",
    "plan_overlap",
    "bidirectional_disagreement",
    "safety_failure",
    "no_localized_effect",
]
EffectFailureReason = Literal[
    "bidirectional_disagreement",
    "no_localized_effect",
]
StateField = Literal[
    "emotion",
    "intensity",
    "primary_need",
    "support_goal",
    "readiness",
    "main_constraint",
    "relationship_context",
]


@dataclass(frozen=True)
class EffectVerification:
    passed: bool
    reason: EffectFailureReason | None = None

    def __post_init__(self) -> None:
        if self.passed and self.reason is not None:
            raise ValueError("successful verification must not have a reason")
        if not self.passed and self.reason is None:
            raise ValueError("failed verification must have a reason")


class InterventionExcluded(Exception):
    def __init__(self, reason: ExclusionReason):
        super().__init__(reason)
        self.reason = reason


def stable_seed(example_id: str, global_seed: int, protocol_version: str) -> int:
    return int(
        protocol_hash(
            {
                "example_id": example_id,
                "global_seed": global_seed,
                "protocol_version": protocol_version,
            }
        ),
        16,
    )


def mask_state_field(
    state: StateBlackboard,
    field: StateField,
) -> tuple[StateBlackboard, Mutation]:
    before = getattr(state, field)
    mutated = state.model_copy(update={field: MASKED_STATE_VALUE})
    return mutated, Mutation(
        operation="mask_state_field",
        field=field,
        before=before,
        after=MASKED_STATE_VALUE,
    )


def select_counterfactual_plan(
    *,
    planned: StrategyPlanSet,
    used: PlanSelection,
    example_id: str,
    global_seed: int,
) -> tuple[PlanSelection, Mutation]:
    k = len(used.strategies)
    selected = [
        strategy
        for strategy in planned.strategies
        if strategy not in used.strategies
    ][:k]
    if len(selected) < k:
        fallback = [
            strategy
            for strategy in STRATEGY_CATALOG
            if strategy not in planned.strategies
        ]
        rng = random.Random(
            stable_seed(example_id, global_seed, ANCHOR_PROTOCOL_VERSION)
        )
        rng.shuffle(fallback)
        selected.extend(fallback[: k - len(selected)])
    counterfactual = PlanSelection(strategies=selected)
    return counterfactual, Mutation(
        operation="replace_plan_categories",
        field="strategies",
        before=list(used.strategies),
        after=list(counterfactual.strategies),
    )


class InterventionBuilder:
    """Build a retained trajectory or raise a stable exclusion reason."""

    def __init__(
        self,
        runner: TeacherRunner,
        verify_effect: Callable[[str, str, str], EffectVerification],
        verify_safety: Callable[[str, str], bool],
    ):
        self.runner = runner
        self.verify_effect = verify_effect
        self.verify_safety = verify_safety

    def build(
        self,
        trace: TeacherTrace,
        function: Literal["STATE", "PLAN"],
        *,
        global_seed: int,
        state_field: StateField | None = None,
    ) -> InterventionRecord:
        mutated_state: StateBlackboard | None = None
        mutated_plan: PlanSelection | None = None
        if function == "STATE":
            if state_field not in STATE_ANCHOR_FIELDS:
                raise InterventionExcluded("state_not_single_field")
            if any(
                getattr(trace.state, field) == MASKED_STATE_VALUE
                for field in STATE_ANCHOR_FIELDS
            ):
                raise InterventionExcluded("state_not_single_field")
            state, mutation = mask_state_field(trace.state, state_field)
            changed = {
                key
                for key, value in trace.state.model_dump(mode="json").items()
                if state.model_dump(mode="json")[key] != value
            }
            if changed != {state_field}:
                raise InterventionExcluded("state_not_single_field")
            mutated_state = state
            downstream = self.runner.rerun_downstream(
                trace.history,
                state,
                example_id=f"{trace.example_id}:intervention:STATE",
            )
            dimension: NonSafetyDimension = {
                "emotion": "emotion",
                "intensity": "emotion",
                "primary_need": "need",
                "support_goal": "intent",
                "readiness": "timing",
                "main_constraint": "effectiveness",
                "relationship_context": "relationship",
            }[state_field]
        else:
            used = PlanSelection.from_final_answer(trace.final_answer)
            selection, mutation = select_counterfactual_plan(
                planned=trace.plan,
                used=used,
                example_id=trace.example_id,
                global_seed=global_seed,
            )
            if len(selection.strategies) != len(used.strategies):
                raise InterventionExcluded("plan_cardinality")
            if set(selection.strategies) & set(used.strategies):
                raise InterventionExcluded("plan_overlap")
            mutated_plan = selection
            downstream = self.runner.run_selected_strategies(
                example_id=f"{trace.example_id}:intervention:PLAN",
                history=trace.history,
                state=trace.state,
                selection=selection,
                reusable_candidates=trace.candidates,
            )
            dimension = "timing"
        counterfactual_response = downstream.final_answer.response
        if not self.verify_safety(trace.final_response, counterfactual_response):
            raise InterventionExcluded("safety_failure")
        safety_reports = [
            report
            for report in (*trace.critiques, *downstream.critiques)
            if report.critic == "safety"
        ]
        if any(
            issues
            for report in safety_reports
            for issues in report.candidate_issues.values()
        ):
            raise InterventionExcluded("safety_failure")
        verification = self.verify_effect(
            trace.final_response, counterfactual_response, dimension
        )
        if not verification.passed:
            assert verification.reason is not None
            raise InterventionExcluded(verification.reason)
        return InterventionRecord(
            example_id=trace.example_id,
            function=function,
            mutation=mutation,
            mutated_state=mutated_state,
            mutated_plan=mutated_plan,
            full_response=trace.final_response,
            counterfactual_response=counterfactual_response,
            target_dimension=dimension,
            localized_degradation=True,
            bidirectional_verified=True,
        )


class MarginPairBuilder:
    """Choose a localized non-safety negative and verify preference in A/B and B/A order."""

    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.caller = StructuredCaller(backend, config)

    @staticmethod
    def _candidate_by_id(trace: TeacherTrace, candidate_id: str) -> Candidate | None:
        return next((item for item in trace.candidates if item.candidate_id == candidate_id), None)

    def _select_negative(self, trace: TeacherTrace):
        unsafe_ids: set[str] = set()
        scored: list[tuple[int, str, str, str]] = []
        for report in trace.critiques:
            for candidate_id, issues in report.candidate_issues.items():
                if report.critic == "safety" and issues:
                    unsafe_ids.add(candidate_id)
                    continue
                if report.critic == "safety":
                    continue
                for issue in issues:
                    if issue.dimension in _NON_SAFETY_DIMENSIONS:
                        scored.append(
                            (issue.severity, candidate_id, issue.dimension, issue.evidence)
                        )
        for _, candidate_id, dimension, evidence in sorted(scored, reverse=True):
            candidate = self._candidate_by_id(trace, candidate_id)
            if candidate is not None and candidate_id not in unsafe_ids:
                return candidate, dimension, evidence
        return None

    def _verify_order_swap(
        self,
        trace: TeacherTrace,
        rejected: Candidate,
        dimension: str,
    ) -> tuple[bool, PairVerdict, PairVerdict]:
        first, _ = self.caller.call(
            "pair_verifier",
            build_messages(
                "pair_verifier",
                trace.history,
                PairVerdict,
                context={"A": trace.final_response, "B": rejected.response, "dimension": dimension},
            ),
            PairVerdict,
            example_id=f"{trace.example_id}:margin:ab",
        )
        second, _ = self.caller.call(
            "pair_verifier",
            build_messages(
                "pair_verifier",
                trace.history,
                PairVerdict,
                context={"A": rejected.response, "B": trace.final_response, "dimension": dimension},
            ),
            PairVerdict,
            example_id=f"{trace.example_id}:margin:ba",
        )
        valid = (
            first.preferred == "A"
            and second.preferred == "B"
            and first.defect_dimension == second.defect_dimension == dimension
        )
        return valid, first, second

    def build(self, trace: TeacherTrace) -> MarginPair | None:
        selected = self._select_negative(trace)
        if selected is None:
            return None
        rejected, dimension, evidence = selected
        verified, first, _ = self._verify_order_swap(trace, rejected, dimension)
        if not verified:
            return None
        return MarginPair(
            example_id=trace.example_id,
            prompt=trace.history.as_prompt(),
            chosen=trace.final_response,
            chosen_strategy_uses=trace.final_answer.strategy_uses,
            rejected_candidate_id=rejected.candidate_id,
            rejected_strategy_id=rejected.strategy_id,
            rejected_strategy=rejected.strategy,
            rejected=rejected.response,
            defect_dimension=first.defect_dimension,
            defect_evidence=evidence,
            order_swap_verified=True,
            safety_filter_passed=True,
        )
