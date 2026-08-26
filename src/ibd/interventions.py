"""STATE/PLAN counterfactual construction for Stage C controllability."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from .schemas import (
    Candidate,
    InterventionRecord,
    MASKED_STATE_VALUE,
    Mutation,
    NonSafetyDimension,
    PlanSelection,
    STATE_ANCHOR_FIELDS,
    StateBlackboard,
    StrictModel,
    TeacherTrace,
)
from .teacher import TeacherRunner


class ConditionalEffectVerdict(StrictModel):
    condition_a_fit: bool
    condition_b_fit: bool
    control_effect_present: bool
    evidence: str = Field(min_length=1)


class StateEffectVerdict(StrictModel):
    condition_a_fit: bool
    condition_b_fit: bool
    field_effect_present: bool
    affected_dimensions: list[NonSafetyDimension] = Field(default_factory=list)
    evidence: str = Field(min_length=1)


ExclusionReason = Literal[
    "state_not_single_field",
    "invalid_state_counterfactual",
    "plan_overlap",
    "bidirectional_disagreement",
    "safety_failure",
    "no_localized_effect",
]
EffectFailureReason = Literal["bidirectional_disagreement", "no_localized_effect"]
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
    affected_dimensions: tuple[NonSafetyDimension, ...] = ()

    def __post_init__(self) -> None:
        if self.passed and self.reason is not None:
            raise ValueError("successful verification must not have a reason")
        if not self.passed and self.reason is None:
            raise ValueError("failed verification must have a reason")
        if not self.passed and self.affected_dimensions:
            raise ValueError("failed verification cannot carry effect metadata")


class InterventionExcluded(Exception):
    def __init__(self, reason: ExclusionReason):
        super().__init__(reason)
        self.reason = reason


def mask_state_field(
    state: StateBlackboard, field: StateField
) -> tuple[StateBlackboard, Mutation]:
    before = getattr(state, field)
    mutated = state.model_copy(update={field: MASKED_STATE_VALUE})
    return mutated, Mutation(
        operation="mask_state_field",
        field=field,
        before=before,
        after=MASKED_STATE_VALUE,
    )


def replace_state_field(
    state: StateBlackboard,
    field: StateField,
    replacement: str,
) -> tuple[StateBlackboard, Mutation]:
    if not isinstance(replacement, str):
        raise ValueError("contrastive STATE replacement must be text")
    normalized = " ".join(replacement.split())
    if not normalized:
        raise ValueError("contrastive STATE replacement must not be empty")
    if normalized == MASKED_STATE_VALUE:
        raise ValueError("contrastive STATE replacement must not use <MASKED>")
    before = getattr(state, field)
    if normalized == before:
        raise ValueError("contrastive STATE replacement must differ from the original")
    payload = state.model_dump(mode="json")
    payload[field] = normalized
    mutated = StateBlackboard.model_validate(payload)
    changed = {
        key for key, value in state.model_dump(mode="json").items()
        if mutated.model_dump(mode="json")[key] != value
    }
    if changed != {field}:
        raise ValueError("contrastive STATE replacement must change exactly one field")
    return mutated, Mutation(
        operation="replace_state_field",
        field=field,
        before=before,
        after=getattr(mutated, field),
    )


def select_counterfactual_plan(
    *,
    candidates: list[Candidate],
    selected_candidate_id: str,
    example_id: str,
    global_seed: int,
) -> tuple[PlanSelection, Mutation]:
    selected = next(
        (item for item in candidates if item.candidate_id == selected_candidate_id),
        None,
    )
    if selected is None:
        raise ValueError("selected candidate is absent from the candidate set")
    alternatives = sorted(
        (item for item in candidates if item.candidate_id != selected_candidate_id),
        key=lambda item: item.candidate_id,
    )
    if len(alternatives) != 2:
        raise ValueError("PLAN counterfactual requires exactly two unselected candidates")
    rng = random.Random(f"{global_seed}:{example_id}")
    counterfactual = PlanSelection.from_candidate(
        alternatives[rng.randrange(len(alternatives))]
    )
    original = PlanSelection.from_candidate(selected)
    return counterfactual, Mutation(
        operation="replace_plan_categories",
        field="strategies",
        before=list(original.strategies),
        after=list(counterfactual.strategies),
    )


class InterventionBuilder:
    """Build only verified conditional-response pairs."""

    def __init__(
        self,
        runner: TeacherRunner,
        *,
        verify_safety: Callable[[str, str], bool],
        verify_state_effect: Callable[
            [StateBlackboard, StateBlackboard, StateField, str, str],
            EffectVerification,
        ],
        verify_plan_effect: Callable[
            [PlanSelection, PlanSelection, str, str], EffectVerification
        ],
    ):
        self.runner = runner
        self.verify_safety = verify_safety
        self.verify_state_effect = verify_state_effect
        self.verify_plan_effect = verify_plan_effect

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
            if state_field not in STATE_ANCHOR_FIELDS or any(
                getattr(trace.state, field) == MASKED_STATE_VALUE
                for field in STATE_ANCHOR_FIELDS
            ):
                raise InterventionExcluded("state_not_single_field")
            dimension: NonSafetyDimension = {
                "emotion": "emotion",
                "intensity": "emotion",
                "primary_need": "need",
                "support_goal": "intent",
                "readiness": "timing",
                "main_constraint": "effectiveness",
                "relationship_context": "relationship",
            }[state_field]
            replacement = self.runner.generate_state_counterfactual(
                trace.history,
                trace.state,
                state_field,
                dimension,
                example_id=(
                    f"{trace.example_id}:intervention:STATE:{state_field}:counterfactual"
                ),
            )
            try:
                mutated_state, mutation = replace_state_field(
                    trace.state, state_field, replacement
                )
            except ValueError as error:
                raise InterventionExcluded("invalid_state_counterfactual") from error
            downstream = self.runner.rerun_downstream(
                trace.history,
                mutated_state,
                example_id=f"{trace.example_id}:intervention:STATE:{state_field}",
                clamped_state_field=state_field,
            )
            counterfactual_response = downstream.response
            verification = self.verify_state_effect(
                trace.state,
                mutated_state,
                state_field,
                trace.final_response,
                counterfactual_response,
            )
        else:
            original_plan = trace.final_selection.to_plan_selection()
            mutated_plan, mutation = select_counterfactual_plan(
                candidates=trace.candidates,
                selected_candidate_id=trace.final_selection.selected_candidate_id,
                example_id=trace.example_id,
                global_seed=global_seed,
            )
            if mutated_plan.strategies == original_plan.strategies:
                raise InterventionExcluded("plan_overlap")
            counterfactual_candidate = next(
                item
                for item in trace.candidates
                if item.strategy == mutated_plan.strategies[0]
            )
            counterfactual_response = counterfactual_candidate.response
            dimension = "effectiveness"
            verification = self.verify_plan_effect(
                original_plan,
                mutated_plan,
                trace.final_response,
                counterfactual_response,
            )
        if not self.verify_safety(trace.final_response, counterfactual_response):
            raise InterventionExcluded("safety_failure")
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
            affected_dimensions=list(verification.affected_dimensions),
            localized_effect=True,
            conditional_correspondence_verified=True,
            bidirectional_verified=True,
        )
