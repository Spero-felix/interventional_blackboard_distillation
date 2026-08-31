"""Typed contracts for the unified Teacher and three-stage Student pipeline."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FunctionName = Literal["STATE", "PLAN"]
StrategyId = Literal["S1", "S2", "S3"]
StrategyName = Literal[
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]
NonSafetyDimension = Literal[
    "emotion", "need", "relationship", "intent", "specificity", "timing",
    "effectiveness", "autonomy", "factuality", "non_template",
]
MASKED_STATE_VALUE = "<MASKED>"
STATE_ANCHOR_FIELDS = (
    "emotion", "intensity", "primary_need", "support_goal", "readiness",
    "main_constraint", "relationship_context",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DialogueTurn(StrictModel):
    role: Literal["seeker", "supporter"]
    content: str = Field(min_length=1)


class History(StrictModel):
    turns: list[DialogueTurn] = Field(min_length=1)

    @model_validator(mode="after")
    def require_seeker_last(self) -> "History":
        if self.turns[-1].role != "seeker":
            raise ValueError("history must end with a seeker turn")
        return self

    def as_prompt(self) -> str:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in self.turns)


class AnalysisView(StrictModel):
    summary: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    uncertainty: str = ""


class MultiViewStateViews(StrictModel):
    emotion: AnalysisView
    need: AnalysisView
    relationship: AnalysisView
    intent: AnalysisView


class StateBlackboard(StrictModel):
    emotion: str
    intensity: str
    primary_need: str
    support_goal: str
    readiness: str
    main_constraint: str
    relationship_context: str

    @field_validator(*STATE_ANCHOR_FIELDS, mode="before")
    @classmethod
    def normalize_compact_value(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("STATE values must not be empty")
        if len(normalized.split()) > 20:
            raise ValueError("STATE values must contain at most 20 words")
        return normalized

    @model_validator(mode="after")
    def require_distinct_values(self) -> "StateBlackboard":
        values = [getattr(self, field) for field in STATE_ANCHOR_FIELDS]
        if values.count(MASKED_STATE_VALUE) > 1:
            raise ValueError("STATE may contain only a single <MASKED> value")
        visible = [value.casefold() for value in values if value != MASKED_STATE_VALUE]
        if len(set(visible)) != len(visible):
            raise ValueError("STATE values must be distinct after normalization")
        return self


class MultiViewStateAnalysis(StrictModel):
    views: MultiViewStateViews
    state: StateBlackboard


class StateCounterfactual(StrictModel):
    replacement: str = Field(min_length=1)


class StrategyPlanSet(StrictModel):
    strategies: Annotated[list[StrategyName], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def require_distinct_strategies(self) -> "StrategyPlanSet":
        if len(set(self.strategies)) != 3:
            raise ValueError("strategy plan must use three distinct strategies")
        return self

    def strategy_for_id(self, strategy_id: StrategyId) -> StrategyName:
        return self.strategies[{"S1": 0, "S2": 1, "S3": 2}[strategy_id]]


class Candidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    strategy_id: StrategyId
    strategy: StrategyName
    response: str = Field(min_length=1)
    seed: int | None = None
    response_goal: str = Field(min_length=1)
    response_act: str = Field(min_length=1)


class PlanSelection(StrictModel):
    strategies: Annotated[list[StrategyName], Field(min_length=1, max_length=1)]
    response_goal: str = Field(min_length=1)
    response_act: str = Field(min_length=1)

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> "PlanSelection":
        return cls(
            strategies=[candidate.strategy],
            response_goal=candidate.response_goal,
            response_act=candidate.response_act,
        )


class FinalSelectionDecision(StrictModel):
    selected_candidate_id: str = Field(min_length=1)
    response_goal: str = Field(min_length=1)
    response_act: str = Field(min_length=1)


class FinalSelection(StrictModel):
    selected_candidate_id: str = Field(min_length=1)
    selected_strategy_id: StrategyId
    selected_strategy: StrategyName
    response_goal: str = Field(min_length=1)
    response_act: str = Field(min_length=1)
    response: str = Field(min_length=1)

    @classmethod
    def from_candidate(
        cls, candidate: Candidate, decision: FinalSelectionDecision | None = None
    ) -> "FinalSelection":
        return cls(
            selected_candidate_id=candidate.candidate_id,
            selected_strategy_id=candidate.strategy_id,
            selected_strategy=candidate.strategy,
            response_goal=decision.response_goal if decision else candidate.response_goal,
            response_act=decision.response_act if decision else candidate.response_act,
            response=candidate.response,
        )

    def to_plan_selection(self) -> PlanSelection:
        return PlanSelection(
            strategies=[self.selected_strategy],
            response_goal=self.response_goal,
            response_act=self.response_act,
        )


class SafetyVerdict(StrictModel):
    original_safe: bool
    counterfactual_safe: bool
    evidence: str = Field(min_length=1)


class CallRecord(StrictModel):
    role: str
    attempt: int = Field(ge=1)
    raw_text: str
    parsed: dict[str, Any]
    schema_retry: bool = False
    cached: bool = False


class TeacherTrace(StrictModel):
    example_id: str
    split: Literal["train", "dev", "test", "diagnostic_holdout"] = "train"
    history: History
    state_analysis: MultiViewStateAnalysis
    state: StateBlackboard
    plan: StrategyPlanSet
    candidates: Annotated[list[Candidate], Field(min_length=3, max_length=3)]
    final_selection: FinalSelection
    final_response: str = Field(min_length=1)
    call_records: list[CallRecord]

    @model_validator(mode="after")
    def validate_provenance(self) -> "TeacherTrace":
        if self.state != self.state_analysis.state:
            raise ValueError("state must equal state_analysis.state")
        if self.final_response != self.final_selection.response:
            raise ValueError("final_response must equal final_selection.response")
        for candidate in self.candidates:
            if self.plan.strategy_for_id(candidate.strategy_id) != candidate.strategy:
                raise ValueError("candidate strategy must match its planned strategy")
        selected = [
            candidate for candidate in self.candidates
            if candidate.candidate_id == self.final_selection.selected_candidate_id
        ]
        if len(selected) != 1:
            raise ValueError("final selection must identify exactly one candidate")
        candidate = selected[0]
        if (
            self.final_selection.selected_strategy_id != candidate.strategy_id
            or self.final_selection.selected_strategy != candidate.strategy
            or self.final_selection.response != candidate.response
        ):
            raise ValueError("final selection must be canonicalized from its candidate")
        return self


class Mutation(StrictModel):
    operation: Literal[
        "mask_state_field", "replace_state_field", "replace_plan_categories"
    ]
    field: str
    before: object
    after: object


class InterventionRecord(StrictModel):
    example_id: str
    function: FunctionName
    mutation: Mutation
    mutated_state: StateBlackboard | None = None
    mutated_plan: PlanSelection | None = None
    full_response: str = Field(min_length=1)
    counterfactual_response: str = Field(min_length=1)
    target_dimension: NonSafetyDimension
    affected_dimensions: list[NonSafetyDimension] = Field(default_factory=list)
    conditioning_contract: Literal[
        "legacy_joint_downstream_v1", "single_variable_v1"
    ] = "legacy_joint_downstream_v1"
    localized_effect: bool = True
    conditional_correspondence_verified: bool = True
    bidirectional_verified: bool

    @model_validator(mode="after")
    def validate_intervention(self) -> "InterventionRecord":
        if not (
            self.localized_effect
            and self.conditional_correspondence_verified
            and self.bidirectional_verified
        ):
            raise ValueError("retained intervention must pass conditional verification")
        if self.function == "STATE":
            if self.mutation.operation != "replace_state_field":
                raise ValueError("STATE intervention must replace a state field")
            if self.mutated_state is None or self.mutated_plan is not None:
                raise ValueError("STATE intervention must carry only mutated_state")
        else:
            if self.mutation.operation != "replace_plan_categories":
                raise ValueError("PLAN intervention must replace its strategy")
            if self.mutated_plan is None or self.mutated_state is not None:
                raise ValueError("PLAN intervention must carry only mutated_plan")
        return self
