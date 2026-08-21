"""Typed, auditable contracts for Teacher traces and Student datasets."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ExpertName = Literal["emotion", "need", "relationship", "intent"]
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
MASKED_STATE_VALUE = "<MASKED>"
STATE_ANCHOR_FIELDS = (
    "emotion",
    "intensity",
    "primary_need",
    "support_goal",
    "readiness",
    "main_constraint",
    "relationship_context",
)
NonSafetyDimension = Literal[
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
]


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


_EXPERT_FIELDS: dict[str, frozenset[str]] = {
    "emotion": frozenset({"emotion", "intensity", "trajectory", "coping_signal"}),
    "need": frozenset({"need", "priority", "readiness", "constraint"}),
    "relationship": frozenset(
        {"relationship_type", "relationship_pattern", "power_dynamic", "boundary_signal"}
    ),
    "intent": frozenset({"intent", "explicit_ask", "implicit_goal", "decision_stage"}),
}


class ExpertOutput(StrictModel):
    expert: ExpertName
    fields: dict[str, Any]
    evidence: list[str]
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def keep_domain_isolated(self) -> "ExpertOutput":
        invalid = set(self.fields) - _EXPERT_FIELDS[self.expert]
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(f"fields {names} are not allowed for {self.expert} expert")
        return self


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
        if len(set(values)) != len(values):
            raise ValueError("STATE values must be distinct")
        return self


class StrategyPlan(StrictModel):
    strategy_id: StrategyId
    strategy: StrategyName
    therapeutic_goal: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    response_acts: list[str] = Field(min_length=1)
    tone: str = Field(min_length=1)
    avoid: list[str] = Field(default_factory=list)


class StrategyPlanSet(StrictModel):
    strategies: Annotated[list[StrategyName], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def require_distinct_strategies(self) -> "StrategyPlanSet":
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("strategy plan must use three distinct strategies")
        return self

    def strategy_for_id(self, strategy_id: StrategyId) -> StrategyName:
        return self.strategies[{"S1": 0, "S2": 1, "S3": 2}[strategy_id]]


class StrategyUse(StrictModel):
    strategy_id: StrategyId
    strategy: StrategyName
    contribution: str = Field(min_length=1)


class FinalAnswer(StrictModel):
    response: str = Field(min_length=1)
    strategy_uses: Annotated[list[StrategyUse], Field(min_length=1, max_length=2)]

    @model_validator(mode="after")
    def require_distinct_strategy_uses(self) -> "FinalAnswer":
        if len({item.strategy for item in self.strategy_uses}) != len(self.strategy_uses):
            raise ValueError("final strategy uses must have distinct strategies")
        return self


class PlanSelection(StrictModel):
    strategies: Annotated[list[StrategyName], Field(min_length=1, max_length=2)]

    @model_validator(mode="after")
    def require_distinct_strategies(self) -> "PlanSelection":
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("plan selection must use distinct strategies")
        return self

    @classmethod
    def from_final_answer(cls, final_answer: FinalAnswer) -> "PlanSelection":
        return cls(strategies=[item.strategy for item in final_answer.strategy_uses])


class Candidate(StrictModel):
    candidate_id: str
    strategy_id: StrategyId
    strategy: StrategyName
    response: str = Field(min_length=1)
    seed: int


class CritiqueIssue(StrictModel):
    dimension: str
    evidence: str
    severity: int = Field(ge=1, le=5)
    suggested_revision: str = ""


CritiqueIssueList = Annotated[list[CritiqueIssue], Field(max_length=1)]


class CritiqueReport(StrictModel):
    critic: Literal["emotion", "effectiveness", "safety"]
    candidate_issues: dict[str, CritiqueIssueList]
    summary: str = ""


class EmotionCritiqueReport(CritiqueReport):
    critic: Literal["emotion"]


class EffectivenessCritiqueReport(CritiqueReport):
    critic: Literal["effectiveness"]


class SafetyCritiqueReport(CritiqueReport):
    critic: Literal["safety"]


class CallRecord(StrictModel):
    role: str
    attempt: int = Field(ge=1)
    request_hash: str
    raw_text: str
    parsed: dict[str, Any]
    schema_retry: bool = False
    cached: bool = False


class TeacherTrace(StrictModel):
    example_id: str
    teacher_protocol_hash: str = Field(min_length=64, max_length=64)
    split: Literal["train", "dev", "test", "diagnostic_holdout"] = "train"
    history: History
    expert_outputs: dict[ExpertName, ExpertOutput]
    state: StateBlackboard
    plan: StrategyPlanSet
    candidates: list[Candidate]
    critiques: list[CritiqueReport]
    final_answer: FinalAnswer
    final_response: str
    call_records: list[CallRecord]

    @model_validator(mode="after")
    def validate_strategy_provenance(self) -> "TeacherTrace":
        if self.final_response != self.final_answer.response:
            raise ValueError("final_response must equal final_answer.response")
        for item in self.final_answer.strategy_uses:
            if self.plan.strategy_for_id(item.strategy_id) != item.strategy:
                raise ValueError("final strategy use must match a planned strategy")
        for candidate in self.candidates:
            if self.plan.strategy_for_id(candidate.strategy_id) != candidate.strategy:
                raise ValueError("candidate strategy must match a planned strategy")
        return self


class Mutation(StrictModel):
    operation: Literal["mask_state_field", "replace_plan_categories"]
    field: str
    before: object
    after: object


class InterventionRecord(StrictModel):
    example_id: str
    function: FunctionName
    mutation: Mutation
    mutated_state: StateBlackboard | None = None
    mutated_plan: PlanSelection | None = None
    full_response: str
    counterfactual_response: str
    target_dimension: NonSafetyDimension
    localized_degradation: bool
    bidirectional_verified: bool

    @model_validator(mode="after")
    def function_matches_mutation(self) -> "InterventionRecord":
        if self.function == "STATE":
            if self.mutation.operation != "mask_state_field":
                raise ValueError("STATE intervention must mask a state field")
            if self.mutated_state is None or self.mutated_plan is not None:
                raise ValueError("STATE intervention must carry only mutated_state")
        else:
            if self.mutation.operation != "replace_plan_categories":
                raise ValueError("PLAN intervention must replace plan categories")
            if self.mutated_plan is None or self.mutated_state is not None:
                raise ValueError("PLAN intervention must carry only mutated_plan")
        return self


class MarginPair(StrictModel):
    example_id: str
    prompt: str
    chosen: str
    chosen_strategy_uses: list[StrategyUse]
    rejected_candidate_id: str
    rejected_strategy_id: StrategyId
    rejected_strategy: StrategyName
    rejected: str
    defect_dimension: NonSafetyDimension
    defect_evidence: str
    order_swap_verified: bool
    safety_filter_passed: bool

    @model_validator(mode="after")
    def require_safe_pair(self) -> "MarginPair":
        if not self.safety_filter_passed:
            raise ValueError("margin pairs must use a safety-filter-passing rejected candidate")
        if not self.order_swap_verified:
            raise ValueError("margin pairs must pass order-swap verification")
        return self
