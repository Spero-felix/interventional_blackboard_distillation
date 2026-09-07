"""Typed contracts for the unified Teacher and three-stage Student pipeline."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
DominantEmotion = Literal[
    "sadness_loss",
    "fear_anxiety",
    "anger_frustration",
    "shame_guilt",
    "hurt_disappointment",
    "loneliness",
    "overwhelm",
    "relief",
    "hope_positive",
    "neutral",
    "mixed",
    "other",
    "unknown",
]
DistressLevel = Literal["low", "moderate", "high", "unknown"]
PrimarySupportNeed = Literal[
    "validation",
    "esteem_support",
    "sensemaking",
    "information",
    "decision_support",
    "action_support",
    "connection",
    "unknown",
]
AdviceReceptivity = Literal[
    "closed", "hesitant", "open", "requested", "unknown"
]
ActionIntent = Literal[
    "not_considering",
    "ambivalent",
    "considering",
    "committed",
    "acting",
    "unknown",
]
ActionCapacity = Literal["blocked", "limited", "adequate", "strong", "unknown"]
ContinuationIntent = Literal[
    "closing",
    "passive_open",
    "engaged",
    "explicitly_continuing",
    "unknown",
]
StateField = Literal[
    "dominant_emotion",
    "distress_level",
    "primary_support_need",
    "advice_receptivity",
    "action_intent",
    "action_capacity",
    "continuation_intent",
]
STATE_ANCHOR_FIELDS: tuple[StateField, ...] = (
    "dominant_emotion",
    "distress_level",
    "primary_support_need",
    "advice_receptivity",
    "action_intent",
    "action_capacity",
    "continuation_intent",
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
    dominant_emotion: DominantEmotion
    distress_level: DistressLevel
    primary_support_need: PrimarySupportNeed
    advice_receptivity: AdviceReceptivity
    action_intent: ActionIntent
    action_capacity: ActionCapacity
    continuation_intent: ContinuationIntent


EvidenceBasis = Literal[
    "explicit", "strong_inference", "absent_or_ambiguous"
]


class StateFieldEvidence(StrictModel):
    evidence: str = ""
    basis: EvidenceBasis


class StateEvidence(StrictModel):
    dominant_emotion: StateFieldEvidence
    distress_level: StateFieldEvidence
    primary_support_need: StateFieldEvidence
    advice_receptivity: StateFieldEvidence
    action_intent: StateFieldEvidence
    action_capacity: StateFieldEvidence
    continuation_intent: StateFieldEvidence


class MultiViewStateAnalysis(StrictModel):
    views: MultiViewStateViews
    state: StateBlackboard
    state_evidence: StateEvidence

    @model_validator(mode="after")
    def require_evidence_consistency(self) -> "MultiViewStateAnalysis":
        for field in STATE_ANCHOR_FIELDS:
            value = getattr(self.state, field)
            evidence = getattr(self.state_evidence, field)
            absent = evidence.basis == "absent_or_ambiguous"
            if (value == "unknown") != absent:
                raise ValueError(f"{field} value and evidence basis disagree")
            if value != "unknown" and not evidence.evidence.strip():
                raise ValueError(f"{field} requires non-blank evidence")
        return self


class StrategyPlanSet(StrictModel):
    strategies: Annotated[list[StrategyName], Field(min_length=1, max_length=3)]

    @model_validator(mode="after")
    def require_distinct_strategies(self) -> "StrategyPlanSet":
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("strategy plan must use distinct strategies")
        return self

    def strategy_for_id(self, strategy_id: StrategyId) -> StrategyName:
        index = {"S1": 0, "S2": 1, "S3": 2}[strategy_id]
        if index >= len(self.strategies):
            raise ValueError(f"{strategy_id} is absent from this strategy plan")
        return self.strategies[index]


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


class CallRecord(StrictModel):
    role: str
    attempt: int = Field(ge=1)
    raw_text: str
    parsed: dict[str, Any]
    schema_retry: bool = False
    metadata_fallback: bool = False
    cached: bool = False


class TeacherTrace(StrictModel):
    example_id: str
    split: Literal["train", "dev", "test", "diagnostic_holdout"] = "train"
    history: History
    state_analysis: MultiViewStateAnalysis
    state: StateBlackboard
    plan: StrategyPlanSet
    candidates: Annotated[list[Candidate], Field(min_length=1, max_length=3)]
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
