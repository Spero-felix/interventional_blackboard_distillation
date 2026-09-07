import pytest
from pydantic import ValidationError

from ibd.schemas import (
    AnalysisView,
    Candidate,
    DialogueTurn,
    FinalSelection,
    FinalSelectionDecision,
    History,
    MultiViewStateAnalysis,
    MultiViewStateViews,
    PlanSelection,
    STATE_ANCHOR_FIELDS,
    StateBlackboard,
    StrategyPlanSet,
)


VALID_STATE = {
    "dominant_emotion": "hurt_disappointment",
    "distress_level": "moderate",
    "primary_support_need": "decision_support",
    "advice_receptivity": "hesitant",
    "action_intent": "considering",
    "action_capacity": "limited",
    "continuation_intent": "engaged",
}

EXPECTED_STATE_VALUES = {
    "dominant_emotion": (
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
    ),
    "distress_level": ("low", "moderate", "high", "unknown"),
    "primary_support_need": (
        "validation",
        "esteem_support",
        "sensemaking",
        "information",
        "decision_support",
        "action_support",
        "connection",
        "unknown",
    ),
    "advice_receptivity": ("closed", "hesitant", "open", "requested", "unknown"),
    "action_intent": (
        "not_considering",
        "ambivalent",
        "considering",
        "committed",
        "acting",
        "unknown",
    ),
    "action_capacity": ("blocked", "limited", "adequate", "strong", "unknown"),
    "continuation_intent": (
        "closing",
        "passive_open",
        "engaged",
        "explicitly_continuing",
        "unknown",
    ),
}

VALID_EVIDENCE = {
    field: {
        "evidence": f"dialogue evidence for {field}",
        "basis": "strong_inference",
    }
    for field in VALID_STATE
}


def _state(**overrides):
    return StateBlackboard(**{**VALID_STATE, **overrides})


def _analysis(*, state=None, evidence=None):
    view = lambda summary: AnalysisView(summary=summary, evidence="dialogue quote")
    return MultiViewStateAnalysis(
        views=MultiViewStateViews(
            emotion=view("hurt"),
            need=view("decision support"),
            relationship=view("relationship tension"),
            intent=view("considering action"),
        ),
        state=VALID_STATE if state is None else state,
        state_evidence=VALID_EVIDENCE if evidence is None else evidence,
    )


def _evidence_with(field, *, evidence, basis):
    payload = {key: dict(value) for key, value in VALID_EVIDENCE.items()}
    payload[field] = {"evidence": evidence, "basis": basis}
    return payload


def test_history_must_end_with_seeker():
    with pytest.raises(ValidationError, match="seeker"):
        History(
            turns=[
                DialogueTurn(role="seeker", content="我有点难过"),
                DialogueTurn(role="supporter", content="发生了什么？"),
            ]
        )


def test_state_has_exactly_seven_ordered_fields():
    assert STATE_ANCHOR_FIELDS == tuple(VALID_STATE)
    assert tuple(_state().model_dump()) == STATE_ANCHOR_FIELDS


def test_state_accepts_every_declared_enum_value():
    for field, values in EXPECTED_STATE_VALUES.items():
        for value in values:
            assert getattr(_state(**{field: value}), field) == value


@pytest.mark.parametrize("field", tuple(VALID_STATE))
def test_state_rejects_value_outside_target_field_enum(field):
    with pytest.raises(ValidationError):
        _state(**{field: "invalid"})


def test_state_json_schema_exposes_each_fields_exact_values():
    properties = StateBlackboard.model_json_schema()["properties"]
    for field, values in EXPECTED_STATE_VALUES.items():
        assert properties[field]["enum"] == list(values)


def test_old_state_shape_is_rejected_directly():
    with pytest.raises(ValidationError) as caught:
        StateBlackboard(
            emotion="失落",
            intensity="中等",
            primary_need="被理解",
            support_goal="准备沟通",
            readiness="愿意探索",
            main_constraint="担心回避",
            relationship_context="亲密关系",
        )
    errors = caught.value.errors()
    assert any(error["type"] == "missing" for error in errors)
    assert any(error["type"] == "extra_forbidden" for error in errors)


@pytest.mark.parametrize(
    "overrides",
    [
        {"advice_receptivity": "requested", "action_intent": "not_considering"},
        {"action_intent": "committed", "action_capacity": "blocked"},
        {"continuation_intent": "engaged", "advice_receptivity": "closed"},
        {"continuation_intent": "closing", "action_intent": "acting"},
        {"dominant_emotion": "neutral", "distress_level": "high"},
    ],
)
def test_state_dimensions_allow_orthogonal_combinations(overrides):
    assert _state(**overrides)


def test_unified_analysis_contains_views_state_and_field_evidence():
    analysis = _analysis()
    assert analysis.state.primary_support_need == "decision_support"
    assert analysis.state_evidence.primary_support_need.basis == "strong_inference"


@pytest.mark.parametrize("field", tuple(VALID_STATE))
def test_unknown_state_requires_absent_or_ambiguous_evidence(field):
    with pytest.raises(ValidationError, match=field):
        _analysis(
            state={**VALID_STATE, field: "unknown"},
            evidence=_evidence_with(
                field,
                evidence="explicit dialogue evidence",
                basis="explicit",
            ),
        )


@pytest.mark.parametrize("field", tuple(VALID_STATE))
def test_known_state_rejects_absent_or_ambiguous_evidence(field):
    with pytest.raises(ValidationError, match=field):
        _analysis(
            evidence=_evidence_with(
                field,
                evidence="no reliable evidence",
                basis="absent_or_ambiguous",
            )
        )


@pytest.mark.parametrize("field", tuple(VALID_STATE))
def test_known_state_requires_non_blank_evidence(field):
    with pytest.raises(ValidationError, match=field):
        _analysis(
            evidence=_evidence_with(
                field,
                evidence=" ",
                basis="strong_inference",
            )
        )


@pytest.mark.parametrize("field", tuple(VALID_STATE))
def test_unknown_state_accepts_absent_or_ambiguous_evidence(field):
    analysis = _analysis(
        state={**VALID_STATE, field: "unknown"},
        evidence=_evidence_with(
            field,
            evidence="",
            basis="absent_or_ambiguous",
        ),
    )
    assert getattr(analysis.state, field) == "unknown"


@pytest.mark.parametrize(
    "strategies",
    [
        ["Question"],
        ["Question", "Reflection of feelings"],
        ["Question", "Reflection of feelings", "Providing Suggestions"],
    ],
)
def test_planner_accepts_one_to_three_distinct_strategies(strategies):
    plan = StrategyPlanSet(strategies=strategies)
    assert plan.strategies == strategies


def test_planner_rejects_duplicate_or_out_of_range_strategies():
    with pytest.raises(ValidationError, match="distinct"):
        StrategyPlanSet(strategies=["Question", "Question", "Information"])
    with pytest.raises(ValidationError):
        StrategyPlanSet(strategies=[])
    with pytest.raises(ValidationError):
        StrategyPlanSet(
            strategies=[
                "Question",
                "Reflection of feelings",
                "Providing Suggestions",
                "Information",
            ]
        )


def test_plan_selection_contains_exactly_one_strategy_and_intent():
    selection = PlanSelection(
        strategies=["Question"],
        response_goal="明确下一步",
        response_act="提出开放式问题",
    )
    assert selection.strategies == ["Question"]
    with pytest.raises(ValidationError):
        PlanSelection(
            strategies=["Question", "Information"],
            response_goal="混合",
            response_act="混合",
        )


def test_final_selection_is_canonicalized_from_candidate():
    candidate = Candidate(
        candidate_id="2",
        strategy_id="S2",
        strategy="Question",
        response="你最希望先改变什么？",
        seed=29,
        response_goal="明确目标",
        response_act="询问优先改变",
    )
    decision = FinalSelectionDecision(
        selected_candidate_id="2",
        response_goal="明确目标",
        response_act="询问优先改变",
    )
    selected = FinalSelection.from_candidate(candidate, decision)
    assert selected.response == candidate.response
    assert selected.to_plan_selection().strategies == ["Question"]


def test_candidate_seed_is_optional_but_accepts_historical_integer_values():
    payload = {
        "candidate_id": "2",
        "strategy_id": "S2",
        "strategy": "Question",
        "response": "What feels most manageable now?",
        "response_goal": "clarify readiness",
        "response_act": "ask one focused question",
    }

    assert Candidate.model_validate({**payload, "seed": 29}).seed == 29
    assert Candidate.model_validate({**payload, "seed": None}).seed is None
    assert Candidate.model_validate(payload).seed is None
