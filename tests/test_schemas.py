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
    StateBlackboard,
    StrategyPlanSet,
)


def _state(**overrides):
    values = {
        "emotion": "失落",
        "intensity": "中等",
        "primary_need": "被理解",
        "support_goal": "准备坦诚沟通",
        "readiness": "愿意探索",
        "main_constraint": "担心对方回避",
        "relationship_context": "亲密关系沟通僵局",
    }
    values.update(overrides)
    return StateBlackboard(**values)


def test_history_must_end_with_seeker():
    with pytest.raises(ValidationError, match="seeker"):
        History(
            turns=[
                DialogueTurn(role="seeker", content="我有点难过"),
                DialogueTurn(role="supporter", content="发生了什么？"),
            ]
        )


def test_state_is_compact_normalized_and_distinct():
    state = _state(primary_need="  希望\n被听见 ")
    assert state.primary_need == "希望 被听见"
    with pytest.raises(ValidationError, match="distinct"):
        _state(emotion="需要 空间", primary_need=" 需要  空间 ")
    with pytest.raises(ValidationError, match="at most 20 words"):
        _state(main_constraint=" ".join(f"w{i}" for i in range(21)))


def test_unified_analysis_contains_four_views_and_one_state():
    view = lambda summary: AnalysisView(summary=summary, evidence="对话证据")
    analysis = MultiViewStateAnalysis(
        views=MultiViewStateViews(
            emotion=view("失落"),
            need=view("被理解"),
            relationship=view("回避沟通"),
            intent=view("准备开口"),
        ),
        state=_state(),
    )
    assert analysis.state.primary_need == "被理解"


def test_planner_requires_exactly_three_distinct_strategies():
    plan = StrategyPlanSet(
        strategies=["Question", "Reflection of feelings", "Providing Suggestions"]
    )
    assert plan.strategy_for_id("S2") == "Reflection of feelings"
    with pytest.raises(ValidationError, match="distinct"):
        StrategyPlanSet(strategies=["Question", "Question", "Information"])


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
