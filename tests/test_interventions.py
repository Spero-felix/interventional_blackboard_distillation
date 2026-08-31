import random

import pytest
from pydantic import ValidationError

from conftest import ScriptedBackend
from ibd.interventions import (
    ConditionalEffectVerdict,
    EffectVerification,
    InterventionBuilder,
    InterventionExcluded,
    StateEffectVerdict,
    replace_state_field,
    select_counterfactual_plan,
)
from ibd.schemas import STATE_ANCHOR_FIELDS
from ibd.teacher import TeacherRunner


@pytest.mark.parametrize("field", STATE_ANCHOR_FIELDS)
def test_state_counterfactual_changes_exactly_one_field(history, app_config, field):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-state", history)
    replacement = "一个不同且有效的值"
    mutated, mutation = replace_state_field(trace.state, field, replacement)
    changed = {
        key
        for key, value in trace.state.model_dump().items()
        if mutated.model_dump()[key] != value
    }
    assert changed == {field}
    assert mutation.operation == "replace_state_field"


@pytest.mark.parametrize("replacement", [" ", "<MASKED>"])
def test_state_counterfactual_rejects_invalid_replacement(
    history, app_config, replacement
):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-state", history)
    with pytest.raises(ValueError):
        replace_state_field(trace.state, "emotion", replacement)


def test_plan_counterfactual_reuses_one_unselected_candidate_deterministically(
    history, app_config
):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-plan", history)
    kwargs = {
        "candidates": trace.candidates,
        "selected_candidate_id": trace.final_selection.selected_candidate_id,
        "example_id": trace.example_id,
        "global_seed": 17,
    }
    random.seed(991)
    state_before = random.getstate()
    first, mutation = select_counterfactual_plan(**kwargs)
    second, _ = select_counterfactual_plan(**kwargs)
    assert first == second
    assert first.strategies[0] != trace.final_selection.selected_strategy
    assert first.strategies[0] in trace.plan.strategies
    assert mutation.before == [trace.final_selection.selected_strategy]
    assert random.getstate() == state_before


def test_plan_intervention_makes_no_new_model_call(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-plan", history)
    backend = ScriptedBackend()
    builder = InterventionBuilder(
        TeacherRunner(backend, app_config),
        verify_safety=lambda original, counterfactual: True,
        verify_state_effect=lambda *args: EffectVerification(passed=True),
        verify_plan_effect=lambda *args: EffectVerification(passed=True),
    )
    record = builder.build(trace, "PLAN", global_seed=17)
    assert backend.calls == []
    expected = next(
        item.response
        for item in trace.candidates
        if item.strategy == record.mutated_plan.strategies[0]
    )
    assert record.counterfactual_response == expected
    assert record.conditioning_contract == "legacy_joint_downstream_v1"


def test_state_intervention_reruns_planner_candidates_and_selector(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-state", history)
    backend = ScriptedBackend()
    builder = InterventionBuilder(
        TeacherRunner(backend, app_config),
        verify_safety=lambda original, counterfactual: True,
        verify_state_effect=lambda *args: EffectVerification(
            passed=True, affected_dimensions=("timing",)
        ),
        verify_plan_effect=lambda *args: EffectVerification(passed=True),
    )
    record = builder.build(
        trace, "STATE", state_field="readiness", global_seed=17
    )
    assert [call["role"] for call in backend.calls] == [
        "state_counterfactual_generator",
        "planner",
        "candidate",
        "candidate",
        "candidate",
        "final_selector",
    ]
    assert record.mutated_state.readiness == "准备立即采取具体行动"
    assert record.affected_dimensions == ["timing"]
    assert record.conditioning_contract == "legacy_joint_downstream_v1"


def test_b2_state_intervention_fixes_the_original_plan(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-b2-state", history)
    backend = ScriptedBackend()
    builder = InterventionBuilder(
        TeacherRunner(backend, app_config),
        verify_safety=lambda original, counterfactual: True,
        verify_state_effect=lambda *args: EffectVerification(passed=True),
        verify_plan_effect=lambda *args: EffectVerification(passed=True),
    )

    record = builder.build(
        trace,
        "STATE",
        state_field="readiness",
        global_seed=17,
        state_plan_policy="fixed_original",
    )

    assert [call["role"] for call in backend.calls] == [
        "state_counterfactual_generator",
        "candidate",
    ]
    candidate_payload = __import__("json").loads(backend.calls[-1]["messages"][1]["content"])
    assert candidate_payload["context"]["state"]["readiness"] == "准备立即采取具体行动"
    assert candidate_payload["context"]["fixed_plan"] == trace.final_selection.to_plan_selection().model_dump(mode="json")
    assert record.counterfactual_response == "候选回复-1"
    assert record.conditioning_contract == "single_variable_v1"


def test_b2_plan_intervention_has_single_variable_contract(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-b2-plan", history)
    builder = InterventionBuilder(
        TeacherRunner(ScriptedBackend(), app_config),
        verify_safety=lambda original, counterfactual: True,
        verify_state_effect=lambda *args: EffectVerification(passed=True),
        verify_plan_effect=lambda *args: EffectVerification(passed=True),
    )

    record = builder.build(
        trace,
        "PLAN",
        global_seed=17,
        state_plan_policy="fixed_original",
    )

    assert record.conditioning_contract == "single_variable_v1"


def test_builder_requires_safety_and_conditional_effect(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-filter", history)
    unsafe = InterventionBuilder(
        TeacherRunner(ScriptedBackend(), app_config),
        verify_safety=lambda original, counterfactual: False,
        verify_state_effect=lambda *args: EffectVerification(passed=True),
        verify_plan_effect=lambda *args: EffectVerification(passed=True),
    )
    with pytest.raises(InterventionExcluded) as caught:
        unsafe.build(trace, "PLAN", global_seed=17)
    assert caught.value.reason == "safety_failure"

    no_effect = InterventionBuilder(
        TeacherRunner(ScriptedBackend(), app_config),
        verify_safety=lambda original, counterfactual: True,
        verify_state_effect=lambda *args: EffectVerification(passed=True),
        verify_plan_effect=lambda *args: EffectVerification(
            passed=False, reason="no_localized_effect"
        ),
    )
    with pytest.raises(InterventionExcluded) as caught:
        no_effect.build(trace, "PLAN", global_seed=17)
    assert caught.value.reason == "no_localized_effect"


def test_verifier_contracts_have_no_preference_direction():
    plan = ConditionalEffectVerdict(
        condition_a_fit=True,
        condition_b_fit=True,
        control_effect_present=True,
        evidence="both fit",
    )
    state = StateEffectVerdict(
        condition_a_fit=True,
        condition_b_fit=True,
        field_effect_present=True,
        evidence="localized",
    )
    assert "preferred" not in plan.model_dump()
    assert "preferred" not in state.model_dump()
    with pytest.raises(ValidationError):
        ConditionalEffectVerdict(
            condition_a_fit=True,
            condition_b_fit=True,
            control_effect_present=True,
            evidence="x",
            preferred="A",
        )
