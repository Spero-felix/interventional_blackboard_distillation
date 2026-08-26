import pytest

from conftest import ScriptedBackend
from ibd.export import intervention_row, sft_row, slot_row
from ibd.interventions import EffectVerification, InterventionBuilder
from ibd.teacher import TeacherRunner


def test_sft_export_contains_natural_response_and_one_strategy(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-sft", history)
    assert sft_row(trace) == {
        "example_id": "e-sft",
        "prompt": trace.history.as_prompt(),
        "response": trace.final_response,
        "selected_strategy": trace.final_selection.selected_strategy,
    }


def test_slot_export_contains_only_state_and_final_single_plan(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-slot", history)
    row = slot_row(trace)
    assert set(row) == {"example_id", "prompt", "state", "plan"}
    assert row["plan"]["strategies"] == [trace.final_selection.selected_strategy]
    assert "views" not in row


@pytest.mark.parametrize("split", ["test", "diagnostic_holdout"])
def test_student_export_rejects_non_trainable_splits(history, app_config, split):
    trace = TeacherRunner(ScriptedBackend(), app_config).run(
        f"e-{split}", history, split=split
    )
    with pytest.raises(ValueError):
        sft_row(trace)


def test_intervention_export_has_no_quality_direction_labels(history, app_config):
    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-int", history)
    record = InterventionBuilder(
        TeacherRunner(ScriptedBackend(), app_config),
        verify_safety=lambda *args: True,
        verify_state_effect=lambda *args: EffectVerification(passed=True),
        verify_plan_effect=lambda *args: EffectVerification(passed=True),
    ).build(trace, "PLAN", global_seed=17)
    row = intervention_row(record, trace.history.as_prompt())
    assert set(row) == {
        "example_id",
        "prompt",
        "function",
        "clamp",
        "full_response",
        "counterfactual_response",
        "target_dimension",
        "affected_dimensions",
        "conditional_correspondence_verified",
    }
    assert "chosen" not in row
    assert "rejected" not in row
    assert "effect_direction" not in row
