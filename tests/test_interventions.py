import json
import random

import pytest

from conftest import ScriptedBackend

from ibd.schemas import (
    STATE_ANCHOR_FIELDS,
    CritiqueIssue,
    CritiqueReport,
    PlanSelection,
    StrategyPlanSet,
)
from ibd.teacher import TeacherRunner


class PairBackend(ScriptedBackend):
    def __init__(self):
        super().__init__()
        self._pair_checks = 0

    def _payload(self, role, seed):
        if role == "pair_verifier":
            self._pair_checks += 1
            return {
                "preferred": "A" if self._pair_checks == 1 else "B",
                "defect_dimension": "timing",
                "evidence": "候选过早给出建议",
            }
        return super()._payload(role, seed)


class SelectedStrategyBackend(ScriptedBackend):
    def __init__(self, final_response="按反事实策略生成的安全回复"):
        super().__init__()
        self.final_response = final_response

    def complete(self, *, role, messages, model_config, json_mode=True, seed=None):
        from ibd.backend import LLMResult

        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "model": model_config.model,
                "json_mode": json_mode,
                "seed": seed,
            }
        )
        context = json.loads(messages[1]["content"])["context"]
        if role.startswith("candidate_"):
            payload = {
                "candidate_id": context["candidate_id"],
                "strategy_id": context["strategy_id"],
                "strategy": context["strategy"],
                "response": f"候选回复-{role[-1]}",
                "seed": context["seed"],
            }
        elif role.endswith("_critic"):
            payload = {
                "critic": role.removesuffix("_critic"),
                "candidate_issues": {
                    item["candidate_id"]: [] for item in context["candidates"]
                },
                "summary": "完成独立检查",
            }
        elif role == "final_integrator":
            payload = {
                "response": self.final_response,
                "strategy_uses": [
                    {
                        "strategy_id": f"S{index}",
                        "strategy": strategy,
                        "contribution": "反事实策略贡献",
                    }
                    for index, strategy in enumerate(
                        context["required_strategies"], start=1
                    )
                ],
            }
        else:
            raise AssertionError(f"unexpected role: {role}")
        return LLMResult(
            text=json.dumps(payload, ensure_ascii=False), usage={"total_tokens": 10}
        )


def _top_level_changes(before, after):
    left = before.model_dump(mode="json")
    right = after.model_dump(mode="json")
    return {key for key in left if left[key] != right[key]}


@pytest.mark.parametrize("field", STATE_ANCHOR_FIELDS)
def test_mask_state_field_changes_only_the_assigned_field(
    history, app_config, field
):
    from ibd.interventions import mask_state_field

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-state", history)
    before = trace.state.model_dump_json()
    mutated, mutation = mask_state_field(trace.state, field)

    assert _top_level_changes(trace.state, mutated) == {field}
    assert getattr(mutated, field) == "<MASKED>"
    assert mutation.operation == "mask_state_field"
    assert mutation.field == field
    assert mutation.before == getattr(trace.state, field)
    assert mutation.after == "<MASKED>"
    restored = mutated.model_copy(update={field: mutation.before})
    assert restored.model_dump_json() == before


@pytest.mark.parametrize("field", STATE_ANCHOR_FIELDS)
def test_replace_state_field_changes_only_target_and_records_contrast(
    history, app_config, field
):
    from ibd.interventions import replace_state_field

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-replace", history)
    replacement = f"contrastive {field} value"
    mutated, mutation = replace_state_field(trace.state, field, replacement)

    assert _top_level_changes(trace.state, mutated) == {field}
    assert getattr(mutated, field) == replacement
    assert mutation.operation == "replace_state_field"
    assert mutation.field == field
    assert mutation.before == getattr(trace.state, field)
    assert mutation.after == replacement


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("  失落  ", "must differ"),
        ("<MASKED>", "must not use"),
        (" \n\t ", "must not be empty"),
        (" ".join(f"word{index}" for index in range(21)), "at most 20 words"),
    ],
)
def test_replace_state_field_rejects_invalid_contrasts(
    history, app_config, replacement, message
):
    from ibd.interventions import replace_state_field

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-invalid", history)

    with pytest.raises(ValueError, match=message):
        replace_state_field(trace.state, "emotion", replacement)


@pytest.mark.parametrize("used_count", [1, 2])
def test_counterfactual_plan_prefers_planned_unused_and_preserves_count(used_count):
    from ibd.interventions import select_counterfactual_plan

    planned = StrategyPlanSet(
        strategies=["Question", "Reflection of feelings", "Providing Suggestions"]
    )
    used = PlanSelection(strategies=planned.strategies[:used_count])
    selected, mutation = select_counterfactual_plan(
        planned=planned,
        used=used,
        example_id="e-plan",
        global_seed=17,
    )

    assert len(selected.strategies) == used_count
    assert selected.strategies[0] == planned.strategies[used_count]
    assert len(set(selected.strategies)) == used_count
    assert set(selected.strategies).isdisjoint(used.strategies)
    assert mutation.operation == "replace_plan_categories"
    assert mutation.before == used.strategies
    assert mutation.after == selected.strategies


def test_counterfactual_plan_fallback_is_deterministic_seeded_and_rng_local():
    from ibd.interventions import select_counterfactual_plan

    planned = StrategyPlanSet(
        strategies=["Question", "Reflection of feelings", "Providing Suggestions"]
    )
    used = PlanSelection(strategies=planned.strategies[:2])
    random.seed(991)
    expected_global = random.getstate()
    first, _ = select_counterfactual_plan(
        planned=planned, used=used, example_id="e-fallback", global_seed=1
    )
    repeat, _ = select_counterfactual_plan(
        planned=planned, used=used, example_id="e-fallback", global_seed=1
    )
    alternatives = {
        tuple(
            select_counterfactual_plan(
                planned=planned,
                used=used,
                example_id="e-fallback",
                global_seed=seed,
            )[0].strategies
        )
        for seed in range(12)
    }

    assert first == repeat
    assert len(alternatives) > 1
    assert len(first.strategies) == len(set(first.strategies)) == 2
    assert set(first.strategies).isdisjoint(used.strategies)
    assert random.getstate() == expected_global


def test_effect_verification_requires_reason_exactly_on_failure():
    from ibd.interventions import EffectVerification

    assert EffectVerification(passed=True).reason is None
    assert EffectVerification(
        passed=False, reason="no_localized_effect"
    ).reason == "no_localized_effect"
    with pytest.raises(ValueError, match="successful verification"):
        EffectVerification(passed=True, reason="bidirectional_disagreement")
    with pytest.raises(ValueError, match="failed verification"):
        EffectVerification(passed=False)


def test_intervention_builder_plan_reuses_candidates_without_planner_calls(
    history, app_config
):
    from ibd.interventions import EffectVerification, InterventionBuilder

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-build", history)
    downstream_backend = SelectedStrategyBackend()
    record = InterventionBuilder(
        TeacherRunner(downstream_backend, app_config),
        lambda full, changed, dimension: EffectVerification(passed=True),
        lambda full, changed: True,
    ).build(trace, "PLAN", global_seed=17)

    roles = [call["role"] for call in downstream_backend.calls]
    assert record.mutated_plan is not None
    assert "planner" not in roles
    assert sum(role.startswith("candidate_") for role in roles) <= 1


def test_intervention_builder_plan_uses_zero_candidate_calls_when_reusable(
    history, app_config
):
    from ibd.interventions import EffectVerification, InterventionBuilder

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-reuse", history)
    trace = trace.model_copy(
        update={
            "final_answer": trace.final_answer.model_copy(
                update={"strategy_uses": trace.final_answer.strategy_uses[:1]}
            )
        }
    )
    downstream_backend = SelectedStrategyBackend()

    InterventionBuilder(
        TeacherRunner(downstream_backend, app_config),
        lambda full, changed, dimension: EffectVerification(passed=True),
        lambda full, changed: True,
    ).build(trace, "PLAN", global_seed=17)

    roles = [call["role"] for call in downstream_backend.calls]
    assert "planner" not in roles
    assert not any(role.startswith("candidate_") for role in roles)


def test_intervention_builder_rejects_a_second_state_mask(history, app_config):
    from ibd.interventions import (
        EffectVerification,
        InterventionBuilder,
        InterventionExcluded,
    )

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-double-mask", history)
    trace = trace.model_copy(
        update={"state": trace.state.model_copy(update={"emotion": "<MASKED>"})}
    )
    builder = InterventionBuilder(
        TeacherRunner(ScriptedBackend(), app_config),
        lambda full, changed, dimension: EffectVerification(passed=True),
        lambda full, changed: True,
    )

    with pytest.raises(InterventionExcluded) as caught:
        builder.build(
            trace,
            "STATE",
            state_field="intensity",
            global_seed=17,
        )
    assert caught.value.reason == "state_not_single_field"


def test_intervention_builder_uses_one_field_qualified_contrastive_replacement(
    history, app_config
):
    from ibd.interventions import EffectVerification, InterventionBuilder

    class RecordingRunner(TeacherRunner):
        def __init__(self, backend, config):
            super().__init__(backend, config)
            self.counterfactual_ids = []

        def generate_state_counterfactual(
            self,
            history,
            state,
            target_field,
            target_dimension,
            *,
            example_id,
        ):
            self.counterfactual_ids.append(example_id)
            return "准备立即采取具体行动"

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-state-build", history)
    downstream_backend = ScriptedBackend()
    runner = RecordingRunner(downstream_backend, app_config)

    record = InterventionBuilder(
        runner,
        lambda full, changed, dimension: EffectVerification(passed=True),
        lambda full, changed: True,
    ).build(
        trace,
        "STATE",
        state_field="readiness",
        global_seed=17,
    )

    assert runner.counterfactual_ids == [
        "e-state-build:intervention:STATE:readiness:counterfactual"
    ]
    assert record.mutation.operation == "replace_state_field"
    assert record.mutation.before == trace.state.readiness
    assert record.mutation.after == "准备立即采取具体行动"
    assert record.mutated_state.readiness == "准备立即采取具体行动"
    assert "<MASKED>" not in record.mutated_state.model_dump(mode="json").values()


@pytest.mark.parametrize(
    "replacement",
    [
        "  愿意探索  ",
        "<MASKED>",
        " ",
        " ".join(f"word{index}" for index in range(21)),
    ],
)
def test_intervention_builder_excludes_invalid_state_counterfactual_before_downstream(
    history, app_config, replacement
):
    from ibd.interventions import (
        EffectVerification,
        InterventionBuilder,
        InterventionExcluded,
    )

    class InvalidCounterfactualBackend(ScriptedBackend):
        def _payload(self, role, seed):
            if role == "state_counterfactual_generator":
                return {"replacement": replacement}
            return super()._payload(role, seed)

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-invalid-cf", history)
    backend = InvalidCounterfactualBackend()
    builder = InterventionBuilder(
        TeacherRunner(backend, app_config),
        lambda full, changed, dimension: EffectVerification(passed=True),
        lambda full, changed: True,
    )

    with pytest.raises(InterventionExcluded) as caught:
        builder.build(
            trace,
            "STATE",
            state_field="readiness",
            global_seed=17,
        )

    assert caught.value.reason == "invalid_state_counterfactual"
    assert [call["role"] for call in backend.calls] == [
        "state_counterfactual_generator"
    ]


def test_intervention_builder_raises_stable_verifier_reason(history, app_config):
    from ibd.interventions import (
        EffectVerification,
        InterventionBuilder,
        InterventionExcluded,
    )

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-excluded", history)
    builder = InterventionBuilder(
        TeacherRunner(ScriptedBackend(), app_config),
        lambda full, changed, dimension: EffectVerification(
            passed=False, reason="bidirectional_disagreement"
        ),
        lambda full, changed: True,
    )

    with pytest.raises(InterventionExcluded) as caught:
        builder.build(trace, "STATE", state_field="emotion", global_seed=17)
    assert caught.value.reason == "bidirectional_disagreement"


def test_intervention_builder_rejects_safety_failure(history, app_config):
    from ibd.interventions import (
        EffectVerification,
        InterventionBuilder,
        InterventionExcluded,
    )

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-unsafe", history)
    safety = next(report for report in trace.critiques if report.critic == "safety")
    unsafe_safety = safety.model_copy(
        update={
            "candidate_issues": {
                **safety.candidate_issues,
                "1": [
                    CritiqueIssue(
                        dimension="boundary",
                        evidence="unsafe boundary crossing",
                        severity=5,
                    )
                ],
            }
        }
    )
    trace = trace.model_copy(
        update={
            "critiques": [
                unsafe_safety if report.critic == "safety" else report
                for report in trace.critiques
            ]
        }
    )
    builder = InterventionBuilder(
        TeacherRunner(ScriptedBackend(), app_config),
        lambda full, changed, dimension: EffectVerification(passed=True),
        lambda full, changed: True,
    )

    with pytest.raises(InterventionExcluded) as caught:
        builder.build(trace, "STATE", state_field="emotion", global_seed=17)
    assert caught.value.reason == "safety_failure"


def test_intervention_builder_rejects_unsafe_final_with_empty_candidate_critiques(
    history, app_config
):
    from ibd.interventions import (
        EffectVerification,
        InterventionBuilder,
        InterventionExcluded,
    )

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-final-unsafe", history)
    safety_calls = []

    def verify_final_safety(full, counterfactual):
        safety_calls.append((full, counterfactual))
        return "unsafe" not in full and "unsafe" not in counterfactual

    builder = InterventionBuilder(
        TeacherRunner(
            SelectedStrategyBackend(final_response="unsafe final output"),
            app_config,
        ),
        lambda full, changed, dimension: EffectVerification(passed=True),
        verify_final_safety,
    )

    with pytest.raises(InterventionExcluded) as caught:
        builder.build(trace, "PLAN", global_seed=17)
    assert caught.value.reason == "safety_failure"
    assert safety_calls == [(trace.final_response, "unsafe final output")]


def test_margin_pair_excludes_unsafe_candidate_and_verifies_both_orders(history, app_config):
    from ibd.interventions import MarginPairBuilder

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-margin", history)
    trace = trace.model_copy(
        update={
            "critiques": [
                CritiqueReport(
                    critic="emotion",
                    candidate_issues={
                        "1": [],
                        "2": [],
                        "3": [
                            CritiqueIssue(
                                dimension="emotion",
                                evidence="没有承接感受",
                                severity=5,
                            )
                        ],
                    },
                ),
                CritiqueReport(
                    critic="effectiveness",
                    candidate_issues={
                        "1": [],
                        "2": [
                            CritiqueIssue(
                                dimension="timing",
                                evidence="过早给出建议",
                                severity=4,
                            )
                        ],
                        "3": [],
                    },
                ),
                CritiqueReport(
                    critic="safety",
                    candidate_issues={
                        "1": [],
                        "2": [],
                        "3": [
                            CritiqueIssue(
                                dimension="boundary",
                                evidence="越过专业边界",
                                severity=5,
                            )
                        ],
                    },
                ),
            ]
        }
    )
    backend = PairBackend()
    pair = MarginPairBuilder(backend, app_config).build(trace)

    assert pair is not None
    assert pair.rejected_candidate_id == "2"
    assert pair.rejected_strategy_id == "S2"
    assert pair.rejected_strategy == "Question"
    assert pair.chosen_strategy_uses == trace.final_answer.strategy_uses
    assert pair.defect_dimension == "timing"
    verifier_calls = [call for call in backend.calls if call["role"] == "pair_verifier"]
    assert len(verifier_calls) == 2
    first = json.loads(verifier_calls[0]["messages"][1]["content"])["context"]
    second = json.loads(verifier_calls[1]["messages"][1]["content"])["context"]
    assert first["A"] == second["B"] == trace.final_response
    assert first["B"] == second["A"] == pair.rejected
