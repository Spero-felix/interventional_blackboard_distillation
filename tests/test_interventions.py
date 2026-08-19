import json

from conftest import ScriptedBackend

from ibd.schemas import CritiqueIssue, CritiqueReport
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


def _top_level_changes(before, after):
    left = before.model_dump(mode="json")
    right = after.model_dump(mode="json")
    return {key for key in left if left[key] != right[key]}


def test_state_mutation_downgrades_exactly_one_field(history, app_config):
    from ibd.interventions import mutate_state

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-state", history)
    mutated, mutation = mutate_state(trace.state)

    assert _top_level_changes(trace.state, mutated) == {"needs"}
    assert mutation.function == "STATE"
    assert mutation.field == "needs.primary"
    assert mutation.changed_field_count == 1


def test_plan_mutation_reorders_exactly_one_field(history, app_config):
    from ibd.interventions import mutate_plan

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-plan", history)
    mutated, mutation = mutate_plan(trace.plan)

    assert _top_level_changes(trace.plan, mutated) == {"response_acts"}
    assert mutated.response_acts[:2] == trace.plan.response_acts[1::-1]
    assert mutation.function == "PLAN"
    assert mutation.changed_field_count == 1


def test_intervention_builder_reruns_only_the_selected_downstream_path(history, app_config):
    from ibd.interventions import InterventionBuilder

    trace = TeacherRunner(ScriptedBackend(), app_config).run("e-build", history)
    downstream_backend = ScriptedBackend()
    runner = TeacherRunner(downstream_backend, app_config)
    record = InterventionBuilder(runner, lambda full, ablated, dimension: True).build(
        trace, "PLAN"
    )

    assert record is not None
    assert record.function == "PLAN"
    assert record.target_dimension == "timing"
    assert [call["role"] for call in downstream_backend.calls] == [
        "candidate_1",
        "candidate_2",
        "candidate_3",
        "emotion_critic",
        "effectiveness_critic",
        "safety_critic",
        "final_integrator",
        "quality_gate",
    ]


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
    assert pair.defect_dimension == "timing"
    verifier_calls = [call for call in backend.calls if call["role"] == "pair_verifier"]
    assert len(verifier_calls) == 2
    first = json.loads(verifier_calls[0]["messages"][1]["content"])["context"]
    second = json.loads(verifier_calls[1]["messages"][1]["content"])["context"]
    assert first["A"] == second["B"] == trace.final_response
    assert first["B"] == second["A"] == pair.rejected
