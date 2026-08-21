import pytest


def test_retention_uses_teacher_gain_over_base_as_denominator():
    from ibd.evaluation import retention

    assert retention(base=2.0, teacher=4.0, student=3.5) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="Teacher gain"):
        retention(base=3.0, teacher=3.1, student=3.2)


def test_functional_fidelity_is_full_minus_ablation_for_state_and_plan():
    from ibd.evaluation import functional_fidelity_matrix

    matrix = functional_fidelity_matrix(
        full_scores={"emotion": 4.5, "timing": 4.0},
        ablated_scores={
            "STATE": {"emotion": 3.0, "timing": 3.8},
            "PLAN": {"emotion": 4.4, "timing": 2.5},
        },
    )

    assert matrix == {
        "STATE": {"emotion": pytest.approx(1.5), "timing": pytest.approx(0.2)},
        "PLAN": {"emotion": pytest.approx(0.1), "timing": pytest.approx(1.5)},
    }


def test_functional_matrix_rejects_critic_row():
    from ibd.evaluation import functional_fidelity_matrix

    with pytest.raises(ValueError, match="STATE and PLAN"):
        functional_fidelity_matrix(
            full_scores={"emotion": 4.0},
            ablated_scores={
                "STATE": {"emotion": 3.0},
                "PLAN": {"emotion": 3.5},
                "CRITIC": {"emotion": 2.0},
            },
        )


def test_matrix_alignment_reports_sign_spearman_and_normalized_l1():
    from ibd.evaluation import matrix_alignment

    teacher = {"STATE": {"a": 1.0, "b": 2.0}, "PLAN": {"a": 3.0, "b": 4.0}}
    student = {"STATE": {"a": 0.5, "b": 1.0}, "PLAN": {"a": 1.5, "b": 2.0}}

    result = matrix_alignment(teacher, student)

    assert result == {
        "sign_agreement": 1.0,
        "spearman": pytest.approx(1.0),
        "normalized_l1": pytest.approx(0.5),
    }


def test_pair_accuracy_treats_ties_as_incorrect_and_rank_gap_is_signed():
    from ibd.evaluation import mean_rank_gap, pair_accuracy

    chosen = [0.2, 0.1]
    rejected = [0.2, 0.0]

    assert pair_accuracy(chosen, rejected) == 0.5
    assert mean_rank_gap(chosen, rejected) == pytest.approx(0.05)


def test_causal_metrics_separate_functions_count_ties_as_errors_and_keep_empty_groups():
    from ibd.evaluation import aggregate_causal_metrics

    result = aggregate_causal_metrics(
        [
            {
                "function": "STATE",
                "original_clamp_original_score": 3.0,
                "original_clamp_counterfactual_score": 1.0,
                "counterfactual_clamp_original_score": 0.5,
                "counterfactual_clamp_counterfactual_score": 2.0,
                "non_target_slot_invariant": True,
            },
            {
                "function": "STATE",
                "original_clamp_original_score": 1.0,
                "original_clamp_counterfactual_score": 1.0,
                "counterfactual_clamp_original_score": 2.0,
                "counterfactual_clamp_counterfactual_score": 1.0,
                "non_target_slot_invariant": False,
            },
        ]
    )

    assert result == {
        "STATE": {
            "pairs": 2,
            "original_clamp_preference_accuracy": 0.5,
            "counterfactual_clamp_preference_accuracy": 0.5,
            "flip_consistency": 0.5,
            "non_target_slot_invariance_rate": 0.5,
        },
        "PLAN": {
            "pairs": 0,
            "original_clamp_preference_accuracy": None,
            "counterfactual_clamp_preference_accuracy": None,
            "flip_consistency": None,
            "non_target_slot_invariance_rate": None,
        },
    }


def test_intervention_audit_reports_fixed_state_coverage_and_plan_fallback_from_planner():
    from ibd.evaluation import aggregate_intervention_audit

    result = aggregate_intervention_audit(
        [
            {
                "example_id": "state-kept",
                "function": "STATE",
                "eligibility": "eligible",
                "status": "retained",
                "state_field": "emotion",
                "exclusion_reason": None,
            },
            {
                "example_id": "state-dropped",
                "function": "STATE",
                "eligibility": "ineligible",
                "status": "excluded",
                "state_field": "intensity",
                "exclusion_reason": "state_not_single_field",
            },
            {
                "example_id": "plan-preferred",
                "function": "PLAN",
                "eligibility": "eligible",
                "status": "retained",
                "counterfactual_plan_categories": ["Question"],
                "exclusion_reason": None,
            },
            {
                "example_id": "plan-fallback",
                "function": "PLAN",
                "eligibility": "eligible",
                "status": "excluded",
                "counterfactual_plan_categories": ["Information"],
                "exclusion_reason": "safety_failure",
            },
        ],
        planner_strategies={
            "plan-preferred": [
                "Question",
                "Reflection of feelings",
                "Providing Suggestions",
            ],
            "plan-fallback": [
                "Question",
                "Reflection of feelings",
                "Providing Suggestions",
            ],
        },
    )

    assert result["STATE"] == {
        "attempted": 2,
        "eligibility": {"eligible": 1, "ineligible": 1},
        "status": {"retained": 1, "excluded": 1},
        "exclusion_reasons": {"state_not_single_field": 1},
        "field_coverage": {
            "emotion": 1,
            "intensity": 1,
            "primary_need": 0,
            "support_goal": 0,
            "readiness": 0,
            "main_constraint": 0,
            "relationship_context": 0,
        },
    }
    assert result["PLAN"] == {
        "attempted": 2,
        "eligibility": {"eligible": 2, "ineligible": 0},
        "status": {"retained": 1, "excluded": 1},
        "exclusion_reasons": {"safety_failure": 1},
        "fallback_count": 1,
        "fallback_frequency": 0.5,
    }
