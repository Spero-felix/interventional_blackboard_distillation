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

