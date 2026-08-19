import json

import pytest

from conftest import ScriptedBackend

from ibd.teacher import NORMAL_ROLES, TeacherRunner


def test_normal_teacher_trace_has_exactly_fourteen_roles(history, app_config):
    backend = ScriptedBackend()
    trace = TeacherRunner(backend, app_config).run("e-1", history)

    assert tuple(call["role"] for call in backend.calls) == NORMAL_ROLES
    assert len(trace.call_records) == 14
    assert set(trace.expert_outputs) == {"emotion", "need", "relationship", "intent"}
    assert not any("risk" in call.role.lower() for call in trace.call_records)


def test_gate_failure_allows_only_one_repair_and_one_recheck(history, app_config):
    backend = ScriptedBackend(fail_first_gate=True)
    trace = TeacherRunner(backend, app_config).run("e-2", history)

    assert [call["role"] for call in backend.calls] == [*NORMAL_ROLES, "repair", "quality_gate_recheck"]
    assert len(trace.call_records) == 16
    assert trace.repaired_response == trace.final_response
    assert trace.quality_gate.accepted is True


def test_each_expert_prompt_sees_history_but_no_peer_output(history, app_config):
    backend = ScriptedBackend()
    TeacherRunner(backend, app_config).run("e-3", history)

    expert_calls = backend.calls[:4]
    prompts = {call["role"]: str(call["messages"]) for call in expert_calls}
    assert all("最近总觉得被忽略" in prompt for prompt in prompts.values())
    assert "relationship_pattern" not in prompts["emotion_expert"]
    assert "decision_stage" not in prompts["need_expert"]


def test_candidate_prompt_freezes_candidate_id_and_seed(history, app_config):
    backend = ScriptedBackend()
    TeacherRunner(backend, app_config).run("e-seed", history)

    candidate_calls = [call for call in backend.calls if call["role"].startswith("candidate_")]
    contexts = [json.loads(call["messages"][1]["content"])["context"] for call in candidate_calls]

    assert [(item["candidate_id"], item["seed"]) for item in contexts] == [
        ("1", 11),
        ("2", 29),
        ("3", 47),
    ]


def test_teacher_rejects_candidate_with_mismatched_frozen_metadata(history, app_config):
    class WrongSeedBackend(ScriptedBackend):
        def _payload(self, role, seed):
            payload = super()._payload(role, seed)
            if role == "candidate_1":
                payload["seed"] = 999
            return payload

    with pytest.raises(ValueError, match="candidate metadata"):
        TeacherRunner(WrongSeedBackend(), app_config).run("e-wrong-seed", history)
