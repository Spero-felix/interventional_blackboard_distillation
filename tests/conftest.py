import json
from typing import Any

import pytest


class ScriptedBackend:
    """Network-free backend that emits complete production-shaped JSON."""

    def __init__(self, fail_first_gate: bool = False):
        self.fail_first_gate = fail_first_gate
        self.calls: list[dict[str, Any]] = []
        self._gate_calls = 0

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
        payload = self._payload(role, seed)
        return LLMResult(text=json.dumps(payload, ensure_ascii=False), usage={"total_tokens": 10})

    def _payload(self, role: str, seed: int | None):
        expert_payloads = {
            "emotion_expert": {
                "expert": "emotion",
                "fields": {"emotion": "失落", "intensity": "medium"},
                "evidence": ["最近总觉得被忽略"],
                "uncertainties": [],
            },
            "need_expert": {
                "expert": "need",
                "fields": {"need": "被理解", "readiness": "愿意表达"},
                "evidence": ["我想知道怎么开口"],
                "uncertainties": [],
            },
            "relationship_expert": {
                "expert": "relationship",
                "fields": {"relationship_type": "亲密关系", "relationship_pattern": "回避沟通"},
                "evidence": ["我们一谈就躲开"],
                "uncertainties": [],
            },
            "intent_expert": {
                "expert": "intent",
                "fields": {"intent": "准备一次沟通", "decision_stage": "探索"},
                "evidence": ["我想知道怎么开口"],
                "uncertainties": [],
            },
        }
        if role in expert_payloads:
            return expert_payloads[role]
        if role == "state_integrator":
            return {
                "emotions": {"primary": "失落"},
                "needs": {"primary": "被理解"},
                "relationship": {"type": "亲密关系", "pattern": "回避沟通"},
                "intent": {"goal": "准备一次沟通"},
                "readiness": "探索",
                "uncertainties": [],
                "evidence": ["最近总觉得被忽略", "我想知道怎么开口"],
            }
        if role == "planner":
            return {
                "support_goals": ["承接感受", "帮助准备沟通"],
                "response_acts": ["具体反映失落", "询问期待", "给出可选表达"],
                "avoid": ["替用户做决定"],
                "rationale": "先承接，再协助行动",
            }
        if role.startswith("candidate_"):
            return {
                "candidate_id": role.removeprefix("candidate_"),
                "response": f"候选回复-{role[-1]}",
                "seed": seed,
            }
        if role.endswith("_critic"):
            critic = role.removesuffix("_critic")
            return {
                "critic": critic,
                "candidate_issues": {"1": [], "2": [], "3": []},
                "summary": "完成独立检查",
            }
        if role in {"final_integrator", "repair"}:
            return {"response": "我能听出那种被忽略后的失落。你更希望先理清自己的期待，还是一起准备一句开场？"}
        if role in {"quality_gate", "quality_gate_recheck"}:
            self._gate_calls += 1
            reject = self.fail_first_gate and self._gate_calls == 1
            return {
                "accepted": not reject,
                "safety_pass": True,
                "defects": ["缺少选择空间"] if reject else [],
                "repair_instruction": "补充选择空间" if reject else "",
            }
        raise AssertionError(f"unexpected role: {role}")


@pytest.fixture
def history():
    from ibd.schemas import DialogueTurn, History

    return History(
        turns=[
            DialogueTurn(role="seeker", content="最近总觉得被忽略。"),
            DialogueTurn(role="supporter", content="你愿意多说一点吗？"),
            DialogueTurn(role="seeker", content="我们一谈就躲开，我想知道怎么开口。"),
        ]
    )


@pytest.fixture
def app_config():
    from ibd.config import AppConfig, ModelConfig

    return AppConfig(default_model=ModelConfig(model="fake-model"))

