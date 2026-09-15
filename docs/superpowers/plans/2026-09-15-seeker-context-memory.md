# Seeker Context Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在模型推理侧增加一个由 seeker 发言驱动、逐轮更新的结构化用户信息层，并严格按照 `Context → STATE → PLAN → response` 的顺序完成教师推理与审计记录。

**Architecture:** `UserContext` 只保存用户明确表达或可近似逐字复述的事实；Context Updater 根据上一轮 Context 和最近的 supporter/seeker 片段生成结构化 patch；确定性 reducer 合并 patch；State Analyzer 再结合完整历史和更新后的 Context 推断用户当前需求、建议接受意愿与行动能力；Planner 和响应生成模块使用 Context 作为事实背景、使用 STATE 作为当前决策依据。首版保持学生训练输入不变，不增加 Context token。

**Tech Stack:** Python 3.11、Pydantic 2、PyYAML、pytest、现有 `StructuredCaller` 与 OpenAI-compatible backend。

## Global Constraints

- 以 [设计文档](/homeb/wangnianxiang/interventional_blackboard_distillation/docs/superpowers/specs/2026-09-15-seeker-context-memory-design.md) 为语义依据；若实现细节与设计冲突，先修订计划或设计，不静默改变语义。
- Context 的九个字段全部必填，运行时条目只能是非空字符串列表；不引入 `kind`、`basis`、`source_turn`、`first_observed_turn`、`last_confirmed_turn`、`status` 或记忆 ID。
- Context 只由最新 seeker 发言触发更新。supporter 的回复本身不得改变 Context 或 STATE。
- Context 只存事实。对“是否愿意听建议、是否准备行动、是否有能力行动”的判断留在 STATE，不能写入 Context。
- 最新 seeker 明确表达的内容优先于持久 Context；持久偏好不能覆盖用户当轮的拒绝、犹豫或边界。
- 首版不修改 `student_data.py`、`model.py`、训练目标或锚点，不向学生暴露 Context，不新增 Context token。
- 当前 SocialSim 预处理结果每个 conversation 只选一个 target，因此不能据此声称完成了跨多轮持续记忆评估。本计划只提供可复用的在线 session 能力，以及通过输入 `context_before` 继续上下文的离线能力；多 target 轨迹数据生成需另立计划。
- 将 teacher protocol 改为 `qwen25-socialsim-seven-state-context-v1`，避免复用旧协议缓存。
- 保留用户现有未跟踪文件 `result/balanced-2500-quality/model_strategy_heatmap.svg`，不纳入任何提交。
- 每个任务都遵循红—绿—重构循环；只在对应测试通过后提交。

---

### Task 1: 定义 Context、Patch 与 Trace 数据契约

**Files:**

- Modify: `src/ibd/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: 为九字段 Context 和完整 Patch 写失败测试**

在 `tests/test_schemas.py` 增加以下测试与辅助函数：

```python
import pytest
from pydantic import ValidationError

from ibd.schemas import (
    CONTEXT_FIELDS,
    ContextPatch,
    ContextReplacements,
    ReplacePair,
    UserContext,
)


def empty_context_payload() -> dict[str, list[str]]:
    return {field: [] for field in CONTEXT_FIELDS}


def test_user_context_requires_all_nine_string_list_fields() -> None:
    payload = empty_context_payload()
    payload["active_concerns"] = ["担心明天的答辩会再次卡住"]

    context = UserContext.model_validate(payload)

    assert tuple(context.model_dump()) == CONTEXT_FIELDS
    assert context.active_concerns == ["担心明天的答辩会再次卡住"]


def test_user_context_rejects_missing_extra_and_blank_values() -> None:
    missing = empty_context_payload()
    missing.pop("communication_preferences")
    with pytest.raises(ValidationError):
        UserContext.model_validate(missing)

    extra = empty_context_payload() | {"readiness": []}
    with pytest.raises(ValidationError):
        UserContext.model_validate(extra)

    blank = empty_context_payload()
    blank["active_concerns"] = ["   "]
    with pytest.raises(ValidationError):
        UserContext.model_validate(blank)


def test_context_patch_requires_complete_add_replace_remove_sections() -> None:
    patch = ContextPatch.empty()

    assert patch.add == UserContext.empty()
    assert patch.remove == UserContext.empty()
    assert patch.replace == ContextReplacements.empty()

    with pytest.raises(ValidationError):
        ContextPatch.model_validate(
            {
                "add": empty_context_payload(),
                "remove": empty_context_payload(),
            }
        )


def test_replace_pair_rejects_identical_old_and_new_values() -> None:
    with pytest.raises(ValidationError):
        ReplacePair(old="我不想听建议", new="我不想听建议")


```

`TeacherTrace` 的向后兼容行为在 Task 4 使用真实 runner 生成的合法 trace 测试，避免在 schema 单测里复制复杂的候选回复与最终选择 fixture。

- [ ] **Step 2: 运行 schema 测试并确认因新类型不存在而失败**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_schemas.py -q
```

Expected: collection 阶段因无法导入 `CONTEXT_FIELDS`、`UserContext` 或 `ContextPatch` 而失败。

- [ ] **Step 3: 在 `schemas.py` 实现严格数据模型**

在现有 typing 和 Pydantic import 中加入 `StringConstraints`，并在 `StrictModel` 之后定义：

```python
ContextText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

CONTEXT_FIELDS = (
    "active_concerns",
    "events_and_triggers",
    "functional_impacts",
    "goals_and_priorities",
    "people_and_relationships",
    "constraints_and_resources",
    "coping_attempts_and_outcomes",
    "support_preferences_and_boundaries",
    "communication_preferences",
)


class UserContext(StrictModel):
    active_concerns: list[ContextText]
    events_and_triggers: list[ContextText]
    functional_impacts: list[ContextText]
    goals_and_priorities: list[ContextText]
    people_and_relationships: list[ContextText]
    constraints_and_resources: list[ContextText]
    coping_attempts_and_outcomes: list[ContextText]
    support_preferences_and_boundaries: list[ContextText]
    communication_preferences: list[ContextText]

    @classmethod
    def empty(cls) -> "UserContext":
        return cls.model_validate({field: [] for field in CONTEXT_FIELDS})


class ReplacePair(StrictModel):
    old: ContextText
    new: ContextText

    @model_validator(mode="after")
    def values_must_differ(self) -> "ReplacePair":
        if self.old == self.new:
            raise ValueError("replace.old and replace.new must differ")
        return self


class ContextReplacements(StrictModel):
    active_concerns: list[ReplacePair]
    events_and_triggers: list[ReplacePair]
    functional_impacts: list[ReplacePair]
    goals_and_priorities: list[ReplacePair]
    people_and_relationships: list[ReplacePair]
    constraints_and_resources: list[ReplacePair]
    coping_attempts_and_outcomes: list[ReplacePair]
    support_preferences_and_boundaries: list[ReplacePair]
    communication_preferences: list[ReplacePair]

    @classmethod
    def empty(cls) -> "ContextReplacements":
        return cls.model_validate({field: [] for field in CONTEXT_FIELDS})


class ContextPatch(StrictModel):
    add: UserContext
    replace: ContextReplacements
    remove: UserContext

    @classmethod
    def empty(cls) -> "ContextPatch":
        return cls(
            add=UserContext.empty(),
            replace=ContextReplacements.empty(),
            remove=UserContext.empty(),
        )
```

给 `TeacherTrace` 增加向后兼容的审计字段：

```python
context_before: UserContext = Field(default_factory=UserContext.empty)
context_patch: ContextPatch = Field(default_factory=ContextPatch.empty)
context_after: UserContext = Field(default_factory=UserContext.empty)
context_merge_errors: list[str] = Field(default_factory=list)
```

这些字段放在 `history` 后、`state_analysis` 前，使 trace 的阅读顺序与运行顺序一致。

- [ ] **Step 4: 运行 schema 测试并确认通过**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_schemas.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交数据契约**

```bash
git add src/ibd/schemas.py tests/test_schemas.py
git commit -m "feat: add seeker context schemas"
```

---

### Task 2: 实现确定性 Context reducer

**Files:**

- Create: `src/ibd/context_memory.py`
- Create: `tests/test_context_memory.py`

- [ ] **Step 1: 为顺序、幂等、无效操作和冲突写失败测试**

创建 `tests/test_context_memory.py`：

```python
from ibd.context_memory import apply_context_patch
from ibd.schemas import ContextPatch, ReplacePair, UserContext


def context_with(**updates: list[str]) -> UserContext:
    payload = UserContext.empty().model_dump()
    payload.update(updates)
    return UserContext.model_validate(payload)


def patch_with(
    *,
    add: dict[str, list[str]] | None = None,
    replace: dict[str, list[dict[str, str]]] | None = None,
    remove: dict[str, list[str]] | None = None,
) -> ContextPatch:
    payload = ContextPatch.empty().model_dump()
    for section, updates in (
        ("add", add),
        ("replace", replace),
        ("remove", remove),
    ):
        if updates is not None:
            payload[section].update(updates)
    return ContextPatch.model_validate(payload)


def test_reducer_applies_replace_remove_add_and_preserves_order() -> None:
    previous = context_with(
        active_concerns=["担心答辩", "担心失眠"],
        goals_and_priorities=["先把论文交上去"],
    )
    patch = patch_with(
        add={"active_concerns": ["担心导师失望", "担心导师失望"]},
        replace={
            "active_concerns": [
                ReplacePair(old="担心答辩", new="担心明天答辩卡住").model_dump()
            ]
        },
        remove={"active_concerns": ["担心失眠"]},
    )

    result = apply_context_patch(previous, patch)

    assert result.context.active_concerns == [
        "担心明天答辩卡住",
        "担心导师失望",
    ]
    assert result.context.goals_and_priorities == ["先把论文交上去"]
    assert result.errors == ()


def test_reducer_treats_duplicate_add_and_absent_remove_as_no_op() -> None:
    previous = context_with(active_concerns=["担心答辩"])
    patch = patch_with(
        add={"active_concerns": ["担心答辩"]},
        remove={"active_concerns": ["不存在的事实"]},
    )

    result = apply_context_patch(previous, patch)

    assert result.context.active_concerns == ["担心答辩"]
    assert result.errors == ()


def test_reducer_skips_replace_when_old_value_is_absent() -> None:
    previous = context_with(active_concerns=["担心答辩"])
    patch = patch_with(
        replace={
            "active_concerns": [
                {"old": "担心面试", "new": "担心明天面试"}
            ]
        }
    )

    result = apply_context_patch(previous, patch)

    assert result.context == previous
    assert result.errors == (
        "active_concerns: replace.old not found: 担心面试",
    )


def test_reducer_rejects_same_fact_replace_remove_and_add_remove_conflicts() -> None:
    previous = context_with(active_concerns=["担心答辩"])
    patch = patch_with(
        add={"active_concerns": ["不想听建议"]},
        replace={
            "active_concerns": [
                {"old": "担心答辩", "new": "担心明天答辩"}
            ]
        },
        remove={"active_concerns": ["担心答辩", "不想听建议"]},
    )

    result = apply_context_patch(previous, patch)

    assert result.context.active_concerns == ["担心答辩"]
    assert result.errors == (
        "active_concerns: replace/remove conflict: 担心答辩",
        "active_concerns: add/remove conflict: 不想听建议",
    )
```

- [ ] **Step 2: 运行 reducer 测试并确认模块缺失**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_context_memory.py -q
```

Expected: collection 阶段报 `ModuleNotFoundError: ibd.context_memory`。

- [ ] **Step 3: 实现纯函数 reducer 与可审计错误**

创建 `src/ibd/context_memory.py`：

```python
from dataclasses import dataclass

from ibd.schemas import CONTEXT_FIELDS, ContextPatch, UserContext


@dataclass(frozen=True)
class ContextMergeResult:
    context: UserContext
    errors: tuple[str, ...]


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def apply_context_patch(
    previous: UserContext,
    patch: ContextPatch,
) -> ContextMergeResult:
    merged = previous.model_dump()
    errors: list[str] = []

    for field in CONTEXT_FIELDS:
        values = list(getattr(previous, field))
        additions = list(getattr(patch.add, field))
        replacements = list(getattr(patch.replace, field))
        removals = list(getattr(patch.remove, field))
        removal_set = set(removals)

        replace_remove_conflicts = {
            pair.old for pair in replacements if pair.old in removal_set
        }
        add_remove_conflicts = {
            value for value in additions if value in removal_set
        }

        for pair in replacements:
            if pair.old in replace_remove_conflicts:
                errors.append(
                    f"{field}: replace/remove conflict: {pair.old}"
                )
                continue
            if pair.old not in values:
                errors.append(
                    f"{field}: replace.old not found: {pair.old}"
                )
                continue
            values[values.index(pair.old)] = pair.new

        for value in removals:
            if value in replace_remove_conflicts:
                continue
            if value in add_remove_conflicts:
                errors.append(f"{field}: add/remove conflict: {value}")
                continue
            if value in values:
                values.remove(value)

        for value in additions:
            if value in add_remove_conflicts:
                continue
            if value not in values:
                values.append(value)

        merged[field] = _deduplicate(values)

    return ContextMergeResult(
        context=UserContext.model_validate(merged),
        errors=tuple(errors),
    )
```

结构性错误在 `ContextPatch.model_validate` 阶段拒绝整个 patch；这个 reducer 只处理已经通过 schema 的 patch，并逐项跳过语义错误。

- [ ] **Step 4: 运行 reducer 与 schema 测试**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_context_memory.py tests/test_schemas.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交 reducer**

```bash
git add src/ibd/context_memory.py tests/test_context_memory.py
git commit -m "feat: add deterministic context reducer"
```

---

### Task 3: 增加 Context Updater prompt 并升级协议配置

**Files:**

- Modify: `src/ibd/prompting.py`
- Modify: `configs/deepseek_teacher.yaml`
- Modify: `tests/conftest.py`
- Modify: `tests/test_teacher.py`

- [ ] **Step 1: 为 prompt 注册表、输入窗口和 Context 使用边界写失败测试**

在 `tests/test_teacher.py` 更新原有角色集合断言，并增加：

```python
import json

from ibd.prompting import PROMPT_ROLES, build_messages
from ibd.schemas import ContextPatch, History, UserContext


def test_prompt_registry_includes_context_updater() -> None:
    assert PROMPT_ROLES == {
        "context_updater",
        "multi_view_state_analyzer",
        "planner",
        "candidate",
        "final_selector",
    }


def test_context_updater_prompt_receives_previous_context_and_recent_turns() -> None:
    recent = History.model_validate(
        {
            "turns": [
                {"role": "supporter", "content": "你愿意说说最担心什么吗？"},
                {"role": "seeker", "content": "我只想先把今晚熬过去，不想听长期建议。"},
            ]
        }
    )
    messages = build_messages(
        role="context_updater",
        history=recent,
        schema=ContextPatch,
        context={"previous_context": UserContext.empty()},
    )
    payload = json.loads(messages[1]["content"])

    assert payload["history"] == recent.model_dump()
    assert payload["context"]["previous_context"] == UserContext.empty().model_dump()
    assert "Only the latest seeker turn may change User Context" in messages[0]["content"]
    assert "Do not infer willingness, readiness, diagnosis, or personality" in messages[0]["content"]


def test_state_and_response_roles_receive_same_context_policy(history) -> None:
    user_context = UserContext.empty()
    for role, schema, context in (
        (
            "multi_view_state_analyzer",
            MultiViewStateAnalysis,
            {"user_context": user_context},
        ),
        (
            "planner",
            StrategyPlanSet,
            {"user_context": user_context, "state": VALID_STATE},
        ),
        (
            "candidate",
            Candidate,
            {
                "user_context": user_context,
                "state": VALID_STATE,
                "candidate_id": "1",
                "strategy_id": "S1",
                "strategy": "Question",
            },
        ),
        (
            "final_selector",
            FinalSelectionDecision,
            {
                "user_context": user_context,
                "state": VALID_STATE,
                "candidates": [],
            },
        ),
    ):
        messages = build_messages(
            role=role,
            history=history,
            schema=schema,
            context=context,
        )
        system = messages[0]["content"]
        assert "User Context is factual background" in system
        assert "the latest explicit seeker message takes priority" in system
```

同时修改 `tests/conftest.py` 中 `ScriptedBackend._payload`：

```python
if role == "context_updater":
    return ContextPatch.empty().model_dump()
```

并补充 `ContextPatch` import，使既有 teacher 测试在角色增加后有合法返回值。

- [ ] **Step 2: 运行 prompt 测试并确认失败**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q
```

Expected: `context_updater` 未注册，或 prompt 中缺失 Context 使用策略。

- [ ] **Step 3: 在 prompt registry 中加入 Updater 与统一 Context 策略**

在 `src/ibd/prompting.py` 增加：

```python
_CONTEXT_UPDATER_PROMPT = _prompt(
    "Context Updater",
    """
Available Inputs:
- Previous User Context with exactly nine fields.
- A recent dialogue window ending in the latest seeker turn.

Task:
Return a ContextPatch with complete add, replace, and remove sections.

Rules:
- Only the latest seeker turn may change User Context.
- Store only facts explicitly stated by the seeker or facts that are near-literal paraphrases.
- Do not copy supporter claims into User Context.
- Do not infer willingness, readiness, diagnosis, personality, demographics, motives, or hidden causes.
- Use add for new facts, replace only when a current fact explicitly corrects an old fact, and remove only when the seeker explicitly retracts or invalidates an old fact.
- If there is no justified update, return empty lists in every field of every section.
""",
)

_CONTEXT_USAGE_POLICY = """
User Context is factual background, not a diagnosis or a current-state label.
Use it only when it is relevant to the current decision.
For willingness, readiness, boundaries, and immediate needs, the latest explicit seeker message takes priority over persistent User Context.
"""

_CONTEXT_CONSUMING_ROLES = {
    "multi_view_state_analyzer",
    "planner",
    "candidate",
    "final_selector",
}
```

将 `context_updater` 加入 `_ROLE_PROMPTS` 和 `PROMPT_ROLES`。在 `_resolved_prompt` 中只为 `_CONTEXT_CONSUMING_ROLES` 追加 `_CONTEXT_USAGE_POLICY`；Updater 使用自己的更严格规则，不追加决策策略。`_STATE_CONSUMING_ROLES` 保持现有职责，不把 Updater 加入其中。

- [ ] **Step 4: 升级 teacher 配置协议并配置轻量 Updater 角色**

在 `configs/deepseek_teacher.yaml` 中写入：

```yaml
protocol_version: qwen25-socialsim-seven-state-context-v1
roles:
  context_updater:
    model: deepseek-v4-flash
    temperature: 0.0
    max_tokens: 1800
    provider_json_mode: true
    thinking_enabled: false
```

保留 `multi_view_state_analyzer`、candidate、final_selector 的现有覆盖项；planner 继续使用 default 配置。Updater 使用零温度以降低 patch 抖动。

同时把 `test_teacher_configs_use_synthetic_self_disclosure_protocol` 重命名为 `test_teacher_config_uses_seeker_context_protocol`，并将期望值改为 `qwen25-socialsim-seven-state-context-v1`。

- [ ] **Step 5: 运行 prompt 测试并确认通过**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q
```

Expected: PASS。

- [ ] **Step 6: 提交 prompt 与协议变更**

```bash
git add src/ibd/prompting.py configs/deepseek_teacher.yaml tests/conftest.py tests/test_teacher.py
git commit -m "feat: add context updater prompt contract"
```

---

### Task 4: 串联 `Context → STATE → PLAN → response` 并提供会话级状态

**Files:**

- Modify: `src/ibd/teacher.py`
- Modify: `tests/test_teacher.py`

- [ ] **Step 1: 为完整角色顺序和 Context 传播写失败测试**

在 `tests/test_teacher.py` 增加一个会记录每个角色 user payload 的 backend，并验证：

```python
def test_teacher_updates_context_before_state_and_passes_it_downstream(
    history,
    app_config,
) -> None:
    patch_payload = ContextPatch.empty().model_dump()
    patch_payload["add"]["support_preferences_and_boundaries"] = [
        "当前不想听长期建议"
    ]
    backend = ContextAwareScriptedBackend(
        updater_patch=ContextPatch.model_validate(patch_payload)
    )
    runner = TeacherRunner(backend, app_config)

    trace = runner.run("example-1", history, split="train")

    assert [call["role"] for call in backend.calls] == list(NORMAL_ROLES)
    assert trace.context_before == UserContext.empty()
    assert trace.context_after.support_preferences_and_boundaries == [
        "当前不想听长期建议"
    ]
    downstream_payloads = [
        json.loads(call["messages"][1]["content"])
        for call in backend.calls
        if call["role"] in {
            "multi_view_state_analyzer",
            "planner",
            "candidate",
            "final_selector",
        }
    ]
    assert len(downstream_payloads) == 6
    assert all(
        payload["context"]["user_context"] == trace.context_after.model_dump()
        for payload in downstream_payloads
    )
```

测试 fixture 中的期望调用顺序改为：

```python
NORMAL_ROLES = (
    "context_updater",
    "multi_view_state_analyzer",
    "planner",
    "candidate",
    "candidate",
    "candidate",
    "final_selector",
)
```

测试中的 `ContextAwareScriptedBackend` 使用以下可执行实现：

```python
class ContextAwareScriptedBackend(ScriptedBackend):
    def __init__(self, updater_patch: ContextPatch) -> None:
        super().__init__()
        self.updater_patch = updater_patch

    def _payload(self, role: str, seed: int | None):
        if role == "context_updater":
            return self.updater_patch.model_dump(mode="json")
        return super()._payload(role, seed)

    def user_payloads(self, role: str) -> list[dict[str, object]]:
        return [
            json.loads(call["messages"][1]["content"])
            for call in self.calls
            if call["role"] == role
        ]
```

- [ ] **Step 2: 为最近两轮窗口写失败测试**

```python
def test_context_updater_sees_only_last_supporter_and_latest_seeker_turn(
    history,
    app_config,
) -> None:
    backend = ContextAwareScriptedBackend(updater_patch=ContextPatch.empty())
    runner = TeacherRunner(backend, app_config)

    runner.run("example-1", history, split="train")

    updater_history = backend.user_payloads("context_updater")[0]["history"]["turns"]
    assert updater_history == history.model_dump()["turns"][-2:]
    assert updater_history[-1]["role"] == "seeker"


def test_first_seeker_turn_is_a_valid_context_update_window(app_config) -> None:
    history = History.model_validate(
        {"turns": [{"role": "seeker", "content": "我今天完全睡不着。"}]}
    )
    backend = ContextAwareScriptedBackend(updater_patch=ContextPatch.empty())
    runner = TeacherRunner(backend, app_config)

    runner.run("example-1", history, split="train")

    assert backend.user_payloads("context_updater")[0]["history"] == history.model_dump()
```

- [ ] **Step 3: 为 Updater 失败降级写失败测试**

```python
def test_context_updater_failure_preserves_previous_context_and_continues(
    history,
    app_config,
) -> None:
    previous_payload = UserContext.empty().model_dump()
    previous_payload["active_concerns"] = ["担心答辩"]
    previous = UserContext.model_validate(previous_payload)
    backend = InvalidUpdaterThenValidTeacherBackend()
    runner = TeacherRunner(backend, app_config)

    trace = runner.run(
        "example-1",
        history,
        split="train",
        context_before=previous,
    )

    assert trace.context_patch == ContextPatch.empty()
    assert trace.context_after == previous
    assert trace.context_merge_errors == [
        "context_updater_failed: context_updater failed schema validation"
    ]
    assert trace.final_response
```

该 backend 对 Updater 的两次结构化调用都返回非法 JSON，对之后角色返回合法 fixture，从而覆盖 `StructuredCaller` 重试耗尽后的降级路径。

测试 backend 使用以下实现，保证非法结果确实经过两次 schema 尝试：

```python
class InvalidUpdaterThenValidTeacherBackend(ScriptedBackend):
    def complete(
        self,
        *,
        role,
        messages,
        model_config,
        json_mode=True,
        seed=None,
    ):
        if role != "context_updater":
            return super().complete(
                role=role,
                messages=messages,
                model_config=model_config,
                json_mode=json_mode,
                seed=seed,
            )
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "model": model_config.model,
                "json_mode": json_mode,
                "seed": seed,
            }
        )
        return LLMResult(text="{", usage={"total_tokens": 1})
```

- [ ] **Step 4: 为会话级串行积累写失败测试**

```python
def test_teacher_session_carries_context_between_seeker_turns(app_config) -> None:
    backend = SequentialContextBackend()
    session = TeacherSession(TeacherRunner(backend, app_config))
    first_history = History.model_validate(
        {"turns": [{"role": "seeker", "content": "我有点担心答辩。"}]}
    )
    second_history = History.model_validate(
        {
            "turns": [
                {"role": "seeker", "content": "我有点担心答辩。"},
                {"role": "supporter", "content": "最担心哪个部分？"},
                {"role": "seeker", "content": "更准确地说，我担心明天答辩时卡住。"},
            ]
        }
    )

    first = session.run("turn-1", first_history, split="train")
    second = session.run("turn-2", second_history, split="train")

    assert first.context_before == UserContext.empty()
    assert second.context_before == first.context_after
    assert session.context == second.context_after
    assert backend.user_payloads("context_updater")[1]["context"][
        "previous_context"
    ] == first.context_after.model_dump()
```

`SequentialContextBackend` 使用以下实现，第一轮 add “担心答辩”，第二轮用 exact-match replace 改为“担心明天答辩卡住”：

```python
class SequentialContextBackend(ContextAwareScriptedBackend):
    def __init__(self) -> None:
        super().__init__(ContextPatch.empty())
        self.update_index = 0

    def _payload(self, role: str, seed: int | None):
        if role != "context_updater":
            return ScriptedBackend._payload(self, role, seed)

        self.update_index += 1
        payload = ContextPatch.empty().model_dump(mode="json")
        if self.update_index == 1:
            payload["add"]["active_concerns"] = ["担心答辩"]
        else:
            payload["replace"]["active_concerns"] = [
                {"old": "担心答辩", "new": "担心明天答辩卡住"}
            ]
        return payload
```

再增加旧 trace 兼容测试：先用 runner 得到合法 trace，删除四个 Context 审计键后重新校验。

```python
def test_legacy_teacher_trace_defaults_to_empty_context_audit(
    history,
    app_config,
) -> None:
    payload = TeacherRunner(ScriptedBackend(), app_config).run(
        "legacy", history
    ).model_dump(mode="json")
    for field in (
        "context_before",
        "context_patch",
        "context_after",
        "context_merge_errors",
    ):
        payload.pop(field)

    trace = TeacherTrace.model_validate(payload)

    assert trace.context_before == UserContext.empty()
    assert trace.context_patch == ContextPatch.empty()
    assert trace.context_after == UserContext.empty()
    assert trace.context_merge_errors == []
```

- [ ] **Step 5: 运行新增 teacher 测试并确认失败**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q
```

Expected: `TeacherRunner.run` 尚不接受 `context_before`、缺少 `TeacherSession`，且调用顺序尚未包含 Updater。

- [ ] **Step 6: 实现最近窗口、失败降级和 Context 合并**

在 `src/ibd/teacher.py` 导入 `apply_context_patch`、`ContextPatch`、`UserContext`，增加：

```python
def _recent_context_history(history: History) -> History:
    return History(turns=history.turns[-2:])
```

修改 `TeacherRunner.run` 签名与首段流程：

```python
def run(
    self,
    example_id: str,
    history: History,
    *,
    split: Literal["train", "dev", "test", "diagnostic_holdout"] = "train",
    context_before: UserContext | None = None,
) -> TeacherTrace:
    before = context_before or UserContext.empty()
    records: list[CallRecord] = []
    merge_errors: list[str] = []

    try:
        patch = self._call(
            "context_updater",
            _recent_context_history(history),
            ContextPatch,
            records,
            example_id=example_id,
            context={"previous_context": before},
        )
    except ValueError as exc:
        patch = ContextPatch.empty()
        merge_errors.append(f"context_updater_failed: {exc}")

    merge_result = apply_context_patch(before, patch)
    after = merge_result.context
    merge_errors.extend(merge_result.errors)

    state_analysis = self._call(
        "multi_view_state_analyzer",
        history,
        MultiViewStateAnalysis,
        records,
        example_id=example_id,
        context={"user_context": after},
    )
```

然后调用 `_run_from_state`，并在 `TeacherTrace` 中显式写入：

```python
context_before=before,
context_patch=patch,
context_after=after,
context_merge_errors=merge_errors,
```

如果 Updater 失败，不中止整个教师流水线；旧 Context 原样进入 State Analyzer。结构化调用失败信息记录在 `context_merge_errors`，不把它混入用户事实。

- [ ] **Step 7: 将更新后的 Context 传给所有下游角色**

给 `_run_from_state`、`_generate_candidate` 和 `_run_final_selector` 增加 `user_context: UserContext` 参数。各角色使用以下精确输入：

```python
analyzer_context = {"user_context": user_context}
planner_context = {
    "user_context": user_context,
    "state": state,
}
candidate_context = {
    "user_context": user_context,
    "state": state,
    "candidate_id": candidate_id,
    "strategy_id": strategy_id,
    "strategy": strategy,
}
selector_context = {
    "user_context": user_context,
    "state": state,
    "candidates": candidates,
}
```

实际调用时分别传入对应字典。不得从任何下游模型返回值反向修改 Context。

- [ ] **Step 8: 增加 `TeacherSession` 作为模型推理侧状态容器**

在 `src/ibd/teacher.py` 增加：

```python
class TeacherSession:
    def __init__(
        self,
        runner: TeacherRunner,
        *,
        context: UserContext | None = None,
    ) -> None:
        self.runner = runner
        self.context = context or UserContext.empty()

    def run(
        self,
        example_id: str,
        history: History,
        *,
        split: Literal["train", "dev", "test", "diagnostic_holdout"] = "train",
    ) -> TeacherTrace:
        trace = self.runner.run(
            example_id,
            history,
            split=split,
            context_before=self.context,
        )
        self.context = trace.context_after
        return trace
```

`TeacherSession` 只持有 Context。STATE 和 PLAN 每轮重新计算，避免把瞬时判断错误地固化为用户属性。

- [ ] **Step 9: 运行 teacher 测试并确认通过**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_teacher.py -q
```

Expected: PASS；所有正常路径角色顺序以 `context_updater` 开始，三个 candidate 仍按现有策略生成。

- [ ] **Step 10: 提交串行教师流水线**

```bash
git add src/ibd/teacher.py tests/test_teacher.py
git commit -m "feat: run teacher from context to response"
```

---

### Task 5: 接入 CLI 审计输入并保护学生导出边界

**Files:**

- Modify: `src/ibd/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_export.py`
- Modify: `README.md`

- [ ] **Step 1: 为可选 `context_before` 离线输入写失败 CLI 测试**

在 `tests/test_cli.py` 的 run-teacher 输入记录中增加完整 Context：

```python
context_payload = UserContext.empty().model_dump(mode="json")
context_payload["active_concerns"] = ["担心答辩"]
record["context_before"] = context_payload
```

运行 CLI 后读取输出 trace，并断言：

```python
assert output["context_before"]["active_concerns"] == ["担心答辩"]
assert output["context_after"]["active_concerns"] == ["担心答辩"]
assert output["context_patch"] == ContextPatch.empty().model_dump()
assert output["context_merge_errors"] == []
```

保留另一个不含 `context_before` 的既有测试，断言它从 `UserContext.empty()` 开始，以确保旧输入文件仍可运行。

- [ ] **Step 2: 运行 CLI 测试并确认自定义 Context 未被读取**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_cli.py -q
```

Expected: 输出 `context_before` 为空，而不是输入记录提供的值。

- [ ] **Step 3: 在 CLI 中解析并传入可选 Context**

在 `src/ibd/cli.py` 导入 `UserContext`，在每条记录调用 runner 前增加：

```python
context_before = (
    UserContext.model_validate(record["context_before"])
    if "context_before" in record
    else None
)
```

调用改为：

```python
trace = runner.run(
    str(record["example_id"]),
    History.model_validate(record["history"]),
    split=record.get("split", "train"),
    context_before=context_before,
)
```

不要在 CLI 内按 `conversation_id` 暗中缓存 Context。离线数据若要跨记录连续更新，必须显式把上一条 `context_after` 写入下一条 `context_before`，这样乱序、resume 和并行执行都不会产生隐式状态错误。

- [ ] **Step 4: 为学生导出不泄漏 Context 写回归测试**

在 `tests/test_export.py` 对三种学生可见导出保留精确键集合断言：

```python
assert set(sft_row) == {
    "example_id",
    "prompt",
    "response",
    "selected_strategy",
}
assert set(slot_row) == {"example_id", "prompt", "state", "plan"}
assert set(visible_row) == {"example_id", "split", "history", "response"}

for row in (sft_row, slot_row, visible_row):
    assert "context_before" not in row
    assert "context_patch" not in row
    assert "context_after" not in row
    assert "context_merge_errors" not in row
```

这些集合与 `src/ibd/export.py` 当前正式 allowlist 一致；实现时不得为迎合测试而改变导出生产代码。

- [ ] **Step 5: 运行 CLI 与 export 测试并确认通过**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_cli.py tests/test_export.py -q
```

Expected: PASS。

- [ ] **Step 6: 在 README 记录模型推理侧使用方式与数据边界**

在 `README.md` 增加“Seeker Context Memory”小节，明确：

```text
每轮处理顺序为 Context → STATE → PLAN → response。在线多轮推理使用
TeacherSession 自动携带上一轮 Context；离线 run-teacher 可在每条输入记录中提供
context_before。当前 SocialSim 默认预处理每个 conversation 只产生一个 target，
因此不会自动形成跨 target 的 Context 轨迹。Context 只进入教师推理与 trace，首版学生
训练输入保持不变。
```

- [ ] **Step 7: 提交 CLI、导出边界和使用文档**

```bash
git add src/ibd/cli.py tests/test_cli.py tests/test_export.py README.md
git commit -m "feat: expose context audit in teacher cli"
```

---

### Task 6: 全量验证与设计一致性检查

**Files:**

- Verify: `src/ibd/`
- Verify: `tests/`
- Verify: `configs/deepseek_teacher.yaml`
- Verify: `docs/superpowers/specs/2026-09-15-seeker-context-memory-design.md`

- [ ] **Step 1: 运行 Context 相关定向测试**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest tests/test_schemas.py tests/test_context_memory.py tests/test_teacher.py tests/test_cli.py tests/test_export.py -q
```

Expected: PASS。

- [ ] **Step 2: 运行完整测试套件**

Run:

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q
```

Expected: 全部 PASS；如果旧 fixture 因新增 `context_updater` 调用失败，只更新测试 backend 的合法结构化返回，不放宽生产 schema。

- [ ] **Step 3: 检查实现中没有被否决的元数据字段**

Run:

```bash
rg -n 'first_observed_turn|last_confirmed_turn|source_turn|\bbasis\b|\bkind\b|memory_id' src tests configs
```

Expected: 无 Context 实现命中；若其他既有模块有同名但无关字段，逐项确认其不进入 `UserContext`、`ContextPatch` 或学生导出。

- [ ] **Step 4: 检查调用顺序与 Context 消费面**

Run:

```bash
rg -n 'context_updater|user_context|context_before|context_patch|context_after|context_merge_errors' src/ibd tests configs/deepseek_teacher.yaml
```

Expected: Updater 位于 Analyzer 前；Analyzer、Planner、Candidate、Final Selector 都接收同一个 `context_after`；只有 Updater 和 reducer 可以改变 Context。

- [ ] **Step 5: 检查学生训练代码没有新增 Context token 或输入字段**

Run:

```bash
git diff -- src/ibd/student_data.py src/ibd/model.py src/ibd/export.py
```

Expected: `student_data.py` 与 `model.py` 无差异；`export.py` 无生产代码差异，或仅有不改变 allowlist 的说明性改动。

- [ ] **Step 6: 检查格式与工作树范围**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` 无输出；工作树不包含 `result/balanced-2500-quality/model_strategy_heatmap.svg` 的删除或修改，不包含计划外文件。

- [ ] **Step 7: 若验证过程中产生必要修复，提交最终修复**

只有确实发生修复时执行：

```bash
git add src tests configs README.md
git commit -m "test: verify seeker context pipeline"
```

最终验收标准：

1. 每个 seeker turn 先更新 Context，再分析 STATE，再制定 PLAN，再生成与选择回复。
2. Context 九字段始终结构完整，条目只有非空字符串。
3. supporter 回复不能直接改变 Context；最新 seeker 明确信息具有最高优先级。
4. Updater schema 失败时保留旧 Context，教师流水线继续运行且 trace 可审计。
5. `TeacherSession` 能在多轮调用间携带 Context，但不会携带上一轮 STATE 或 PLAN。
6. 当前学生训练数据格式完全不变。
7. 全量测试通过，且没有把当前单 target SocialSim 数据误表述为跨轮记忆轨迹。
