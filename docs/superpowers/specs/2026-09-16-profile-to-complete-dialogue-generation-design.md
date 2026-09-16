# Profile 驱动的完整 Seeker–Supporter 对话生成设计

## 状态

本设计于 2026-09-16 经用户确认。它定义一次输入一个用户 profile、系统内部自动生成完整 seeker–supporter 多轮对话的架构。

本设计只规定 profile 的可见性边界和适配器接口，不规定 profile 的详细字段。profile schema、字段语义、信息拆分方式和可选披露账本应作为后续独立设计处理。

## 目标

将当前面向单个 supporter target 的增量式 Teacher 推理，扩展为 profile 级完整对话生成：

~~~text
一次外部调用输入一个 profile
        |
        v
系统内部交替生成 seeker 和 supporter
        |
        v
Dialogue Manager 每轮选择大体对话方向
        |
        v
对话自然进入 closing，或以非完成状态终止
        |
        v
输出完整 ConversationTrace 和逐回合 TeacherTrace
~~~

生成结果用于批量构造完整对话数据。系统内部允许多次本地模型调用。

## 核心约束

1. 原始 profile 只对 Seeker Simulator 可见。
2. Supporter 只能使用 seeker 在公开对话中已经表达的信息。
3. Seeker 不维护由模型自由更新的私有心理状态。
4. 不增加在线 ProfileConsistencyChecker；seeker 行为仅由 prompt 约束。
5. Supporter 每轮继续使用完整的 Context、STATE、PLAN、candidate 和 selection 流程。
6. Dialogue Manager 只选择 DialogueMode，不输出 DialogueGoal 或具体策略。
7. Mode 是软方向，不是固定阶段序列或策略映射。
8. hard max 只表示生成被截断，不能伪装成正常 closing。
9. 每个 supporter 回合的 Dialogue Manager 决策与 TeacherTrace 按回合绑定。
10. 现有 TeacherRunner.run、TeacherTrace 导出和 Stage A/B 训练接口保持兼容。

## 非目标

- 本设计不确定 profile 的详细 schema。
- 本设计不确定 demographic、personality、event、goal 或 coping 等字段如何标准化。
- 本设计不引入 SeekerPrivateState。
- 本设计不引入 trust、engagement、hidden emotion 或未披露问题等可变私有状态。
- 本设计不引入在线 profile 一致性分类器或生成后拒绝器。
- 本设计不定义跨会话长期记忆。
- 本设计不把完整对话生成拆分为 full_teacher 和 scalable_dialogue 两种模式。
- 本设计不修改现有 Student 的 STATE / PLAN 槽位定义。

## 相关工作

ESConv 基于 Helping Skills Theory 将情绪支持组织为 Exploration、Comforting 和 Action 三个阶段，并证明支持策略对有效情绪支持的重要性：
<https://aclanthology.org/2021.acl-long.269/>。

MultiESC 指出多轮情绪支持需要动态建模用户状态并进行面向长期效果的策略规划：
<https://aclanthology.org/2022.emnlp-main.195/>。

TransESC 从语义、策略和情绪三个角度建模逐轮转移，说明多轮支持不适合只依靠固定阶段规则：
<https://aclanthology.org/2023.findings-acl.420/>。

PAL 说明 seeker persona 会影响情绪支持生成，并采用动态 persona 建模改善个性化支持：
<https://aclanthology.org/2023.findings-acl.34/>。

SocialSim 使用 persona bank 促进 seeker 的社会披露，并以认知推理增强 supporter 的社会意识，为 profile 驱动的完整 ESC 仿真提供直接参考：
<https://ojs.aaai.org/index.php/AAAI/article/view/32116>。

SPASM 将 persona 构建、Client–Responder 对话生成和 termination detection 分开，并通过角色视角投影降低 persona drift 和 role confusion：
<https://arxiv.org/abs/2604.09212>。

本设计吸收这些工作的共同原则，但保留现有项目的 Context、七维 STATE、PLAN 和多候选选择结构。

## 当前系统基线

当前 TeacherRunner.run 对一段以 seeker 结尾的 History 执行：

~~~text
Context Updater
    -> deterministic Context reducer
    -> Multi-view State Analyzer
    -> Planner
    -> 1–3 Candidate Generators
    -> Final Selector
    -> TeacherTrace
~~~

TeacherSession 会把上一轮 context_after 作为下一轮 context_before，因此已经具备跨 seeker turn 的公开事实记忆。

当前 SocialSim 预处理每个 conversation 只选择一个 supporter target。profile 内容只用于 ID 完整性校验，不进入生成或训练 artifact。

本设计新增的是 profile 驱动的 Seeker Simulator、Dialogue Manager、完整对话轮转、终止状态和 ConversationTrace 容器。

## 总体架构

~~~text
Raw Profile
    |
    v
SeekerProfileAdapter
    |
    v
ValidatedProfile
    |
    +------------------------------+
    |                              |
    v                              |
SeekerSimulator                    |
profile + public history + mode    |
    |                              |
    v                              |
seeker utterance                   |
    |                              |
    v                              |
Public History                     |
    |                              |
    v                              |
TeacherSession.observe             |
Context -> STATE                   |
    |                              |
    v                              |
Dialogue Manager                   |
DialogueMode only                  |
    |                              |
    v                              |
TeacherSession.respond             |
PLAN -> candidates -> selection    |
    |                              |
    v                              |
supporter response ----------------+
~~~

ConversationGenerator 负责轮转、checkpoint、完成状态和结果组装。它不解释 profile 字段，也不实现 supporter 策略。

## 可见性边界

| 信息 | Seeker Simulator | Dialogue Manager | Supporter |
|---|---:|---:|---:|
| 原始或标准化 profile | 是 | 否 | 否 |
| 公开对话历史 | 是 | 是 | 是 |
| UserContext | 否 | 是 | 是 |
| StateBlackboard | 否 | 是 | 是 |
| STATE evidence | 否 | 是 | Planner 可间接使用 |
| 最近选择的 supporter strategies | 否 | 是 | Planner 自身可用 |
| DialogueMode | 是，作为弱提示 | 产生 | Planner 接收 |
| PLAN / candidates / selector decision | 否 | 可读历史摘要 | 是 |

原始 profile 不得出现在 Context Updater、State Analyzer、Dialogue Manager、Planner、Candidate 或 Selector 的 prompt 中。

Supporter 获得 profile 信息的唯一合法路径是：

~~~text
profile
  -> seeker utterance
  -> public history
  -> Context Updater
  -> UserContext
~~~

## Profile 适配器边界

profile 详细定义延后，但当前架构需要一个稳定边界：

~~~python
class ValidatedProfile(Protocol):
    @property
    def profile_id(self) -> str: ...

    def render_for_seeker(self) -> str: ...


class SeekerProfileAdapter(Protocol):
    def validate(
        self,
        raw_profile: Mapping[str, Any],
    ) -> ValidatedProfile: ...
~~~

ConversationGenerator 只依赖 profile_id 和 render_for_seeker。它不读取具体字段，也不计算 profile complexity。

第一版统一使用全局轮数配置，不根据 profile 自动分配不同长度。

## Seeker Simulator

### 输入

Seeker Simulator 每轮接收：

- 完整私有 ValidatedProfile；
- 当前公开对话历史；
- 当前 DialogueMode；
- 当前轮数。

它不接收 UserContext、STATE、PLAN、候选回复、selector decision 或 Dialogue Manager 的 transition reason。
轮数预算只提供给 Dialogue Manager，不直接提示 seeker 因接近上限而结束。

### 输出

Seeker Simulator 只输出下一条 seeker 文本：

~~~python
def generate_seeker_turn(
    profile: ValidatedProfile,
    history: Sequence[DialogueTurn],
    mode: DialogueMode,
    *,
    round_index: int,
    seed: int,
) -> str: ...
~~~

不输出：

- closure_signal；
- private_state_patch；
- trust_change；
- engagement_change；
- disclosed_item_ids；
- remaining_issue_ids；
- suggestion usefulness；
- 自由文本隐藏推理。

### Prompt 合同

Seeker system prompt 必须要求：

~~~text
You are simulating the seeker described by the private profile.

- Speak only as the seeker.
- Treat the profile as private background, not text to quote or summarize.
- Never mention that you were given a profile.
- Remain consistent with the profile throughout the conversation.
- Respond naturally to the latest supporter message.
- Reveal information gradually and only when conversationally relevant.
- Do not disclose every known fact in the opening turns.
- Do not invent events, relationships, goals, symptoms, or experiences
  that contradict or materially extend the profile.
- Preserve uncertainty present in the profile.
- You may disagree with, reject, question, or only partially accept support.
- Do not become satisfied merely because the supporter offered one suggestion.
- When the immediate conversational need has naturally been met, express
  closure through the utterance itself.
- Return only the next seeker utterance.
~~~

Mode 只提供弱提示：

- opening：自然开始互动并引出最初困扰；
- exploration：回应当前问题，逐步补充相关背景；
- comforting：自然表现是否感到被理解，但不强制好转；
- action：根据 profile 和公开对话接受、犹豫、拒绝或澄清建议；
- closing：在当前需要已经自然得到回应时表达结束，不引入重大新问题。

### TextCaller

Seeker 使用非 JSON 的 TextCaller：

~~~python
class TextCaller:
    def call(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        seed: int,
        example_id: str,
    ) -> tuple[str, CallRecord]: ...
~~~

TextCaller：

- 调用 LLMBackend.complete，并设置 json_mode=False；
- 去除首尾空白；
- 拒绝空输出；
- 记录原始文本和调用元数据；
- 按 conversation、round 和 protocol 缓存；
- 只对空输出、传输错误和 provider failure 重试；
- 不执行 profile 一致性判断或内容改写。

## Dialogue Manager

### 职责

Dialogue Manager 回答：

~~~text
根据当前公开对话、Context、STATE 和近期策略，
整段对话此刻最合适的大体方向是什么？
~~~

它不回答下一条回复应使用哪个具体策略，也不生成回复内容。

### DialogueMode

~~~python
DialogueMode = Literal[
    "opening",
    "exploration",
    "comforting",
    "action",
    "closing",
]
~~~

Mode 是每轮重新判断的软对话方向，不构成固定的有限状态机。Manager 可以保持、前进、回退或跳过 Mode。

### 输出 Schema

~~~python
class DialogueManagementDecision(StrictModel):
    mode: DialogueMode
    transition_reason: str = Field(min_length=1)
~~~

不包含 DialogueGoal 和 should_close。终止由 mode == closing 唯一表示。

transition_reason 只用于审计，不传给 Planner 或 Seeker。

### 输入

Dialogue Manager 只接收公开信息：

~~~json
{
  "history": {},
  "user_context": {},
  "state_analysis": {},
  "previous_mode": "exploration",
  "recent_selected_strategies": [
    "Question",
    "Reflection of feelings"
  ],
  "round_index": 5,
  "min_rounds": 6,
  "soft_max_rounds": 16,
  "hard_max_rounds": 20
}
~~~

其中 state_analysis 必须完全复用现有 MultiViewStateAnalysis：

~~~text
MultiViewStateAnalysis
├── views: MultiViewStateViews
│   ├── emotion: AnalysisView
│   ├── need: AnalysisView
│   ├── relationship: AnalysisView
│   └── intent: AnalysisView
├── state: StateBlackboard
└── state_evidence: StateEvidence
~~~

每个 AnalysisView 包含：

~~~python
summary: str
evidence: str
uncertainty: str = ""
~~~

每个 StateFieldEvidence 包含：

~~~python
evidence: str = ""
basis: Literal[
    "explicit",
    "strong_inference",
    "absent_or_ambiguous",
]
~~~

### Prompt 合同

~~~text
Choose exactly one broad dialogue mode.

The modes are soft conversational orientations, not a mandatory sequence.
You may keep the previous mode, move forward, move backward, or skip a mode.

- opening: establish the interaction and invite the initial concern.
- exploration: understand the situation, emotion, needs, constraints, or
  unresolved questions.
- comforting: prioritize understanding, validation, emotional accompaniment,
  and reducing interpersonal pressure.
- action: collaboratively consider information, choices, coping approaches,
  or feasible next steps.
- closing: provide the final supporter response because the seeker has
  explicitly or naturally indicated that the immediate conversation can end.

Use the latest seeker turn as the primary evidence.
Do not choose action merely because the conversation is long.
Do not choose closing for a brief acknowledgment when the seeker remains
engaged or has an unresolved request.
Do not keep exploring when further questioning would be repetitive.
Return only JSON matching the supplied schema.
~~~

### 与 Planner 的边界

Planner 只接收 DialogueMode，不接收 transition_reason：

~~~json
{
  "user_context": {},
  "state": {},
  "dialogue_mode": "comforting"
}
~~~

Planner prompt 必须声明：

~~~text
Dialogue mode describes the broad direction of the conversation.
It is not a required strategy and does not override the latest seeker turn,
User Context, or STATE. Select strategies according to the current evidence.
Different strategies may be appropriate within the same mode.
~~~

同一策略允许跨 Mode 使用。Mode 与 ESConv strategy 之间不存在固定映射。

## Teacher 的兼容拆分

当前 TeacherRunner.run 把观察和响应绑定在一个调用链中。为在 STATE 之后插入 Dialogue Manager，内部拆成：

~~~python
class TurnObservation(StrictModel):
    example_id: str
    history: History
    context_before: UserContext
    context_patch: ContextPatch
    context_after: UserContext
    context_merge_errors: list[str]
    state_analysis: MultiViewStateAnalysis
    call_records: list[CallRecord]


class TurnResponse(StrictModel):
    plan: StrategyPlanSet
    candidates: list[Candidate]
    final_selection: FinalSelection
    call_records: list[CallRecord]
~~~

TeacherRunner 新增：

~~~python
def observe(
    self,
    example_id: str,
    history: History,
    *,
    context_before: UserContext | None = None,
) -> TurnObservation: ...

def respond(
    self,
    observation: TurnObservation,
    *,
    dialogue_mode: DialogueMode | None = None,
) -> TurnResponse: ...
~~~

原入口保留：

~~~python
def run(...) -> TeacherTrace:
    observation = self.observe(...)
    response = self.respond(observation)
    return TeacherTrace.from_parts(observation, response)
~~~

不使用完整对话生成时，dialogue_mode 默认为 None，行为必须与当前 run 一致。

TeacherTrace.from_parts 必须保持现有 provenance：

- 顶层 state 等于 state_analysis.state；
- final_response 等于 final_selection.response；
- selected candidate 与 final selection 一致；
- candidate strategy 与 plan 一致。

## 单回合生成顺序

~~~text
1. Seeker Simulator 生成 seeker utterance
2. utterance 追加到公开 History
3. TeacherSession.observe：
   - Context Updater
   - deterministic Context reducer
   - Multi-view State Analyzer
4. Dialogue Manager：
   - 读取公开 History、Context、STATE 和近期策略
   - 产生 DialogueMode 和审计 reason
5. TeacherSession.respond：
   - Planner 接收 Mode 作为弱先验
   - Candidate generation
   - Final selection
6. supporter response 追加到公开 History
7. 保存 ConversationRoundAudit checkpoint
8. 如果 Mode 为 closing，正常完成
9. 如果达到 hard max 且 Mode 不是 closing，标记 truncated
10. 否则继续下一轮 seeker
~~~

最后一条成功或截断的轨迹都以 supporter 结束。

## 动态长度和完成状态

### 默认轮数配置

现有 3,229 条 SSConv 的 supporter 回合数分布为：

~~~text
min=9
P05=10
median=12
P95=13
max=20
mean=11.98
~~~

第一版配置：

~~~yaml
min_rounds: 6
soft_max_rounds: 16
hard_max_rounds: 20
~~~

min 和 soft max 是提供给 Dialogue Manager 的软参考，不直接覆盖其判断。

hard max 是执行预算，不是正常对话事件。

### 状态

~~~python
ConversationStatus = Literal[
    "completed",
    "truncated",
]

TerminationReason = Literal[
    "dialogue_manager_closing",
    "hard_max_without_closing",
]
~~~

判断规则：

~~~python
if decision.mode == "closing":
    status = "completed"
    reason = "dialogue_manager_closing"
elif round_index >= hard_max_rounds:
    status = "truncated"
    reason = "hard_max_without_closing"
else:
    continue_dialogue()
~~~

不得在 hard max 时把非 closing 决策强制改为 closing。

如果 Manager 恰好在第 hard max 轮根据对话选择 closing，结果仍为 completed。

truncated conversation 不进入正常训练数据。

## Conversation 数据结构

### 对话配置

~~~python
class ConversationGenerationConfig(StrictModel):
    min_rounds: int = Field(default=6, ge=1)
    soft_max_rounds: int = Field(default=16, ge=1)
    hard_max_rounds: int = Field(default=20, ge=1)
    split: Literal[
        "train",
        "dev",
        "diagnostic_holdout",
    ] = "train"
~~~

必须满足：

~~~text
min_rounds <= soft_max_rounds <= hard_max_rounds
~~~

### 公开 Turn

~~~python
class ConversationTurn(StrictModel):
    turn_index: int = Field(ge=1)
    round_index: int = Field(ge=1)
    role: Literal["seeker", "supporter"]
    content: str = Field(min_length=1)
~~~

Mode 不重复写入每条 turn。当前回合 seeker 使用的 Mode 与 supporter 使用的
Mode 可能不同，二者通过 DialogueDecisionRecord.mode_before 和
DialogueDecisionRecord.decision.mode 精确区分。

### Dialogue Manager 审计

~~~python
class DialogueDecisionRecord(StrictModel):
    round_index: int = Field(ge=1)
    mode_before: DialogueMode
    decision: DialogueManagementDecision
    call_records: list[CallRecord]
~~~

mode_before 是生成当前 seeker turn 时使用的 Mode；decision.mode 是 Dialogue
Manager 在看到该 seeker turn 后为当前 supporter response 选择的 Mode。如果对话
继续，decision.mode 将成为下一回合的 mode_before。

### 回合绑定

~~~python
class ConversationRoundAudit(StrictModel):
    round_index: int = Field(ge=1)
    seeker_turn_index: int = Field(ge=1)
    supporter_turn_index: int = Field(ge=1)
    dialogue_decision: DialogueDecisionRecord
    teacher_trace: TeacherTrace
~~~

ConversationRoundAudit 避免 supporter_traces 和 dialogue_decisions 两个平行数组发生错位。

teacher_trace 完整保存当前既有字段：

- history；
- context_before；
- context_patch；
- context_after；
- context_merge_errors；
- state_analysis.views；
- state_analysis.state；
- state_analysis.state_evidence；
- 顶层 state；
- plan；
- candidates；
- final_selection；
- final_response；
- call_records。

顶层 state 保持等于 state_analysis.state，以兼容现有 schema、导出和训练代码。

### 终止记录

~~~python
class ConversationTermination(StrictModel):
    reason: TerminationReason
    round_count: int = Field(ge=1)
    final_mode: DialogueMode
~~~

### 完整输出

~~~python
class ConversationTrace(StrictModel):
    protocol_version: str
    conversation_id: str
    profile_id: str
    seed: int
    split: Literal[
        "train",
        "dev",
        "diagnostic_holdout",
    ]
    status: ConversationStatus
    turns: list[ConversationTurn]
    rounds: list[ConversationRoundAudit]
    termination: ConversationTermination
    seeker_call_records: list[CallRecord]
~~~

ConversationTrace 默认只保存 profile_id，不复制原始 profile。

模型或系统调用重试耗尽时不构造 status=failed 的 ConversationTrace，而是写入：

~~~python
class ConversationFailure(StrictModel):
    conversation_id: str
    profile_id: str
    seed: int
    failed_stage: str
    error_type: str
    error: str
    completed_rounds: int = Field(ge=0)
    checkpoint_path: str | None = None
~~~

## 数据存储

~~~text
artifacts/conversations/
├── conversations.jsonl
├── truncated-conversations.jsonl
├── conversation-failures.jsonl
└── checkpoints/
~~~

- conversations.jsonl：只包含 status=completed；
- truncated-conversations.jsonl：包含 hard max 未自然 closing 的对话；
- conversation-failures.jsonl：包含 ConversationFailure；
- checkpoints：保存逐回合可恢复状态。

完整对话可以扁平导出为每个 supporter 回合一条 TeacherTrace，供现有 Stage A/B 管线消费。

同一 profile 的多个 seed 必须保持在同一数据 split 中。

## Checkpoint 与恢复

每生成一个完整 seeker–supporter 回合后原子保存：

~~~json
{
  "conversation_id": "profile-1-seed-42",
  "profile_id": "1",
  "seed": 42,
  "history": {},
  "rounds": [],
  "last_context": {},
  "previous_mode": "comforting",
  "next_round_index": 7,
  "seeker_call_records": []
}
~~~

恢复时：

1. 重建公开 History；
2. 用最后一个 TeacherTrace.context_after 初始化 TeacherSession；
3. 恢复 previous_mode；
4. 从 next_round_index 继续；
5. 不重新调用已完成回合；
6. 不重复追加已完成输出。

每个模型调用的 seed 使用稳定哈希派生：

~~~text
protocol_version
+ conversation_id
+ base_seed
+ round_index
+ role
+ cache variant
~~~

不得使用 Python 内置 hash。

## 失败处理

| 失败位置 | 处理 |
|---|---|
| Profile adapter 拒绝输入 | 不调用模型，写 failure |
| Seeker 空输出或 provider failure | 重试；耗尽后写 ConversationFailure |
| Context Updater 失败 | 沿用现有空 patch fallback，并记录错误 |
| State / Planner / Candidate / Selector 失败 | 使用现有 structured retry；耗尽后写 ConversationFailure |
| Dialogue Manager 非法 JSON | structured retry；耗尽后保持上一 Mode |
| Dialogue Manager 在 hard max 前持续非 closing | 正常继续 |
| 到达 hard max 且 Mode 非 closing | status=truncated |
| 进程中断 | 保留 checkpoint，可恢复 |

失败或截断对话不得混入 completed JSONL。

## CLI

新增：

~~~bash
python -m ibd.cli generate-conversations \
  --config configs/local_teacher.yaml \
  --profiles data/profiles.json \
  --output artifacts/conversations/conversations.jsonl \
  --truncated artifacts/conversations/truncated-conversations.jsonl \
  --failures artifacts/conversations/conversation-failures.jsonl \
  --checkpoint-dir artifacts/conversations/checkpoints \
  --seed 42 \
  --min-rounds 6 \
  --soft-max-rounds 16 \
  --hard-max-rounds 20 \
  --resume
~~~

第一版语义是一条 profile 生成一条 ConversationTrace。

同一 profile 多 seed 生成不属于第一版；未来增加时必须保持 split 隔离。

## 模块划分

~~~text
src/ibd/
├── conversation.py
├── conversation_schemas.py
├── seeker.py
├── teacher.py
├── prompting.py
├── cli.py
└── storage.py
~~~

- conversation.py：ConversationGenerator、回合循环、状态分类和恢复；
- conversation_schemas.py：Mode、Decision、RoundAudit、ConversationTrace；
- seeker.py：profile adapter protocol、TextCaller、SeekerSimulator；
- teacher.py：observe/respond 拆分与 run 兼容包装；
- prompting.py：Dialogue Manager prompt，并让 Planner 接收 Mode；
- cli.py：generate-conversations；
- storage.py：原子 checkpoint 辅助函数。

AppConfig 增加两个 role：

~~~yaml
roles:
  seeker_simulator:
    model: local-model
    temperature: 0.7
    max_tokens: 256
    provider_json_mode: false

  dialogue_manager:
    model: local-model
    temperature: 0.0
    max_tokens: 256
    provider_json_mode: true
~~~

Supporter 始终走现有完整 Teacher 路径，不另设快速模式。

## 测试设计

### Teacher 兼容性

- TeacherRunner.run 与 observe + respond 产生等价 TeacherTrace；
- 没有 DialogueMode 时行为与当前版本一致；
- TeacherSession 正确携带上一轮 context_after；
- 现有单 target CLI、导出和训练测试继续通过。

### 信息隔离

使用只存在于 profile 中的唯一哨兵字符串：

~~~text
PRIVATE_PROFILE_SENTINEL
~~~

测试 seeker backend 输出不含该字符串的普通 utterance，并断言：

- seeker system prompt 包含哨兵；
- Dialogue Manager 的所有 messages 不包含哨兵；
- Context、STATE、Planner、Candidate 和 Selector messages 不包含哨兵；
- ConversationTrace 的公开 turns 不自动包含原始 profile；
- supporter 只能看到 seeker 实际生成的文本。

该测试验证架构泄漏，不增加在线 ProfileConsistencyChecker。

### Dialogue Manager

- 输入只包含公开 History、Context、STATE、策略历史和轮数信息；
- 输出严格匹配 DialogueManagementDecision；
- 能保持、前进、回退或跳过 Mode；
- transition_reason 不进入 Planner 或 Seeker prompt；
- Planner 只接收 Mode；
- 非法 JSON 触发 structured retry；
- retry 耗尽后保持上一 Mode；
- Mode 为 closing 时，在当前 supporter 回复后完成。

### State Schema

- Dialogue Manager 接收现有 MultiViewStateAnalysis；
- views 严格包含 emotion、need、relationship 和 intent；
- state 严格使用当前七维 StateBlackboard；
- state_evidence 严格使用现有 EvidenceBasis；
- TeacherTrace 顶层 state 等于 state_analysis.state；
- conversation 层不定义第二套 STATE schema。

### Conversation 不变量

- 第一条 turn 为 seeker；
- 角色严格交替；
- 最后一条 turn 为 supporter；
- 每个 supporter turn 恰好对应一个 ConversationRoundAudit；
- 每个 RoundAudit 恰好包含一个 DialogueDecisionRecord 和一个 TeacherTrace；
- turn index 与 round index 连续；
- TeacherTrace.history 等于对应 supporter 回复前的公开历史前缀；
- Context 可以沿 rounds 逐轮重放；
- completed 的 final_mode 为 closing；
- hard max 非 closing 的结果只能是 truncated；
- truncated 不写入 completed JSONL。

### Resume

- 在第 N 回合后模拟中断；
- resume 不重复前 N 回合；
- 恢复结果与不中断运行一致；
- 已完成 profile 不再次调用模型；
- 不同 profile、seed 或 protocol 不共享缓存结果。

### 离线数据审计

小规模生成后统计：

- completed、truncated 和 failed 比例；
- 对话长度分布；
- Mode 分布和转移矩阵；
- strategy 分布；
- 连续重复问题和重复建议；
- closing 自然度；
- Context 是否只包含 seeker 已公开信息。

这些审计不参与每轮在线生成。

## 迁移顺序

### Phase 1：拆分 Teacher

增加 TurnObservation、TurnResponse、observe、respond 和 TeacherTrace.from_parts，同时保持 run 完全兼容。

### Phase 2：Dialogue Manager

增加 DialogueMode、DialogueManagementDecision、prompt 和结构化调用。Planner 只接收 Mode 作为弱先验。

### Phase 3：Seeker Simulator

增加 ValidatedProfile protocol、SeekerProfileAdapter protocol、TextCaller 和 SeekerSimulator。具体 profile schema 不在此阶段确定。

### Phase 4：ConversationGenerator

实现首次 seeker turn、逐轮 observe–manage–respond、动态 closing、truncated 分类和 ConversationRoundAudit。

### Phase 5：Checkpoint、CLI 和导出

增加逐回合原子 checkpoint、resume、三类输出文件，以及 ConversationTrace 到 TeacherTrace rows 的扁平导出。

### Phase 6：小规模验收

先生成约 50 条完整对话，检查完成率、截断率、长度、Mode、strategy、重复、closing 和信息隔离，再决定批量规模。

## 验收标准

1. 外部输入一次 profile，系统生成一条完整 seeker–supporter 多轮轨迹。
2. 原始 profile 只进入 seeker prompt。
3. Supporter 每轮执行 Context -> STATE -> Mode -> PLAN -> candidates -> selection -> response。
4. Dialogue Manager 由额外 LLM 调用产生 Mode。
5. Dialogue Manager 不输出 Goal 或具体 strategy。
6. Mode 是弱方向，可保持、回退、前进或跳过。
7. 每个 supporter 回合绑定一个 DialogueDecisionRecord 和一个完整 TeacherTrace。
8. TeacherTrace 使用当前准确的 MultiViewStateAnalysis 和 StateBlackboard schema。
9. 正常完成必须由 Dialogue Manager 选择 closing。
10. hard max 非 closing 结果标记 truncated，不进入正常训练数据。
11. 进程中断后可从最后一个完整回合恢复。
12. ConversationTrace 可扁平导出给现有 Stage A/B 管线。
13. 现有 TeacherRunner.run、CLI、导出和训练行为保持兼容。

## 延后决策

以下内容明确不属于本设计，将通过后续独立设计确定：

- profile 的完整 Pydantic schema；
- profile 字段的可见性和伦理边界；
- personality 字段如何影响 seeker 表达；
- profile 是否拆成原子披露单元；
- 是否引入确定性的 DisclosureLedger；
- 是否允许一个 profile 生成多个 seed 的对话；
- 是否需要专门训练或微调 Seeker Simulator。
