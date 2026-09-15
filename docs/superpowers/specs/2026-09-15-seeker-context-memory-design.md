# Seeker Context Memory 串行推理设计

## 状态

本设计于 2026-09-15 经用户确认。它只定义模型推理侧的动态用户信息维护，不实现用户模拟器、跨会话长期记忆或训练代码。

## 目标

在现有七维 `STATE -> PLAN -> response` 流程之前增加一层动态 `Seeker Context Memory`，使模型先维护用户明确披露的处境，再判断用户当前的支持需求与准备度，最后选择下一步支持行为。

三层分别回答不同问题：

```text
Context：用户正在面对什么，受哪些目标、关系、经历和现实条件影响？
STATE：用户此刻需要什么、愿意接受什么、想不想行动、能不能行动？
PLAN：基于上述信息，支持者下一步应该做什么？
```

`Context` 是模型依据可见对话形成的事实性工作记忆，不是用户的私有真值。它只能保存用户明确表达或可直接改写的信息；所有不确定推断继续由 `STATE` 承担。

## 非目标

- 不维护跨会话、跨日期的持久用户档案。
- 不推断或存储 MBTI、Big Five、临床诊断或深层人格。
- 不默认存储年龄、性别、职业、宗教等人口统计属性；只有它们构成当前问题事实时，才以问题相关描述进入相应 Context 字段。
- 不把 emotion、support need、advice receptivity、action intent、action capacity 或 continuation intent 重复写入 Context。
- 不把 supporter 的判断、承诺或建议写成用户事实。
- 不在 supporter 回复之后自行推断用户已经好转；必须等待下一条 seeker 表达。
- 初版不增加 memory summarizer、数值置信度、事实 ID、生命周期时间戳或状态标签。

## 相关工作边界

PAL 从对话历史动态抽取自由文本 persona sentences，用于个性化 ESC 回复，但没有为本设计所需的决策相关事实提供严格槽位和增量更新协议：<https://aclanthology.org/2023.findings-acl.34/>。

PACEP 已经逐轮维护职业、年龄段、情绪困境、理想支持者和补充描述，因此“动态用户画像”本身不能作为本项目的核心创新：<https://www.sciencedirect.com/science/article/pii/S0957417426019081>。

USP 将用户画像分为人口统计、关系、习惯、目标、任务、Big Five 和语言风格等客观与主观属性，适合通用用户模拟，但范围过宽：<https://aclanthology.org/2025.acl-long.1025/>。

EmoHarbor 使用 demographics、preferences/personality、counseling attributes 和 scenario script 模拟用户私有世界。这类预设真值不应直接暴露给本设计中的 supporter 推理侧：<https://aclanthology.org/2026.acl-long.53/>。

PsyProbe 使用 Presenting、Predisposing、Precipitating、Perpetuating、Protective 和 Impact 槽位维护咨询 formulation，为问题、诱因、资源和功能影响提供了结构参考，但本项目保持非临床 ESC 边界：<https://aclanthology.org/2026.findings-eacl.336/>。

Seeker simulator 工作还会刻画 coping、resistance、engagement、self-disclosure 和 reaction。这里仅吸收可由用户明确表达、且不与七维 STATE 重复的事实信息：<https://aclanthology.org/2026.findings-acl.1146/>。

因此，本设计不复刻宽泛 persona，而采用 **decision-relevant context memory**：仅维护那些一旦改变，就可能合理改变 STATE 判断或 PLAN 选择的用户事实。

## 总体架构

采用串行更新，而不是联合更新或并行更新：

```text
previous User Context
        +
latest seeker turn and recent dialogue
        |
        v
Context Updater
        |
        v
Context Patch
        |
        v
Deterministic Validator + Reducer
        |
        v
updated User Context
        +
visible dialogue history
        |
        v
State Analyzer
        |
        v
current seven-dimensional STATE
        |
        v
Planner -> Candidate Generation -> Selection -> Response
```

串行设计保证：

1. Context 只负责事实更新；
2. State Analyzer 可以使用更新后的事实，但最新 seeker 明示状态仍然优先；
3. Context、STATE 和 PLAN 可以分别消融；
4. 错误 Context patch 可以在进入 STATE 和 PLAN 之前被拒绝。

## Context 数据结构

所有九个字段在 JSON 中都必须存在，值必须是字符串数组，数组允许为空。当前 Context 快照只保存仍然有效的信息。

```json
{
  "active_concerns": [],
  "events_and_triggers": [],
  "functional_impacts": [],
  "goals_and_priorities": [],
  "people_and_relationships": [],
  "constraints_and_resources": [],
  "coping_attempts_and_outcomes": [],
  "support_preferences_and_boundaries": [],
  "communication_preferences": []
}
```

每个字符串必须是一个简短、中性、第三人称的原子事实。不能在同一字符串中混合多个彼此可独立变化的事实。

### `active_concerns`

保存用户当前面对、尚未解决的核心问题、疑问或冲突。

包含：工作压力、关系冲突、决策困难、尚未解决的具体担忧。

排除：当前情绪、支持策略、模型推断的深层心理原因、已经解决的问题。

### `events_and_triggers`

保存与当前困境相关的关键事件、近期变化、持续情境或用户明确指出的诱因。

它只回答“发生了什么”，不回答“用户现在有多痛苦”。

### `functional_impacts`

保存问题对睡眠、工作、学习、社交、日常生活、身体状态或决策能力造成的具体影响。

它保存影响事实；`distress_level` 仍负责概括当前痛苦强度。

### `goals_and_priorities`

保存用户希望实现的结果、当前优先事项、明确价值取向和需要权衡的目标。

它保存“用户想要什么结果”；`action_intent` 负责判断用户此刻是否准备为该结果行动。

### `people_and_relationships`

保存与当前问题直接相关的人物、关系、现实支持、冲突或依赖。

只能记录用户描述的关系事实，不能推断第三方动机或人格。

### `constraints_and_resources`

保存影响行动的具体障碍和可用资源，包括时间、精力、经济、信息、环境、社会支持、身体条件和用户明确表达的安全顾虑。

它保存 capacity 的原因；`action_capacity` 负责将这些事实与当前表达综合为离散状态。

### `coping_attempts_and_outcomes`

保存用户已经尝试的应对方式以及用户报告的结果。

supporter 刚提出但用户尚未尝试的建议不能写入；模型预测的效果也不能写入。

### `support_preferences_and_boundaries`

保存用户对支持方式的相对稳定偏好、反感方式、明确边界和不愿讨论的内容。

它不替代当前 `advice_receptivity`。长期偏好与本轮明确请求冲突时，本轮明确表达决定当前 STATE。

### `communication_preferences`

保存用户明确表达的对话节奏、详细程度、语气、提问方式和直接程度偏好。

该字段写入门槛最高。初版仅接受明确表达，不根据文本长短、单次简短回复或刻板印象推断交流风格。

## Context Patch 数据结构

Updater 不输出完整新 Context，只输出三类变更。`add`、`replace` 和 `remove` 都必须包含完整九字段；没有对应操作的字段使用空数组。

概念结构如下：

```json
{
  "add": {
    "active_concerns": [],
    "events_and_triggers": [],
    "functional_impacts": [],
    "goals_and_priorities": [],
    "people_and_relationships": [],
    "constraints_and_resources": [],
    "coping_attempts_and_outcomes": [],
    "support_preferences_and_boundaries": [],
    "communication_preferences": []
  },
  "replace": {
    "active_concerns": [],
    "events_and_triggers": [],
    "functional_impacts": [],
    "goals_and_priorities": [
      {
        "old": "用户正在考虑是否离职",
        "new": "用户已经明确希望离职"
      }
    ],
    "people_and_relationships": [],
    "constraints_and_resources": [],
    "coping_attempts_and_outcomes": [],
    "support_preferences_and_boundaries": [],
    "communication_preferences": []
  },
  "remove": {
    "active_concerns": [],
    "events_and_triggers": [],
    "functional_impacts": [],
    "goals_and_priorities": [],
    "people_and_relationships": [],
    "constraints_and_resources": [],
    "coping_attempts_and_outcomes": [],
    "support_preferences_and_boundaries": [],
    "communication_preferences": []
  }
}
```

`add` 和 `remove` 的元素是字符串；`replace` 的元素是严格的 `{old, new}` 对象。

操作语义：

- `add`：最新 seeker turn 明确披露了此前不存在的新事实；
- `replace`：最新 seeker turn 修正、细化或更新了一条现有事实；
- `remove`：用户明确否认旧事实，或明确说明该问题、约束或目标已经不再成立；
- 未被 patch 提及的信息原样保留；
- 最新一轮没有新事实时，三个操作中的所有数组都为空。

## Context 更新器的输入和规则

Updater 输入：

```json
{
  "previous_context": {},
  "recent_dialogue": [
    {"role": "supporter", "content": "..."},
    {"role": "seeker", "content": "..."}
  ]
}
```

初版使用最近两轮对话。较早信息已经由 `previous_context` 承载；recent dialogue 只用于解析代词、省略和对 supporter 问题的回答。

Updater 必须遵守：

1. 只有最新 seeker turn 能触发事实变更；
2. supporter 内容只能帮助理解指代，不能作为用户事实来源；
3. 只记录用户明确陈述或可以近似逐字改写的内容；
4. 不推断人格、诊断、隐含动机或第三方心理；
5. 保留用户原始的不确定性、否定和条件限制；
6. 不把七维 STATE 标签改写成 Context 事实；
7. 每个字符串只表达一个原子事实；
8. 不重述未变化的已有 Context；
9. 新信息修正旧信息时必须使用 `replace`；
10. 不因本轮没有提及就删除旧信息。

## 合并器与校验规则

Reducer 是确定性的，不由 LLM 自由生成合并结果。

校验顺序：

1. patch 必须包含 `add`、`replace`、`remove`；
2. 每个操作必须包含完整九字段，禁止额外字段；
3. 所有字符串必须去除首尾空白且非空；
4. 同一字段内的重复 `add` 视为 no-op；
5. `replace.old` 必须与旧 Context 中某个字符串完全一致，否则拒绝该条操作；
6. `remove` 目标不存在时视为 no-op；
7. 同一旧事实不能在同一 patch 中同时被 replace 和 remove；
8. 同一新事实不能在同一 patch 中同时被 add 和 remove；
9. 任一字段内的结果去重但保持原有顺序；
10. 一个非法条目只拒绝该条目，其余合法条目继续合并；结构级非法或 JSON 解析失败才拒绝整个 patch。

合并后得到唯一 `context_after`。运行时只向下游传递 `context_after`，不传递历史 patch。

## STATE 与 PLAN 的输入边界

State Analyzer 接收：

- 当前可见 dialogue history；
- 更新后的 `context_after`；
- 现有七维 STATE label guide。

State Analyzer 必须优先使用最新 seeker 明确表达。Context 是背景证据，不能机械映射为 STATE，也不能覆盖当前明确表达。例如，Context 中的“用户通常不喜欢别人替其做决定”不能覆盖本轮“请直接告诉我该怎么办”，后者应使当前 `advice_receptivity` 倾向 `requested`。

Planner 接收可见 dialogue history、`context_after` 和当前 STATE。PLAN 仍然只描述 supporter 的下一步支持目标、策略和 response act，不写回 Context。

supporter response 生成后，本轮 Context 和 STATE 均冻结。只有下一条 seeker turn 能启动下一次 Context 更新和 STATE 重估。

## 逐轮审计记录

每个 supporter 目标轮保存以下概念记录，其中 `UserContext` 与 `ContextPatch` 分别采用前文定义的完整数据结构：

```text
turn_id: 9
context_before: UserContext
context_patch: ContextPatch
context_after: UserContext
state: StateBlackboard
plan: PlanSelection
response: string
```

`turn_id` 只存在于整轮审计记录中，不重复写入每条 Context 信息。该记录允许离线重放 reducer、审查错误更新，并构造训练样本。

## 失败处理

- Updater 返回非法 JSON：沿用现有 structured-output 重试机制；重试耗尽后使用空 patch。
- patch 结构非法：拒绝整个 patch，保留 `context_before`。
- 单个 replace/remove 目标非法：仅跳过该操作，记录校验错误。
- Updater 最终失败：STATE 继续使用旧 Context 和可见历史运行，不阻断回复生成。
- State Analyzer 缺少足够证据：相应字段使用 `unknown`，Planner 转向探索、澄清或低假设回复。
- Context 与最新 seeker 明示内容冲突：最新明示内容优先；Updater 应在同轮用 replace/remove 修正 Context。
- 初版不自动总结过长 Context。先统计长度、重复率和对性能的影响，再决定是否单独设计压缩机制。

## 测试设计

### 数据结构测试

- Context 和每个 patch operation 都包含完整九字段；
- 不允许额外字段；
- add/remove 只接受非空字符串；
- replace 只接受非空 `{old, new}`；
- 全空 patch 合法。

### 合并器单元测试

- 新增一个事实；
- 重复新增为 no-op；
- 精确替换现有事实；
- replace.old 不存在时跳过；
- 删除现有事实；
- 删除不存在事实为 no-op；
- 同轮冲突操作被拒绝；
- 合并后去重并保持顺序；
- 旧 Context 在空 patch 下逐字保持不变。

### 边界行为测试

- seeker 明确事实可以进入 Context；
- supporter 建议不能进入 Context；
- emotion、need、receptivity、intent、capacity 和 continuation 不进入 Context；
- 未被本轮提及的旧事实被保留；
- 用户纠正自己时旧事实被替换；
- 用户带有不确定性的表达不能被升级为确定事实；
- 单轮文本风格不能触发 `communication_preferences`；
- 人口统计和人格属性不因刻板推断进入 Context。

### 串行集成测试

- Context Updater 在 State Analyzer 之前运行；
- State Analyzer 接收 reducer 产生的 `context_after`；
- Planner 接收同一版本的 Context 和 STATE；
- supporter response 不反向修改 Context 或 STATE；
- Updater 失败时旧 Context 能安全传递到下游；
- 审计记录可以从 `context_before + context_patch` 重放得到 `context_after`。

## 实验与成功标准

至少比较：

```text
History only
History + STATE
History + Context
History + Context + STATE
History + Context + STATE + PLAN
```

Context Updater 单独评估：

- schema valid rate；
- add/replace/remove 操作正确性；
- explicit-fact precision；
- unsupported-information rate；
- stale-fact retention 和 correction accuracy；
- STATE 信息误写入 Context 的 contamination rate。

端到端评估关注：

- STATE 判断是否因相关 Context 获得改善；
- PLAN 是否对现实约束、既往尝试和用户边界更敏感；
- 回复是否减少重复建议和不切实际建议；
- 无关人口统计信息是否不改变本应相同的情绪理解与支持决策。

数据划分必须按完整 conversation 切分。同一对话的 Context 演化轨迹不得跨 train/dev/test。

## 与现有代码的预期集成点

后续实现计划应围绕以下最小改动展开：

1. 在 `src/ibd/schemas.py` 增加 Context、ReplacePair 和 ContextPatch 严格 schema；
2. 在 `src/ibd/prompting.py` 增加 `context_updater` role 和规则；
3. 新增纯确定性的 Context reducer；
4. 在 `src/ibd/teacher.py` 中把 Context 更新置于现有 `multi_view_state_analyzer` 之前；
5. 扩展 Teacher trace/export，使其保存 context before、patch 和 after；
6. 让 State Analyzer、Planner 和候选生成器接收更新后的 Context；
7. 为 schema、reducer、prompt contract、串行调用顺序和 export 添加测试。

不在第一版引入新的 student Context token。第一阶段先验证显式 Context 是否改善 STATE、PLAN 和 response；只有在贡献得到验证后，再单独设计 Context latent-slot distillation，避免一次改变数据、teacher 和 student 三个核心变量。

## 最终决策

采用以下已确认设计：

- 模型推理侧维护 `Seeker Context Memory`，不宣称获得用户真实内在状态；
- Context、STATE、PLAN 三层职责严格分离；
- 使用 `Context -> STATE -> PLAN -> response` 串行更新；
- Context 包含九组决策相关事实字符串；
- 运行时 Context 不保存 kind、basis、source turn、first/last observed turn、status 或 memory ID；
- Context 只保存用户明确表达的信息，推断留在 STATE；
- Updater 只产生 add、replace、remove patch；
- reducer 确定性合并并验证；
- supporter 回复不能自行改变用户 Context 或 STATE；
- 通过模块级消融和逐轮边界测试证明每一层的独立贡献。
