# Others 对话管理策略设计

## 目标

在保留 ESConv 八类策略体系兼容性的前提下，为 `Others` 建立一个具有正面语义、可用于后续 Teacher 数据生成与 Final Selector 判定的严格定义，避免将其继续用作其他七类之外的兜底标签。

## 标签与展示名称

代码、结构化输出和数据中的规范标签继续使用：

```text
Others
```

提示词、文档和人工审查界面使用带解释性的展示名称：

```text
Others (Dialogue Management and Social Courtesy)
```

中文名称为：

```text
Others（对话管理与社交回应）
```

这样既保持既有 schema、数据和实验的标签兼容性，又避免模型把 `Others` 理解为无法分类时的默认选项。

## 核心定义

`Others` 的主要功能是处理对话关系和对话流程，而不是处理用户问题本身。

判断时关注回复的主要作用对象：

- 如果回复主要用于开启、维持、暂停、恢复或结束当前互动，可使用 `Others`。
- 如果回复主要作用于用户的处境、情绪、认知、决定或行动，应使用其他七类中更具体的策略。

`Others` 包括：

- 开场问候；
- 简短社交性回应；
- 回应感谢；
- 对话节奏或轮次管理；
- 适当的结束、告别及回应告别。

`Others` 不是其他七类的补集，也不能作为 Planner 为凑足三个不同候选而使用的 fallback 类别。

## 正式英文定义

> **Others (Dialogue Management and Social Courtesy):**
> Use this strategy only when the primary function of the response is to manage the conversational interaction itself rather than to address the seeker's situation, emotions, beliefs, decisions, or actions. This includes greetings, brief social acknowledgments, responses to gratitude, conversational pacing or transitions, and appropriate closing or farewell messages.
>
> Do not select Others if the response primarily asks for information, restates the seeker's meaning, reflects the seeker's feelings, shares personal experience, provides affirmation or reassurance, offers factual information, or gives suggestions. When another strategy clearly describes the primary response act, that more specific strategy takes precedence over Others.

## 正式中文定义

> **Others（对话管理与社交回应）**：
> 仅当回复的主要功能是管理对话互动本身，而不是处理用户的处境、情绪、认知、决定或行动时使用。包括开场问候、简短社交性回应、回应感谢、对话节奏或过渡管理，以及适当的结束和告别。
>
> 如果回复的主要功能是询问信息、复述用户含义、反映用户情绪、分享个人经历、给予肯定或安慰、提供事实信息或提出建议，则不得选择 `Others`。当其他任一策略能够明确描述回复的主要行为时，应优先选择该更具体的策略。

## 边界示例

| 回复 | 策略 | 原因 |
|---|---|---|
| “Hi, nice to meet you.” | `Others` | 纯开场问候 |
| “You're very welcome.” | `Others` | 回应感谢 |
| “Take your time; we can continue whenever you're ready.” | `Others` | 管理对话节奏 |
| “We can stop here if you'd like.” | `Others` | 管理对话是否继续 |
| “Bye, have a good day.” | `Others` | 告别和结束 |
| “Is there anyone you trust that you could talk to?” | `Question` | 询问用户可获得的支持 |
| “Maybe try writing down how you feel first.” | `Providing Suggestions` | 提供行动建议 |
| “It makes sense that you'd feel hurt after that.” | `Reflection of feelings` 或 `Affirmation and Reassurance` | 主要作用于用户情绪 |
| “Sometimes people withdraw when they're overwhelmed.” | `Information` | 提供解释性信息 |
| “I've been through something similar myself.” | `Self-disclosure` | 分享个人经历 |

当一个回复同时包含礼貌表达和实质性支持时，按照主要回复行为标注；只要其他策略能够更准确地描述该主要行为，就不选择 `Others`。

## Teacher 各角色的使用规则

### Planner

Planner 仅在最新用户轮次主要表现为问候、感谢、告别、暂停交流、明确表示没有其他问题，或显式讨论对话流程时，才应将 `Others` 纳入三个候选策略。

当用户仍在陈述问题、表达情绪、询问信息或寻求行动支持时，Planner 不应为了候选多样性选择 `Others`。

### Candidate Generator

分配到 `Others` 时，Candidate Generator 应生成一个完整但简洁的对话管理或社交回应。生成内容应清楚体现问候、致谢回应、节奏管理、结束或告别中的一种主要行为，不得用新的实质性支持行为替代指定策略。

### Final Selector

Final Selector 应根据候选回复的实际主要功能判断，而不能因为标签模糊或其他候选难以比较就选择 `Others`。

当用户的最新轮次已经转向感谢、告别或其他明确的对话管理行为时，`Others` 可以与其他候选正常竞争；当用户仍需要实质性支持时，应优先选择能准确满足该需要的具体策略。

## 非目标

- 不改变八类策略集合或结构化输出中的规范标签。
- 不把 `Others` 扩展成未分类内容的兜底类别。
- 不引入基于删除礼貌部分的额外判定测试。
- 不在本设计中增加平台或数据采集噪声的过滤规则。
- 不修改 Final Selector 的单候选选择输出协议。

## 验收标准

- `Others` 在提示词中具有独立、正面的功能定义，不再以“其他七类之外”定义。
- Planner 只在对话管理语境中考虑 `Others`，不会将其作为候选数量补足手段。
- Candidate Generator 能生成纯粹且可识别的对话管理或社交回应。
- Final Selector 在存在明确实质性支持策略时优先选择更具体的策略。
- 代码和数据仍使用规范标签 `Others`，保持现有 schema 与历史实验可比性。
