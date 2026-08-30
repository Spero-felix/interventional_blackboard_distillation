# SSConv 2000 条链路策略坍缩审计

## 结论

当前链路已经出现可观测的策略/措辞坍缩，不只是理论风险。第一根因是旧
`prepared.json` 固定截断到最后一个可回复位置；第二个明确收缩点位于 Teacher 的
planner + final selector：`Providing Suggestions` 出现在 98.65% 的 plan 中，并在
出现时以 89.36% 的概率胜出，最终占 2000 条回复的 88.15%。Student 训练继续放大
了 Teacher 的固定句式：旧 holdout 上 Stage C 有 47.67% 的回复以
“Would you like to” 开头。

此外还有一个独立的数据污染缺陷：candidate 后端没有解析 Markdown fenced JSON，
而是把整段 JSON 当成 `response`。原始 6000 个候选中有 156 个受污染；原 final
selector 选中 15 个，而固定 S1 数据集强制选中其中 66 个。重新生成 Teacher 数据
前应先修复或加零容忍校验。

本报告使用已有 artifact 做追溯。新生成的 phase-balanced prepared artifact 尚未
运行 Teacher，因此不能假定其策略分布已经改善。

## 审计范围与数据

- 旧数据：`artifacts/full-2000/prepared.json`、Teacher traces、interventions 和
  intervention manifest。
- 固定 S1 实验：`artifacts/full-2000-first-plan/`。
- 已有 holdout 生成：`artifacts/quality/full-2000-holdout/generation/responses.jsonl`
  与 `artifacts/quality/first-plan/generation/responses.jsonl`。
- 既有策略诊断：
  `artifacts/quality/full-2000-holdout/plan-strategy-diagnostics/analysis_summary.json`。
- 代码边界：SocialSim preparation、Teacher planner/candidates/final selector、
  intervention 验收、Stage A/B/C loss、DataLoader 和 checkpoint selection。

所有计数均在 2026-08-30 重新从 JSON/JSONL artifact 计算。

## 1. 已观察：旧截断把任务推向收尾阶段

旧 2000 条中 2000/2000 都属于 late，1998/2000 恰好使用最后一个候选回复位置。
旧 history 长度中位数为 23 turns。已有 holdout 语义诊断又把 300 条分成：

| 对话状态 | 数量 | 占比 |
|---|---:|---:|
| 明确收尾 | 239 | 79.67% |
| 行动准备 | 34 | 11.33% |
| 中段求助 | 27 | 9.00% |

明确收尾的 239 条中，211 条选择 `Providing Suggestions`（88.28%）。这说明旧采样
不仅丢失中段上下文，还把 Teacher 长期置于“给最后一步建议”的局部任务分布。

本次实现后的临时 v2 artifact 为严格 `200/1600/200`，target progress 中位数为
0.5833，2000 条 target-response 泄漏为 0。该修复解决输入分布，但尚未证明
Teacher 策略分布恢复。

## 2. 已观察：Planner 覆盖收缩，Final Selector 进一步收缩

Planner 的 6000 个 strategy slots：

| 策略 | slot 数 | 出现在会话中的比例 | 最终选择数 | 出现后胜率 |
|---|---:|---:|---:|---:|
| Providing Suggestions | 1973 | 98.65% | 1763 | 89.36% |
| Affirmation and Reassurance | 1845 | 92.25% | 97 | 5.26% |
| Question | 1435 | 71.75% | 108 | 7.53% |
| Reflection of feelings | 743 | 37.15% | 32 | 4.31% |
| Information | 3 | 0.15% | 0 | 0% |
| Self-disclosure | 1 | 0.05% | 0 | 0% |
| Restatement or Paraphrasing | 0 | 0% | 0 | — |
| Others | 0 | 0% | 0 | — |

Planner 虽保证单条 trace 内三个策略不同，但没有跨数据集覆盖约束。其 prompt 只要求
“three plausible but different response approaches”
（`src/ibd/prompting.py:50-60`）。Final selector 又把 concrete support 作为显式优先级
并在 ready/action 情况下偏好 actionable response
（`src/ibd/prompting.py:68-87`）。在旧 late/closure-heavy history 上，这一规则与
`Providing Suggestions` 高度对齐。

候选位置也不平衡：S1/S2/S3 最终选择比例为 11.80%/68.15%/20.05%。这不完全由策略
位置解释：同为 `Providing Suggestions` 时，S1/S2/S3 胜率仍为
98.70%/94.77%/73.65%；同为 `Question` 时为 11.61%/30.00%/0.97%。代码把三个候选
按固定顺序交给 selector（`src/ibd/teacher.py:209-228,299-327`），每个位置还固定绑定
candidate role 和 seed（`src/ibd/teacher.py:231-272`）。这是可疑的位置/seed 放大器，
但仅凭观察数据不能区分两者。

## 3. 已观察：句式坍缩已传播到 Student

Teacher 全量 2000 条虽然有 1999 个不同的完整字符串，但前四词高度集中：

- `Would you like to`：592/2000（29.60%）；
- 在 `Providing Suggestions` 内：592/1763（33.58%）；
- 其次为 `Would you prefer to`：106、`You could try a`：90。

完整字符串“几乎都不同”因此不能排除模板坍缩。holdout 实际生成的 top-prefix 比例：

| 模型 | 最常见四词前缀 | 300 条中的比例 |
|---|---|---:|
| Teacher | Would you like to | 35.00% |
| Base | I'm glad to hear | 13.00% |
| IBD Stage B | Would you prefer to | 40.33% |
| IBD Stage C | Would you like to | 47.67% |
| Standard SFT | Would you prefer to | 43.00% |

Stage B、Stage C 和普通 SFT 都比 Teacher 更集中，说明主要坍缩信号来自监督目标，
而不是 latent slot 独有问题；Stage C 最严重，说明后续训练没有自动恢复多样性。

固定 S1 实验把策略分布改为 Affirmation 49.30%、Question 38.75%、Reflection
8.10%、Suggestions 3.85%，其 IBD Stage B top-prefix 降到 14.67%。这说明 Teacher
目标分布对 Student 输出风格有直接影响。但固定位置不是通用修复：脚本明确把
2000/2000 都改为 S1（`scripts/select_first_plan_candidate.py:38-56,78-86`），会引入
新的单位置/单 seed 实验偏差。

## 4. 已观察：fenced JSON 被当作自然回复

candidate 配置关闭 provider JSON mode 后，`_PlainTextCandidateBackend` 先直接
`json.loads`；解析失败且文本不以 `{` 或 `[` 开头时，就把完整文本包装进
`response`（`src/ibd/teacher.py:39-80`）。Markdown fenced JSON 以反引号开头，
因此进入错误分支，并且外层包装后的对象能通过 Candidate schema。

重新统计结果：

- 6000 个候选中 156 个 `response` 以 fenced JSON 开头；
- 按位置为 S1=66、S2=35、S3=55；
- 原 final selector 选中 15 个；
- 固定 S1 artifact 强制选中全部 66 个 S1 污染候选。

这是数据正确性问题，不应等待新 phase 数据再判断。重新生成前必须做到：正确提取
fence 内对象，或拒绝 candidate 并触发 schema retry；prepared/trace 发布门槛要求
fenced JSON response 数量严格为 0。

## 5. 已观察：Intervention 验收产生 function 失衡

构造器按排序后的 example ID 交替分配 STATE/PLAN，初始各 1000 条
（`src/ibd/pipeline.py:555-581`）。验收后：

| function | 尝试 | 保留 | 保留率 |
|---|---:|---:|---:|
| PLAN | 1000 | 999 | 99.90% |
| STATE | 1000 | 559 | 55.90% |

STATE 的 441 个排除由 bidirectional disagreement 313、no localized effect 126、
invalid counterfactual 2 构成。训练 split 最终为 PLAN 746、STATE 410，即
64.53%/35.47%。验收逻辑要求 safety 和 effect verification 通过
（`src/ibd/interventions.py:238-290`），这个门槛本身合理；问题是 Stage C 随后直接
消费全部 retained rows（`src/ibd/pipeline.py:135-172`），DataLoader 只 shuffle，
没有 function-balanced sampler（`src/ibd/pipeline.py:1073-1083`）。因此它是已观察到
的控制功能失衡及潜在放大器。

它不是当前“建议策略占 88%”的主要原因：intervention 前后 Suggestions share 仅从
88.15% 变为 87.74%，各策略保留率为 77.54%–82.41%。它主要威胁 STATE 控制学习，
而不是进一步改变自然回复策略分布。

## 6. 可能放大：同一 Teacher 目标被多阶段重复强化

`build_stage_rows` 让 A/B 都使用同一 `final_response`
（`src/ibd/pipeline.py:118-134`）。当前配置 A=5 epochs、B=5 epochs
（`configs/experiments/ab2000-lr-8e-5.yaml:30-38`）；Stage B 的 loss 仍含完整
response loss，再加 slot alignment（`src/ibd/training.py:99-122`，
`src/ibd/trainer.py:172-190,374-379`）。因此 dominant Teacher 句式最多连续得到十轮
自然回复监督。

Stage C 又对 retained subset 的 original response 做自然路径 CE；其
`normal_loss = original CE + 0.1 * replay`，而 replay 内再次包含 response loss
（`src/ibd/trainer.py:466-480`）。这不是数学 bug，但会继续放大已有目标分布。
模型生成统计证明“放大”存在；不能仅靠降低 epoch 断言能解决根因。

## 7. 监控缺口：checkpoint loss 看不到生成坍缩

当前 checkpoint 分别按 A 的 dev SFT loss、B 的 replay loss、B2/C 的 total loss
选择（`src/ibd/pipeline.py:874-881,1130-1144`）。保存的 dev metrics 也只有 loss
组件（`src/ibd/pipeline.py:1112-1128`），没有：

- phase × selected strategy 分布；
- top-prefix、distinct-n 或语义模板比例；
- candidate position bias；
- STATE/PLAN 分 function 指标；
- 固定 generation probe 的多样性和质量。

低 dev loss 可以与更强模板化同时发生，所以 early stopping 不能作为 anti-collapse
机制。

## 8. 已排除的怀疑点

- 对话不会跨 split：prepared schema 对 conversation ID 做全局唯一校验。
- Student 编码不会静默截掉长 history；超过 `max_length` 会直接报错
  （`src/ibd/student_data.py:73-78`）。
- 完整回复的 exact duplication 很低（旧 Teacher 仅 1 条重复），但这不等于句式
  多样，因为 prefix collapse 很强。
- intervention filter 没有显著扩大 Suggestions 占比，因此不是当前策略坍缩根因。

## 发布门槛与后续顺序

### 重新生成 Teacher 前的硬门槛

1. 使用 v2 prepared artifact；每个 split phase counts 必须精确匹配 manifest。
2. 修复/拒绝 fenced JSON candidate，候选与 final response 污染数必须为 0。
3. 不复用旧 trace、intervention、anchor 或旧 cache namespace。

### 新 Teacher traces 生成后的 stop-and-review 门槛

这些是触发人工复核的诊断阈值，不是强制人为配平标签：

1. 输出 overall 和 phase × strategy 表；任一策略 overall 超过 60% 时暂停训练复核。
2. 输出 strategy × candidate position 胜率；任一位置总体超过 50% 或同策略跨位置
   胜率差超过 20 percentage points 时，先做候选顺序随机化实验。
3. top four-word prefix 超过 20% 时暂停训练，检查 Teacher prompt/template。
4. intervention train rows 的 STATE/PLAN 任一低于 40% 时使用 balanced sampler 或
   显式 loss weighting，并分别报告两类 dev 指标。

### 若 phase-balanced Teacher 仍坍缩，再做的实验

1. 在送入 final selector 前确定性随机打乱候选顺序，选择后映射回原 candidate ID，
   用相同 traces 比较 position-conditioned selection rate。
2. 保留质量选择，不直接强制全局策略配额；先对 selector 的 readiness/action 偏置做
   phase 分层校准。
3. checkpoint selection 增加固定 probe generation 的 top-prefix、distinct-n、
   strategy proxy 和人工/LLM quality guardrail，不能只按 loss。
4. 对 Stage C 先平衡 STATE/PLAN batch，再讨论 margin、replay weight 或 epoch；否则
   调 loss 会掩盖数据验收造成的 function 失衡。
