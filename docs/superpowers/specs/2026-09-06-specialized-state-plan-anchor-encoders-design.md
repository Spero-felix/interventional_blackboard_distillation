# 专用 STATE/PLAN Anchor Encoder：原理与 Stage B 训练设计

## 0. 文档范围

本文提出一种替代“将 Teacher 标签 JSON 交给冻结通用语言模型编码”的方案：训练两个专用 target encoder，分别为 `STATE` 和 `PLAN` 产生稳定的 anchor。

本文只讨论 Stage B。Student 根据完整对话生成回复，同时让两个指定 slot 的 hidden state 对齐到 Teacher 标签导出的 anchor；不涉及 B2、Stage C 或反事实训练。

为保证任何 Markdown 阅读器都能显示，本文不用 LaTeX/MathJax/KaTeX。独立公式一律使用普通 `text` 代码块；行内符号使用反引号。

## 1. 要解决的问题

一条训练样本写作：

```text
(H_n, S_n, P_n, y_n)
```

- `n`：样本编号。
- `H_n`：完整可见对话历史。
- `S_n`：Teacher 给出的七维 STATE 标注。
- `P_n`：Teacher 给出的最终单策略 PLAN 标注。
- `y_n`：Teacher 最终回复。

Student 正常学习的是：

```text
H_n -> y_n
```

Stage B 额外希望两个特殊 token 位置的向量承担较窄功能：

```text
H_n -> z_state_n
H_n -> z_plan_n
```

`z_state_n` 与 `z_plan_n` 是 Student 某层的 hidden state，形状均为 `[d]`，其中 `d` 是该层 hidden size，例如 3584。

Teacher 标签却是结构化字段和短文本，无法直接与 `[d]` hidden state 比较。因此另行构造两个 target：

```text
S_n --StateEncoder--> a_state_n in R^d
P_n --PlanEncoder----> a_plan_n  in R^d
```

`a_state_n`、`a_plan_n` 就是 anchor。最关键的边界是：两个 target encoder 只读 Teacher 标签，绝不读取对话 `H_n`。anchor 表达的是标签条件，不是整段对话的副本。

## 2. 两套独立坐标系

STATE 与 PLAN 的数据类型不同：

- STATE 是七个具名、有限取值的类别字段。
- PLAN 是策略类别、回复目标和回复动作。

因此分别训练：

```text
E_state(S; phi_state) = a_state
E_plan(P; phi_plan)   = a_plan
```

虽然两个输出的形状都为 `[d]`，它们并不处于天然可比较的同一空间。不同 encoder 可以通过旋转或重参数化得到同样可预测的表示，却让每个维度的数值完全不同。

所以只允许以下两种比较：

```text
Student STATE slot <-> StateEncoder anchor
Student PLAN slot  <-> PlanEncoder anchor
```

不要计算或解释 `cosine(a_state, a_plan)`。STATE bank 与 PLAN bank 要分开保存、分开取样、分开计算 loss。两者同为 `[d]`，只是为了能与 Student 同一层的向量直接对齐。

## 3. 标签 schema

### 3.1 STATE

将 STATE 表示为七元组：

```text
S = (s_1, s_2, ..., s_7)
```

每个 `s_i` 从对应字段自己的词表 `V_i` 取值：

```text
dominant_emotion: overwhelm
distress_level: high
primary_support_need: validation
advice_receptivity: hesitant
action_intent: ambivalent
action_capacity: limited
continuation_intent: engaged
```

字段名是标签的一部分。`distress_level=unknown` 与 `action_capacity=unknown` 不是同一个无类型的 `unknown`。

### 3.2 PLAN

PLAN 写作：

```text
P = (strategy, goal, act)
```

优先将 `goal` 和 `act` 规范为受控标签：

```text
goal: emotional_validation | clarify_need | support_decision | ...
act:  name_emotion | ask_open_question | offer_one_option | ...
```

这样 PLAN 也变成小型类别表格，含义稳定、可恢复、可审计。若必须保留自由文本，见第 6 节。

## 4. StateEncoder：将七个字段编码成向量

### 4.1 字段 token

设 encoder 内部维度为 `r`，例如 256。对字段 `i`，学习：

```text
field_embedding[i]        in R^r
value_embedding[i][value] in R^r
```

输入 token 是二者相加：

```text
t_i = field_embedding[i] + value_embedding[i][s_i]
```

其中：

- `field_embedding[i]` 表示“这是哪个字段”；
- `value_embedding[i][s_i]` 表示“该字段取了什么值”；
- 两者相加后，相同字符串值出现在不同字段中仍可被区分。

### 4.2 字段交互与汇聚

构造 token 序列：

```text
[STATE] t_1 t_2 t_3 t_4 t_5 t_6 t_7
```

送入一个小型 Transformer：

```text
[h_cls, h_1, ..., h_7] = StateTransformer([STATE], t_1, ..., t_7)
```

`self-attention` 让字段可以相互影响。例如 `high distress` 搭配 `advice_receptivity=requested`，可与搭配 `advice_receptivity=closed` 形成不同的综合表示。TabTransformer 是“类别字段嵌入后用 self-attention 建模字段交互”的直接架构参考。[TabTransformer](https://arxiv.org/abs/2012.06678)

取 `[STATE]` 的输出 `h_cls`，用线性层映射到 Student hidden size：

```text
raw_anchor_state = W_state @ h_cls + b_state
```

- `W_state` 的形状为 `[d, r]`。
- `b_state` 的形状为 `[d]`。
- `raw_anchor_state` 的形状为 `[d]`。

随后进行 L2 归一化：

```text
anchor_state = raw_anchor_state / l2_norm(raw_anchor_state)

l2_norm(v) = sqrt(sum over k of v[k]^2)
```

归一化后 anchor 长度为 1。后续点积就是 cosine similarity，模型不能单靠把向量长度变大来降低 loss。

## 5. StateEncoder 如何学习“确实保留了字段”

只输出向量不能证明向量带有七个字段的信息。因此，从同一个 `anchor_state` 接七个字段恢复头。

对字段 `i`：

```text
logits_i = W_i @ anchor_state + b_i
```

`logits_i` 给出该字段所有候选值的未归一化分数。softmax 将其变为概率：

```text
p_i(value | anchor_state)
  = exp(logits_i[value])
    / sum(exp(logits_i[candidate]) for candidate in V_i)
```

若真实值为 `s_i`，该字段的交叉熵为：

```text
loss_i = -log(p_i(s_i | anchor_state))
```

例子：模型为真实值给 0.70 概率时，loss 约为 `0.357`；只给 0.01 时，loss 约为 `4.605`。所以优化会迫使 anchor 保留能恢复真实字段的信息。

七个字段的总损失：

```text
loss_state_encoder = sum(weight_i * loss_i for i in 1..7)
```

`weight_i` 是字段权重。若一个字段里 `unknown` 占绝大多数，应使用类别权重或重采样，避免模型只猜多数类却获得表面高分。

该 loss 的限制：若两个字段在数据中高度相关，模型可能用字段 A 猜字段 B。因此验收应逐字段报告 held-out accuracy 和 balanced accuracy，而不能只报告平均 loss。

## 6. PlanEncoder：两种实现路线

### 6.1 路线 A：受控 PLAN taxonomy，推荐

当 `strategy`、`goal`、`act` 都是离散字段时，PlanEncoder 可复用 StateEncoder：

```text
[PLAN] strategy-token goal-token act-token
```

从 `[PLAN]` 输出投影到 `[d]`，再用三个分类头恢复三个字段。优点是 anchor 的含义完全由 schema 定义。

这与 Concept Bottleneck Model 的思路一致：使用训练时给定的高层概念作为中间表示，而不是让模型自行发明不可审计的隐藏概念。[Koh et al., 2020](https://proceedings.mlr.press/v119/koh20a.html)

### 6.2 路线 B：自由文本 goal/act

当 `goal`、`act` 必须保留自由文本时：

```text
text_representation
  = TextEncoder([GOAL], goal, [ACT], act)

strategy_representation
  = strategy_embedding[strategy]

plan_representation
  = FusionMLP(concat(strategy_representation, text_representation))

anchor_plan
  = l2_normalize(W_plan @ plan_representation + b_plan)
```

`concat(x, y)` 表示拼接向量。文本子 encoder 可使用 Sentence-BERT 风格句向量模型初始化；这类模型的目的就是产生可使用 cosine 比较的定长句向量。[Sentence-BERT](https://aclanthology.org/D19-1410.pdf)

训练至少需要策略分类：

```text
loss_strategy = -log(p(strategy | anchor_plan))
```

若希望 anchor 保留 goal/act 的具体内容，可加自回归文本重建。对 goal token 序列 `g_1 ... g_T`：

```text
loss_goal = -sum(
  log p(g_t | g_1 ... g_(t-1), anchor_plan)
  for t in 1..T
)
```

`loss_act` 同理：

```text
loss_plan_encoder
  = loss_strategy
  + lambda_goal * loss_goal
  + lambda_act * loss_act
```

自由文本在小数据下容易记忆原句而不形成稳定语义。因此应优先规范 taxonomy；若必须使用文本，则冻结大部分文本 encoder，只训练 adapter、fusion 和投影层。

## 7. 可选的对比学习

当 StateEncoder 直接接收 canonical 字段 ID 时，同一 STATE 的输入已经完全一致，因此对比学习不是必须项。

若输入格式可能变化，可为同一标签构造不改变语义的两个视图：

```text
a_1 = Encoder(label, view_1)
a_2 = Encoder(label, view_2)
```

例如字段顺序或等价格式不同。InfoNCE 的普通写法是：

```text
loss_nce = -log(
  exp(sim(a_1, a_2) / temperature)
  /
  sum(exp(sim(a_1, candidate) / temperature)
      for candidate in batch)
)
```

- `sim(u, v)`：单位向量的点积，即 cosine similarity。
- `temperature`：正数；更小时，更强调正样本得分必须明显大于候选负样本。

监督对比学习的基本思想是拉近同类表示、推远异类表示。[Khosla et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

但不应把所有不同 STATE 都视作强负样本：两个 STATE 可能只改一个字段，按定义应保留部分相似性。因此该项应是小权重的稳定项；字段恢复仍是主要监督。

## 8. target encoder 的训练、冻结与预计算

先只使用 Teacher 标签训练 target encoder：

```text
train StateEncoder using:
  loss_state_encoder + lambda_state_contrast * loss_state_contrast

train PlanEncoder using:
  loss_plan_encoder + lambda_plan_contrast * loss_plan_contrast
```

训练完成后冻结：

```text
freeze(phi_state)
freeze(phi_plan)
```

然后为每条样本预计算：

```text
anchor_state_n = StateEncoder(S_n; frozen phi_state)
anchor_plan_n  = PlanEncoder(P_n; frozen phi_plan)
```

不要在 Student 训练中继续更新 target encoder。若同时最小化：

```text
min over theta, phi:
  squared_l2_norm(z_theta(H) - a_phi(S))
```

Student 和 target 可以一起漂移，甚至一起塌缩到无意义表示，loss 仍可能变小。

冻结后的目标才是：

```text
min over theta:
  squared_l2_norm(z_theta(H) - a_frozen_phi(S))
```

此时 anchor 是固定答案，Student 必须从对话预测它。以固定中间表征指导 Student hidden state 属于表示蒸馏思路；FitNets 使用 Teacher 中间“hint”监督 Student 的中间表示。[Romero et al., 2015](https://mlanthology.org/iclr/2015/romero2015iclr-fitnets/)

重新随机训练 target encoder，即使得到相同字段恢复分数，也会定义新坐标系。因此不同 checkpoint 生成的 anchor 不能混用。

## 9. Stage B：Student 如何对齐 anchor

Student 的输入仍是：

```text
完整对话 H + <|ibd_state|> + <|ibd_plan|> + 回复 y
```

在选定层取得：

```text
z_state_n = StudentStateSlot(H_n)
z_plan_n  = StudentPlanSlot(H_n)
```

两者都为 `[d]`。归一化后，最简单的 alignment loss：

```text
loss_state_align = mean over n of (
  1 - dot(normalize(z_state_n), anchor_state_n)
)

loss_plan_align = mean over n of (
  1 - dot(normalize(z_plan_n), anchor_plan_n)
)
```

对于单位向量：

```text
dot(same direction) =  1  -> loss = 0
dot(orthogonal)     =  0  -> loss = 1
dot(opposite)       = -1  -> loss = 2
```

回复损失仍是普通 token-level cross entropy：

```text
loss_response = -sum(
  log p(y_t | H, STATE-token, PLAN-token, y_before_t)
  for response token t
)
```

总目标：

```text
loss_stage_B
  = loss_response
  + lambda_state * loss_state_align
  + lambda_plan * loss_plan_align
```

`lambda_state`、`lambda_plan` 控制结构监督强度。过大可能牺牲回复能力；过小则 slot 不会稳定保留标签信息。

部署时不需要 Teacher 标签、target encoder 或 anchor artifact。它们只在训练期定义目标；推理时由 Student 根据对话自然计算两个 slot。

## 10. 为什么相同 STATE anchor 不会让回复相同

例如：

```text
H1: “我快被工作压垮了，不知道是否该辞职。”
H2: “照顾家人让我喘不过气，也不知道该怎么办。”
```

二者可能有相同 STATE：

```text
overwhelm + high distress + validation need + hesitant advice
```

因此按定义：

```text
anchor_state(H1) = anchor_state(H2)
```

但这不要求最终回复相同。两段对话本身的 token hidden states 仍不同，所以 Student 可以对 H1 讨论工作压力，对 H2 回应照顾责任。STATE slot 只压缩 schema 指定的支持状态；未定义的具体情节仍保留在完整对话上下文中。

如果某个细节应由 STATE slot 区分，应把该维度加入 schema；不能要求同一个七维 STATE anchor 同时保存全部对话细节。

## 11. 重复标签与多正样本

若两条样本的完整 STATE 相同：

```text
S_i = S_j
```

则其 target anchor 应相同：

```text
anchor_state_i = anchor_state_j
```

若 Stage B 使用 batch 内对比损失，`j` 不能被错误当作 `i` 的负样本。应把所有具有同一完整 STATE 的样本作为正样本：

```text
positive_set(i) = { j | S_j equals S_i }
```

多正样本对比损失的通用写法：

```text
loss_i = -log(
  sum(exp(sim(z_state_i, anchor_state_j) / temperature)
      for j in positive_set(i))
  /
  sum(exp(sim(z_state_i, anchor_state_k) / temperature)
      for k in batch)
)
```

这表示任何同 STATE anchor 都是正确目标。若使用纯 cosine regression 而非 batch contrastive loss，则没有这种 false negative 问题。PLAN 被离散化为 taxonomy 后，同样适用。

## 12. PyTorch 级伪代码与张量形状

`state_ids` 的形状为 `[B, 7]`，其中 `B` 是 batch size：

```python
def encode_state(state_ids):
    # state_ids: [B, 7]
    field_tokens = []
    for i in range(7):
        value = value_embeddings[i](state_ids[:, i])  # [B, r]
        field_tokens.append(value + field_embeddings[i])

    x = torch.stack(field_tokens, dim=1)               # [B, 7, r]
    cls = state_cls.expand(x.shape[0], 1, -1)           # [B, 1, r]
    hidden = state_transformer(torch.cat([cls, x], 1))  # [B, 8, r]
    anchor = F.normalize(project(hidden[:, 0]), dim=-1) # [B, d]
    logits = [head(anchor) for head in field_heads]
    return anchor, logits
```

预训练 StateEncoder：

```python
anchor, logits = encode_state(state_ids)
state_loss = sum(
    F.cross_entropy(logits[i], state_ids[:, i])
    for i in range(7)
)
state_loss.backward()
optimizer.step()
```

冻结并预计算：

```python
state_encoder.eval()
for parameter in state_encoder.parameters():
    parameter.requires_grad_(False)

with torch.inference_mode():
    state_anchor = state_encoder(state_ids)[0]
```

Stage B 的 alignment：

```python
state_slot = output.slots.STATE  # [B, d]
plan_slot = output.slots.PLAN    # [B, d]

state_alignment = 1 - (
    F.normalize(state_slot, dim=-1) * state_anchor
).sum(dim=-1).mean()

plan_alignment = 1 - (
    F.normalize(plan_slot, dim=-1) * plan_anchor
).sum(dim=-1).mean()

loss = response_cross_entropy \
    + lambda_state * state_alignment \
    + lambda_plan * plan_alignment
```

## 13. Anchor artifact 应保存什么

```text
state: [num_examples, d]
plan:  [num_examples, d]
example_id -> row mapping

metadata:
  schema_version
  state_encoder_checkpoint_hash
  plan_encoder_checkpoint_hash
  target_encoder_training_split
  anchor_dimension
  normalization = l2
  state_vocabularies
  plan_vocabularies or text_encoder_revision
```

这些元数据防止不安全混用：

- 新 schema 的标签配旧 encoder；
- 不同随机初始化训练出的 target encoder 生成的 anchor；
- 不同 hidden size 的 anchor 与 Student layer；
- 不同 vocabularies 或 text encoder revision。

## 14. 训练 Student 前的验收

### 14.1 字段可恢复性

在 held-out 标签上逐字段报告 STATE accuracy 和 balanced accuracy；PLAN 则报告 strategy、goal、act 的对应恢复指标。不要只报告一个总平均值。

### 14.2 格式不变性

若 encoder 接收文本或多种序列化方式，同一条件的等价格式应获得高 cosine similarity。若输入是 canonical 字段 ID，这一性质按构造成立。

### 14.3 单字段敏感性

只替换第 `i` 个字段：

```text
S_prime = (s_1, ..., replacement_for_s_i, ..., s_7)
```

测量：

```text
delta_i = 1 - dot(anchor_state(S), anchor_state(S_prime))
```

`delta_i` 应明显大于数值噪声。若某字段替换后 anchor 几乎不变，且字段恢复准确率也低，说明 encoder 没有编码该字段。

### 14.4 碰撞检查

若 `S_i` 与 `S_j` 不同，却有：

```text
cosine(anchor_state_i, anchor_state_j) approximately 1
```

需检查两个标注是否实际近义、encoder 容量是否不足、字段是否严重不平衡，或恢复头是否只靠字段相关性猜测。

## 15. 局限

1. target encoder 定义的是训练期坐标约定，不是 STATE/PLAN 的天然本体。
2. 字段恢复高分不等于字段在向量中完全独立；数据相关性可造成捷径。
3. 自由文本 PLAN 的可审计性与稳定性弱于受控 taxonomy。
4. schema 若遗漏重要差异，anchor 会按定义丢弃该差异；encoder 不能自动补救。
5. checkpoint、词表或 schema 任一变化，都要求重新生成全部 anchor。

## 16. 推荐的最小可行版本

```text
STATE
- 7 个独立 value embedding 表
- 7 个 field embedding
- 2 层、4 或 8 heads 的小 Transformer
- 投影至 Student slot layer hidden size
- 7 个字段恢复头

PLAN
- 首选：strategy / goal / act 全部受控标签，复用小型类别 encoder
- 次选：strategy embedding + 冻结句向量文本子 encoder + 小型 fusion MLP

训练
- 先仅以 Teacher 标签训练 target encoder
- 在 dev 标签上验收后冻结
- 预计算并版本化 anchor artifact
- Stage B 仅让 Student 对应 slot 对齐自己的 anchor
```

该方案将 anchor 的含义从“通用语言模型对一段 JSON 文本的隐藏层输出”收紧为：

> 由固定 schema 和显式恢复目标训练得到、只承载 Teacher STATE 或 PLAN 标签的专用连续目标表示。
