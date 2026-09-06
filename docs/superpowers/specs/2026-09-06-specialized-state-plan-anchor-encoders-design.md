# 专用 STATE/PLAN Anchor Encoder：原理与 Stage B 训练设计

## 0. 文档范围

本文说明一种替代“将 Teacher 标签 JSON 送入冻结通用语言模型并取某层最后 token hidden state”的 anchor 构造方法：训练两个专用 target encoder，分别为 `STATE` 和 `PLAN` 定义稳定的向量目标空间。

本文只讨论 Stage B：Student 从完整对话预测回复，并将两个指定 slot 的中间 hidden state 分别对齐到 Teacher 标签导出的 anchor。它不包含 B2、Stage C 或任何反事实/干预训练方案。

本文是设计与教程，不改变当前实现。所有公式均以单个样本为例；批量训练时再对样本取均值。

## 1. 问题：为什么需要专用 target encoder

一条训练样本记作：

\[
(H_n, S_n, P_n, y_n)
\]

符号含义如下：

- \(n\)：样本编号。
- \(H_n\)：完整可见对话历史。
- \(S_n\)：Teacher 对该对话给出的七维 STATE 标注。
- \(P_n\)：Teacher 的最终单策略 PLAN 标注。
- \(y_n\)：Teacher 的最终回复文本。

Student 的正常输入仍是对话，任务是生成回复：

\[
H_n \longrightarrow y_n
\]

而 Stage B 希望两个特殊 token 所在位置的 hidden state 分别承担更窄的功能：

\[
H_n \longrightarrow z^S_n
\]

\[
H_n \longrightarrow z^P_n
\]

其中 \(z^S_n\) 是 Student 的 `STATE` slot 向量，\(z^P_n\) 是 `PLAN` slot 向量。二者是 Student 某一层的连续向量，例如 \(\mathbb{R}^{3584}\)。

Teacher 标签却是结构化字段与短文本，无法直接与 hidden state 相比。因此需要将标签映射为同维度目标：

\[
S_n \xrightarrow{E_S} a^S_n \in \mathbb{R}^{d}
\]

\[
P_n \xrightarrow{E_P} a^P_n \in \mathbb{R}^{d}
\]

其中 \(E_S\) 和 \(E_P\) 是专用 target encoder，\(a^S_n\)、\(a^P_n\) 是 anchor，\(d\) 等于 Student slot 所在层的 hidden size。

关键点是：target encoder 只读取 Teacher 标签，绝不读取原对话 \(H_n\)。所以 anchor 表达的是“Teacher 已经抽取出的标签条件”，不是整段对话的另一份副本。

## 2. 两套独立坐标系，而不是一个共享语义空间

STATE 和 PLAN 的数据类型不同：

- STATE 是七个具名、有限取值的类别字段。
- PLAN 是策略类别，以及回复目标、回复动作。

因此采用两个 encoder：

\[
E_S(S;\phi_S)=a^S
\]

\[
E_P(P;\phi_P)=a^P
\]

\(\phi_S\)、\(\phi_P\) 是两套各自训练的参数。即使两个输出都属于 \(\mathbb{R}^d\)，也不代表其第 \(k\) 维具有相同含义。

两个独立 encoder 的坐标存在不可识别性：若 \(R\) 是任意正交矩阵，\(z\) 和 \(Rz\) 都可以携带相同可预测信息，但坐标数值已完全变化。因此以下跨空间比较没有定义良好的语义：

\[
\operatorname{cos}(a^S, a^P)
\]

正确的比较只有：

\[
z^S \leftrightarrow a^S
\]

\[
z^P \leftrightarrow a^P
\]

也就是说，STATE bank 和 PLAN bank 应分开保存、分开采样、分开计算损失。维度相同仅仅是为了让 anchor 能与 Student 的同层向量直接比较或替换。

## 3. 数据 schema

### 3.1 STATE

将 STATE 写成七元组：

\[
S=(s_1,s_2,\ldots,s_7)
\]

每个 \(s_i\) 从该字段自己的有限词表 \(\mathcal V_i\) 中取值。例如：

```text
dominant_emotion: overwhelm
distress_level: high
primary_support_need: validation
advice_receptivity: hesitant
action_intent: ambivalent
action_capacity: limited
continuation_intent: engaged
```

字段名是 schema 的一部分。即使两个字段都有 `unknown`，也不能将它们当作同一个无类型值。

### 3.2 PLAN

PLAN 写成：

\[
P=(p^{strategy},p^{goal},p^{act})
\]

其中 `strategy` 是离散策略类别，`goal` 和 `act` 可为受控标签或短文本。

优先级最高的方案是将 `goal` 与 `act` 规范成有限本体，例如：

```text
goal: emotional_validation | clarify_need | support_decision | ...
act: name_emotion | ask_open_question | offer_one_option | ...
```

如此 PLAN 也是小型类别表格，含义最稳定、最可审计。若必须保留自由文本，见第 6 节。

## 4. StateEncoder 架构

### 4.1 字段 token

令内部 embedding 维度为 \(r\)，例如 256。对每个字段 \(i\) 学习：

\[
e_i^{field}\in\mathbb{R}^{r}
\]

并为该字段的每个可能取值 \(v\in\mathcal V_i\) 学习：

\[
e_{i,v}^{value}\in\mathbb{R}^{r}
\]

字段 \(i\) 的输入 token 定义为：

\[
t_i=e_i^{field}+e_{i,s_i}^{value}
\]

`field embedding` 告诉 encoder“这是哪一个字段”；`value embedding` 告诉 encoder“该字段取了什么值”。两者相加后，`distress_level=unknown` 与 `action_capacity=unknown` 仍是不同 token。

### 4.2 聚合字段交互

构造输入序列：

```text
[STATE] t1 t2 t3 t4 t5 t6 t7
```

送入一个小型 Transformer：

\[
[h_{cls},h_1,\ldots,h_7]
=
T_S([STATE],t_1,\ldots,t_7)
\]

这里的 self-attention 允许字段相互影响。例如 `high distress` 在 `advice_receptivity=requested` 时与在 `advice_receptivity=closed` 时，可以形成不同的总体表示。TabTransformer 是这类“类别字段嵌入后使用 self-attention 建模字段交互”的直接架构参考。[TabTransformer](https://arxiv.org/abs/2012.06678)

取 `[STATE]` 位置的输出 \(h_{cls}\in\mathbb R^r\)，投影到 Student hidden size \(d\)：

\[
\tilde a^S=W_Sh_{cls}+b_S
\]

\[
W_S\in\mathbb{R}^{d\times r},\qquad b_S\in\mathbb{R}^{d}
\]

最后做 L2 归一化：

\[
a^S=\frac{\tilde a^S}{\|\tilde a^S\|_2}
\]

其中：

\[
\|\tilde a^S\|_2=
\sqrt{\sum_{k=1}^{d}(\tilde a^S_k)^2}
\]

归一化让每个 anchor 的长度为 1，后续内积可直接解释为 cosine similarity，模型无法仅靠无限增大向量长度来降低损失。

## 5. StateEncoder 的训练目标

仅输出一个向量不能保证它带有七个字段的信息。因此，从同一个 anchor 接七个字段恢复头。

第 \(i\) 个字段的 logits 为：

\[
\ell_i=W_i a^S+b_i
\]

`logits` 是每个候选值的未归一化分数。softmax 将其转换为概率：

\[
p_i(v\mid a^S)=
\frac{\exp(\ell_{i,v})}
{\sum_{u\in\mathcal V_i}\exp(\ell_{i,u})}
\]

若真实标签是 \(s_i\)，交叉熵为：

\[
L_i=-\log p_i(s_i\mid a^S)
\]

直觉上，模型对真实值给 0.70 概率时损失为 \(-\log 0.70\approx0.357\)；若只给 0.01 概率，损失为 \(-\log0.01\approx4.605\)。因此优化会要求 anchor 保留能够恢复真实值的信息。

StateEncoder 总损失：

\[
L_{state-encoder}=\sum_{i=1}^{7}w_iL_i
\]

其中 \(w_i\) 是字段权重。若某字段的 `unknown` 远多于其他值，可通过类别权重或采样修正降低“只猜多数类”的虚假高分。

该目标的限制也应明确：若两个字段在训练数据中高度相关，一个字段有可能被模型用另一个字段猜出。故验收必须逐字段报告 held-out balanced accuracy，而不能只报告总 loss。

## 6. PlanEncoder 架构与训练

### 6.1 受控 PLAN taxonomy

当 `strategy`、`goal`、`act` 都是离散字段时，PlanEncoder 可直接复用 StateEncoder 的形式：

```text
[PLAN] strategy-token goal-token act-token
```

随后从 `[PLAN]` 输出投影到 \(\mathbb R^d\)，并以三个分类头恢复对应字段。这是最推荐的版本，因为每个 anchor 的含义完全由 schema 决定。

这种“先预测显式高层概念，再使用概念表示”的思路与 Concept Bottleneck Models 一致。[Koh et al., 2020](https://proceedings.mlr.press/v119/koh20a.html)

### 6.2 自由文本 goal/act

若 `goal` 和 `act` 必须保留为自由文本，使用文本子 encoder：

\[
h^{text}=T_{text}([GOAL],goal,[ACT],act)
\]

并从 strategy 查得类别 embedding：

\[
h^{strategy}=E_{strategy}(p^{strategy})
\]

将其拼接并融合：

\[
h^P=\operatorname{MLP}([h^{strategy};h^{text}])
\]

\([·;·]\) 表示向量拼接。然后投影并归一化：

\[
a^P=\frac{W_Ph^P+b_P}{\|W_Ph^P+b_P\|_2}
\]

文本子 encoder 可采用 Sentence-BERT 风格句向量模型初始化，并优先冻结其主体、只训练 adapter 与 fusion MLP。Sentence-BERT 的目的正是构造可用 cosine 进行相似度比较的定长句向量。[Reimers and Gurevych, 2019](https://aclanthology.org/D19-1410.pdf)

训练目标至少包括策略恢复：

\[
L_{strategy}=-\log p(p^{strategy}\mid a^P)
\]

若要使 anchor 保留 goal/act 的实际文本内容，还可加入自回归重建。令 goal 的 token 序列是 \(g_1,\ldots,g_T\)，则：

\[
L_{goal}=-\sum_{t=1}^{T}
\log p(g_t\mid g_{<t},a^P)
\]

\(g_{<t}\) 表示第 \(t\) 个 token 之前的真实 token。`act` 使用相同的损失。总损失可写为：

\[
L_{plan-encoder}=L_{strategy}
+\lambda_gL_{goal}
+\lambda_aL_{act}
\]

自由文本方案的风险是小数据下可能记忆原句、而非学习稳定语义。因此实际优先级应是：先规范 PLAN taxonomy，再考虑训练自由文本 encoder。

## 7. 可选的对比学习

对比学习并非 StateEncoder 的必要条件。若输入已经是 canonical 的字段 ID，则同一 STATE 的输入天然完全一致。

若输入格式仍可能变化，可为同一条件创建两种不改变语义的视图：

\[
a^{(1)}_n=E(S_n;view_1),\qquad
a^{(2)}_n=E(S_n;view_2)
\]

例如字段呈现顺序或等价格式不同。以 \(a_n^{(1)}\) 为查询、\(a_n^{(2)}\) 为正样本，batch 内其他条件为候选的 InfoNCE 损失为：

\[
L_{NCE,n}=-\log
\frac{\exp(\operatorname{sim}(a_n^{(1)},a_n^{(2)})/\tau)}
{\sum_{j\in B}
\exp(\operatorname{sim}(a_n^{(1)},a_j^{(2)})/\tau)}
\]

其中：

- \(B\)：当前 batch 的样本集合。
- \(\tau>0\)：temperature；更小的值会更强调正样本必须显著优于负样本。
- \(\operatorname{sim}(u,v)=u^\top v\)：二者均 L2 归一化时，该内积就是 cosine similarity。

监督对比学习的基本机制是拉近同类表示、推远异类表示。[Khosla et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

但不应无条件把所有不同 STATE 当作强负样本：两个 STATE 可能只改了一个字段，在定义上应保留部分相似性。因此，对比项宜作为小权重稳定项，字段恢复才是主要监督。

## 8. 先预训练 target encoder，再冻结

分别训练：

\[
\phi_S^\star=
\arg\min_{\phi_S}
\left(L_{state-encoder}+\lambda_{SC}L_{state-contrast}\right)
\]

\[
\phi_P^\star=
\arg\min_{\phi_P}
\left(L_{plan-encoder}+\lambda_{PC}L_{plan-contrast}\right)
\]

然后冻结 \(\phi_S^\star\)、\(\phi_P^\star\)，为每条样本预计算：

\[
a^S_n=E_S(S_n;\phi_S^\star)
\]

\[
a^P_n=E_P(P_n;\phi_P^\star)
\]

不能在 Student 训练时继续训练 target encoder。若同时优化：

\[
\min_{\theta,\phi}
\|z_\theta(H)-a_\phi(S)\|^2
\]

Student 和 target 可以一起漂移，甚至一起塌缩，loss 变小却不意味着标签信息被保留。冻结后，优化变成：

\[
\min_{\theta}
\|z_\theta(H)-a_{\phi^\star}(S)\|^2
\]

此时 anchor 是固定答案，Student 必须从对话中学习预测它。使用固定中间表征监督 Student 隐层是表示蒸馏的常见思想；FitNets 即以 Teacher 中间“hint”指导 Student 表征。[Romero et al., 2015](https://mlanthology.org/iclr/2015/romero2015iclr-fitnets/)

重新随机训练 target encoder 即使取得相同指标，也会定义不同坐标系。因此不同 checkpoint 生成的 anchor artifact 不能混用。

## 9. Stage B 的 Student 对齐

Student 的实际输入仍是：

```text
完整对话 H + <|ibd_state|> + <|ibd_plan|> + 回复 y
```

在指定层捕获：

\[
z^S_n=f_{\theta}^{state}(H_n)
\]

\[
z^P_n=f_{\theta}^{plan}(H_n)
\]

将它们各自 L2 normalize 后，最简单的对齐损失为：

\[
L_{state-align}=
\frac{1}{N}\sum_{n=1}^{N}
\left(1-(z^S_n)^\top a^S_n\right)
\]

\[
L_{plan-align}=
\frac{1}{N}\sum_{n=1}^{N}
\left(1-(z^P_n)^\top a^P_n\right)
\]

单位向量的内积等于 cosine similarity：同方向为 1，正交为 0，反方向为 -1。因此 `1 - inner product` 分别为 0、1、2。

回复仍以 token-level cross entropy 训练：

\[
L_{response}=-\sum_{t=1}^{T}
\log p_\theta(y_t\mid H,
\texttt{STATE},\texttt{PLAN},y_{<t})
\]

\(y_{<t}\) 表示第 \(t\) 个回复 token 前的真实 token。总目标：

\[
L_B=L_{response}
+\lambda_S L_{state-align}
+\lambda_P L_{plan-align}
\]

\(\lambda_S\)、\(\lambda_P\) 控制结构监督强度。过大时模型可能过度优先对齐、损害回复能力；过小时 slot 不会稳定承载标签信息。

部署时不需要 Teacher 标签、target encoder 或 anchor artifact。它们只在训练期定义目标；推理时由 Student 根据对话自然计算两个 slot。

## 10. 相同 STATE 不等于相同对话或相同回复

考虑两段对话：

```text
H1: “我快被工作压垮了，不知道是否该辞职。”
H2: “照顾家人让我喘不过气，也不知道该怎么办。”
```

它们可有相同 STATE：

```text
overwhelm + high distress + validation need + hesitant advice
```

因此按设计：

\[
a^S(H1)=a^S(H2)
\]

这不要求最终回复相同。完整对话的其他 token hidden states 仍不同，因而 Student 可针对“工作”或“照顾家人”生成不同的具体措辞。STATE slot 只压缩 schema 指定的支持状态；其余细节保留在正常对话上下文中。

若某种细节应由 STATE slot 区分，正确做法是将其加入 schema，而不是要求同一个七维 STATE anchor 同时保存未定义的全部对话内容。

## 11. 重复标签与多正样本

若两条样本具有完全相同的 STATE：

\[
S_i=S_j
\]

则应有：

\[
a^S_i=a^S_j
\]

若 Stage B 使用 batch 内对比损失，不能将 \(j\) 错当成 \(i\) 的负样本。定义同条件正样本集合：

\[
\mathcal P(i)=\{j:S_j=S_i\}
\]

使用多正样本形式：

\[
L_i=-\log
\frac{
\sum_{j\in\mathcal P(i)}
\exp(\operatorname{sim}(z_i^S,a_j^S)/\tau)
}{
\sum_{k\in B}
\exp(\operatorname{sim}(z_i^S,a_k^S)/\tau)
}
\]

这表示“任何具有同一完整 STATE 的 anchor 都是正确目标”。若采用纯 cosine regression 而不是 batch contrastive loss，则不会出现这一类 false negative 问题。

PLAN 若转为受控 taxonomy，也应使用相同原则。

## 12. 伪代码与张量形状

StateEncoder 的输入 `state_ids` 形状为 `[B, 7]`，其中 `B` 是 batch size：

```python
def encode_state(state_ids):
    # state_ids: [B, 7]
    field_tokens = []
    for i in range(7):
        value = value_embeddings[i](state_ids[:, i])  # [B, r]
        field_tokens.append(value + field_embeddings[i])

    x = torch.stack(field_tokens, dim=1)              # [B, 7, r]
    cls = state_cls.expand(x.shape[0], 1, -1)          # [B, 1, r]
    hidden = state_transformer(torch.cat([cls, x], 1)) # [B, 8, r]
    anchor = F.normalize(project(hidden[:, 0]), dim=-1) # [B, d]
    logits = [head(anchor) for head in field_heads]
    return anchor, logits
```

预训练 target encoder：

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

Stage B 取 Student slot 并计算对齐：

```python
state_slot = output.slots.STATE                  # [B, d]
plan_slot = output.slots.PLAN                    # [B, d]

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

## 13. Artifact 应保存的内容

每份 anchor artifact 至少应包含：

```text
state: [num_examples, d]
plan:  [num_examples, d]
example_id -> row 映射

metadata:
  schema_version
  state_encoder_checkpoint_hash
  plan_encoder_checkpoint_hash
  target_encoder_training_split
  anchor_dimension
  normalization = l2
  state_vocabularies
  plan_vocabularies / text-encoder revision
```

这些信息用于阻止以下不安全混用：

- 将新 schema 的标签送入旧 encoder；
- 用不同随机初始化训练出的 target encoder 替换原 anchor；
- 将不同 hidden size 的 anchor 注入或对齐到同一 Student 层；
- 未经确认地混合不同 vocabularies 或不同文本 encoder revision。

## 14. 在训练 Student 前必须完成的验收

### 14.1 字段可恢复性

在 held-out 标签上报告每个 STATE 字段的 accuracy 与 balanced accuracy，以及 PLAN strategy/goal/act 的对应恢复指标。仅报告一个总平均值会掩盖弱字段。

### 14.2 格式不变性

若 target encoder 接收文本或多种序列化方式，同一条件的等价格式应获得高 cosine similarity。若直接输入 canonical 字段 ID，该性质按构造成立。

### 14.3 单字段敏感性

只替换一个字段：

\[
S'=(s_1,\ldots,s_i',\ldots,s_7)
\]

测量：

\[
\Delta_i=1-(a^S)^\top a^{S'}
\]

该值应显著大于数值噪声；若某字段替换后 anchor 几乎不变，且其恢复准确率也低，说明 encoder 没有编码该字段。

### 14.4 碰撞与近邻检查

若 \(S_i\ne S_j\)，却有：

\[
\operatorname{cos}(a^S_i,a^S_j)\approx1
\]

需人工检查：两个标注是否实为近义、encoder 容量是否不足、字段是否极不平衡，或恢复头是否仅靠字段相关性猜测。

## 15. 主要局限

1. target encoder 定义的是训练期坐标约定，不是 STATE/PLAN 的“天然本体”。
2. 字段恢复高分不等于字段在向量中完全独立；训练数据相关性会造成捷径。
3. 自由文本 PLAN 的可审计性与稳定性明显弱于受控 taxonomy。
4. 若 schema 本身遗漏重要差异，anchor 会按定义丢弃该差异；这不是 encoder 可自动补救的问题。
5. target encoder 的 checkpoint、词表和 schema 任何一项变化，都要求重新生成所有 anchor。

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
- 对 dev 标签验收后冻结
- 预计算并版本化 anchor artifact
- Stage B 仅让 Student 的相应 slot 对齐各自 anchor
```

该方案将 anchor 的含义从“通用语言模型对一段 JSON 文本的隐藏层输出”收紧为：

> 由固定 schema 和显式恢复目标训练得到、只承载 Teacher STATE 或 PLAN 标签的专用连续目标表示。
