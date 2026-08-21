# Interventional Blackboard Distillation

这是“功能干预式黑板蒸馏”的独立参考实现。目标不是让小模型复现多 Agent 的完整讨论过程，而是让它学习两类可验证的内部功能：`STATE` 对上下文状态的表征，以及 `PLAN` 对回复行为顺序的控制。部署仍是一个 Student、一次生成。

## 已实现的范围

- Teacher 的四个互相不可见专家：情绪、需求、关系、意图；
- `STATE` 整合、`PLAN` 规划、三个候选、三个评论视角、最终整合和盲门控；
- Teacher 侧安全批评、质量过滤和最终安全门禁；
- `STATE/PLAN` 最小干预及下游重跑；
- 从多个候选与评论证据构造 Stage D 的离线 chosen/rejected pair；
- Student 的两个隐式槽和训练期钳制；
- Stage A–D 所需的损失原语、数据白名单与评测指标。

项目明确不声称 Student 保留了多个评论 Agent 的交流讨论过程。评论只在 Teacher 侧用于产生更好的答案和可定位的近负例；Stage D 督促 Student 向好输出靠拢，但不增加 `CRITIC` 槽，也没有 Student-in-the-loop repair。

## 安装与测试

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test,train]'
pytest -q
```

测试全部使用脚本化后端和随机初始化的 tiny Qwen2，不需要网络或模型下载。可以直接复用 supervisor 的 Python 环境：

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q
```

该环境已经包含 PyTorch 2.6、Transformers 4.57、PEFT 0.19、BitsAndBytes 0.49 和 Safetensors 0.8。运行真实 Teacher 前仍需让当前 shell 提供 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`；`configs/deepseek_teacher.yaml` 会读取这两个变量。

## Teacher 架构与调用计数

正常轨迹固定包含 14 个逻辑调用：

1. 情绪、需求、关系、意图四专家；
2. 状态整合器与规划器；
3. 三个带固定 seed 的候选生成器；
4. 情绪、有效性、安全三个评论视角；
5. 最终整合器和盲质量门控。

首次门控失败时只允许一次修订和一次复检，因此上限为 16。Stage D pair 的 A/B 与 B/A 顺序交换验证另加两次调用：正常轨迹加 pair 验证为 16，发生门控修订再加 pair 验证为 18。功能干预的下游重跑单独计账。

系统没有风险专家，也没有生成前风险路由。安全批评属于 Teacher 的候选过滤与最终评测路径；它不会成为 Student slot、干预目标、margin 缺陷维度或导出字段。

## 四阶段训练

### Stage A — Outcome Warm-up

用 `(H, y*)` 做 response-only SFT，先建立语言和任务能力。

### Stage B — Latent State–Plan Acquisition

对齐 `<|ibd_state|>` 与 `<|ibd_plan|>` 两个隐式槽。Teacher 的状态黑板和支持计划是训练期特权监督，部署时不作为外部输入。

### Stage C — Functional Intervention Distillation

分别钳制 `STATE` 或 `PLAN`，让正常槽生成 `y*`，让被干预槽复现 `y^(-r)` 的定向行为变化。只保留目标维度出现可定位退化、且通过顺序交换验证的样本。

### Stage D — Critique-Grounded Margin Alignment

Teacher 的多个候选提供不同解法，情绪与有效性评论定位非安全缺陷，安全评论仅负责淘汰不合格候选。对剩余候选选择缺陷最强且可定位的近负例，再通过 A/B、B/A 两次验证冻结 `(H, chosen, rejected)`。

训练目标为：

```text
L_D = L_chosen_SFT
    + relu(margin - score(chosen) + score(rejected))
    + replay_weight * L_StageB/C
```

`score` 是仅覆盖回复 token 的平均 log-probability，因此不会因候选长度不同产生直接偏置。DPO 只作为对照，不是主训练机制。

## CLI

### 本地 Qwen2.5-7B 流程

以下命令均可使用 supervisor 的解释器。`prepare-socialsim` 产出的分组 JSON 可以直接作为 `run-teacher` 输入，diagnostic holdout 会保留在轨迹中，但 Student 导出和训练入口会拒绝将其用于更新参数。

耗时命令会在交互终端的 stderr 显示进度；重定向输出或在非交互环境运行时会自动静默，stdout 中的 JSON 和摘要保持不变。

```bash
export PYTHONPATH=src
PY=/home/wangnianxiang/supervisor/.venv/bin/python

$PY -m ibd.cli prepare-socialsim \
  --output artifacts/socialsim/prepared.json

$PY -m ibd.cli run-teacher \
  --config configs/deepseek_teacher.yaml \
  --input artifacts/socialsim/prepared.json \
  --output artifacts/teacher/traces.jsonl

$PY -m ibd.cli build-interventions \
  --config configs/deepseek_teacher.yaml \
  --input artifacts/teacher/traces.jsonl \
  --output artifacts/student/interventions.jsonl \
  --margins-output artifacts/student/margins.jsonl \
  --manifest artifacts/student/manifest.json

CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli precompute-anchors \
  --config configs/qwen25_7b_qlora_3090.yaml \
  --traces artifacts/teacher/traces.jsonl \
  --interventions artifacts/student/interventions.jsonl \
  --output artifacts/student/anchors.safetensors
```

先用一张 3090 完成 Stage A-D、checkpoint 和评估门禁：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli train-pipeline \
  --run-name smoke-seed-42 --seed 42 \
  --config configs/qwen25_7b_qlora_3090_smoke.yaml \
  --traces artifacts/teacher/traces.jsonl \
  --interventions artifacts/student/interventions.jsonl \
  --margins artifacts/student/margins.jsonl \
  --anchors artifacts/student/anchors.safetensors

CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli evaluate \
  --run-name smoke-seed-42 \
  --config configs/qwen25_7b_qlora_3090_smoke.yaml \
  --checkpoint runs/smoke-seed-42/stage-D-step-4 \
  --traces artifacts/teacher/traces.jsonl \
  --interventions artifacts/student/interventions.jsonl \
  --margins artifacts/student/margins.jsonl \
  --anchors artifacts/student/anchors.safetensors \
  --output runs/smoke-seed-42/evaluation.json
```

smoke 配置把每个 stage 限制为一个 optimizer step。命令输出分别列出总参数、LoRA 参数、两个 input/output token 行参数、loss 和最大 gradient norm。checkpoint 只在完整 stage 边界发布；同 stage `--resume` 表示从下一 epoch 继续，并使用新的确定性 shuffle。

门禁通过后，三个任务各自占用一张逻辑 GPU，不进行分布式通信：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli train-pipeline --run-name seed-41 --seed 41 --config configs/qwen25_7b_qlora_3090.yaml --traces artifacts/teacher/traces.jsonl --interventions artifacts/student/interventions.jsonl --margins artifacts/student/margins.jsonl --anchors artifacts/student/anchors.safetensors
CUDA_VISIBLE_DEVICES=1 $PY -m ibd.cli train-pipeline --run-name seed-42 --seed 42 --config configs/qwen25_7b_qlora_3090.yaml --traces artifacts/teacher/traces.jsonl --interventions artifacts/student/interventions.jsonl --margins artifacts/student/margins.jsonl --anchors artifacts/student/anchors.safetensors
CUDA_VISIBLE_DEVICES=2 $PY -m ibd.cli train-pipeline --run-name seed-43 --seed 43 --config configs/qwen25_7b_qlora_3090.yaml --traces artifacts/teacher/traces.jsonl --interventions artifacts/student/interventions.jsonl --margins artifacts/student/margins.jsonl --anchors artifacts/student/anchors.safetensors
```

`generate` 默认是正常一次生成。anchor artifact 会为可变换的 diagnostic holdout 同时保存 STATE 与 PLAN 诊断向量，因此同一个 holdout 历史可以依次运行 normal、STATE clamp 和 PLAN clamp；钳制命令需同时提供 `--clamp STATE|PLAN --anchors ... --example-id ...`。这些 holdout 向量不进入 Stage A-D 参数更新。

### 原有协议工具

计算冻结协议哈希：

```bash
python -m ibd.cli protocol-hash --config configs/demo.yaml
```

Teacher 输入是单个 JSON、JSON 数组或 JSONL，每条形如：

```json
{"example_id":"e-1","split":"train","history":{"turns":[{"role":"seeker","content":"我不知道怎么开口。"}]}}
```

运行 Teacher：

```bash
export OPENAI_API_KEY=...
python -m ibd.cli run-teacher \
  --config configs/demo.yaml \
  --input data/input.jsonl \
  --output outputs/traces.jsonl
```

验证轨迹并导出 Student 数据：

```bash
python -m ibd.cli validate-trace --input outputs/traces.jsonl
python -m ibd.cli export-student \
  --kind sft \
  --input outputs/traces.jsonl \
  --output outputs/student-sft.jsonl
```

`export-student` 支持 `sft`、`slot`、`intervention` 和 `margin`。所有导出都从允许字段重新构造字典，而不是从完整 Teacher trace 中删字段。

## 评测与消融

除 `Retention=(S-B)/(T-B)` 外，必须分别报告 Teacher 与 Student 的两行功能保真矩阵：

```text
E[r,d] = Q_d(y_full) - Q_d(y_ablate(r)),  r ∈ {STATE, PLAN}
```

矩阵比较包括符号一致率、Spearman 秩相关和以 Teacher L1 为分母的归一化 L1 距离。Stage D 另外报告严格不计 tie 的 PairAcc 和平均 chosen–rejected 分数差。

建议消融序列：

```text
S1 = SFT
S2 = SFT + DPO
S3 = SFT + STATE/PLAN auxiliary heads
S4 = S3 + latent STATE/PLAN tokens
S5 = S4 + function-specific negatives
S6 = S5 + intervention-effect distillation
S7 = S6 + critique-grounded margin alignment
```

安全性独立进行最终评测和硬门禁，不进入 Student 功能监督；正式结论仍需人工盲评，PairAcc 上升不能单独证明回复质量提升。

## 目录

```text
src/ibd/          核心实现
tests/            离线单元与集成测试
configs/          冻结协议示例
docs/superpowers/plans/  实施计划
```
