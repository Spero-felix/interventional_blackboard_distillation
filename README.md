# Interventional Blackboard Distillation

这是“功能干预式黑板蒸馏”的参考实现。目标不是让 Student 复现可见思维链，而是让两个特殊 token 的上下文隐状态承担可验证的内部功能：`STATE` 表征当前支持状态，`PLAN` 控制最终回复策略。部署时仍然是一个 Student、一次生成。

这两个隐向量可以称为 **CoT-inspired latent process compression**：它们压缩的是 Teacher 的分析与决策过程，不是逐 token 复现经典 Chain-of-Thought，也不要求 Student 输出可见推理文本。

## Teacher 架构

正常轨迹固定为 6 个逻辑调用：

```text
对话历史 H
  → Multi-view State Analyzer（一次调用，输出四个审计视角 + STATE）
  → Planner（生成三个不同策略）
  → Candidate 1 / 2 / 3（每个候选只执行一个策略）
  → Final Selector（只选择一个候选 ID）
  → 控制器原样返回该候选回复
```

最终选择器不能合并或重写候选。模型只输出 `selected_candidate_id` 以及简短的 `response_goal`、`response_act`；最终策略、策略 ID 和回复文本由控制器从候选表中重新构造。因此最终答案最多包含一个策略，中间仍保留三个策略与三个回复用于比较。

正常 Teacher 路径不再运行情绪、有效性、安全三个多维 critic，也不再构造 critique-grounded Stage D。训练数据离线构造阶段保留两个窄功能组件：

- 独立安全过滤器：只判断两个条件回复能否安全保留，不做多维评分；
- 条件效应验证器：判断每个回复是否匹配自己的 STATE/PLAN 条件，不判断反事实回复是否“更差”。

## 三阶段训练

### Stage A — Response Warm-up

对自然对话的 Teacher 最终回复做 response-only SFT，先建立基础支持能力：

```text
(H → y_original)
```

### Stage B — Latent State–Plan Acquisition

继续学习自然回复，同时将 `<|ibd_state|>` 与 `<|ibd_plan|>` 位置的动态隐状态分别对齐到 Teacher 的 STATE 与最终单策略 PLAN anchor。

Stage B 的动机是把“产生好回复所需的内部中间变量”压进两个可定位位置。只做 Stage A 时，模型可能会回答，但无法说明两个特殊 token 是否真正承载了状态和计划；Stage B 先建立这种结构对应关系，为 Stage C 的干预检验提供对象。

### Stage C — Latent Controllability

Stage C 不验证“换成反事实后回复应该变差”，而是验证“换控制条件后，回复是否按条件改变”。原始条件和反事实条件都是有效控制：

```text
自然路径：              SFT(y_original)
原始 anchor clamp：     score(y_original) > score(y_counterfactual)
反事实 anchor clamp：   score(y_counterfactual) > score(y_original)
```

训练使用 teacher-forced、长度归一化的回复分数和双向 hinge loss。只钳制被干预的目标槽，另一个槽保持由当前上下文自然计算。PLAN 反事实直接从原先未被最终选择的两个候选中按 `example_id + global_seed` 确定性选择一个，不额外生成回复；STATE 反事实只替换一个字段，然后重新运行 Planner、三个 Candidate 和 Final Selector。

自然无 clamp 的 SFT 始终以 `y_original` 为目标。`y_counterfactual` 不是 rejected、negative 或 degraded response，而是 **counterfactual-conditioned response**。

## 安装与测试

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test,train]'
pytest -q
```

也可以复用 supervisor 环境：

```bash
PYTHONPATH=src /home/wangnianxiang/supervisor/.venv/bin/python -m pytest -q
```

## CLI

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
  --manifest artifacts/student/manifest.json \
  --global-seed 42

CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli precompute-anchors \
  --config configs/qwen25_7b_qlora_3090.yaml \
  --traces artifacts/teacher/traces.jsonl \
  --interventions artifacts/student/interventions.jsonl \
  --output artifacts/student/anchors.safetensors \
  --diagnostic-state-field readiness \
  --global-seed 42

CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli train-pipeline \
  --run-name seed-42 --seed 42 \
  --config configs/qwen25_7b_qlora_3090.yaml \
  --traces artifacts/teacher/traces.jsonl \
  --interventions artifacts/student/interventions.jsonl \
  --anchors artifacts/student/anchors.safetensors
```

`train --stage` 和 `train-pipeline` 的运行主链只包含 A、B、C。Stage D、margin pair 及其训练损失已从代码中删除。

`export-student` 支持 `sft`、`slot` 和 `intervention`。Student 数据由显式白名单构造，不会泄漏 Teacher 的审计视角。

工程约束：除 LLM 调用缓存的键外，不在代码、artifact、manifest 或 checkpoint 中生成、保存或校验 hash。

## Stage C 评测

训练集只用于拟合；controllability 结论必须在 dev/diagnostic holdout 上报告：

- 原始 clamp preference accuracy；
- 反事实 clamp preference accuracy；
- 双向同时成立的 flip consistency；
- 非目标槽 invariance；
- 干预 locality；
- normal、STATE clamp、PLAN clamp 与去掉各训练项的消融。

这套指标回答的核心问题是：两个特殊 token 的隐状态是否真能作为可操纵的内部控制变量，而不是仅仅出现在输入中的装饰 token。

## 目录

```text
src/ibd/          核心实现
tests/            离线单元与集成测试
configs/          冻结协议示例
docs/superpowers/specs/  设计说明
docs/superpowers/plans/  实施计划
```
