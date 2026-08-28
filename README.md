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

## 生成质量对比

生成质量评测与 Stage C 因果评测相互独立。它在同一组 `dev` 和
`diagnostic_holdout` histories 上比较三类正常回复：训练后 Student、对应的
原始 Qwen Instruct Base，以及已有 `TeacherTrace.final_response`。Teacher 回复
直接复用，不重新运行 Teacher，因此不会产生额外 Teacher token 消耗。

先复制并编辑 `configs/quality_eval.yaml`，尤其是 Student 的 checkpoint 和
`run_name`。Base 始终从 Qwen training config 的 `model_path` 加载原始权重与原始
tokenizer：不加载 LoRA/checkpoint，也不添加 `<|ibd_state|>`、
`<|ibd_plan|>`；只有 Student 使用训练后的结构 token。

```bash
PY=/homeb/wangnianxiang/supervisor/.venv/bin/python
export PYTHONPATH=src

CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli quality-generate \
  --config configs/quality_eval.yaml \
  --traces artifacts/teacher/traces.jsonl \
  --output-dir artifacts/quality/generation \
  --device 0

$PY -m ibd.cli quality-judge \
  --config configs/quality_eval.yaml \
  --responses artifacts/quality/generation/responses.jsonl \
  --output-dir artifacts/quality/judge

$PY -m ibd.cli quality-human-export \
  --config configs/quality_eval.yaml \
  --responses artifacts/quality/generation/responses.jsonl \
  --output-dir artifacts/quality/human

$PY -m ibd.cli quality-human-summarize \
  --annotations artifacts/quality/human/pairs.csv \
  --mapping artifacts/quality/human/private_mapping.jsonl \
  --output artifacts/quality/human/report.json
```

前两个命令支持独立 artifact；生成与 Judge 还支持 `--resume` 和
`--continue-on-error`。固定 Judge 对每条匿名回复分别给出 1–5 分的共情、相关性、
连贯性、即时有效性和自主性评分，`overall` 由程序取五维算术平均。主报告分别呈现
dev/holdout 的模型均值、覆盖率、同 history 配对分差以及胜/平/负比例，不包含安全
维度或安全门槛。

人工评审不是必经步骤。`quality-human-export` 生成匿名 A/B CSV 和单独的私有映射；
评审者只需填写 `A`、`B` 或 `tie`。完成后才运行汇总命令。要增加另一个本地 Base
或 Student checkpoint，可在 quality config 的 `models` 列表中添加唯一
`model_id` 与相应 `response_source` 配置，Judge 无需修改。

## 目录

```text
src/ibd/          核心实现
tests/            离线单元与集成测试
configs/          冻结协议示例
docs/superpowers/specs/  设计说明
docs/superpowers/plans/  实施计划
```
## 固定第一个 PLAN 候选的 2000 条实验

这组实验不重新生成 Teacher 的 STATE、PLAN 或三个候选，而是从现有 2000 条
完整轨迹中固定选择 `strategy_id == "S1"` 的候选，重建每条轨迹的
`final_selection` 和 `final_response`。原始 `train/dev/diagnostic_holdout` 划分
保持为 `1500/200/300`，旧数据和旧 artifact 不会被覆盖。

先在 CPU 上构造并校验新轨迹：

```bash
export PYTHONPATH=src
PY=/home/wangnianxiang/supervisor/.venv/bin/python

$PY scripts/select_first_plan_candidate.py \
  --input artifacts/full-2000/teacher-traces.jsonl \
  --output artifacts/full-2000-first-plan/teacher-traces.jsonl
```

成功时摘要应显示总数为 2000，三个 split 分别为 1500、200、300，并且转换后
`selected_after` 为 `{"S1": 2000}`。脚本会拒绝覆盖已有输出；如果需要重跑，请
先人工确认并移走旧的 `artifacts/full-2000-first-plan/teacher-traces.jsonl`。

旧 intervention 的自然回复和 PLAN 条件来自旧的最终选择，不能复用。必须基于新
轨迹重新构造：

```bash
$PY -m ibd.cli build-interventions \
  --config configs/deepseek_teacher.yaml \
  --input artifacts/full-2000-first-plan/teacher-traces.jsonl \
  --output artifacts/full-2000-first-plan/interventions.jsonl \
  --manifest artifacts/full-2000-first-plan/intervention-manifest.json \
  --global-seed 42
```

该命令会调用 Teacher API，当前不支持断点续跑。正式运行前请确认配置和 API
环境；如中途失败，应检查输出后再决定是否移走不完整文件并整批重跑。

随后重新计算 train 和 dev 的自然 anchor，以及干预和 diagnostic holdout 所需的
clamp anchor：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli precompute-anchors \
  --config configs/experiments/c2000-from-b-lr-8e-5.yaml \
  --traces artifacts/full-2000-first-plan/teacher-traces.jsonl \
  --interventions artifacts/full-2000-first-plan/interventions.jsonl \
  --output artifacts/full-2000-first-plan/anchors-train-dev.safetensors \
  --original-splits train dev \
  --diagnostic-state-field readiness \
  --global-seed 42
```

`c2000-from-b-lr-8e-5.yaml` 已启用 A、B、C 三个阶段，可直接从 Base 启动完整
训练：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli train-pipeline \
  --run-name first-plan-2000-lr-8e-5 --seed 42 \
  --config configs/experiments/c2000-from-b-lr-8e-5.yaml \
  --traces artifacts/full-2000-first-plan/teacher-traces.jsonl \
  --interventions artifacts/full-2000-first-plan/interventions.jsonl \
  --anchors artifacts/full-2000-first-plan/anchors-train-dev.safetensors
```

训练完成后，根据 `runs/first-plan-2000-lr-8e-5/` 中的开发集选择记录确定需要
评测的 checkpoint，再运行：

```bash
CHECKPOINT=runs/first-plan-2000-lr-8e-5/stage-C-step-替换为实际步数

CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli evaluate \
  --config configs/experiments/c2000-from-b-lr-8e-5.yaml \
  --run-name first-plan-2000-lr-8e-5 \
  --checkpoint "$CHECKPOINT" \
  --traces artifacts/full-2000-first-plan/teacher-traces.jsonl \
  --interventions artifacts/full-2000-first-plan/interventions.jsonl \
  --anchors artifacts/full-2000-first-plan/anchors-train-dev.safetensors \
  --intervention-manifest artifacts/full-2000-first-plan/intervention-manifest.json \
  --output artifacts/full-2000-first-plan/evaluation.json
```
