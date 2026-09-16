# Interventional Blackboard Distillation

该项目训练 Qwen Student 在支持性对话中生成最终回复，并在内部 STATE / PLAN 槽位上对齐 Teacher anchor。

## 训练阶段

- **Stage A — Response Warm-up**：用 Teacher 的最终回复进行回复监督。
- **Stage B — Slot Alignment**：在回复监督之外，将 STATE 与 PLAN 槽位对齐到冻结的 Teacher anchor。

`train-pipeline` 按配置依次运行 A、B；`train --stage A|B` 可单独运行启用的阶段。B 阶段必须使用冻结 anchor artifact。

## Seeker Context Memory

教师推理每轮按照 `Context → STATE → PLAN → response` 的顺序执行。在线多轮推理使用 `TeacherSession` 自动携带上一轮 Context；离线 `run-teacher` 可在每条输入记录中显式提供 `context_before`。

当前 SocialSim 默认预处理每个 conversation 只产生一个 target，因此不会自动形成跨 target 的 Context 轨迹。Context 只进入教师推理与 trace；首版学生训练输入保持不变。

## Profile 驱动的完整对话生成

`generate-conversations` 对每条输入 profile 发起一次外部任务，在任务内部交替生成 seeker 和 supporter，直到 Dialogue Manager 自然选择 `closing`，或达到执行预算：

```text
opaque profile -> Seeker Simulator -> public seeker utterance
                                      |
                                      v
Context -> seven-field STATE -> DialogueMode -> PLAN -> candidates -> selection
                                      |
                                      v
                              supporter response
```

原始 profile 只进入 Seeker Simulator。Dialogue Manager 和完整 Teacher 路径只读取公开 History、逐轮积累的 UserContext 与当前 `MultiViewStateAnalysis`。当前 profile adapter 只要求非空 `ID`；详细 profile schema 仍留作独立设计。

DialogueMode 是每轮由额外 LLM 调用选择的软方向，可保持、前进、回退或跳过：`opening`、`exploration`、`comforting`、`action`、`closing`。它只作为 Planner 的弱先验，不映射到固定策略。

默认轮数参考为 `min=6`、`soft-max=16`、`hard-max=20`。前两项只提示 Manager；hard max 不会强制伪造 closing：

- Manager 选择 `closing`：写入 `conversations.jsonl`，状态为 `completed`；
- 到达 hard max 仍未 closing：写入 `truncated-conversations.jsonl`，状态为 `truncated`；
- 模型或输入失败：写入 `conversation-failures.jsonl`，不构造伪造的 ConversationTrace。

`truncated` 数据不得作为正常训练数据。每个完整 seeker–supporter 回合都会原子保存 checkpoint；`--resume` 从最后一个完整回合继续，并跳过已经进入 completed/truncated 文件的 conversation ID。

```bash
PY=/homeb/wangnianxiang/supervisor/.venv/bin/python

PYTHONPATH=src $PY -m ibd.cli generate-conversations \
  --config configs/deepseek_teacher.yaml \
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
```

完整 ConversationTrace 可扁平导出为现有 Stage A/B 能直接读取的逐 supporter 回合 `TeacherTrace`：

```bash
PYTHONPATH=src $PY -m ibd.cli flatten-conversations \
  --input artifacts/conversations/conversations.jsonl \
  --output artifacts/conversations/teacher-traces.jsonl
```

正式批量生成前建议先运行约 50 条 profile 的 pilot，检查 completed/truncated/failed 比例、对话长度、Mode 分布与转移、strategy 分布、连续重复问题或建议、closing 自然度，以及 Context 是否只包含 seeker 已公开的信息。

## 常用命令

```bash
PY=.venv/bin/python

CUDA_VISIBLE_DEVICES=0 $PY -m ibd.cli train-pipeline \
  --run-name seed-42 --seed 42 \
  --config configs/qwen25_7b_qlora_3090.yaml \
  --traces artifacts/teacher/traces.jsonl \
  --anchors artifacts/student/anchors.safetensors
```

## 质量评测

质量生成、Judge 和报告合并是独立流程。`quality_merge` 可用于将后续补充实验的完成结果合并到既有质量报告中。历史生成物保留在 `artifacts/` 与 `result/`，但不再由训练命令消费。
