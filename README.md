# Interventional Blackboard Distillation

该项目训练 Qwen Student 在支持性对话中生成最终回复，并在内部 STATE / PLAN 槽位上对齐 Teacher anchor。

## 训练阶段

- **Stage A — Response Warm-up**：用 Teacher 的最终回复进行回复监督。
- **Stage B — Slot Alignment**：在回复监督之外，将 STATE 与 PLAN 槽位对齐到冻结的 Teacher anchor。

`train-pipeline` 按配置依次运行 A、B；`train --stage A|B` 可单独运行启用的阶段。B 阶段必须使用冻结 anchor artifact。

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
