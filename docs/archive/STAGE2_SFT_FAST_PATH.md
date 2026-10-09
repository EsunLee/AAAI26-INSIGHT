# Stage 2 最快语义恢复路线

## 目标

在不等待缺失的 ego hand-object detector 权重、不重新提取 125 万张 EK55 帧的前提下，先解决已经确认的 Stage 2 问题：GRPO adapter 学会了格式，但没有学会未来动作语义。

本路线保留当前完整数据来源：

```text
现有 EK55 特征 → 现有 Stage 1 预测历史 → Stage 2 SFT → 语义验收
```

它不会修复 Stage 1 和 HOI 特征，因此是“最快改进实验”，不是最终严格复现。

## 为什么先做 SFT

当前 GRPO 直接从 base Qwen 开始，模型可以通过重复 20 个合法动作获得格式/长度奖励。SFT 直接使用 10,748 条训练样本的 GT 未来 20 动作作为 assistant target，先让模型学会 EK55 动作词表、序列结构和历史到未来的映射，再考虑短 GRPO。

## 时间预算（3×RTX 3090）

| 阶段 | 预计时间 | 是否必须 |
|---|---:|:--:|
| 数据转换与校验 | 1–3 分钟 | 是 |
| 2-step smoke | 8–20 分钟 | 是 |
| 完整 SFT，1 epoch | 1.5–3 小时 | 是 |
| 100 样本确定性验收 | 20–40 分钟 | 是 |
| 完整 876 样本 Top-1/ED | 约 3 小时 | 通过验收后 |
| 5 个候选 | 约 6 小时 | 最后再做 |

最快约 2–4 小时可以判断路线是否有效；完整现有协议结果约 8–12 小时。

## 1. 构造 SFT 数据

```bash
cd ~/ldy/git/AAAI26-INSIGHT

/data/conda_envs/insight_env/bin/python3 \
  feature_extraction/build_stage2_sft_dataset.py \
  --train /data/datasets/EPIC-KITCHENS/stage2_stage1_history/stage2_train_stage1_history.jsonl \
  --val /data/datasets/EPIC-KITCHENS/stage2_stage1_history/stage2_val_stage1_history.jsonl \
  --output-dir /data/datasets/EPIC-KITCHENS/stage2_stage1_history_sft \
  --val-subset-size 100
```

验收：

```bash
wc -l /data/datasets/EPIC-KITCHENS/stage2_stage1_history_sft/*.jsonl
```

期望主文件为 train 10,748、val 876、val100 100，同时生成 32/12 的 smoke 文件。

## 2. 运行 smoke

```bash
cd ~/ldy/git/AAAI26-INSIGHT

RUN_MODE=smoke nohup bash scripts/run_stage2_sft_fast.sh \
  > /tmp/stage2_sft_smoke.log 2>&1 &

echo "SFT_SMOKE_PID=$!"
```

监控：

```bash
tail -f /tmp/stage2_sft_smoke.log
```

必须看到 `STAGE2_SFT_SMOKE_COMPLETE`，且不能出现 `Traceback`、`OutOfMemory`、`ChildFailedError`。

## 3. 运行完整 SFT

```bash
cd ~/ldy/git/AAAI26-INSIGHT

RUN_MODE=full nohup bash scripts/run_stage2_sft_fast.sh \
  > /tmp/stage2_sft_fast.log 2>&1 &

echo "SFT_FULL_PID=$!"
```

监控：

```bash
watch -n 10 'nvidia-smi -i 0,1,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo; grep -E "Train:|Eval:|eval_loss|Saving model checkpoint|Traceback|RuntimeError|OutOfMemory|train_runtime" /tmp/stage2_sft_fast.log | tail -25'
```

## 4. 只做 100 样本语义验收

训练完成后先查 checkpoint：

```bash
find /data/datasets/EPIC-KITCHENS/stage2_sft_fast_output \
  -type d -name 'checkpoint-*' | sort -V
```

把选中的 checkpoint 赋给 `ADAPTER`：

```bash
cd ~/ldy/git/AAAI26-INSIGHT

ADAPTER=/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/实际运行目录/checkpoint-实际步数 \
SAMPLES=100 \
OUTPUT_DIR=/data/datasets/EPIC-KITCHENS/stage2_sft_diagnostic_100 \
nohup bash scripts/evaluate_stage2_adapter_fast.sh \
  > /tmp/stage2_sft_diagnostic_100.log 2>&1 &

echo "SFT_EVAL_PID=$!"
```

当前旧 GRPO 在 predicted-history 100 样本上的基线是 Top-1=0%、Action ED=1.0。满足下面任一语义条件才继续完整推理：

- Top-1 大于 0%；或
- Action ED 低于 0.98；

同时 `valid20` 应保持在 80% 以上。若语义门没有通过，应先调整 SFT 学习率/目标格式，不运行耗时的 876×5 推理。

## 5. 完整确定性评测

100 样本通过后：

```bash
ADAPTER=/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/实际运行目录/checkpoint-实际步数 \
SAMPLES=876 \
OUTPUT_DIR=/data/datasets/EPIC-KITCHENS/stage2_sft_eval_876 \
nohup bash scripts/evaluate_stage2_adapter_fast.sh \
  > /tmp/stage2_sft_eval_876.log 2>&1 &
```

这一步给出确定性的 Hit@1、编辑距离和格式指标。只有确定性语义结果明显改善后，才值得继续运行五候选脚本。

## 实验边界

- 输入历史仍来自当前 Stage 1，而不是 GT 历史；属于完整 Stage1→Stage2 数据来源。
- SFT 不会修复 Stage 1 的 50% 特征覆盖、HOI mask 失效和时间窗口展平。
- 当前 Hit@1 仍是 876 条自定义子集上的生成诊断，不等同于官方短期 Action Top-1。
- 五次采样仍只能叫 sampled Hit@5，不能叫 ranked Top-5。
