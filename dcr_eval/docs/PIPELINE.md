# DCR 对接评测流程详解（Top-1 only）

> 目标：用我们的 Stage 1 + Stage 2 权重，在 DCR 项目（CVPR 2022）的 EK55 官方 valid 划分上
> 测出 **Next-Action Exact-Match@1**（生成式 Top-1）。

## 1. 完整流程图

```text
DCR valid 标注（validation_videos.csv 过滤，有 verb/noun 标签）
 │
 │ 00_audit.py（CPU）：泄漏检查 → 导出 all/unseen 子集
 ▼
评测子集（无泄漏 = 全部；有泄漏 = unseen_valid_videos.txt）
 │
 │ 01_prepare_features.py：每段 4 帧均匀采样（RGB）→ CLIP 1024 → .pt
 ▼
features/val/{video}_{uid}.pt  +  annotations/val.json
 │
 │ 02_stage1_infer.py
 ▼
Stage 1 推理（best_model.pth + 共现矩阵）
   → 每段 Top-1 pred_verb/pred_noun + 【Top-1 门控指标】
   → 视频内 index ≥ 8 的段：前 8 段预测文本 = 历史 + 第 8 段 stop_frame = 图像
 ▼
stage1_metrics.json  +  stage1_val_predictions.jsonl  +  stage2_input.jsonl
 │
 │ 03_stage2_generate.sh（swift infer，温度 0，三 GPU 分片）
 ▼
candidate_0.jsonl（合并分片后）
 │
 │ 04_evaluate.py（严格行数校验 + 样本 ID 对齐）
 ▼
Next-Action Top-1（evaluation/metrics.json）
```

## 2. 关键设计决策

| 决策 | 方案 | 理由 |
|------|------|------|
| 数据划分 | DCR valid（官方划分，有标签） | 能自评 Top-1；test s1/s2 无标签只能提交官方 |
| 泄漏处理 | 00 导出 unseen 子集 → 01/02 用 `--videos-file` 过滤 | 模型见过 valid 视频则结果虚高，不能叫验证性能 |
| 历史切分 | 用 DCR 标注的时间戳段边界（前 8 段） | 段边界现成，无需滑动窗口；与训练"8 个完整动作"语义一致 |
| 特征 | 段内 4 帧均匀采样 → RGB → CLIP 1024 | 复制 Stage 1 训练提取器的实际行为；TSM 特征分布不匹配不可用 |
| 无泄漏 | 观测窗口（前 8 段）不含目标动作 | ✅ 天然满足（历史段比目标动作更早） |
| 评测单位 | 目标段 = 1 个样本；预测第 1 个动作 vs GT | DCR 的"一段 → 一类"结构 |
| 生成策略 | 单候选（温度 0 确定性） | 只报 Top-1；确定性输出 = 模型最有把握的答案 |
| mask 特征 | 与 frame 共用同一 .pt 文件 | 无真实 HOI；02 两个分支输入同一文件（诚实口径） |

## 3. 评测口径（必须说清楚）

```text
Stage 1 在流程中的角色：识别【观测窗口内】的动作（历史）——不是预测目标
Stage 2 的角色：基于 8 历史预测【未来 20 个动作】
GT：目标段的 verb/noun（DCR valid 标注）
Top-1：candidate_0（温度 0）的 answer 第一个动作 == GT（精确文本匹配）

与 DCR 论文 T1 的差异：
  - DCR 论文：τa=1s 预测口径（输入锚点前 3.5s 特征序列，分类概率排序）
    官方 EK55 valid Top-1 = 19.2
  - 本项目：Stage 1 识别口径构造历史 + Stage 2 生成式预测
    预期量级 ~1%
  - 两者数字不可直接对比
```

## 4. 已知限制与预期

| 限制 | 说明 |
|------|------|
| 覆盖不足 | 每条视频前 8 段无足够历史 → 跳过 |
| 识别质量上限 | Stage 1 Top-1 3.56% → 8 历史几乎全错 → Stage 2 语义 0.8% |
| 预期 Top-1 | 量级 ~1%（与 876 条 val 全链路 0.80% 一致） |
| 划分泄漏 | 若训练见过 DCR valid 视频 → 结果虚高；用 unseen 子集 + 双口径报告 |
| 无 GT 的 test | 只能出预测文件（candidate_0），需提交 EPIC 官方评测（未实现） |

## 5. 运行顺序（按此执行）

```bash
# 00 CPU 审计（必须先跑）：泄漏检查 + 导出子集（产物在 runs/all_valid/annotations/）
python scripts/00_audit.py

# 01 特征提取（有泄漏时加 --videos-file；不同口径用 DCR_RUN_NAME 隔离）
DCR_RUN_NAME=all_valid   python scripts/01_prepare_features.py
DCR_RUN_NAME=unseen_only python scripts/01_prepare_features.py \
  --videos-file runs/all_valid/annotations/unseen_valid_videos.txt

# 02 Stage 1 推理 + 门控指标 + 历史构造（与 01 同口径同 --videos-file）
DCR_RUN_NAME=unseen_only python scripts/02_stage1_infer.py \
  --videos-file runs/all_valid/annotations/unseen_valid_videos.txt
#    ← 看 stage1_metrics.json：action_top1 接近随机/坍缩则停止，不跑 Stage 2

# 03 smoke（100 条，~20-40 分钟；输出与全量隔离）
head -100 runs/all_valid/stage1_predictions/stage2_input.jsonl > /tmp/stage2_input_100.jsonl
DATASET=/tmp/stage2_input_100.jsonl OUTPUT_DIR=/tmp/stage2_smoke_out \
  GPU=0 SPLIT_TOTAL=1 SPLIT_INDEX=0 bash scripts/03_stage2_generate.sh
mv /tmp/stage2_smoke_out/candidate_0_part_0.jsonl /tmp/stage2_smoke_out/candidate_0.jsonl

# 04 评测 smoke（参数化输入/输出）
python scripts/04_evaluate.py \
  --input /tmp/stage2_input_100.jsonl \
  --predictions-dir /tmp/stage2_smoke_out \
  --output /tmp/stage2_smoke_out/metrics.json

# 03 全量 Top-1（三 GPU 分片 ~8-10h；必须用 GPU= 变量，脚本内部读 GPU，
#    外部 CUDA_VISIBLE_DEVICES 会被脚本覆盖成 0）
GPU=0 SPLIT_TOTAL=3 SPLIT_INDEX=0 bash scripts/03_stage2_generate.sh &
GPU=1 SPLIT_TOTAL=3 SPLIT_INDEX=1 bash scripts/03_stage2_generate.sh &
GPU=3 SPLIT_TOTAL=3 SPLIT_INDEX=2 bash scripts/03_stage2_generate.sh &
wait

# 合并分片 + 行数校验（必须与输入行数一致）
wc -l runs/all_valid/stage2_generated/candidate_0_part_*.jsonl
cat runs/all_valid/stage2_generated/candidate_0_part_*.jsonl \
  > runs/all_valid/stage2_generated/candidate_0.jsonl
wc -l runs/all_valid/stage2_generated/candidate_0.jsonl
wc -l runs/all_valid/stage1_predictions/stage2_input.jsonl   # 两数必须相等

# 04 评测全量
python scripts/04_evaluate.py
```

## 6. 时间预估（实测校准：876 条 × 5 候选 ≈ 6h）

| 环节 | 规模 | 预估 |
|------|------|------|
| 00 审计 | CPU | 10-20 分钟 |
| 01 特征 | ~8000 段 × 4 帧 | 1-2h（可复用已有特征则更快） |
| 02 Stage 1 | ~8000 段 | ~30 分钟 |
| 03 smoke | 100 条单候选 | 20-40 分钟 |
| 03 全量 Top-1 | ~8000 条，3 GPU 分片 | **8-10 小时** |

## 7. 未实现/占位说明

- **严格 τa=1s 无泄漏版**：输入仅锚点前 3.5s 特征、不含段内帧——需要 Stage 1
  在锚点前帧上重新评估/训练，本项目未实现（当前为识别口径）
- **test s1/s2 提交官方评测**：只产出预测文件，官方 T1 需 EPIC 官网提交（未实现）
