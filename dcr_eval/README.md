# dcr_eval — 用 INSIGHT 权重评测 DCR 项目的 EK55 数据（Top-1 only）

> 用我们训练好的 Stage 1（动作识别）+ Stage 2（Qwen2.5-VL SFT）权重，
> 在 DCR 项目（CVPR 2022, [arXiv:2204.02587](https://arxiv.org/abs/2204.02587)）的
> EK55 官方 valid 划分上测 **Next-Action Exact-Match@1（Top-1）**。
> 代码完整、可运行；**数据与权重不内置**（见 `data/README.md`、`weights/README.md`）。

## 远端代码约束（重要）

```text
/data/sp/projects/csj/DCR-main/ 是别人的项目（老师的代码），只读不写！
本项目所有代码在本目录（dcr_eval/），所有输出在 /data/datasets/EPIC-KITCHENS/dcr_eval/runs/。
对 DCR-main 仅做只读数据访问（pd.read_pickle / pd.read_csv）。
需要任何适配 DCR 数据的代码 → 在本项目新建脚本，绝不修改 DCR-main 的任何文件。
```

## 结果命名（口径声明）

```text
正式名称：INSIGHT Next-Action Exact-Match@1 on DCR-valid
不要写成：DCR Top-1
原因：DCR 官方 T1 是 τa=1s 预测口径（输入锚点前 3.5s 特征 + 分类概率排序），
      本方案是"Stage1 识别历史 + Stage2 生成式预测"，两者不可直接对比
      （DCR 官方 EK55 valid T1 = 19.2，与本方案量级完全不同）
```

## 项目结构

```text
dcr_eval/
├── README.md                 ← 本文件
├── config.py                 ← 全部路径/参数集中配置
├── data/README.md            ← 数据集说明（DCR 标注 + 我们的 EK55 帧）
├── weights/README.md         ← 权重清单与校验命令
├── docs/PIPELINE.md          ← 流程详解（决策、口径、限制、预期）
└── scripts/
    ├── 00_audit.py              ← CPU 审计：泄漏检查 + 导出评测子集
    ├── 01_prepare_features.py   ← 特征提取（--videos-file 支持子集）
    ├── 02_stage1_infer.py       ← Stage 1 推理 + 门控指标 + 8 历史构造
    ├── 03_stage2_generate.sh    ← Stage 2 Top-1 生成（三 GPU 分片）
    ├── 04_evaluate.py           ← 评测 → Next-Action Top-1
    └── run_top1_all_valid.sh    ← 一键串行 00–02、并行 Stage2 三分片、合并并评测
```

## 最快跑通（推荐）

先跑 100 条，用来确认整条数据流和评测脚本正常：

```bash
cd ~/ldy/git/AAAI26-INSIGHT/dcr_eval
DCR_RUN_NAME=top1_v1 RUN_MODE=smoke STAGE1_GPU=0 STAGE2_GPUS=0,1,3 \
  nohup bash scripts/run_top1_all_valid.sh > /tmp/insight_top1_smoke.log 2>&1 &
echo "TOP1_SMOKE_PID=$!"
```

smoke 成功后再跑全量：

```bash
cd ~/ldy/git/AAAI26-INSIGHT/dcr_eval
DCR_RUN_NAME=top1_v1 RUN_MODE=full STAGE1_GPU=0 STAGE2_GPUS=0,1,3 \
  nohup bash scripts/run_top1_all_valid.sh > /tmp/insight_top1_full.log 2>&1 &
echo "TOP1_FULL_PID=$!"
```

实时监控：

```bash
tail -f /tmp/insight_top1_smoke.log
# 或
tail -f /tmp/insight_top1_full.log
```

## 实验隔离（每个口径一个目录）

```bash
DCR_RUN_NAME=all_valid     # 全量 DCR valid（默认）
DCR_RUN_NAME=unseen_only   # Stage1 train/val 与 Stage2 train 均未见的视频
```

`DCR_RUN_NAME` 是实验/缓存隔离名；100 条测试由 `RUN_MODE=smoke` 选择。
修改特征或模型配置后应使用新名称（如 `top1_v2`），不要混用旧缓存。

输出统一在 `runs/${DCR_RUN_NAME}/` 下（features / annotations / stage1_predictions /
stage2_generated / evaluation），不同口径互不覆盖。

## 执行顺序（按此运行，不要跳步）

```text
00 审计（CPU，10-20 分钟）→ 决定评测子集
→ 01 提特征（1-2h；可复用已有特征则更快）
→ 02 Stage 1 推理 + 门控指标（~30min）
→ 100 条 smoke（~20-40 分钟）
→ 全量 Top-1（三 GPU 分片 ~8-10h）
```

### 步骤 0：CPU 审计（必须先跑）

```bash
/data/conda_envs/insight_env/bin/python3 scripts/00_audit.py
```

审计输出（在 runs/all_valid/annotations/ 下）：
- DCR valid 结构与 UID/类别映射校验
- **划分泄漏检查**（DCR valid ∩ Stage1/Stage2 训练视频；训练划分文件缺失时审计直接失败）
- 导出 `all_valid_videos.txt` 与 `unseen_valid_videos.txt`
- 已有特征复用率

**根据泄漏结论选子集**：
- 无泄漏 → 用全部 DCR valid（DCR_RUN_NAME=all_valid）
- 有泄漏且 strict-unseen 非空 → 用 `unseen_valid_videos.txt`（DCR_RUN_NAME=unseen_only），并同时报告两个口径：
  all-valid proxy 与 unseen-only proxy
- strict-unseen 为空 → 只能报告 all-valid proxy，明确声明模型选择/训练泄漏

### 步骤 1：特征提取

```bash
# 无泄漏：全部 valid
DCR_RUN_NAME=all_valid \
  /data/conda_envs/insight_env/bin/python3 scripts/01_prepare_features.py

# 有泄漏：只处理训练未见视频（审计产物在 runs/all_valid/annotations/）
DCR_RUN_NAME=unseen_only \
  /data/conda_envs/insight_env/bin/python3 scripts/01_prepare_features.py \
  --videos-file runs/all_valid/annotations/unseen_valid_videos.txt
```

注：只有一个 CLIP 特征文件（`features/val/{video}_{uid}.pt`）；02 推理时
frame/mask 两个分支输入同一文件（无真实 HOI，双流退化为单流）。
断点续跑：已存在的 .pt 自动跳过，中断后重跑同一条命令即可。

### 步骤 2：Stage 1 推理 + 门控

```bash
DCR_RUN_NAME=unseen_only \
  /data/conda_envs/insight_env/bin/python3 scripts/02_stage1_infer.py \
  --videos-file runs/all_valid/annotations/unseen_valid_videos.txt   # 有泄漏时；无泄漏可省略
```

输出（runs/${DCR_RUN_NAME}/stage1_predictions/）：
- `stage1_metrics.json`：verb/noun/action Top-1 + 预测多样性
- `stage1_val_predictions.jsonl`：每段的 Top-1 pred_verb/pred_noun
- `stage2_input.jsonl`：8 历史 + 完整 20 步未来窗口的构造结果
- `stage2_input_metrics.json`：候选位置、缺历史/缺图像数与最终覆盖率

**门控判定**：若 action_top1 接近随机（~0.2%）或 top_common 占比过高（预测坍缩），
先修复识别，不必立即跑 Stage 2。

### 步骤 3：Stage 2 Top-1 生成

小样本 smoke（100 条，先做）：

```bash
head -100 runs/all_valid/stage1_predictions/stage2_input.jsonl > /tmp/stage2_input_100.jsonl
DATASET=/tmp/stage2_input_100.jsonl OUTPUT_DIR=/tmp/stage2_smoke_out \
  GPU=0 SPLIT_TOTAL=1 SPLIT_INDEX=0 bash scripts/03_stage2_generate.sh
mv /tmp/stage2_smoke_out/candidate_0_part_0.jsonl /tmp/stage2_smoke_out/candidate_0.jsonl
```

全量三 GPU 分片（~8-10h）——**注意用 `GPU=` 变量（脚本内部读取 GPU，
不要用 CUDA_VISIBLE_DEVICES，否则会被脚本覆盖成 0）**：

```bash
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
```

### 步骤 4：评测（smoke 与全量分别跑，输出互不覆盖）

```bash
# smoke（100 条）
/data/conda_envs/insight_env/bin/python3 scripts/04_evaluate.py \
  --input /tmp/stage2_input_100.jsonl \
  --predictions-dir /tmp/stage2_smoke_out \
  --output /tmp/stage2_smoke_out/metrics.json

# 全量
/data/conda_envs/insight_env/bin/python3 scripts/04_evaluate.py
```

输出 `runs/${DCR_RUN_NAME}/evaluation/metrics.json`：Next-Action Top-1
（candidate_0 首动作精确匹配 GT，大小写不敏感、连续空白规范化）。

## 时间预估（实测校准）

| 环节 | 规模 | 预估 |
|------|------|------|
| 00 审计 | CPU | 10-20 分钟 |
| 01 特征 | ~8000 段 × 4 帧 | 1-2h（可复用已有特征则更快） |
| 02 Stage 1 | ~8000 段 | ~30 分钟 |
| 03 smoke | 100 条 × 单候选 | 20-40 分钟 |
| 03 全量 Top-1 | ~8000 条，3 GPU 分片 | **8-10 小时** |

## 核心口径（汇报必读）

- **Top-1**：candidate_0（温度 0 确定性）的 answer 第一个动作 == GT（精确匹配）
- **识别 vs 预测**：Stage 1 识别观测窗口内动作（构造历史）；GT 是目标段动作
- **与 DCR 论文不可直接对比**（DCR 官方 T1 = 19.2，τa=1s 分类口径）
- **预期**：全链路 Top-1 量级 ~1%（Stage 1 3.56% 上限 → Stage 2 语义 0.8%）
- **test s1/s2**：无标签，只能出预测文件提交 EPIC 官方评测（本项目未实现）

## 诚实边界

1. mask 特征 = 帧副本（无真实 HOI，双流退化为单流）——推理时两个分支用同一文件
2. 每条视频前 8 段因历史不足被跳过（覆盖不全）
3. 若存在划分泄漏，结果虚高，需用 unseen 子集并双口径报告
4. 生成式 Top-1 无分类分数，与分类器 Top-1 判定机制不同
