# INSIGHT 项目代码、数据、权重与实验产物位置总说明

> 更新时间：2026-08-03  
> 项目：AAAI 2026 INSIGHT（EK55 复现与 DCR-valid Top-1 对接）  
> 本文用途：说明“代码在哪里、数据从哪里来、每一步读写什么、当前结果属于什么协议”。  
> 事实来源：当前仓库代码、远端审计输出及 [`WORKLOG.md`](./WORKLOG.md)。

## 1. 最重要的结论

当前真正完成的全量 Top-1 实验是：

```text
DCR EK55 validation（有 GT 时间边界和动作标签）
  → 按 GT 动作段均匀取 4 帧
  → EgoVideo large_best 中的 CLIP 图像编码器（1024 维）
  → 重建 Stage1 识别历史动作
  → 8 个 Stage1 预测历史 + 最后一段历史图像
  → Qwen2.5-VL-7B + SFT checkpoint-200
  → 生成 20 个未来动作
  → 第一个生成动作与目标 GT verb+noun 精确匹配
```

全量运行名为 `all_valid_top1_id_v1`，1,472 个样本中命中 15 个：

```text
INSIGHT Next-Action Exact-Match@1 Top-1 = 1.019%
```

这不是程序错误：1,472 条输入/预测 ID 完整对应、解析率 100%。但它是当前不完整复现的 **DCR-valid 生成式 proxy**，不是 DCR 官方 `tau_a=1s` 分类协议，也不是论文可以直接对比的官方 Top-1。

## 2. 本地、远端和数据盘根目录

| 类型 | 位置 | 用途 |
|------|------|------|
| Mac 本地代码 | `/Users/esunlee/clone_workspace/AAAI26-INSIGHT` | 代码编辑、文档和工作日志 |
| 远端代码 | `/home/amax/ldy/git/AAAI26-INSIGHT` | GPU 服务器实际运行代码 |
| 主数据根目录 | `/data/datasets/EPIC-KITCHENS` | EK55 帧、特征、训练集、权重和实验输出 |
| 预训练模型根目录 | `/data/pretrain_model` | EgoVideo、Qwen、SAM2、手物检测器等外部模型 |
| DCR 项目（只读） | `/data/sp/projects/csj/DCR-main` | DCR EK55 划分、标注和官方/已有特征 |
| Python 环境 | `/data/conda_envs/insight_env` | PyTorch、Transformers、ms-swift 和全部运行依赖 |
| Python 入口 | `/data/conda_envs/insight_env/bin/python3` | 当前统一 Python |
| ms-swift 入口 | `/data/conda_envs/insight_env/bin/swift` | Stage2 训练和推理 |
| 临时日志 | `/tmp/*.log` | nohup 驱动日志和诊断日志；不是长期归档位置 |

本地与远端不是自动同步目录。代码在本地修改后通过文本传输到远端；关键运行文件应通过 SHA256 和语法检查确认一致。远端数据和权重不会自动回传 Mac。

## 3. 仓库代码结构

### 3.1 上游 INSIGHT 主体

```text
AAAI26-INSIGHT/
├── HandObject/               # Stage1：手物语义动作识别
├── CognitiveReasoning/       # Stage2：GRPO 奖励与序列评测
├── assets/                   # 论文框架图和结果图
├── README.md                 # 上游项目总说明
└── requirements.txt
```

#### `HandObject/`

| 文件 | 作用 | 当前状态 |
|------|------|---------|
| `model.py` | 双流 MLP + Transformer + verb/noun 分类头 | 当前 Stage1 架构来源 |
| `dataset.py` | 读取 frame/mask 特征和动作窗口 | 原训练数据读取器 |
| `train.py` | Stage1 训练/验证循环 | 被 `main.py`/重建脚本调用 |
| `main.py` | 原 Stage1 训练入口 | 已适配 1024 维、125 verb、352 noun |
| `rebuild_stage1.py` | 原 checkpoint 损坏后重建 Stage1 | 生成当前可用 Stage1 权重 |
| `export_stage1_predictions.py` | 导出 train/val 动作预测 | 构造历史 Stage2 数据使用 |
| `evaluate_topk.py` | Stage1 verb/noun/action Top-k 诊断 | 历史诊断；当前只要求 Top-1 |
| `train_anticipation_topk.py` | 独立短期 anticipation 分类尝试 | 非当前 DCR 生成式线路 |

#### `CognitiveReasoning/`

| 文件 | 作用 | 当前状态 |
|------|------|---------|
| `plugin_in_log_intention.py` | ms-swift 外部 reward 插件 | GRPO 训练使用 |
| `my_rewards_intention.py` | action/intention/格式 reward | 参考与历史训练 |
| `run_external_reward_func_7B_qwen_intention.sh` | 原 Stage2 GRPO 启动脚本 | 已被服务器参数化命令替代 |
| `cal_ED.py` | action/verb/noun 编辑距离 | 历史序列诊断 |
| `evaluate_generated_topk*.py` | 多候选生成结果评测 | 历史 5 候选实验 |

### 3.2 EK55 复现补充代码

#### `feature_extraction/`

| 文件 | 输入 → 输出 | 用途 |
|------|-------------|------|
| `convert_annotations.py` | EK55 CSV → `insight_annotations/*.json` | 原 Stage1 标注转换 |
| `extract_features.py` | 原始帧 + CLIP/SAM2/检测器 → `.pt` | 原复现特征流程；真实 HOI 不完整 |
| `build_stage2_from_stage1.py` | Stage1 JSONL → Stage2 预测历史数据 | 生成 10,748/876 条历史训练数据 |
| `build_stage2_lta_dataset.py` | 动作标注 → GT-history LTA 数据 | 历史探针/对照数据 |
| `build_stage2_sft_dataset.py` | Stage2 数据 → 带 assistant GT 的 SFT 数据 | SFT 快速修复线路 |
| `extract_ek55_anticipation_rgb.py` | EK55 帧 → anticipation 特征 | 非当前 DCR proxy 主线 |
| `export_rulstm_lmdb_sequences.py` | RULSTM LMDB → 时序序列 | 下载失败后暂停的官方短期线路 |

#### `evaluation/`

| 文件 | 作用 |
|------|------|
| `audit_pipeline_exact.py` | 权重、数据量、哈希和 Stage2 产物审计 |
| `plot_sft_training_curves.py` | 从 `trainer_state.json` 导出 SFT 曲线 |
| `prepare_ek55_anticipation.py` | 构造 EK55 anticipation 协议数据 |
| `train_ek55_future_verb_map.py` | EGO-TOPO 风格未来动词 mAP baseline |

#### 根目录 `scripts/`

| 文件 | 作用 | 是否当前 Top-1 主线 |
|------|------|:--:|
| `run_stage2_sft_fast.sh` | Stage2 SFT 训练 | 当前 adapter 来源 |
| `evaluate_stage2_adapter_fast.sh` | Stage2 SFT 快速诊断 | 否 |
| `run_stage2_generated_topk.sh` | 历史 5 候选生成 | 否 |
| `run_stage2_diagnostic_100.sh` | GT/pred history 与 base/adapter 对照 | 诊断用 |
| `run_ek55_future_verb_map.sh` | P25/P50/P75 mAP | 独立指标线路 |
| `run_ek55_anticipation_topk.sh` | 短期 Top-k 尝试 | 已暂停 |
| `run_ek55_topk_rulstm_features.sh` | RULSTM 特征短期 Top-k | 已暂停 |

### 3.3 当前 DCR-valid Top-1 主线：`dcr_eval/`

远端代码位置：

```text
/home/amax/ldy/git/AAAI26-INSIGHT/dcr_eval
```

| 文件 | 阶段 | 读取 | 写出 |
|------|------|------|------|
| `config.py` | 集中配置 | 数据、权重和 run name 环境变量 | 全部脚本共享路径 |
| `data_utils.py` | 数据校验 | DCR CSV/PKL | 统一 DataFrame/视频集合 |
| `scripts/00_audit.py` | Phase 0 | DCR 标注、已有训练划分和帧 | 审计 JSON、视频清单、泄漏结论 |
| `scripts/01_prepare_features.py` | Phase 1 | GT 动作段、原始 JPEG、CLIP 权重 | `features/val/*.pt`、`annotations/val.json`、meta |
| `scripts/02_stage1_infer.py` | Phase 2 | 特征、Stage1 权重、共现矩阵、GT 时间边界 | Stage1 指标/预测和 Stage2 输入 |
| `scripts/03_stage2_generate.sh` | Phase 3 | Stage2 输入、Qwen、SFT adapter | 三卡分片 `candidate_0_part_N.jsonl` |
| `scripts/attach_prediction_ids.py` | Phase 3 后处理 | 分片输入 + swift 原始输出 | 带 `clip_uid/action_idx` 的预测 |
| `scripts/validate_generated_jsonl.py` | 校验 | 候选 JSONL | 数量、结构和唯一 ID 判定 |
| `scripts/04_evaluate.py` | Phase 4 | Stage2 输入 + 合并预测 | `metrics.json` |
| `scripts/05_export_results_csv.py` | 结果导出 | Stage1/Stage2/GT 全部已有产物 | 逐样本总 CSV + 汇总 CSV |
| `scripts/run_top1_all_valid.sh` | 总入口 | 上述全部资源 | Phase 0–4 一键运行 |
| `tests/test_evaluate.py` | 单元测试 | 合成输入 | 评测逻辑检查 |
| `tests/test_stage2_shell.sh` | Shell 测试 | 合成/替代命令 | 分片与恢复逻辑检查 |

## 4. 原始数据和标注位置

### 4.1 EK55 原始帧

```text
/data/datasets/EPIC-KITCHENS/{participant_id}/rgb_frames/frame_{frame:010d}.jpg
```

已记录规模：约 1,251,563 张 JPEG、228 GB。

当前目录只有 participant，没有 video_id。例如：

```text
/data/datasets/EPIC-KITCHENS/P07/rgb_frames/frame_0000012345.jpg
```

这不是理想的 EK55 视频级布局。DCR valid 中有 13 个 participant 对应多个视频，同一帧号可能来源不明确，所以当前结果必须标记为 frame-path-ambiguous proxy。

### 4.2 原 EK55 标注

```text
/data/datasets/EPIC-KITCHENS/annotations/EPIC_train_action_labels.csv
/data/datasets/EPIC-KITCHENS/insight_annotations/train.json
/data/datasets/EPIC-KITCHENS/insight_annotations/val.json
```

CSV 是原动作标注；`insight_annotations` 是 `convert_annotations.py` 转成 Stage1 格式后的文件。

### 4.3 DCR EK55 标注与划分（只读）

```text
/data/sp/projects/csj/DCR-main/data/EK55/
├── validation_videos.csv
├── EPIC_train_action_labels.pkl
├── EPIC_verb_classes.csv
├── EPIC_noun_classes.csv
├── EPIC_test_s1_timestamps.pkl
└── EPIC_test_s2_timestamps.pkl
```

用途：

- `validation_videos.csv`：当前评测的 40 个有标签视频清单。
- `EPIC_train_action_labels.pkl`：动作 UID、video_id、participant、start/stop frame、verb/noun class。
- verb/noun CSV：125 个动词、352 个名词的 ID→文本映射。
- test s1/s2：官方 challenge 测试时间戳，没有本地 GT，当前没有用来计算 Top-1。

### 4.4 DCR 已有 TSM 特征

```text
/data/sp/projects/csj/DCR-main/data/feature/EK55_RGB_TSM
```

这是 DCR/RULSTM 体系的 2048 维特征，和当前 Stage1 的 1024 维输入权重不兼容，因此当前主线没有使用。

## 5. “测试集识别与分割”实际做法

### 5.1 当前不是自动动作分割

项目没有从完整视频中自动检测动作开始/结束。动作段由 DCR/EPIC 的 GT 标注直接给出：

```text
(video_id, uid, start_frame, stop_frame, verb_class, noun_class)
```

因此 Stage1 当前测的是 **oracle temporal segmentation 下的动作识别**，不是“无标注视频中的动作检测 + 分类”。

### 5.2 数据集选择

1. 读取 `validation_videos.csv` 得到 40 个 DCR valid 视频。
2. 读取 `EPIC_train_action_labels.pkl`。
3. 只保留 `video_id` 位于 valid 清单的动作，得到 4,979 条有标签动作。
4. 不使用无标签 `EPIC_test_s1_timestamps.pkl`/`s2` 计算本地 Top-1。

### 5.3 每个动作段如何取帧

对标注段 `[start_frame, stop_frame)`：

```python
n = stop_frame - start_frame
if n <= 4:
    indices = range(start_frame, stop_frame)
else:
    step = n / 4
    indices = [int(start_frame + i * step) for i in range(4)]
```

每段最多均匀取 4 帧；至少成功读取 2 帧才保留。图像经 BGR→RGB、224×224 resize、CLIP mean/std 归一化，输入 CLIP；多个帧的 `pooler_output` 求平均得到一个 `[1024]` 特征。

输出：

```text
runs/{RUN_NAME}/features/val/{video_id}_{uid}.pt
```

### 5.4 Stage1 如何识别

每个 GT 动作段对应一个 1024 维特征：

```text
frame branch = 该 CLIP 特征
mask branch  = 同一个 CLIP 特征（临时替代，非真实 HOI）
```

Stage1 输出 125 类 verb logits 和 352 类 noun logits，再乘 verb-noun 共现先验选择联合 action Top-1。

### 5.5 Stage2 如何保证不看目标动作

对同一视频按 `(start_frame, uid)` 排序，目标为动作 `A_i`：

- 历史：`A_(i-8)` 到 `A_(i-1)` 的 Stage1 预测文本。
- 因果门：任何历史段满足 `history.stop_frame >= target.start_frame`，整条样本排除。
- 图像：历史最后一段 `A_(i-1)` 的 `stop_frame`，失败时依次回退 `stop-1`、`start`。
- GT：`A_i` 的 verb+noun，只保存在评分字段，不写进 user prompt。
- Stage2 仍生成 20 个动作，但 Top-1 只要求存在 `A_i` 这一条 GT。

## 6. 特征位置

### 6.1 原 Stage1 训练特征

```text
/data/datasets/EPIC-KITCHENS/features/
├── frame_features/
│   ├── train/*.pt
│   └── val/*.pt
├── mask_features/
│   ├── train/*.pt
│   └── val/*.pt
└── cooccurrence_matrix.pt
```

已记录约 15,480 个 1024 维动作特征。绝大多数 mask 特征与 frame 特征相同；真实 HOI 只覆盖极少部分，因此论文的双流优势没有被忠实复现。

### 6.2 当前 DCR run 的独立特征

```text
/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1/features/val/
```

命名：

```text
{video_id}_{uid}.pt
```

审计得到 4,979 条 DCR valid 动作中约 2,201 条可从当前 JPEG 布局提取。特征规格记录在：

```text
runs/all_valid_top1_id_v1/annotations/feature_spec.json
runs/all_valid_top1_id_v1/annotations/meta.json
```

## 7. 权重位置和使用状态

| 权重 | 远端位置 | 状态/用途 |
|------|----------|----------|
| EgoVideo CLIP large | `/data/pretrain_model/EgoVideo/checkpoints/large_best.pt` | 当前可用；实际转换为 1024 维 CLIP 图像编码器 |
| 当前 Stage1 | `/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth` | 当前 DCR 主线使用；89 tensors，可加载 |
| verb-noun 共现矩阵 | `/data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt` | Stage1 联合重排使用 |
| Qwen 基座 | `/data/pretrain_model/Qwen2.5-VL-7B-Instruct` | 当前 Stage2 基座 |
| 当前 SFT adapter | `/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/checkpoint-200` | 当前全量 `1.019%` 使用；验证损失最优 |
| SFT 末轮 adapter | `.../stage2_sft_fast_output/v0-20260730-225536/checkpoint-895` | 过拟合末轮对照，不是当前默认 |
| GRPO 全量 adapter | `/data/datasets/EPIC-KITCHENS/stage2_stage1_history_full_3gpu_v3_output/v0-20260729-235831/checkpoint-1773` | 历史 GRPO/五候选实验，不是当前 DCR 主线 |
| GRPO probe best | `/data/datasets/EPIC-KITCHENS/stage2_lta_context_probe_output/v0-20260729-092844/checkpoint-75` | 100-step 探针历史产物 |
| 原 Stage1 checkpoint | `/data/datasets/EPIC-KITCHENS/checkpoint/best_model.pth` | **禁止使用**：PyTorch zip central directory 损坏 |
| SAM2 | `/data/pretrain_model/SAM2/sam2.1_hiera_large.pt` | 曾验证可加载；当前 DCR 主线未用 |
| Hand Object Detector | `/data/pretrain_model/Hand_Object_Detector` | 缺少作者第一人称 `handobj_100K+ego` 权重 |

论文依赖但当前未获得：

- EgoVideo-V 1408 维视频级权重/完整提取函数。
- 第一人称 `handobj_100K+ego` 手物检测权重。
- 作者完整组合特征提取实现。

## 8. Stage1 训练、预测及其产物

### 8.1 当前 Stage1 训练数据

```text
/data/datasets/EPIC-KITCHENS/insight_annotations/
/data/datasets/EPIC-KITCHENS/features/frame_features/
/data/datasets/EPIC-KITCHENS/features/mask_features/
/data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt
```

### 8.2 重建权重

```text
/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth
```

原 40-epoch 权重文件损坏后，重建脚本实际在第 6 epoch 提前结束并保存当前权重。该事实必须和“原论文官方 Stage1 权重”区分。

### 8.3 原 EK55 Stage1 预测历史

```text
/data/datasets/EPIC-KITCHENS/stage1_predictions/
├── stage1_train_predictions.jsonl   # 14,179 行
├── stage1_val_predictions.jsonl     # 1,263 行
├── stage1_topk_metrics.json
└── stage1_topk_metrics.txt
```

### 8.4 当前 DCR run 的 Stage1 输出

```text
/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1/stage1_predictions/
├── stage1_metrics.json
├── stage1_val_predictions.jsonl
├── stage2_input.jsonl
└── stage2_input_metrics.json
```

`stage2_input_metrics.json` 已确认：

| 项目 | 数量 |
|------|-----:|
| candidate positions | 4,663 |
| written | 1,472 |
| missing history | 1,754 |
| noncausal history | 1,437 |
| missing image | 0 |
| coverage | 31.568% |

## 9. Stage2 训练数据、权重和历史实验

### 9.1 Stage1 预测历史数据

```text
/data/datasets/EPIC-KITCHENS/stage2_stage1_history/
├── stage2_train_stage1_history.jsonl   # 10,748
└── stage2_val_stage1_history.jsonl     # 876
```

每条数据包含：8 个 Stage1 预测动作历史、1 张观察图像、未来 20 动作 GT/监督字段。

### 9.2 SFT 数据

```text
/data/datasets/EPIC-KITCHENS/stage2_stage1_history_sft/
├── stage2_train_sft.jsonl          # 10,748
├── stage2_val_sft.jsonl            # 876
├── stage2_val_sft_100.jsonl        # 100
├── stage2_train_sft_smoke.jsonl    # 32
└── stage2_val_sft_smoke.jsonl      # 12
```

### 9.3 SFT 训练输出

```text
/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/
├── checkpoint-200/   # 当前最佳
└── checkpoint-895/   # 末轮
```

训练共 895 steps，约 36 分 43 秒。checkpoint-200 的验证损失最低，因此当前 DCR 推理使用它。

### 9.4 历史 GRPO 输出

```text
/data/datasets/EPIC-KITCHENS/stage2_stage1_history_full_3gpu_v3_output/
/data/datasets/EPIC-KITCHENS/stage2_generated_topk_checkpoint1773/
```

这些目录属于早期 GRPO 和五候选实验，不应与当前 SFT `all_valid_top1_id_v1` 混合。

## 10. 当前全量 DCR Top-1 运行目录

```text
/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1/
├── annotations/
│   ├── audit_summary.json
│   ├── all_valid_videos.txt
│   ├── unseen_valid_videos.txt
│   ├── feature_spec.json
│   ├── meta.json
│   └── val.json
├── features/
│   └── val/{video_id}_{uid}.pt
├── stage1_predictions/
│   ├── stage1_metrics.json
│   ├── stage1_val_predictions.jsonl
│   ├── stage2_input.jsonl                 # 1,472 行
│   └── stage2_input_metrics.json
├── stage2_generated/
│   ├── generation_spec.txt
│   ├── shard_0.log
│   ├── shard_1.log
│   ├── shard_2.log
│   ├── candidate_0_part_0.jsonl           # 490 行 + 唯一 ID
│   ├── candidate_0_part_1.jsonl           # 491 行 + 唯一 ID
│   ├── candidate_0_part_2.jsonl           # 491 行 + 唯一 ID
│   └── candidate_0.jsonl                  # 1,472 行 + 唯一 ID
├── evaluation/
│   ├── metrics.json                       # Top-1 = 1.019%
│   ├── all_valid_top1_results.csv         # 远端导出待确认
│   └── all_valid_top1_summary.csv         # 远端导出待确认
└── logs/
    └── 05_export_results_csv.log           # CSV 导出后生成
```

已确认文件哈希：

```text
stage2_input.jsonl:
97c003224fe86ab042d2c60bf215728c00e9f79b779bbbfe51a93674aa813b5d

candidate_0.jsonl:
c698c1ba65f897b709c2e3366456bc7825b9ea4001f26e0478e5d64f6f7540c2
```

### 10.1 最终指标

```text
samples_labeled    = 1472
top1_correct       = 15
next_action_top1   = 1.019%
parse_failures     = 0
parse_success_rate = 100%
```

## 11. 其他已完成指标和产物

### 11.1 EGO-TOPO 风格 future-verb mAP（feature-limited）

```text
/data/datasets/EPIC-KITCHENS/ek55_future_verb_map_output/metrics.json
/data/datasets/EPIC-KITCHENS/final_metrics_20260730/
```

结果：

| 指标 | 数值 |
|------|-----:|
| Average All mAP | 46.245 |
| Average Freq mAP | 53.753 |
| Average Rare mAP | 41.107 |

边界：只有 96 个 val 阶段样本、64 个正类且使用 1024 维 proxy 特征，不等价于论文完整覆盖。

### 11.2 本地汇报文件

```text
docs/INSIGHT_总结果.xlsx
docs/Stage1_结果.xlsx
docs/Stage2_结果.xlsx
docs/stage_results_topk_map.csv
docs/stage2_sft_charts.xlsx
docs/stage2_sft_training_report.html
docs/REPORT_TO_PROFESSOR.md
docs/EK55_METRICS_PROTOCOL.md
docs/TECHNICAL_EXPLANATION.md
docs/WORKLOG.md
```

这些是汇总与历史记录，不是模型直接读取的数据。

## 12. 指标口径

### 12.1 当前生成式 Top-1

```text
Stage2 的 <answer> 中第一个动作 == 目标 GT action text
```

比较前只做小写和连续空白规范化。verb 和 noun 必须同时正确；解析失败也进入分母但不命中。

### 12.2 可从总 CSV 附带统计

- Stage2 Action/Verb/Noun Top-1。
- overall 和逐视频命中率。
- `<answer>` 解析率。
- 恰好 20 个动作率。
- 20 个动作全部满足“两 token action”格式的比例。
- 平均生成动作数、首动作预测多样性和常见预测。
- 每条样本的 8 个 Stage1 历史、20 个 Stage2 动作、think/intention、GT 和命中标记。
- Stage1 总体 Verb/Noun/Action Top-1，以及存在时目标段的 Stage1 预测分数。

### 12.3 当前产物不能直接给出的指标

- DCR 官方 `tau_a=1s` Action Top-1/Top-5。
- 2513 类分类概率下的官方 Top-5。
- 需要多候选或类别置信度的 Recall@5。
- 论文完整覆盖下的 P25/P50/P75 All/Freq/Rare mAP。

原因不是缺少计算公式，而是当前输入锚点、样本协议和模型输出类型不同。

## 13. 数据泄漏与报告边界

DCR valid 与当前训练资产存在真实重叠：

```text
DCR valid: 40 videos / 4,979 actions
与 Stage1 train 精确 (video_id, uid) 重叠: 4,602
与 Stage1 val 精确重叠: 377
与 Stage2 SFT train 重叠: 10 videos
严格 video-unseen valid 子集: 0 videos
```

所以结果不能声称“严格测试集泛化”或“无泄漏性能”。正确表述：

> 使用当前不完整 INSIGHT 复现权重，在 DCR EK55 有标签 validation 动作段上运行的 recognition-gated generative proxy；1,472 个可构造样本的 Next-Action Exact-Match@1 为 1.019%。

## 14. 当前主要缺陷按数据流排序

1. **帧路径不含 video_id**：可能读取同 participant 其他视频的同名帧。
2. **使用 GT 动作时间边界**：没有测试自动动作分割能力。
3. **图像级 1024 维 CLIP 替代论文 1408 维视频特征**：缺少时序动态。
4. **无真实 HOI**：mask branch 复用 frame 特征，双流退化。
5. **Stage1 权重是损坏后重建版本**：不是作者官方权重。
6. **Stage1 误差进入 8 步历史**：错误级联到 Stage2。
7. **Stage2 SFT 主要学会格式，动作语义仍弱**。
8. **训练/validation 重叠**：不能当严格泛化指标。
9. **当前 Top-1 协议不是 DCR 官方 tau_a=1s**。

## 15. 推荐的唯一当前入口

全量运行：

```bash
cd /home/amax/ldy/git/AAAI26-INSIGHT/dcr_eval

DCR_RUN_NAME=all_valid_top1_id_v1 \
DCR_EVAL_FUTURE_REQUIRED=1 \
RUN_MODE=full \
STAGE1_GPU=0 \
STAGE2_GPUS=0,1,3 \
bash scripts/run_top1_all_valid.sh
```

当前 run 已完成，重复执行会按 generation spec、行数和 ID 校验安全复用完整缓存；不要通过删除或覆盖现有 run 来启动不同协议。新设置应使用新的 `DCR_RUN_NAME`。

导出总 CSV：

```bash
DCR_RUN_NAME=all_valid_top1_id_v1 \
/data/conda_envs/insight_env/bin/python3 \
  scripts/05_export_results_csv.py
```

## 16. 文件管理规则

1. `/data/sp/projects/csj/DCR-main` 只读，不写入任何产物。
2. 每种协议使用独立 `DCR_RUN_NAME`。
3. 不使用损坏的 `/data/datasets/EPIC-KITCHENS/checkpoint/best_model.pth`。
4. 当前 Stage2 正式 adapter 固定为 SFT `checkpoint-200`，不要与 GRPO `checkpoint-1773` 混用。
5. 每次生成必须保留 `generation_spec.txt`。
6. 合并候选必须通过 `unique_ids == input rows`。
7. `metrics.json`、CSV、SHA256 和日志应与 run 一起归档。
8. `/tmp` 只放可丢失日志或传输包；最终结果必须位于 run 的 `evaluation/`。
9. GPU 2 曾出现 Xid 79 “fallen off the bus”；稳定线路使用物理 GPU 0/1/3。

## 17. 快速定位表

| 想找的内容 | 路径 |
|------------|------|
| 当前总代码 | `/home/amax/ldy/git/AAAI26-INSIGHT` |
| 当前 DCR 代码 | `/home/amax/ldy/git/AAAI26-INSIGHT/dcr_eval` |
| 原始帧 | `/data/datasets/EPIC-KITCHENS/Pxx/rgb_frames/` |
| DCR valid 视频清单 | `/data/sp/projects/csj/DCR-main/data/EK55/validation_videos.csv` |
| DCR 动作标注 | `/data/sp/projects/csj/DCR-main/data/EK55/EPIC_train_action_labels.pkl` |
| CLIP 权重 | `/data/pretrain_model/EgoVideo/checkpoints/large_best.pt` |
| Stage1 权重 | `/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth` |
| 共现矩阵 | `/data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt` |
| Qwen 基座 | `/data/pretrain_model/Qwen2.5-VL-7B-Instruct` |
| 当前 SFT adapter | `/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/checkpoint-200` |
| 当前全量 run | `/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1` |
| Stage2 输入 | `.../stage1_predictions/stage2_input.jsonl` |
| Stage2 完整预测 | `.../stage2_generated/candidate_0.jsonl` |
| 最终 Top-1 | `.../evaluation/metrics.json` |
| 逐样本总 CSV | `.../evaluation/all_valid_top1_results.csv` |
| 汇总 CSV | `.../evaluation/all_valid_top1_summary.csv` |
| 本地工作日志 | `/Users/esunlee/clone_workspace/AAAI26-INSIGHT/docs/WORKLOG.md` |

---

如路径、权重或协议发生变化，应先修改 `dcr_eval/config.py`，使用新的 `DCR_RUN_NAME`，再同步更新本文和 `WORKLOG.md`；不要在旧 run 内混合不同输入或 adapter 的结果。
