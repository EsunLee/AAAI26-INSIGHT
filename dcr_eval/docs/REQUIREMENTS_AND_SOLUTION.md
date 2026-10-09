# DCR 对接评测 — 需求、现状与解决方案（逻辑确认文档）

> 本文档完整记录：① 用户要求 ② 当前项目状态（原项目可复用资产 + 目标项目情况）
> ③ 解决方案（dcr_eval 的设计与决策依据）。用于确认整体逻辑后再执行。

---

## 一、用户要求

### 1.1 原始需求

- 用**现有的模型权重**（Stage 1 动作识别 + Stage 2 Qwen2.5-VL SFT）评测
  **另一个数据集**：远程 `/data/sp/projects/csj/DCR-main`（CVPR 2022 项目）
- 目标指标：**Top-1**（"就是测出这个项目的 TOP1"）
- 测试集形态：单标签（每个动作段一个标签，GT 存在）

### 1.2 用户对流程的理解（已确认正确）

```text
① 难点：怎么识别动作 + 选动作发生之前的几秒当观测窗口
② 把目标动作当 GT
③ 前几秒画面 → Stage 1 语义识别（识别观测窗口内的动作 = 历史）
④ 历史 → Stage 2 预测未来
⑤ 预测动作 vs GT 比较 → 计算 Top-1
```

### 1.3 约束与已知风险

- 目标数据的特征维度 `[batch, 48, 2048]`（TSM）与我们的输入（1024 CLIP）**不一定对得上**
- 模型训练时输入长度定死（8 历史 → 20 未来），不能重训、参数不能动
- 需要"动作发生之前"的窗口（无泄漏）与 GT 对齐

---

## 二、当前项目状态

### 2.1 原项目（INSIGHT 复现）可复用资产

| 资产 | 位置（远程） | 状态 | 本方案中的用途 |
|------|------|:--:|------|
| EK55 原始帧（125 万 JPEG，60fps） | `/data/datasets/EPIC-KITCHENS/{Pxx}/rgb_frames/` | ✅ | 特征提取的唯一数据源 |
| CLIP ViT-L/14 权重 | `/data/pretrain_model/EgoVideo/checkpoints/large_best.pt` | ✅ | 帧 → 1024 维特征 |
| 特征提取逻辑 | `feature_extraction/extract_features.py` | ✅ | 移植 build_clip / 4 帧采样 |
| Stage 1 权重 | `/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth` | ✅ | 动作识别 |
| 共现矩阵 | `/data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt` | ✅ | 公式5 语义先验重排 |
| Stage 1 评测代码 | `HandObject/evaluate_topk.py`（全参数化） | ✅ | 评测逻辑移植 |
| Stage 2 基座 + SFT adapter | Qwen2.5-VL-7B + `.../stage2_sft_fast_output/.../checkpoint-200` | ✅ | 未来动作生成 |
| Stage 2 生成流程 | `scripts/run_stage2_generated_topk.sh` | ✅ | 五候选生成逻辑移植 |
| 原项目已知限制 | HOI 缺失（mask=帧副本）、Stage1 Top-1 3.56%、Stage2 语义 0.8% | ⚠️ | 决定方案预期（见 3.5） |

### 2.2 目标项目（DCR）状态

- **DCR** = *Learning to Anticipate Future with Dynamic Context Removal*（CVPR 2022），EK55 短期动作预测（τa=1s）
- 数据目录 `/data/sp/projects/csj/DCR-main/data/EK55/`：

| 文件 | 内容 | 是否可用 |
|------|------|:--:|
| `EPIC_train_action_labels.pkl` | 28472 条标注（verb/noun 类、起止帧） | ✅ |
| `validation_videos.csv` | **官方 valid 划分（有标签 → 可自评）** | ✅ |
| `EPIC_test_s1/s2_timestamps.pkl` | 官方 test（**无标签**，challenge） | ✅（只能提交官方） |
| `EPIC_verb/noun_classes.csv` | 类 id → 名称 | ✅ |
| `data/feature/EK55_RGB_TSM/data.mdb` | TSM 特征 19.3GB（2048 维 8fps） | ⚠️ 与我们的权重不兼容（见 2.3） |

- **DCR 数据没有原始视频帧**（只有特征 mdb）——但我们的 EK55 目录有全部帧（test 28 个 participant 已确认）

### 2.3 必须解决的三处不匹配

| 不匹配 | 目标数据（DCR） | 我们的模型 | 处理方式 |
|------|------|------|------|
| 特征维度 | TSM **2048** | Stage 1 输入 **1024** | 不用 TSM，用 CLIP 重新提 1024 维 |
| 特征语义 | TSM 视频动作特征 | CLIP 图像语义特征 | 同上（分布不匹配，直接喂会崩） |
| 任务形态 | τa=1s 单动作预测（T1/T5） | 8 历史 → 20 未来（生成式） | 以目标段为评测单位，预测首动作 vs GT |

---

## 三、解决方案（dcr_eval 项目）

### 3.1 核心决策（每条都有依据）

| # | 决策 | 依据 |
|:--:|------|------|
| D1 | 数据划分用 **DCR valid**（有标签） | 自评 Top-1 的前提；test s1/s2 无标签只能提交官方 |
| D2 | 特征用 **我们的 CLIP 管线从原始帧重提**（1024 维） | 权重训练分布 = CLIP；TSM 特征分布不匹配不可用 |
| D3 | 每段 **4 帧均匀采样 → 平均**（与训练一致） | 复用 extract_features.py 逻辑，保证输入分布一致 |
| D4 | 历史切分用 **DCR 标注的时间戳段边界**（前 8 段） | 段边界现成，无需滑动窗口；与训练"8 个完整动作"语义一致 |
| D5 | 观测窗口 = 目标段**之前**的 8 个段（不含目标） | 满足无泄漏；符合用户"动作发生之前的几秒当窗口" |
| D6 | Stage 1 识别历史段的动作文本 → Stage 2 生成 20 个 → 取第 1 个 vs GT | 全链路对接用户要求；评测单位对齐 DCR"一段 → 一类" |
| D7 | Top-1 = candidate_0（温度 0）首动作精确匹配 GT | 生成式无分类分数，确定性输出 = 最有把握的答案 |
| D9 | 图像 = 历史最后一段的 stop_frame 帧 | 与 Stage 2 训练输入一致（观察窗口末帧） |
| D10 | mask 特征 = 帧副本 | 无真实 HOI（诚实口径，双流=单流） |

### 3.2 四步流程（对应 4 个脚本）

```text
DCR valid 标注（pkl，有 verb/noun 标签）
 │
 │ ① 01_prepare_features.py
 │    每段 4 帧采样 → CLIP 1024 → .pt（mask=帧副本）
 │    复用：extract_features.py 的 build_clip / 采样逻辑
 ▼
features/val/{video}_{uid}.pt + annotations/val.json
 │
 │ ② 02_stage1_infer.py
 │    Stage 1 推理（best_model.pth + 共现矩阵）→ 每段 pred_verb/pred_noun
 │    对 index ≥ 8 的段：前 8 段预测文本 = 历史 + 第 8 段 stop_frame = 图像
 │    复用：evaluate_topk.py 的模型加载 / 共现矩阵逻辑
 ▼
stage2_input.jsonl（ms-swift 格式，含 GT 文本）
 │
 │ ③ 03_stage2_generate.sh
 │    swift infer + SFT adapter → candidate_0.jsonl（温度 0）
 ▼
candidate_0.jsonl
 │
 │ ④ 04_evaluate.py
 │    candidate_0 首动作 == GT → Top-1
 ▼
metrics.json（Next-Action Top-1）
```

### 3.3 原项目资产 → 方案映射（复用清单）

| 原项目资产 | 方案中对应 | 形态 |
|------|------|------|
| extract_features.py（CLIP 加载/4 帧采样） | 01 脚本主体 | 代码移植 |
| evaluate_topk.py（模型/共现/推理） | 02 脚本主体 | 代码移植 |
| run_stage2_generated_topk.sh | 03 脚本 | 参数化复制（改 DATASET/ADAPTER/OUTPUT_DIR） |
| evaluate_generated_topk_compact.py（评测口径） | 04 脚本 | 逻辑移植（Top-1/Hit@5 判定） |
| Stage 1 权重 + 共现矩阵 + CLIP 权重 + SFT adapter | 全程 | 直接加载 |
| EK55 原始帧 | 01 数据源 | 直接使用 |

### 3.4 评测口径（必须说清楚）

```text
Stage 1 的角色：识别观测窗口内动作（构造历史）——不是预测目标
Stage 2 的角色：基于 8 历史预测未来 20 个动作
GT：目标段的 verb/noun（DCR valid 标注）

Top-1：candidate_0 的 answer 第一个动作 == GT（精确文本匹配）

与 DCR 论文 T1 的差异：DCR 论文是 τa=1s 预测口径（输入锚点前特征，无泄漏训练）；
本项目是 Stage 1 识别口径构造历史 + Stage 2 生成式预测。两者不可直接对比。
```

### 3.5 诚实预期（基于原项目已知限制）

| 限制 | 来源 | 对结果的影响 |
|------|------|------|
| Stage 1 Top-1 3.56% | 原项目评测 | 8 个历史动作几乎全错 → Stage 2 拿到错题 |
| Stage 2 语义 0.8% | 原项目 876 条 val | 全链路 Top-1 预期 **~1% 量级** |
| 覆盖不足 | 每条视频前 8 段无历史 | valid 8047 条会跳过一批 |
| mask=帧副本 | HOI 缺失 | 双流退化为单流（识别质量上限） |

### 3.6 未实现 / 占位（需要但没有的代码）

1. **严格 τa=1s 无泄漏版**：输入仅锚点前 3.5s 特征、不含段内帧——需要 Stage 1 在锚点前帧上重新评估/训练，**未实现**（当前为识别口径）
2. **test s1/s2 提交官方评测**：只产出预测文件，官方指标需 EPIC 官网提交——**流程未实现**
3. **无样本 ID 的 swift 输出**：04 会严格校验输入/输出行数，且三分片按原始连续区间顺序合并；若未来 ms-swift 内部不再保持行序，需先让其回传样本 ID，不能盲算指标。

---

## 四、确认清单

| 项 | 结论 | 状态 | 实测结果（all_valid_top1_id_v1，2026-08-03） |
|------|------|:--:|------|
| 数据划分用 DCR valid（有标签） | ✅ | ✅ 已确认 | 40 videos / 4,979 actions；可构造 Top-1 目标 1,472（覆盖 31.6%） |
| 特征用 CLIP 重提（不用 TSM） | ✅ | ✅ 已确认 | 1024-dim CLIP ViT-L/14（EgoVideo 权重）；2,201/4,979 动作可提取帧 |
| 历史 = 目标段前 8 段（标注时间戳切分） | ✅ | ✅ 已确认 | history=8、horizon=20、future GT=1；noncausal 排除 1,437 |
| 评测 = candidate_0 首动作 vs GT（Top-1） | ✅ | ✅ 已确认 | Next-Action Top-1 **1.019%**（15/1,472）；verb 13.383% / noun 2.446%；解析率 100% |
| 预期 Top-1 ~1% 量级（诚实边界） | ✅ | ✅ 已验证 | 1.019% 落在 ~1% 预期区间；先验坍塌（open cupboard 350 次）与诊断一致 |
| 口径 = 识别+生成式，与 DCR 论文 T1 不可直接对比 | ✅ | ✅ 已确认 | 维持口径；不得与 DCR τa=1s Top-1 19.2 直接比较 |

> 六项全部确认并执行完毕；结果与预期边界一致。详见 WORKLOG（2026-08-03）与 `runs/all_valid_top1_id_v1/` 产物。
