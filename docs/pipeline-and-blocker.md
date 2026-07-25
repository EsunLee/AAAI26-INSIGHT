# INSIGHT 完整流程 & 阻塞点说明

## 从原始数据到最终模型，共需要 5 个步骤

```
[步骤1] 原始视频帧        → 工具: 直接用 (数据集下载)
[步骤2] 特征提取           → 工具: HandObjDet + SAM2 + EgoVideo
[步骤3] Stage 1 训练       → 工具: INSIGHT/HandObject
[步骤4] Stage 2 训练       → 工具: INSIGHT/CognitiveReasoning
[步骤5] 评测              → 工具: INSIGHT/CognitiveReasoning/cal_ED.py
```

---

## 步骤 1：下载原始视频帧

### 要做什么
从 EPIC-Kitchens-55 官方下载 RGB 帧（~220 GB，约 1150 万张 JPEG）。

### 我们的状态
✅ **已完成。** 432/434 个 tar 文件已下载，解压后约 125 万帧，228 GB，在 `/data/datasets/EPIC-KITCHENS/`。

---

## 步骤 2：特征提取（⚠️ 卡在这里）

### 要做什么
把每张 JPEG 帧变成 Stage 1 训练代码能读的 `.pt` 特征文件。这需要依次运行三个开源工具：

<div align="center">

```
JPEG 帧 (1.25M 张)
    │
    ├──[1] Hand Object Detector ──→ 输出: 手和物体的边界框 (bbox)
    │
    ├──[2] SAM2 ──→ 输入: 原始帧 + bbox ──→ 输出: 手物交互区域的精细 mask
    │
    ├──[3] EgoVideo-V ──→ 输入: 原始帧 + mask ──→ 输出: 1408 维特征向量
    │
    └──▶ 最终: {clip_uid}_{action_idx}.pt  特征文件
             按 train/val/test 分目录存放
             每个 .pt 文件是一个 1408 维的 float32 tensor
```

</div>

三个工具的开源代码和预训练权重**我们都有**。缺失的是**串联这三个工具的调度脚本**——也就是 Step 1 怎么接到 Step 2、Step 2 的输出怎么给 Step 3、最终怎么组织成 Stage 1 需要的目录结构。

### 我们的状态
❌ **无法进行。** INSIGHT 仓库中没有提供这个调度脚本。README 只有一句话："依次运行 Hand Object Detector、SAM2、EgoVideo，参考各自的官方教程"，但没有给出任何集成代码。

---

## 步骤 3：Stage 1 训练（Hand-Object Semantic Action Recognition）

### 要做什么
用步骤 2 产出的 `.pt` 特征文件 + 动作标注（verb/noun labels），训练一个 Transformer 模型来识别手物交互动作。

### 我们的状态
⏸️ **代码完整（HandObject/ 目录），被步骤 2 阻塞。** 训练入口在 `HandObject/main.py`，需要传入三类路径：

| 路径 | 内容 | 我们有吗 |
|------|------|:--:|
| `frame_features_dir` | EgoVideo 提取的全局帧特征 (.pt) | ❌ 等步骤 2 |
| `mask_features_dir` | EgoVideo 提取的 HOI 区域特征 (.pt) | ❌ 等步骤 2 |
| `annotation_dir` | train.json / val.json（动作标注） | ❌ 需从 EK55 CSV 转换 |

---

## 步骤 4：Stage 2 训练（Cognitive Reasoning via GRPO）

### 要做什么
用步骤 3 产出的动作识别结果 + Qwen2.5-VL-7B 大模型，通过 GRPO 强化学习进行 think→reason→answer 的认知推理训练。

### 我们的状态
⏸️ **代码完整（CognitiveReasoning/ 目录），被步骤 3 阻塞。** 训练脚本是 `run_external_reward_func_7B_qwen_intention.sh`，使用 ms-swift 框架，6 张 GPU 跑 GRPO。

---

## 步骤 5：评测

### 我们的状态
⏸️ **代码完整（cal_ED.py），被步骤 4 阻塞。** 用 Damerau-Levenshtein 编辑距离计算预测动作与真实动作的相似度。

---

## 总结：一张图看清

```
✅步骤1  [已完成] 数据集下载 (125万帧 JPEG)
         │
❌步骤2  [卡住]   特征提取
         │         ├── HandObjDet ✅有工具 ✅有权重
         │         ├── SAM2       ✅有工具 ✅有权重  
         │         ├── EgoVideo   ✅有工具 ✅有权重
         │         └── 调度脚本  ❌仓库中没有
         │
⏸️步骤3  [阻塞]   Stage 1 训练 (代码完整)
         │
⏸️步骤4  [阻塞]   Stage 2 训练 (代码完整)
         │
⏸️步骤5  [阻塞]   评测 (代码完整)
```

> **关键**：步骤 3/4/5 的代码都在仓库里，但它们的输入必须由步骤 2 产出。**步骤 2 的调度代码不在仓库中。**

---

## 出口

1. **联系作者** 获取步骤 2 的调度脚本（最直接，几小时走通）
2. **自行编写** 调度脚本（三个工具都是开源的，可行但需 5-10 天额外开发）
