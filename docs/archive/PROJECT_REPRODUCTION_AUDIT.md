# INSIGHT 项目复现说明与数据流审计

> 项目：Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation（AAAI 2026）  
> 官方仓库：[CorrineQiu/INSIGHT](https://github.com/CorrineQiu/INSIGHT)  
> 审计日期：2026-07-30  
> 文档性质：当前工作区与远端服务器的**实际复现记录**，不是官方仓库说明的替代品

## 1. 文档目的

本文说明以下内容：

- 项目解决什么任务，两阶段模型分别做什么；
- Mac 本地项目、远端服务器代码、数据集、预训练权重和实验结果的位置；
- 本次复现中数据从 EK55 原始帧流向 Stage 1、Stage 2 和指标文件的实际过程；
- 当前 Top-1、Top-5、编辑距离和 mAP 分别代表什么，哪些结果可以比较、哪些只能诊断；
- 当前实现与论文/官方设计的差异，以及这些差异如何沿数据流逐级放大误差。

本文以已经读取的源码、运行命令、文件数量、checkpoint 配置和生成结果为依据。没有经过审计确认的推测会明确标注。

## 2. 项目与运行位置

### 2.1 代码位置

| 环境 | 位置 | 用途 |
|---|---|---|
| Mac 本地工作区 | `/Users/esunlee/clone_workspace/AAAI26-INSIGHT` | 查看和修改源码、维护文档 |
| 远端服务器代码 | `/home/amax/ldy/git/AAAI26-INSIGHT` | 实际训练、推理和评测 |
| 官方上游 | `https://github.com/CorrineQiu/INSIGHT` | 论文作者公开代码 |
| 当前远端提交 | `3322abec9d8fd5452df1b98c1bfdacc2c4c12df1` | 本次审计时的代码基线 |

远端工作树包含为 EK55、特征提取、Stage 1 导出、Stage 2 数据构造和指标评测增加的本地修改。因此复现实验并非“官方仓库零修改直接运行”，报告结果时应同时保存代码提交、未提交补丁、训练参数和实际插件副本。

### 2.2 主要源码目录

| 目录 | 作用 |
|---|---|
| `feature_extraction/` | EK55 标注转换、RGB/HOI 特征提取、Stage 2 数据构造 |
| `HandObject/` | Stage 1 手物语义动作识别、训练、预测导出与 Top-k 诊断 |
| `CognitiveReasoning/` | Stage 2 GRPO 奖励、生成结果解析和编辑距离评测 |
| `evaluation/` | 补充的精确审计、EK55 anticipation 和 future-verb mAP 实验 |
| `scripts/` | 训练、推理、诊断和评测入口 |
| `docs/WORKLOG.md` | 按时间记录的操作日志 |
| `docs/EK55_METRICS_PROTOCOL.md` | EK55 两类指标协议及可报告边界 |

### 2.3 远端运行环境

| 项目 | 实际环境 |
|---|---|
| Conda | `/data/conda_envs/insight_env` |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu118 |
| TorchVision | 0.20.1+cu118 |
| Transformers | 4.49.0 |
| ms-swift | 3.4.1 |
| TRL | 0.17.0 |
| PEFT | 0.14.0 |
| DeepSpeed | 0.19.2 |
| GPU | 4×RTX 3090 24 GiB；最终 Stage 2 使用 physical GPU 0、1、3 |
| 已知硬件问题 | physical GPU 2 曾出现 Xid 79（fallen off the bus）和 Xid 48（L2 DBE） |

## 3. 数据集说明

### 3.1 原始数据

本次复现使用 **EPIC-KITCHENS-55（EK55）**。数据根目录为：

```text
/data/datasets/EPIC-KITCHENS
```

已经审计的原始标注为：

```text
/data/datasets/EPIC-KITCHENS/annotations/EPIC_train_action_labels.csv
```

标注统计：

| 项目 | 数量 |
|---|---:|
| action segments | 28,472 |
| videos | 272 |
| participants | 28 |
| 实际出现的 verb | 119 |
| 实际出现的 noun | 321 |
| 标签输出空间 | 125 verbs / 352 nouns |

“实际出现类别数”小于“输出空间大小”是因为官方 class ID 存在未使用位置，不是代码矛盾。

RGB 数据共约 1,251,563 张 JPEG。当前帧组织形式为：

```text
/data/datasets/EPIC-KITCHENS/<participant>/rgb_frames/frame_<frame_id>.jpg
```

这里没有保留 `<video_id>` 子目录。不同视频可能出现相同 frame ID，因此 participant-level 扁平目录存在文件名碰撞或覆盖风险。该风险需要通过原始 tar 清单或抽样哈希进一步验证。

### 3.2 当前 Stage 1 划分

`feature_extraction/convert_annotations.py` 从 EK55 train CSV 按视频随机生成 90%/10% 划分：

| split | actions | videos |
|---|---:|---:|
| train | 25,982 | 245 |
| val | 2,490 | 27 |

输出位置：

```text
/data/datasets/EPIC-KITCHENS/insight_annotations/train.json
/data/datasets/EPIC-KITCHENS/insight_annotations/val.json
```

脚本调用了 `random.shuffle()`，但原脚本没有固定 seed。因此现存 JSON 必须作为实验资产保存；重新运行脚本不保证得到同一划分。该划分也不是 RULSTM 短期 anticipation 或 EGO-TOPO 长期预测的官方 split。

## 4. 预训练权重与实验权重

预训练权重统一位于：

```text
/data/pretrain_model
```

| 权重/目录 | 实际位置 | 当前作用 | 状态或限制 |
|---|---|---|---|
| EgoVideo ViT-L/14 | `/data/pretrain_model/EgoVideo/checkpoints/large_best.pt` | 将 RGB 或 mask 图像编码为 1024 维特征 | 已使用；通过自编写参数映射载入 HF `CLIPVisionModel`，移除了 temporal 参数并将位置编码插值到 224×224 |
| EgoVideo base | `/data/pretrain_model/EgoVideo/checkpoints/base_best.pt` | 备用 768 维模型 | 未用于当前主链路 |
| Hand Object Detector | `/data/pretrain_model/Hand_Object_Detector/` | 检测手部 bbox，为 SAM2 提供提示 | 当前使用 `faster_rcnn_1_8_89999.pth`；缺少作者期望的 ego 优化版 `faster_rcnn_1_8_132028.pth` |
| SAM2 | `/data/pretrain_model/SAM2/sam2.1_hiera_large.pt` | 根据手部 bbox 生成 mask | 已部署；当前每个 action 只对第一张采样帧计算一次 mask |
| Qwen2.5-VL-7B | `/data/pretrain_model/Qwen2.5-VL-7B-Instruct` | Stage 2 图像、历史动作和未来动作生成 | 已使用 |
| MiniLM | `/data/pretrain_model/all-MiniLM-L6-v2` | intention 文本相似度奖励 | 已使用；路径在安装到 ms-swift 的插件中硬编码修正 |
| Stage 1 重建权重 | `/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth` | 输出 verb/noun 及联合 action 候选 | 89 tensors，约 24.4M 参数，可正常加载 |
| Stage 2 LoRA | `/data/datasets/EPIC-KITCHENS/stage2_stage1_history_full_3gpu_v3_output/v0-20260729-235831/checkpoint-1773` | 当前 Stage 2 推理 | 最后 checkpoint；训练记录中的 best checkpoint 为 1750 |

原 Stage 1 checkpoint 已损坏，缺少 PyTorch zip central directory，不能恢复；因此当前主链路使用重新训练得到的权重。

## 5. 实际数据流

```text
EK55 CSV + JPEG
        │
        ├─ convert_annotations.py
        │       └─ 随机 train/val JSON
        │
        ├─ extract_features.py
        │       ├─ EgoVideo→HF CLIP：frame feature [1024]
        │       └─ Hand detector→SAM2→EgoVideo：mask feature [1024]
        │
        ├─ Stage 1：HandObject action recognition
        │       └─ 每个 action 的 verb/noun Top-5 与语义先验重排序
        │
        ├─ export_stage1_predictions.py
        │       └─ 去重后的 Stage 1 action predictions JSONL
        │
        ├─ build_stage2_from_stage1.py
        │       └─ 8 个预测历史动作 + 1 张图 → GT 未来 20 动作
        │
        ├─ Stage 2：Qwen2.5-VL-7B + LoRA + GRPO
        │       └─ <think><intention><answer>未来 20 动作</answer>
        │
        └─ 生成结果解析
                ├─ next-action Hit@1 / sampled Hit@5
                ├─ Action/Verb/Noun normalized edit distance
                └─ 输出格式合规率
```

### 5.1 特征提取

当前 `feature_extraction/extract_features.py` 的实际逻辑是：

1. 每个已标注 action segment 最多均匀采样 4 帧；
2. 将 EgoVideo `large_best.pt` 的视觉参数映射到 HF `CLIPVisionModel`；
3. 去掉 checkpoint 中的 temporal 参数，将每帧独立编码后求均值，得到一个 `[1024]` frame feature；
4. Hand Object Detector 只在第一张采样帧上检测手部；
5. SAM2 根据最高置信度手部 bbox 生成一个 mask；
6. 把同一个 mask 重复用于该 action 的所有采样帧，再编码并求均值；
7. 若没有可靠 mask，当前数据中大量 mask 文件实际来自 frame feature 的复制或历史临时替代结果。

输出目录：

```text
/data/datasets/EPIC-KITCHENS/features/frame_features/{train,val}/
/data/datasets/EPIC-KITCHENS/features/mask_features/{train,val}/
```

逐文件审计结果：

| split | 标注 | frame | mask | 成对可用 | frame 与 mask 完全相同 | 不同 |
|---|---:|---:|---:|---:|---:|---:|
| train | 25,982 | 14,217 | 14,179 | 14,179（54.57%） | 13,891（97.97%） | 288（2.03%） |
| val | 2,490 | 1,263 | 1,263 | 1,263（50.72%） | 1,263（100%） | 0 |

“不同的 288 对”只能证明 mask tensor 不是 frame tensor 的直接副本，不能证明 288 个 mask 都是正确的手物交互区域。验证集完全没有独立 HOI mask 信号。

### 5.2 Stage 1：手物语义动作识别

当前 Stage 1 配置：

| 参数 | 值 |
|---|---:|
| input dimension | 1024 |
| frame/mask projection | 1024→2048→256，各一支 |
| Transformer | 4 layers，8 heads，FFN 2048 |
| verb classes | 125 |
| noun classes | 352 |
| loss | verb/noun cross entropy，label smoothing 0.1 |
| optimizer | AdamW，lr `8e-5`，weight decay `2e-5` |
| batch size | 8 |
| checkpoint | 重新训练 6 epochs 后保存的 best model |

意图设计是使用 8-action window 建模上下文，但当前 `custom_collate_fn` 会把窗口内 action 展平；进入 `ActionRecognitionModel.forward()` 后，每个 action 的 `combined` 只有一个 token，再与 CLS 拼成长度 2 的序列。因此当前 Transformer 主要对单动作的 frame/mask 融合建模，没有真正学习 8 步动作时间关系。

语义共现矩阵位于：

```text
/data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt
```

它包含 125×352 verb-noun 先验，用于导出时对联合概率重排序。由于训练窗口中的 action 会重复进入统计，`raw_cooccurrence` 总和 105,588，不等于 25,982 个唯一训练 action。

Stage 1 去重预测输出：

```text
/data/datasets/EPIC-KITCHENS/stage1_predictions/stage1_train_predictions.jsonl
/data/datasets/EPIC-KITCHENS/stage1_predictions/stage1_val_predictions.jsonl
```

| split | 输出 action | 视频 | 预测 verb 种类 | 预测 noun 种类 |
|---|---:|---:|---:|---:|
| train | 14,179 | 151 | 40 | 116 |
| val | 1,263 | 18 | 17 | 57 |

预测类别明显集中于高频动作，说明 Stage 1 除覆盖率问题外还存在严重的类别塌缩/长尾偏置。

### 5.3 Stage 1 → Stage 2 数据构造

`feature_extraction/build_stage2_from_stage1.py` 要求连续 8 个 observed action 都具有 Stage 1 预测，然后取随后 20 个 GT action 作为答案。

输出位置：

```text
/data/datasets/EPIC-KITCHENS/stage2_stage1_history/stage2_train_stage1_history.jsonl
/data/datasets/EPIC-KITCHENS/stage2_stage1_history/stage2_val_stage1_history.jsonl
```

| split | Stage 2 样本 | 可覆盖视频 | history | future | 因历史不完整丢弃 |
|---|---:|---:|---:|---:|---:|
| train | 10,748 | 96 | 8 | 20 | 9,603 |
| val | 876 | 11 | 8 | 20 | 1,050 |

这一步不是“把空历史换成 Stage 1 数据”，而是严格把 Stage 1 的预测结果作为 Stage 2 的历史文本输入。未来 20 步仍来自 GT，仅用于训练与评测目标。

由于必须连续 8 步都有特征和预测，Stage 1 的单步缺失会在这里放大为整条 Stage 2 样本缺失。最终 val 只覆盖原 27 个随机验证视频中的 11 个。

### 5.4 Stage 2：认知推理与未来 20 动作生成

实际训练配置：

| 参数 | 值 |
|---|---|
| base model | Qwen2.5-VL-7B-Instruct |
| train type | LoRA |
| LoRA | `r=8`，`alpha=32`，dropout `0.05` |
| LoRA targets | 语言模型 q/k/v/o/up/down/gate projections |
| RL method | GRPO |
| generations per prompt | 2 |
| max input / completion | 1000 / 256 tokens |
| reward funcs | action-intention、action continuity、soft-overlong |
| reward weights | 0.85 / 0.05 / 0.1 |
| learning rate | `3e-6` |
| effective training | 1 epoch，1773 steps，约 6h19m |
| distributed | 3 GPUs + DeepSpeed ZeRO-2 |

训练命令没有显式传入独立 `val_dataset`；`val_dataset=[]`、`split_dataset_ratio=0.01`。训练日志里的约 107 个 eval 样本来自训练 JSONL 的内部 1% 切分，不是独立的 876 条 Stage 2 val 数据。因此训练时的 `eval_reward` 不能当作最终验证集指标。

另一个复现风险是：实际训练调用的是安装到

```text
/data/conda_envs/insight_env/lib/python3.10/site-packages/swift/plugin/
```

下的插件，而不是直接调用仓库内副本。安装版修改了 MiniLM 路径、模块导入和合法 action 正则。若只复制当前仓库而不保存安装版插件，无法严格复现实验。

## 6. 当前指标及其边界

### 6.1 Stage 1 covered-subset 识别指标

在具备成对特征的 1,263 个随机 val action segment 上：

| 指标 | Top-1 | Top-5 |
|---|---:|---:|
| Verb | 18.6857% | 59.6200% |
| Noun | 8.0760% | 22.4070% |
| Action，raw joint probability | 3.3254% | 7.2051% |
| Action，加入共现先验 | **3.5629%** | **11.1639%** |

这些数值是真实计算结果，但任务是“读取 action segment 内部图像后识别当前 action”。它不是在目标动作开始前 1 秒做 anticipation，因此不能直接填入 DCR/CPM 等短期 Action Top-1/Top-5 表格。

### 6.2 Stage 1 → Stage 2 生成指标

使用 checkpoint-1773 在 876 条独立 Stage 2 val 子集上生成 5 份结果：

| 指标 | 结果 |
|---|---:|
| next-action Hit@1 | 0.1142%（1/876） |
| sampled Hit@5 | 0.1142%（1/876） |
| candidate-1 Action / Verb / Noun ED | 0.9995 / 0.9920 / 0.9870 |
| oracle-min-over-5 Action / Verb / Noun ED | 0.9985 / 0.9804 / 0.9441 |
| exactly 20 valid actions | 89.1553% |

五个候选文件没有 `logprobs`，只是一次 temperature=0 生成和四次不同 seed 的随机采样。因此这里应写 **Hit@1 / sampled Hit@5**，不能写成标准 ranked Top-1/Top-5。

抽查结果出现将 `Add onion` 重复 20 次的模式坍塌。它满足标签和长度要求，却没有未来动作语义。这与训练末期 action-continuity reward 接近 0、格式合规率很高的现象一致。

### 6.3 P=25/50/75 mAP 补充实验

当前 future-verb mAP 实验使用 EGO-TOPO 标签协议，但只复用有限的 Stage 1 RGB action features：train 776、val 96，而不是协议过滤后的完整 train 1,251、val 172。

| P | All | Freq | Rare | 实际 val 样本 |
|---:|---:|---:|---:|---:|
| 25 | 60.16 | 65.09 | 56.78 | 31 |
| 50 | 46.17 | 54.09 | 40.76 | 34 |
| 75 | 32.40 | 42.07 | 25.78 | 31 |
| Average | 46.24 | 53.75 | 41.11 | 96 |

该结果必须标为 **official-label/protocol, feature-limited baseline**。它绕过了完整 Stage 1→Stage 2 生成链，不能作为论文 INSIGHT 主结果。

## 7. 三个已知问题是否导致 Top-1/Top-5 下降

### 7.1 “数据集是 EK55”不是缺陷

不是直接原因。论文和官方项目本身支持 EK55，只要比较对象、划分、观察时间和类别定义相同，使用 EK55 是正确实验设置。

EK55 的训练规模比 Ego4D 小，确实可能让大模型更难学习，但这属于数据集固有难度，不能解释“同在 EK55 上，当前复现显著低于论文或基线”。真正的问题是当前使用了随机 split、约 50% 特征覆盖和非官方任务定义，导致结果与标准 EK55 Top-k 协议不一致。

### 7.2 缺少 ego 优化版第一人称权重是重要原因，但不是唯一原因

如果这里指缺少 Hand Object Detector 的 `faster_rcnn_1_8_132028.pth`（handobj_100K+ego），它会直接降低第一人称视角中的手部检测率：

```text
检测权重不匹配
→ 手 bbox 检出少或不准
→ SAM2 没有可靠提示
→ HOI mask 缺失/错误
→ mask feature 被 RGB feature 替代
→ Stage 1 失去手物交互信息
→ verb/noun 识别下降
→ Stage 2 的 8 步历史一起变差
```

审计结果中 train 97.97%、val 100% 的 mask 与 frame 完全相同，与这条因果链一致。因此该权重缺失是**高影响因素**。

但即使补齐权重，也不会自动解决帧缺失、只检测第一帧、重复同一 mask、Stage 1 时间窗口被展平、Stage 2 奖励漏洞和指标协议错误，所以它不是唯一根因。

### 7.3 官方仓库没有组合特征提取函数，本身不降分；替代实现的偏差会降分

“缺少函数”是工程缺口，不是一个能直接改变准确率的数学因素。真正影响准确率的是：为了填补缺口，本次自行编写的 `extract_features.py` 与论文预期管线存在显著差异：

- 只采 4 帧；
- 只在第一帧检测手；
- 一个 mask 重复应用到所有采样帧；
- EgoVideo temporal 参数被跳过，逐帧 CLIP 后直接平均；
- 原模型预期的 1408 维输入被改为 1024 维；
- 检测失败时大量使用复制的 RGB 特征作为 mask 特征。

所以更准确的说法是：**官方缺少端到端特征提取实现迫使本次使用近似实现，而近似实现造成的信息损失是 Top-k 下降的主要原因之一。**

## 8. 按数据流排序的完整降分原因

### 第一步：原始数据和划分

1. 当前 train/val 是无固定 seed 的随机视频划分，不是目标 Top-k 表格的官方 split。
2. 当前 RGB 帧按 participant 扁平保存，可能发生跨视频同名 frame 冲突。
3. 部分原始帧未下载，直接造成约一半 action 没有可用特征。

影响：训练数据变少，验证集不标准，而且实验不容易重复。

### 第二步：特征提取

1. 成对特征覆盖率仅 train 54.57%、val 50.72%。
2. train 只有 2.03% mask 与 frame 不同，val 为 0%。
3. 缺少 ego 优化版 detector 权重使第一人称手部检测更困难。
4. 当前 EgoVideo 被当作去掉 temporal 模块的逐帧 CLIP 使用。
5. 第一帧 mask 被重复到全部采样帧，无法描述动作过程中手物关系的变化。

影响：Stage 1 看不到可靠的动态信息和 HOI 信息。这是第一处主要瓶颈。

### 第三步：Stage 1 训练和推理

1. 设计上的 8-step window 被 collate 展平，Transformer 实际处理单 action。
2. 共现矩阵受重复窗口统计影响，高频 verb/noun 先验更强。
3. val 只预测出 17 种 verb 和 57 种 noun，类别塌缩明显。
4. 当前输入来自 action segment 内部，解决的是 recognition，不是标准的提前 1 秒 anticipation。

影响：Stage 1 单步动作预测低、偏向高频类别，且 Top-k 指标与论文表格任务不一致。这是第二处主要瓶颈。

### 第四步：Stage 1 历史传给 Stage 2

Stage 2 使用连续 8 个 Stage 1 Top-1 动作文本。只要其中若干步错误，历史叙事就会偏离真实任务；只要任意一步缺少特征，整个样本就会被删除。

结果是 val 从 27 个视频缩小到 11 个视频。Stage 1 的覆盖率和错误在这里产生组合放大效应。

### 第五步：Stage 2 训练

1. 当前直接使用 GRPO，没有先通过强监督 SFT 学会 EK55 动作词表和未来序列。
2. LoRA 只训练语言层，没有适配视觉模块。
3. 每个 prompt 只有 2 个 generation，组内探索有限。
4. 内容连续性奖励权重仅 0.05，末期实际接近 0。
5. 长度/格式门控使“重复 20 个合法动作”也能得到非零奖励。
6. 没有显式独立 val，checkpoint 选择依据来自训练集内部 1% 切分。
7. 最终推理使用 checkpoint-1773，而训练记录中的 best 是 checkpoint-1750。

影响：模型学会了固定输出格式，却没有学会未来动作语义；出现重复 `Add onion` 的 reward hacking。这是第三处、也是 Stage 2 最直接的瓶颈。

### 第六步：指标计算

1. 五次随机生成不是带概率排序的 Top-5。
2. Stage 1 covered-subset recognition 不是提前 1 秒 anticipation。
3. 876 条 Stage 2 子集也不是标准短期 Top-k split。

影响：即使数字被正确计算，也不能与截图中 DCR/CPM 的标准 Top-1/Top-5 横向比较。这里既有模型性能问题，也有指标定义问题。

## 9. 原因重要性排序

| 优先级 | 问题 | 对当前结果的影响 |
|:--:|---|---|
| 极高 | mask 几乎全部等于 frame，val 无独立 HOI | Stage 1 的核心手物分支失效 |
| 极高 | Stage 2 无 SFT且 reward 可被重复动作利用 | 直接导致生成语义坍塌 |
| 极高 | Stage 1 时间窗口被展平 | 失去连续动作上下文 |
| 高 | 成对特征覆盖率约 50% | 训练样本减少，并在 8-step history 阶段进一步放大 |
| 高 | 缺少 handobj_100K+ego 权重 | 降低 ego 手部检测和 mask 质量 |
| 高 | 当前 Top-k 协议与目标表格不一致 | 数字无法合法横向比较 |
| 中高 | EgoVideo temporal 信息被移除并改成逐帧平均 | 动作动态信息损失 |
| 中 | 随机且不可重新生成的 train/val split | 可复现性和公平比较受损 |
| 中 | 使用最后 checkpoint 而非 best checkpoint | 可能进一步降低 Stage 2，但不是 0.114% 的根本解释 |
| 非缺陷 | 使用 EK55 | 论文支持的数据集，数据集名称本身不导致异常下降 |

## 10. 后续修复顺序

1. 固化数据：保存现有 split，并验证 participant-level JPEG 是否发生跨视频覆盖。
2. 重建特征：补齐 ego detector 权重；每帧检测/分割；检测失败时记录缺失，不再静默复制 frame 为 mask。
3. 恢复时序：保留 EgoVideo temporal 编码，并修复 Stage 1 collate，使 Transformer 真正输入 8-step sequence。
4. 建立正确基线：先在官方 split 和目标动作前 1 秒输入上计算标准 Action Top-1/Top-5。
5. Stage 2 先 SFT：使用 8-step predicted history → 20-step GT 序列做监督学习。
6. 再做 GRPO：提高逐位置语义奖励，加入重复惩罚、合法词表约束和序列多样性奖励。
7. 显式验证：传入独立 val JSONL，按语义指标选择 checkpoint。
8. 正确计算 Top-5：保存分类 logits、token log-prob 或受约束 beam score；随机采样只能报告 sampled Hit@5。

## 11. 主要结果文件

| 结果 | 远端位置 |
|---|---|
| Stage 1 checkpoint | `/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth` |
| Stage 1 train/val predictions | `/data/datasets/EPIC-KITCHENS/stage1_predictions/` |
| Stage 1 Top-k metrics | `/data/datasets/EPIC-KITCHENS/stage1_predictions/stage1_topk_metrics.json` |
| Stage 2 train/val JSONL | `/data/datasets/EPIC-KITCHENS/stage2_stage1_history/` |
| Stage 2 final adapter | `/data/datasets/EPIC-KITCHENS/stage2_stage1_history_full_3gpu_v3_output/v0-20260729-235831/checkpoint-1773/` |
| Stage 2 five sampled candidates | `/data/datasets/EPIC-KITCHENS/stage2_generated_topk_checkpoint1773/` |
| Stage 2 metrics | `/data/datasets/EPIC-KITCHENS/stage2_generated_topk_checkpoint1773/metrics_all.json` |
| 100-sample semantic diagnostic | `/data/datasets/EPIC-KITCHENS/stage2_diagnostic_100_checkpoint1773/` |
| feature-limited mAP | `/data/datasets/EPIC-KITCHENS/ek55_future_verb_map_output/metrics.json` |
| 汇总归档 | `/data/datasets/EPIC-KITCHENS/final_metrics_20260730/` |

## 12. 当前复现的准确结论

本次工作已经跑通：

```text
EK55 JPEG/标注
→ 自建 RGB/HOI 特征
→ Stage 1 重建与预测导出
→ Stage 1 预测历史构造 Stage 2 数据
→ Qwen2.5-VL LoRA-GRPO
→ 未来 20 动作生成与诊断评测
```

但“链路跑通”不等于“严格复现论文性能”。当前低分不是由 EK55 这个数据集名称造成的，而是由特征提取近似、HOI 分支失效、特征覆盖不足、Stage 1 时序失效、Stage 2 奖励漏洞以及指标协议不一致共同造成，并且这些问题会沿 Stage 1→Stage 2 逐层放大。

因此，当前结果适合作为**工程链路复现与失败诊断基线**；在完成特征、时序、SFT/reward 和官方评测协议修复前，不应将其作为论文 INSIGHT 的严格性能复现结果。
