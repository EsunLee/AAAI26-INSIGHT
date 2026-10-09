# INSIGHT 技术详解（复现者视角）

> 本文以复现者口吻，从基础技术到论文方法，系统讲解 INSIGHT（AAAI 2026）涉及的全部技术细节：技术名称、原理、在本项目中的角色，以及我们复现时踩过的坑。
> 论文：*Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation*（arXiv:2508.01742）

---

## 0. 项目全景

### 0.1 任务定义

**Egocentric Long-Term Action Anticipation（第一人称长时动作预测）**：给出一段第一人称（胸戴相机）视频中已经发生的动作序列，预测未来一段时间内将发生的一系列动作。

- 输入：观察窗口内已完成的动作（如 8 个动作片段）+ 当前画面
- 输出：未来 20 个动作（每个动作 = 动词 + 名词，如 "take knife"、"open fridge"）
- 数据集：EPIC-Kitchens-55（EK55）——厨房第一人称视频，125 个动词 × 352 个名词，28,472 条动作标注

### 0.2 核心设计思想

论文的关键洞察：**单纯"看"不够，要"想"**。早期方法（EGO-TOPO、PALM、AntGPT 等）直接让模型从视频特征映射到未来动作，没有显式的认知过程。INSIGHT 把推理拆成三个阶段：

```text
Perception（感知）→ Intention Inference（意图推断）→ Action Prediction（动作预测）
     think           →        reason          →        answer
```

对应两阶段架构：

| 阶段 | 角色 | 模型 | 输入 → 输出 |
|------|------|------|------------|
| Stage 1 | 感知：把视频翻译成动作文本 | 手物语义动作识别（Transformer 双流） | 帧特征+HOI特征 → 每片段一个 verb+noun |
| Stage 2 | 认知：从历史动作推理未来 | Qwen2.5-VL-7B（多模态大模型） | 8 个历史动作文本+当前帧 → 20 个未来动作 |

**为什么分两阶段**：把"理解当前发生了什么"（感知）和"推断接下来发生什么"（认知）解耦。感知错误不会直接污染认知模型的推理能力；认知模型可以专注于语义逻辑（做饭的流程、意图驱动）。

---

## 1. 基础技术（项目的地基）

### 1.1 自注意力与 Transformer

**Transformer**（Vaswani et al., 2017）是几乎所有现代模型的骨架。

**自注意力（Self-Attention）**：让序列中每个 token 与所有其他 token 交互，计算相关性权重：

```text
Q = X·W_Q,  K = X·W_K,  V = X·W_V        （Query/Key/Value 线性投影）
Attention(Q,K,V) = softmax(QKᵀ/√d) · V
```

- `QKᵀ/√d`：query 与 key 的点积衡量"我该关注谁"，除以 √d 防止数值过大
- softmax 归一化成权重，加权求和 V 得到输出
- 与 RNN/LSTM 的差异：**无顺序依赖**，可并行；靠**位置编码**注入顺序信息；能建模长距离依赖

**多头注意力（Multi-Head）**：把 d 维分成 h 个头分别算注意力，每个头学不同关注模式（语法、语义、局部、全局），再拼接。

**Transformer Encoder 块**：多头注意力 + 前馈网络（FFN）+ 残差连接 + LayerNorm：

```text
x ← LayerNorm(x + MultiHeadAttention(x))   （Pre-Norm 结构）
x ← LayerNorm(x + FFN(x))
```

**在本项目中的角色**：
- Stage 1 的分类器是 4 层 Transformer Encoder（8 头），把动作片段的特征序列编码成全局表示
- Stage 2 的 Qwen2.5-VL 底层就是 Transformer Decoder（自回归生成）

### 1.2 对比学习与 CLIP

**CLIP**（Radford et al., 2021，OpenAI）是"文字-图像"双塔对比学习模型。

**对比学习（Contrastive Learning）**：不预测标签，而是学会"哪些样本应该靠近、哪些应该远离"。CLIP 的训练目标：从 4 亿个（图片，描述文字）对中，让**匹配对的图像编码和文本编码相似度高，不匹配对低**——InfoNCE 损失（温度缩放后的交叉熵）：

```text
L = -log[ exp(sim(I, T₊)/τ) / Σⱼ exp(sim(I, Tⱼ)/τ) ]
```

**模型结构**：图像塔（ViT 或 ResNet）+ 文本塔（Transformer），在对比空间中把图像和文本嵌入到同一向量空间。**零样本能力**：任意文本类名可以直接和图像比相似度分类。

**ViT-L/14**（Vision Transformer Large，patch 14）：把图像切成 14×14 像素的 patch，展平成 token 序列（224×224 图像 → 16×16=256 个 patch + CLS token），过 24 层 Transformer，输出 1024 维特征。

**在本项目中的角色**：EgoVideo 的 backbone 就是 CLIP 风格结构，我们用其视觉编码器把视频帧编码成 1024 维特征（每动作 4 帧平均）。

### 1.3 目标检测：Faster R-CNN

**Faster R-CNN**（Ren et al., 2015）是经典两阶段检测器：

1. **Backbone**（ResNet-101）：提取图像特征图
2. **RPN（Region Proposal Network）**：滑动窗口锚框 → 判断前景/背景 + 回归框位置，产出候选框
3. **ROI Pooling + 分类头**：对每个候选框池化出固定尺寸特征 → 分类（是什么）+ 回归（精修框）

**在本项目中的角色**：handobj_100K 检测器基于 Faster R-CNN（ResNet-101），同时检测**手**和**被握持物体**的边界框——这是 Stage 1 提取 HOI 特征的入口。

### 1.4 图像分割：SAM2

**SAM2（Segment Anything 2）**（Meta, 2024）：提示驱动的分割大模型。输入图像/视频 + 提示（点、框、掩码），输出目标的分割掩码。训练数据 SA-1B（11M 图像、10 亿掩码）。

**提示分割原理**：图像编码器（Hiera）提取特征 → **提示编码器**（框/点编码成 token）→ **掩码解码器**（Transformer 解码）融合图像特征与提示 → 生成掩码。视频版带记忆机制（streaming memory）跨帧传播掩码。

**在本项目中的角色**：拿到手部检测框后，用 SAM2 做**分割细化**——把 bbox 变成像素级手部掩码，用于裁剪手物交互区域提取 HOI 特征。

### 1.5 大语言模型与自回归生成

**LLM 的自回归原理**：预测下一个 token：

```text
P(x₁x₂…xₙ) = Π P(xₜ | x₁…xₜ₋₁)
```

训练损失：交叉熵，只惩罚被预测位置的 token：

```text
L = -Σ log P(xₜ | x₁…xₜ₋₁)
```

**Decoder-only 架构**：每层因果注意力（只能看左边，掩码矩阵实现），训练和推理一致（teacher forcing + 生成）。

### 1.6 视觉语言模型（VLM）

**VLM** 把视觉和语言统一：图像 → 视觉编码器 → 视觉 token → 与文本 token 一起输入 LLM。

**Qwen2.5-VL** 的具体设计：
- **视觉编码器**：Qwen2-VL 的 ViT（约 6.75 亿参数），输出 1280 维视觉特征
- **位置感知适配器（Position-Aware Adapter）**：两行 1×1 卷积，将 ViT 特征压缩成 LLM 能用的维度（把 2D 特征展平成 1D token 序列），同时保留绝对位置信息
- **动态分辨率**：支持任意分辨率输入，按需切分（动态 patch 合并），分辨率越高视觉 token 越多
- **语言骨干**：Qwen2.5（7B，decoder-only），词汇表 15 万，上下文 32K token

图像进入 LLM 时附带特殊 token（`<|vision_start|>...<|vision_end|>`），跨模态注意力（Qwen2.5-VL 用 QKV 统一投影，类似 Qwen2-VL 的 cross-attention 融合视觉和语言分支）。

**在本项目中的角色**：Stage 2 的推理引擎。输入 = 历史动作文本 + 当前帧图像（图像 token），输出 = think→intention→answer 结构化的未来动作。

---

## 2. Stage 1：手物语义动作识别（感知）

### 2.1 输入管线

每个动作片段（segment）的完整处理链：

```text
动作片段（start_frame ~ stop_frame）
   │ 采样 4 帧
   ├─→ 帧特征：CLIP/EgoVideo 编码 → frame feature [1024]
   └─→ HOI 特征：HandObjDet 检手 → SAM2 分割 → 掩码图像编码 → mask feature [1024]
```

### 2.2 手物交互检测（HOI）

**HOI（Hand-Object Interaction）检测**：检测手、物体以及它们之间的交互关系。

**handobj_100K**（Shan et al.）：一个手物交互数据集（约 100K 张图像），训练出的检测器同时输出手框和物体框。

**handobj_100K+ego**：在 ego（第一人称）视角图像上微调过的版本——**对第一人称视频关键**：胸戴相机拍到的画面里，手是画面主体；第三人称训练的手检测器在 ego 画面上大量漏检（我们复现中实测漏检率 99.4%！）。

**原理补充**：第一人称画面与第三人称画面的统计差异巨大（视角、手部尺度、遮挡、运动模糊），所以需要 ego 数据微调。这就是为什么作者开源与否如此关键。

### 2.3 双流 Transformer 架构

Stage 1 的模型（HandObject/model.py 的 ActionRecognitionModel）：

```text
frame features [1024] ──→ frame_stage1（Transformer 编码）──┐
                                                           ├─→ 融合 → verb 头（125类）
mask features  [1024] ──→ mask_stage1（Transformer 编码）──┘          noun 头（352类）
```

- **双流（Two-Stream）**：一条流看整帧（场景上下文），一条流只看手物交互区域（局部细节）——互补
- 融合后分别接 verb 分类头和 noun 分类头（多任务学习，共享编码器）

**为什么双流有效**：动作的判定依赖"手在做什么"（刀+手 → 切）和"环境是什么"（灶台 → 做饭），两路信息分开编码再融合，比单流信息利用率高。

### 2.4 verb-noun 共现先验（论文公式 5）

分类器输出独立的两组分数（verb logits + noun logits），但**动作 = 动词×名词联合**。论文用数据集统计的**共现矩阵**（co-occurrence matrix，690KB）做语义修正：

```text
P(action) = P(verb) · P(noun) · P(verb,noun 共现先验)
```

即：即使 verb 和 noun 各自的分数都不错，如果这对组合在数据集里极少出现（如 "cut fridge"），就惩罚它；反之高频组合（如 "take knife"）加权。**这利用了动作语义的先验分布，本质是贝叶斯修正**——我们的复现中它把 Action Top-5 从 91/1263 提升到 141/1263。

### 2.5 Stage 1 的定位与局限

- **定位**：识别"当前片段正在发生什么"（对预测未来而言是"历史"）——是 Stage 2 的**出题人**
- **局限（复现实测）**：动作联合分类 44,000 类（125×352），联合概率是两分数乘积，任一个错则全错 → Top-1 仅 3.56%

---

## 3. Stage 2：认知推理（Qwen2.5-VL）

### 3.1 输入构造

```text
用户消息：
  历史动作列表：[open fridge, take milk:soy, pour milk:glass, ...]（8 个，来自 Stage 1）
  当前帧图像：<image>（当前时刻的画面）
  指令：请思考并预测未来 20 个动作

模型输出（结构化）：
  <think>  我看到……正在…… </think>
  <intention> 用户的意图是准备早餐…… </intention>
  <answer> take knife; cut bread; ...（20 个动作）</answer>
```

**think → intention → answer 的设计动机**：强制模型先描述感知（think）、再显式推理意图（intention）、最后给出预测（answer）。意图是中间监督信号——不直接跳到答案，逼模型走"推理路径"而非"模式匹配"，减少幻觉。这是论文声称的核心创新之一。

### 3.2 SFT（监督微调）原理

**SFT = 监督模仿学习（行为克隆）**。用标注好的（输入 → 标准答案）对微调模型：

- 数据：10,748 条（8 历史动作 + 帧 → 标准 think/intention/answer）
- 损失：标准自回归交叉熵——**最大化模型复述标注答案的逐 token 概率**
- 效果：模型学会"输出长什么样"（格式、结构、语言习惯）

**本质**：把大模型从"通用助手"校准成"领域专家"。它学到的是**答案的分布**，不是因果理解——这就是为什么它能 200 步学会格式（91.6%）却学不会语义（0.8%）。

### 3.3 GRPO（Group Relative Policy Optimization）原理

**GRPO**（DeepSeekMath, 2024）是强化学习训练 LLM 的方法，比 PPO 更简单高效。

**强化学习基础**：智能体（模型）根据状态（输入）输出动作（生成的文本），环境/奖励函数打分，智能体更新策略以最大化期望奖励。

**PPO 的做法**：需要训练一个 critic 网络估计状态价值（value function），算优势函数（advantage）指导更新——**critic 和策略一样大，训练开销翻倍**。

**GRPO 的简化**：去掉 critic，改用**组内相对奖励**：

```text
对同一输入 x，采样 G 个输出 {y₁...y_G}（G 是组大小，如 8）
每个 yᵢ 由奖励函数打分 r(yᵢ)
组内归一化：Âᵢ = (r(yᵢ) - mean(r)) / std(r)      ← 相对优势
更新：最大化 Σ E[ min( πθ(yᵢ)/πref(yᵢ) · Âᵢ, clip(...) ) ]
      - β · KL(πθ || πref)                        ← 防止跑太远
```

关键点：
- **组内相对**：不要求绝对分数，只要"组内谁更好"——奖励函数有噪声时更鲁棒
- **重要性采样比** πθ/πref：用旧策略采样的数据更新新策略的修正系数
- **clip 机制**：限制每次更新幅度（0.2 截断），防止策略崩溃
- **KL 惩罚**：β 控制与参考策略（SFT 模型）的距离，避免生成崩溃/语言退化
- 策略梯度：`∇L = E[Âᵢ · ∇log πθ(yᵢ)]`——奖励高的输出概率增大，奖励低的减小

**为什么我们复现时 GRPO 失败**：奖励函数由几个子奖励组成（格式奖励 + 动作命中奖励 + 意图奖励 + 长度惩罚）。模型初期生成完全不符合格式 → **内容奖励恒为 0** → 组内所有样本分数相同 → Âᵢ≈0 → 梯度消失 → 学不动。1,773 步只学会了长度/标签皮毛（5.8%）。**本质：奖励信号稀疏 = 老师从不批改 = 学生学不动。**

### 3.4 LoRA（Low-Rank Adaptation）

**LoRA**（Hu et al., 2021）：参数高效微调。冻结原权重 W，只训练低秩增量：

```text
W' = W + ΔW = W + B·A     （A: r×d, B: d×r, 秩 r 很小，如 8）
```

- 参数量：7B 模型只训练 25.9M（0.31%）
- **原理**：微调产生的权重变化通常是低秩的（有效自由度小），低秩分解足够表达
- 好处：显存小、多个任务可并行（不同 adapter 切换）、不改变原模型推理速度
- 我们配置：r=8, alpha=32, dropout=0.05，目标 q/k/v/o/up/down/gate 投影

### 3.5 ms-swift 与 DeepSpeed

- **ms-swift**（ModelScope）：阿里开源的大模型微调/RLHF 框架，封装了 SFT、GRPO 等训练器、数据格式、多模态支持。我们用 v3.4.1。
- **DeepSpeed ZeRO-2**：分布式显存优化。ZeRO（Zero Redundancy Optimizer）把优化器状态、梯度、参数分片到各 GPU——ZeRO-2 分片优化器状态+梯度，每张卡只存 1/N。7B 模型 + LoRA 才能塞进 3×RTX 3090。

### 3.6 与 Stage 1 的关系（因果链）

```text
Stage 1 是出题人，Stage 2 是考生。
Stage 1 Top-1 3.56% → 8 个历史动作期望正确 0.28 个 → 考生拿到全错题
→ 背答案（SFT）只能学格式；考试（GRPO）奖励恒零学不动
→ 解锁语义的唯一路径：先提升 Stage 1 识别率
```

---

## 4. 评测技术

### 4.1 Top-1 / Top-5（分类准确率）

- Top-1：模型分数最高的类 == 真值才算对
- Top-5：真值在分数最高的前 5 个类里就算对
- 应用：Stage 1 分类器（125×352 联合类）、官方 RULSTM τa=1s 协议（2,513 类）
- 我们的 Stage 2 是生成式（无分类分数），"Top-5"实为 5 次采样命中最优（oracle Hit@5），不是标准排名 Top-5

### 4.2 mAP（平均精度均值，多标签指标）

用于"预测视频剩余部分会出现哪些动词"的多标签任务（EGO-TOPO 协议）：

```text
对每个类：按分数降序排列所有样本 → 逐点算 Precision@k 与 Recall@k
        → AP = PR 曲线下面积（积分）
mAP = 所有类的 AP 平均
```

- 多标签：一个视频的剩余部分可能同时出现多个动词 → 不能用 Top-1 的单标签思路
- 分组：All（全部）/ Freq（训练频次>100）/ Rare（低频）——低频类更难，Rare 反映模型泛化
- 观察比例 P=25/50/75：看 25% 的视频预测其余 75%，越早看越难

### 4.3 编辑距离（ED，Ego4D 协议）

**Damerau-Levenshtein 距离**：把预测序列变成真值序列所需的最少编辑操作数（插入/删除/替换/相邻交换），归一化到 [0,1]。0=完全一致，1=完全无关。论文在 Ego4D 主表用它评估 20 步动作序列。我们的 Stage 2 输出 ED≈0.98——接近随机，语义未突破的直接证据。

### 4.4 官方协议

| 协议 | 任务 | 划分 | 指标 | 论文是否报告 |
|------|------|------|------|:--:|
| RULSTM | 观察 3.5s，预测 τa=1s 后动作 | train 232 / val 40 视频 | Top-1/Top-5（2,513 类） | 否 |
| EGO-TOPO | 观察 P%，预测剩余视频动词 | S1：train 205 / val 67 视频 | mAP（125 类多标签） | ✅ 主表 |

---

## 5. 复现技术决策记录

| 技术点 | 论文 | 我们 | 影响 |
|--------|------|------|------|
| 帧特征 | EgoVideo-V 1408-dim（视频级时序） | CLIP ViT-L/14 1024-dim（图像级） | 容量 -27%，无时序建模 |
| HOI 特征 | handobj_100K+ego 真实检测 | 0.6% 真实 + 99.4% 帧副本 | 双流退化为单流 |
| Stage 2 训练 | GRPO | GRPO 失败 → SFT 修复格式 | 格式 ✅ 语义 ❌ |
| 评测 | EGO-TOPO mAP + Ego4D ED | mAP（feature-limited）✅，Ego4D 未跑 | 协议对齐部分完成 |

---

*配套文档：WORKLOG.md（过程日志）、REPORT_TO_PROFESSOR.md（汇报稿）、Stage1/Stage2/INSIGHT_总结果.xlsx（数据表）*
