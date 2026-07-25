# INSIGHT 项目复现受阻报告

## 1. 项目简介

复现论文：**[Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](https://arxiv.org/abs/2508.01742)**，AAAI 2026，CVPR 2025 Ego4D LTA Challenge 第一名。

官方开源仓库：https://github.com/CorrineQiu/INSIGHT

框架分为两个阶段：

| 阶段 | 功能 | 模型规模 |
|------|------|------|
| Stage 1 | 手-物体语义动作识别（Hand-Object Semantic Action Recognition） | Transformer ~10M 参数 |
| Stage 2 | 显式认知推理（Cognitive Reasoning via GRPO） | Qwen2.5-VL-7B + GRPO 强化学习 |

## 2. 我们已完成的工作

### 2.1 环境搭建 ✅

| 项目 | 值 |
|------|-----|
| 服务器 | 4×NVIDIA RTX (24GB 显存/卡) |
| Python | 3.10 |
| PyTorch | 2.5.1+cu118 |
| Conda 环境 | `/data/conda_envs/insight_env` |

### 2.2 预训练模型下载 ✅

所有 6 个外部预训练模型已就绪（共约 35 GB）：

- SAM2 (sam2.1_hiera_large.pt)
- EgoVideo (base_best.pt + large_best.pt)
- Hand Object Detector (faster_rcnn_1_8_89999.pth + resnet101_caffe.pth)
- Qwen2.5-VL-7B-Instruct (完整 safetensors)
- all-MiniLM-L6-v2 (文本嵌入)

注：Hand Object Detector 缺少 `faster_rcnn_1_8_132028.pth`（handobj_100K+ego版），已邮件联系作者 Dandan Shan，其学校账号过期正在恢复中。当前使用 handobj_100K 版（手检测性能 89.8 vs 90.5 AP，差距可接受）。

### 2.3 Hand Object Detector 适配 ✅

原版 ddshan/hand_object_detector 需要 CUDA C 扩展编译，与服务器 CUDA 11.5/PyTorch 2.5 不兼容。已换用 larsrpe/hand_object_detector_pytorch（纯 PyTorch 实现），权重文件完全兼容，模型加载验证通过。

### 2.4 EPIC-Kitchens-55 数据集下载 ✅

通过 Bristol 大学官方服务器 + 代理下载，历时 4 天：

- RGB 帧：432/434 个 tar 完整下载，2 个因服务器 SSL 错误无法获取
- 总大小：228 GB
- JPEG 帧：~125 万帧

## 3. 代码分析：实际可运行的部分

### 仓库文件结构

```
INSIGHT/
├── HandObject/           # Stage 1 训练代码 ← 可用
│   ├── main.py           #   入口：train + test
│   ├── model.py          #   ActionRecognitionModel (Transformer)
│   ├── dataset.py        #   ActionDataset
│   └── train.py          #   train_one_epoch / validate
├── CognitiveReasoning/   # Stage 2 训练代码 ← 可用
│   ├── run_external_reward_func_7B_qwen_intention.sh  # GRPO 训练脚本
│   ├── my_rewards_intention.py                        # 奖励函数
│   ├── plugin_in_log_intention.py                     # 日志插件
│   └── cal_ED.py                                      # 评测（编辑距离）
└── README.md
```

**Stage 1 和 Stage 2 的训练/评测代码是完整的。**

### Stage 1 的数据接口

`dataset.py` 第 63-64 行明确规定了输入格式：

```python
frame_path = self.frame_features_dir / split / f"{clip_uid}_{action_idx}.pt"
mask_path  = self.mask_features_dir  / split / f"{clip_uid}_{action_idx}.pt"
```

即：每个动作片段必须有一个 1408 维特征向量（`.pt` 文件），分为 `frame_features`（全局帧特征）和 `mask_features`（HOI 区域特征）两类。

## 4. 阻塞点：为什么现在跑不了

### 仓库中缺失的代码

Stage 1 和 Stage 2 的训练代码均在仓库中。但仓库**完全不包含**从原始视频/帧生成输入特征的代码。

README 对此的描述只有一句：

> *"Run Hand Object Detector, SAM2, and EgoVideo in sequence, following their official tutorials, to obtain both frame features and HOI (hand-object interaction) features."*

翻译：依次运行三个外部工具提取特征。但**如何串联这三个工具、如何将它们编排成 INSIGHT 需要的 `.pt` 文件格式，仓库中没有提供任何代码或脚本**。

### 缺失的完整链路

```
原始 JPEG 帧 (1.25M)
    │
    ├──→ Hand Object Detector  →  手/物体 bbox        ← 有代码 + 权重
    ├──→ SAM2                  →  HOI 分割 mask        ← 有代码 + 权重
    ├──→ EgoVideo-V            →  帧级 1408-dim 特征   ← 有代码 + 权重
    │
    ├ ─ ─ ─ ─ 调度/编排脚本 ─ ─ ─ ─ ─ →   【缺失】
    │      - 如何按 EK55 annotation 组织帧
    │      - SAM2 如何接收 HandObjDet 的 bbox 输出
    │      - 如何给 EgoVideo 传入原始帧 + mask
    │      - 如何将 EgoVideo 输出存成 {clip_uid}_{action_idx}.pt
    │      - 如何划分 train/val/test split
    │
    ▼
 .pt 特征文件 → Stage 1 训练 → Stage 2 训练    ← 此部分可用
```

### 其他技术问题

| 问题 | 严重程度 | 说明 |
|------|:--:|------|
| 类数不匹配 | 中 | Stage 1 写死 117 verb / 521 noun（Ego4D FHO），EK55 是 125 verb / 352 noun，需修改模型输出层 |
| Annotation 格式不同 | 低 | EK55 标注为 CSV，需转换为 JSON |
| EgoVideo 推理 API | 高 | stillfast 代码主要面向训练，单帧推理需自行编写 |
| SAM2 bbox-prompt 推理 | 低 | SAM2 开源文档有现成示例，易适配 |

## 5. 出口方案

### 最直接的方式：联系作者

联系 INSIGHT 一作 **Qiaohui Chu**（GitHub: CorrineQiu），请求：

1. **特征提取/编排脚本**（最核心，几小时即可走通）
2. 或**预提取好的 EK55 特征文件**（跳过所有特征提取步骤）
3. EK55 适配版本的代码或配置

### 替代方式：自行开发

三个工具均为开源，自行编写编排脚本也是可行的。估计额外开发时间 **5-10 天**，主要工作量在 EgoVideo-V 的推理适配和三个工具的 I/O 衔接。

## 6. 已投入时间估算

| 工作 | 耗时 |
|------|:--:|
| 环境搭建、权重下载与安装 | ~1 天 |
| HandObjDet 编译/适配 | ~0.5 天 |
| 网络代理配置 | ~0.5 天 |
| EK55 数据下载（等待为主）| ~4 天 |
| 数据解压 | ~2 小时 |
| 代码分析、问题定位 | ~1 天 |
| **合计** | **~7 天** |
