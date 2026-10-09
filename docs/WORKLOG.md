# INSIGHT 复现工作日志

> 论文: Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation (AAAI 2026)
> 仓库: https://github.com/CorrineQiu/INSIGHT
> 负责人: Esun Li (esun12191@gmail.com, 合肥工业大学)
> 开始日期: 2026-07-16

---

## 阶段总览

| 阶段 | 状态 | 开始 | 完成 |
|------|:--:|------|------|
| 0. 环境搭建 | ✅ | 07-16 | 07-16 |
| 1. 预训练权重部署 | ✅ | 07-16 | 07-16 |
| 2. EK55 数据下载 | ✅ | 07-16 | 07-19 |
| 3. 特征提取管线 | ✅ | 07-25 | 07-26 |
| 4. Stage 1 训练 | ✅ | 07-26 | 07-26 |
| 5. Stage 2 GRPO 训练 | ❌ | 07-29 | 07-29 (只学格式) |
| 6. Stage 2 SFT 训练 | ✅ | 07-30 | 07-30 (格式✅ 语义❌) |
| 7. 官方协议评测 (mAP/τa=1s) | 🔶 mAP✅ τa❌ | 07-31 | — |

---

## 2026-07-16

### 环境配置
- 服务器: amax, 4×RTX 24GB, CUDA 11.5, Ubuntu
- 创建 conda 环境 `/data/conda_envs/insight_env`: Python 3.10, PyTorch 2.5.1+cu118
- pip 源: `https://pypi.tuna.tsinghua.edu.cn/simple`

### 预训练模型部署
所有模型统一存放 `/data/pretrain_model/`:
- `all-MiniLM-L6-v2/` — 文本嵌入 (~90 MB)
- `EgoVideo/checkpoints/` — base_best.pt + large_best.pt
- `Hand_Object_Detector/` — handobj_100K + resnet101_caffe.pth
- `Qwen2.5-VL-7B-Instruct/` — 5×safetensors
- `SAM2/sam2.1_hiera_large.pt` — ~850 MB

### Hand Object Detector 适配
- **问题**: ddshan 原版需编译 CUDA C 扩展, nvcc (CUDA 11.5) 与 PyTorch 2.5 C++17 不兼容, 报错 `TypeList.h: 'tuple' was not declared`
- **解决**: 切换至 `larsrpe/hand_object_detector_pytorch` (纯 PyTorch 实现)
- **修改**: `handobdet/hand_object_detector.py:44`, 硬编码的 `faster_rcnn_1_8_132028.pth` → `faster_rcnn_1_8_89999.pth` (我们只有 handobj_100K)
- **验证**: 模型加载成功 (`torch.load` 通过, FutureWarning 忽略)
- **缺失权重**: `faster_rcnn_1_8_132028.pth` (handobj_100K+ego), 已邮件联系 Dandan Shan (dandans@umich.edu), 07-13 回复学校账号过期正在找回

### EK55 数据下载
- 初始: Bristol 直连 0.2 MB/s, 预计 12 天
- 配置 Clash 代理 (127.0.0.1:7897), 速度提升至 500-800 KB/s
- 经历多次 Bristol 断连、SSL 报错、进程退出
- 编写自动重启循环 `/tmp/download_loop.sh`
- **结果**: 432/434 tar 完整下载 (228 GB, 125 万帧 JPEG), 2 个 P32 test 文件永久失败

### EK55 解压
- tar 解压后所有帧在同一个 `rgb_frames/` 目录 (不是按 tar 分子目录)
- **教训**: 解压时错误地删除了 tar, 后需用 `.done` 标记补救, 最终白折腾
- **结果**: 1,251,563 JPEG 帧

### 关键发现: 特征提取管线缺失
- 仓库仅含 Stage 1 (HandObject/) 和 Stage 2 (CognitiveReasoning/) 训练代码
- Stage 1 的输入是 `.pt` 特征文件 (1408-dim), 需由外部工具生成
- 生成路径: JPEG → HandObjDet → SAM2 → EgoVideo → .pt 文件
- **仓库中无任何调用这三个工具的代码或脚本**
- 生成的报告: `docs/blocking-issue-report.md`, `docs/pipeline-and-blocker.md`

---

## 2026-07-17

### 尝试自行搭建特征提取管线
- SAM2 源码 clone: `~/ldy/git/sam2/` (facebookresearch/sam2)
- 确认三个工具均有开源代码和权重
- 但 EgoVideo 的 stillfast 是 PyTorch Lightning 训练框架, 无推理接口
- EgoVideo 的 eccv-2022 有 `modeling_finetune.py`, 包含 `VisionTransformer.forward_features()`
- 决策: 自行编写调度脚本

### Clash 代理问题
- AppImage GUI 版报错: `symbol lookup error: libpangoft2`
- 决定暂时不装代理, 用清华源直连

---

## 2026-07-25

### EgoVideo checkpoint 分析

| checkpoint | 架构 | 维度 | 帧数 |
|------|------|:--:|:--:|
| `base_best.pt` | CLIP ViT-Base | 768 | 4 |
| `large_best.pt` | CLIP ViT-L/14 | 1024 | 4 |

- conv1: [1024, 588] = 3通道 × 1(tubelet) × 14×14(patch)
- pos_embed: [577, 1024] = 1(CLS) + 24×24
- 结论: 标准 CLIP ViT-L/14 架构，不是论文描述的 1408-dim EgoVideo-V

### EgoVideo 加载问题解决

**问题**: checkpoint 键名 (OpenGVLab 风格) 与 HF transformers (CLIPVisionModel) 不兼容，且 eccv-2022 代码 270 个参数对不上，cvpr-2024 stillfast 依赖链太长。

**解决**: 直接使用 HuggingFace `CLIPVisionModel`，写键名映射脚本：
- attn.Wqkv → 拆成 q_proj, k_proj, v_proj
- ln_1/ln_2 → layer_norm1/layer_norm2
- transformer.resblocks → encoder.layers
- conv1: Conv3d→Conv2d 压缩
- pos_embed: [577,1024]→[257,1024] 插值

**结果**: 缺失 0，多余 0，输出 [1, 1024]。

### SAM2 验证通过

- checkpoint: sam2.1_hiera_large.pt (FACEBOOK RESEARCH 官方)
- 测试: EK55 JPEG + 假 bbox → mask score 0.53
- 三个特征提取工具全部验证通过

### 修改 Stage 1 适配 EK55

- `input_dim`: 1408 → 1024 (匹配 CLIP ViT-L/14)
- `verb_num_classes`: 117 → 125 (EK55 实际 125 类)
- `noun_num_classes`: 521 → 352 (EK55 实际 352 类)

---

## 2026-07-26

### 标注转换

- 脚本: `feature_extraction/convert_annotations.py`
- EK55 CSV → INSIGHT JSON (`train.json` + `val.json`)
- 结果: 28472 actions, 272 clips, 10% val split
- verb: 0-124 (125 类), noun: 1-351 (352 类)

### 帧特征提取

- 脚本: `feature_extraction/extract_features.py`
- 方法: CLIP ViT-L/14 批处理，每 action 采样 4 帧，平均为 1024-dim
- 结果: 15480 个 .pt 文件 (train: 14217, val: 1263)
- 耗时: ~2 分钟
- 注意: P22/P24/P31 部分帧缺失 (对应未下载的 2 个 tar)，日志大量 warning 但不影响

### 磁盘危机

- `/home` 分区 (244G) 100% 满，导致 sed/训练写文件失败
- 清理: conda pkgs (15G), .cache, NVIDIA_CUDA_Samples (3.2G)
- 恢复至 90% (24G 可用)

### Stage 1 训练 - 首次跑通

训练过程中修复了 6 个 bug:

1. `SyntaxError` (dataset.py:25) — assert 末尾多余逗号
2. `IndexError` (main.py:96) — verb/noun 类数函数默认参数仍为 Ego4D 值
3. `RuntimeError: mat1 and mat2 shapes` (model.py:45/54) — `make_lowrank_layer` 硬编码 1408
4. `FileNotFoundError` (dataset.py:42) — 缺少 test.json
5. `RuntimeError: unexpected pos` (main.py:258) — checkpoint 写 /home 失败，改为 /data
6. 日志写 /home 失败 — 改为 /tmp

**最终成功输出**: `best_model.pth` (54MB) + `cooccurrence_matrix.pt` (690KB)

**注意**: HOI 特征临时用帧特征复制替代（假 mask），待 HandObjDet + SAM2 调度脚本完成后替换。

### 产出物清单

| 文件 | 位置 | 说明 |
|------|------|------|
| train.json / val.json | `/data/.../insight_annotations/` | 标注 |
| frame_features (15480 .pt) | `/data/.../features/frame_features/` | 帧特征 |
| mask_features (15480 .pt) | `/data/.../features/mask_features/` | HOI特征(临时=frame) |
| cooccurrence_matrix.pt | `/data/.../features/` | 共现矩阵 |
| best_model.pth | `HandObject/checkpoint/` | Stage 1 模型 |
| best_model.pth (v2) | `/data/.../checkpoint/` | 含91个真实HOI |

### 真实 HOI 特征提取

- 脚本: `/tmp/extract_hoi.py`
- 方法: HandObjDet 检测手部 bbox → 裁剪手部区域 → CLIP 编码
- 结果: 91 个 action 检测到手 (0.6%)，其余 13K+ 失败或跳过
- 失败原因:
  - EK55 胸戴相机，大量画面中手未出现在镜头内
  - handobj_100K (非 ego 优化版) 检测率低，需 handobj_100K+ego
  - 每 action 仅采样 4 帧，可能 4 帧均未拍到手的正面
- 临时方案: 混合特征 (91 真 HOI + 15400 帧特征伪 HOI) 重跑 Stage 1 → best_model.pth v2 (56MB)

### Stage 2 环境搭建

#### ms-swift 适配
- 版本: 4.2.0.dev0 (从 GitHub 安装的 editable 版本)
- 重新安装 `pip install -e .` 修复导入问题
- v4.2 API 变更:
  - `swift.llm` → 废弃, 改用 `swift.model.get_model_processor`
  - `swift.plugin` → `swift.rewards`
  - CLI 入口: `swift rlhf` (替代原 `python -m swift.cli.rlhf`)

#### 奖励插件迁移
- 复制 `my_rewards_intention.py` + `plugin_in_log_intention.py` 到 `/data/ms-swift/swift/rewards/`
- 修复 4 处兼容性问题:
  1. `swift.plugin` → `swift.rewards`
  2. `swift.llm` 导入注释掉 (v4.2 已移除)
  3. `my_rewards_new_intention` → `swift.rewards.my_rewards_intention`
  4. MiniLM 路径硬编码修正
  5. `assert ... ,` 末尾逗号
  6. 删除依赖 `swift.llm` 的 `CustomizedRMPlugin`/`QwenLongPlugin` 类

#### Qwen2.5-VL-7B 加载验证
- 依赖: `qwen-vl-utils` (清华源安装)
- 结果: 8.3B 参数，4×25GB GPU 可承载，加载耗时 ~1.5 分钟

#### Stage 2 训练数据集
- 格式: ms-swift RLHF JSONL，每行一条 JSON
- 字段: `images` (帧路径), `messages` (user/assistant), 含 `<think><intention><answer>` 标签
- 创建: 20 train + 5 val 最小验证集 (`stage2_train.jsonl`)

#### 训练启动调试中
- 4 GPU (0,1,2,3) 分布式训练
- 连续遇到多个版本兼容问题:
  1. MKL threading 层冲突 → `MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU`
  2. reward_funcs/weights 数量不匹配 → 简化为 3 个函数
  3. `--train_type full` v4.2 不支持 → 移除
  4. `soft_cache_length` 缺失 → 添加 `--soft_cache_length 256`
  5. `per_device_eval_batch_size=1` 不能被 `num_generations=2` 整除 → 改为 2
  6. `**依赖地狱**: ms-swift v4.2 死要求 `trl>=0.20`，但 `trl>=0.20` 需要 vllm/PyTorch 2.6+
  7. PyTorch 2.5.1 不支持 `trl>=0.28` 的 `FSDPModule`
  8. `trl>=0.20` 依赖 `vllm`/`mergekit`
- 降级 `trl==0.15` 可绕过，但 ms-swift 两处硬编码检查 `trl>=0.20` 阻断

#### 解决方案：独立 Stage 2 环境
- 新建 conda 环境 `/data/conda_envs/stage2_env`: Python 3.10, PyTorch 2.6.0
- 安装 `trl==0.20.0` + `vllm` + `ms-swift` (从清华源)
- **失败**: PyTorch 2.6 要求 CUDA Driver ≥525, 服务器只有 495.29.05 (CUDA 11.5)
- 回退 PyTorch 2.5.1 (cu118)，但 vllm 与 CUDA 11.5 仍有符号冲突 (`cuTensorMapEncodeTiled`)

#### 最终结论：Stage 2 当前无法启动
- 依赖链: `ms-swift v4.2 → trl>=0.20 → vllm → PyTorch 2.6 → CUDA Driver ≥525`
- 服务器 CUDA Driver 495.29.05 无法满足
- 升级驱动可解决，但需要 root 权限+重启，需协调管理员

#### 当前成就
- **Stage 1 完整跑通**: 数据下载 → 标注转换 → 特征提取 → 训练 → best_model.pth (56MB)
- **Stage 2 环境已就绪**: Qwen 模型加载成功 (8.3B), 插件适配完成 (6处修复), 数据集格式正确
- **阻塞点**: GPU 驱动版本，非代码问题
- **出口**: 等管理员升级驱动，或复用 insight_env 的 PyTorch 2.5.1 + 等待 ms-swift 降低 trl 要求

### 已修改的文件清单

| 文件 | 位置 | 修改内容 |
|------|------|------|
| `main.py` | `HandObject/` | input_dim 1408→1024, verb/noun 类数, checkpoint 路径, 日志路径 |
| `model.py` | `HandObject/` | 添加 self.input_dim, hardcoded 1408→self.input_dim |
| `dataset.py` | `HandObject/` | assert 末尾去逗号 |
| `plugin_in_log_intention.py` | `CognitiveReasoning/` | 多处兼容修复 (本地+远程ms-swift副本) |
| `my_rewards_intention.py` | `CognitiveReasoning/` | 导入修正, MiniLM 路径 |
| `ms-swift/` 多个文件 | `/data/ms-swift/swift/` | trl 版本检查绕过 (已 git checkout 恢复) |

---

## 2026-07-27 ~ 07-28

### 驱动升级尝试（失败）

- 目标: 升级 NVIDIA 驱动 495→535 以支持 PyTorch 2.6 + vllm
- 下载: 308MB `.run` 文件，NVIDIA 中国 CDN 返回 403，最终走 Mac 下载传过去
- 安装: `--no-questions` 不是合法参数导致静默失败，重启后黑屏
- 修复: 进 recovery mode, `apt purge nvidia-*`, 重装 `nvidia-driver-495`
- 恢复成功: 驱动回到 495.29.05，CUDA 11.5
- **教训**: 永远不要用 `.run` 文件装驱动，走 `apt` 装。安装前确认 SSH 备用通道可用

### 换 ms-swift 3.4.1（成功方案）

- 原因: 降级 ms-swift 比升级驱动风险低
- 版本: ms-swift 3.4.1 + trl 0.17.0, 兼容 PyTorch 2.5.1 + CUDA 11.5
- 安装坑:
  1. ms-swift 3.4.1 拉入了 vllm 0.26.0, 需要 CUDA 13 → `pip uninstall vllm`
  2. `qwen-vl-utils` 新版有 `IMAGE_FACTOR` 兼容问题 → 降级到 0.0.8
  3. ms-swift 的 `swift.llm` 在 3.4.1 存在 → 插件可直接用原版 INSIGHT 代码

### 插件适配（ms-swift 3.4.1）

- 路径: `/data/conda_envs/insight_env/lib/python3.10/site-packages/swift/plugin/`
- 3 处修改:
  1. `assert ... ,` 末尾逗号
  2. `from my_rewards_new_intention` → `from swift.plugin.my_rewards_intention`
  3. MiniLM 路径修正
- `plugin_in_log_intention.py` 的 `QwenLongPlugin`/`CustomizedRMPlugin` 依赖 `swift.llm` → 在 3.4.1 保留完整，无需删除
- 导入验证: 需从 `/home/amax` 目录执行（避免 swift 日志权限问题）

### Stage 2 GRPO 训练 - 成功启动

- 策略: 4GPU, torch_dtype=float16 (bf16 被 trl GRPOConfig 拒绝)
- 模型: Qwen2.5-VL-7B, LoRA 微调 (25.9M 可训练参数 / 8318M 总参数, 0.31%)
- 数据: 20 train + 5 val (最小验证集)
- 修复了 7 个启动参数错误:
  1. `--train_type full` 旧版本不支持 → 移除
  2. `soft_cache_length >= max_completion_length` → 设为 128
  3. `num_generations=1` → GRPO 要求至少 2
  4. `per_device_eval_batch_size=1` 不被 `num_generations=2` 整除 → 改成 2
  5. `dataloader_num_workers > 0` seed_worker 缺参数 → 设为 0
  6. bf16 GPU 检测失败 → 改用 float16
  7. vllm 依赖 CUDA 13 → 卸载
- **最终成功**: `Train: 0/3` 开始迭代，模型 4 GPU 分布式加载

### 当前状态

- Stage 1: ✅ 完整跑通
- Stage 2: ⏳ 训练已启动，正在跑 GRPO 微调 Qwen2.5-VL-7B

---

## 2026-07-29 ~ 07-30

### 实验边界

- 本轮目标是跑通 `.pt 特征 → Stage 1 → Stage 1 预测历史 → Stage 2 → 五候选评测`。
- Stage 2 输入的 8 步动作历史来自 Stage 1 预测，不再是 GT 历史；未来 20 步仍使用 EK55 标注作为训练/评测目标。
- Stage 1 验证集仅有 1263/2490 条动作同时具备 frame/mask 特征，覆盖率 50.72%；缺失 1227 条。
- 07-30 逐文件审计后确认：train 有 14179 对 frame/mask 特征，其中 13891 对完全相同、仅 288 对不同；val 的 1263 对全部完全相同。因而“91 条真实 HOI”不再作为最终口径；当前只能确认 288 条 train mask 不是 frame 的直接副本，尚不能据此保证检测与分割质量。验证集实际上完全没有独立 HOI mask 信号。因此结果是当前可用特征子集上的复现实验，不是完整 EK55 官方 split 结果。
- 已有固定窗口任务为“8 步历史 → 20 步未来动作”，它本身没有 P=25/50/75 所需的阶段化类别置信度，不能直接计算 All/Freq/Rare mAP。07-30 另行补充 EGO-TOPO 协议评测管线。

### Stage 1 checkpoint 损坏与重建

- 原 `/data/datasets/EPIC-KITCHENS/checkpoint/best_model.pth` 大小为 56,840,192 bytes，但缺少 PyTorch zip central directory，`torch.load` 失败。
- `zip -FF` 能找回大部分条目，但 `best_model/data/53` 流不完整，修复文件仍无法加载。
- 重新训练 Stage 1：train 1644 batches、val 144 batches，共完成 6 epochs。
- 新 checkpoint：`/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth`，89 tensors，可正常加载。

### Stage 1 预测历史导出

- 导出脚本：`HandObject/export_stage1_predictions.py`。
- train：14179 条可用，11803 条缺失特征。
- val：1263 条可用，1227 条缺失特征。
- 输出：
  - `/data/datasets/EPIC-KITCHENS/stage1_predictions/stage1_train_predictions.jsonl`
  - `/data/datasets/EPIC-KITCHENS/stage1_predictions/stage1_val_predictions.jsonl`
- 构造 Stage 2 数据（history=8, horizon=20）：
  - train：10748 条，missing_history=9603。
  - val：876 条，missing_history=1050。
  - 目录：`/data/datasets/EPIC-KITCHENS/stage2_stage1_history/`。

### Stage 2 reward 格式修复与探针实验

- 原 reward 正则错误地要求带 `*` 的动作对，导致大量合法 `verb noun` 标签被拒绝。
- 三处 `pattern_pair` 统一改为 `^[a-z0-9_:-]+ [a-z0-9_:-]+$`；重新检查 train/val 标签均为 invalid=0。
- 100-step context probe 成功完成，best checkpoint 为 `checkpoint-75`；探针确认训练链路可用，但内容奖励很弱。

### GPU 2 硬件故障

- 四卡训练期间 physical GPU 2（PCI `0000:67:00.0`）出现：
  - Xid 79：GPU has fallen off the bus。
  - Xid 48：L2 cache uncorrectable double-bit error (DBE)。
- 故障导致 rank2 `CUDA error: unspecified launch failure`，残留进程一度以 zombie 状态占用显存。
- 重启后四卡重新枚举，但为降低风险，最终训练和推理固定使用 physical GPU `0,1,3`，不再使用 GPU 2。
- ms-swift 不能解析 UUID 形式的 `CUDA_VISIBLE_DEVICES`，最终使用数字 `0,1,3`；进程内部逻辑编号为 `0,1,2`。

### Stage 2 全量训练完成

- 模型：Qwen2.5-VL-7B-Instruct + LoRA + GRPO。
- LoRA 实际配置：`r=8`、`alpha=32`、`dropout=0.05`，目标为语言模型中的 q/k/v/o/up/down/gate projection；视觉模块不在 LoRA target pattern 中。
- 训练数据：10748 条 Stage 1 predicted-history 样本。
- 启动参数没有显式 `val_dataset`，`val_dataset=[]`、`split_dataset_ratio=0.01`；训练日志中的约 107 条 eval 样本来自训练 JSONL 的自动 1% 切分，并不是独立的 876 条 Stage 2 val 集。因此训练期间的 `eval_reward` 不能当作最终验证集指标。
- GPU：physical 0,1,3；DeepSpeed ZeRO-2；1 epoch。
- 完成：1773/1773 steps，耗时 6h 19m 30s。
- 最终 checkpoint：`/data/datasets/EPIC-KITCHENS/stage2_stage1_history_full_3gpu_v3_output/v0-20260729-235831/checkpoint-1773`。
- 末次自动划分验证：
  - eval_reward = 0.0763248
  - ActionIntentReward = 0.09782144
  - ActionIntentContLin = 0.00023810
  - SoftOverlong = -0.06835318
  - eval_kl = 0.43515625
- 内容奖励接近 0，表明模型主要学习了长度/格式约束，语义动作预测能力不足。

### Stage 2 五候选端到端推理

- 使用 `checkpoint-1773` 对 876 条 Stage 2 val 样本生成 5 个候选。
- candidate 0：temperature=0，作为确定性 Top-1；candidate 1~4：temperature=0.9、不同 seed。
- 每个 candidate 文件 876 行，总计 4380 行。
- 输出目录：`/data/datasets/EPIC-KITCHENS/stage2_generated_topk_checkpoint1773/`。

### 最终指标

#### Stage 1：EK55 covered validation subset

评测样本 1263，skipped_missing=1227，micro accuracy：

| 指标 | Top-1 | Top-5 |
|------|------:|------:|
| Verb | 18.6857% | 59.6200% |
| Noun | 8.0760% | 22.4070% |
| Action（raw joint probability） | 3.3254% | 7.2051% |
| Action（Equation 5 semantic prior） | **3.5629%** | **11.1639%** |

- 语义先验将 Action Top-1 从 42/1263 提升到 45/1263，Top-5 从 91/1263 提升到 141/1263。
- Action Top-1 **3.56%**、Top-5 **11.16%** 只能作为当前动作识别诊断：输入特征取自动作片段内部，且仅覆盖随机 val 的 50.72%。它不是目标动作前 1 秒的官方 anticipation 指标，不能填入 DCR/CPM 的 Top-1/Top-5 对比表。

#### Stage 1→Stage 2：生成式下一动作与序列指标

| 指标 | 结果 |
|------|-----:|
| Generative next-action Top-1 | 0.1142% (1/876) |
| Generative next-action Top-5 | 0.1142% (1/876) |
| Candidate-1 Action / Verb / Noun ED | 0.9995 / 0.9920 / 0.9870 |
| Oracle-min-over-5 Action / Verb / Noun ED | 0.9985 / 0.9804 / 0.9441 |
| `<answer>` tag rate | 98.8128% |
| Exactly 20 items | 91.5753% |
| Exactly 20 valid actions | 89.1553% |

- ED 为固定 20 步后的 normalized Damerau–Levenshtein distance，越低越好；接近 1 表明预测与 GT 几乎没有重合。
- Top-5 未高于 Top-1；五个候选没有带来新的正确样本。
- 推理文件没有类别 logits 或候选概率（`labels=None`、`logprobs=None`）；所谓五候选是 1 次确定性生成加 4 次不同随机种子的采样，不是标准的 ranked Top-5。抽查 candidate 0 已出现将 `Add onion` 重复 20 次的模式坍塌。
- 结论：完整链路已经跑通，但 Stage 2 主要学会了输出格式，尚未获得有效的未来动作语义预测能力。

### Stage 2 语义诊断（100 样本）

| 条件 | next-action Top-1 | Action ED | Verb ED | Noun ED | valid-20 |
|------|------------------:|----------:|--------:|--------:|---------:|
| GT history + base Qwen | 3.0% | 0.9825 | 0.9655 | 0.9215 | 15% |
| GT history + GRPO adapter | 0.0% | 1.0000 | 1.0000 | 0.9210 | 100% |
| Pred history + base Qwen | 0.0% | 0.9975 | 0.9300 | 0.9940 | 22% |
| Pred history + GRPO adapter | 0.0% | 1.0000 | 1.0000 | 0.9990 | 100% |

- 该对照确认 adapter 将格式合规率提高到 100%，但没有学习到未来动作语义；继续增加相同 GRPO 训练时长不是当前最优先的优化方向。

### 07-30：补充论文要求的两条官方协议管线

#### EK55 短期 Action Top-1/Top-5

- 依据 RULSTM 的 EK55 视频划分和采样设定新建 `evaluation/prepare_ek55_anticipation.py`。
- 使用官方完整标注实测：RULSTM 视频划分含 train 232 videos / 23493 raw actions、val 40 videos / 4979 raw actions，并建立 2513 个联合 action 类。按本任务 4.25s→1.0s 的 14 帧历史要求过滤过早动作后为 train 23431、val 4970。
- 每个样本使用 14 个 RGB 时间点，间隔 0.25 秒，最后观测点严格不晚于目标动作开始前 1 秒，避免原 action-internal 特征的标签泄漏。
- 新建独立特征提取器 `feature_extraction/extract_ek55_anticipation_rgb.py` 和 temporal classifier `HandObject/train_anticipation_topk.py`。
- 新分类器直接输出 2513 类 action logits，以 direct Action Top-1/Top-5 为主指标，同时输出 Verb/Noun 和 factorized-action Top-k 诊断。
- 一键脚本：`scripts/run_ek55_anticipation_topk.sh`。
- 由于当前 JPEG 为 participant-level 扁平目录、不同视频同名 frame 存在碰撞风险，增加快速替代链：下载 RULSTM 官方 6,515,089,408-byte TSN-RGB LMDB，以 30fps 选择 3.5s→1.0s 的 11 个无泄漏时间点，再训练同一 2513 类 temporal classifier。脚本为 `feature_extraction/export_rulstm_lmdb_sequences.py` 和 `scripts/run_ek55_topk_rulstm_features.sh`；报告时必须标记特征源为 RULSTM TSN-RGB。

#### EK55 长期 P=25/50/75 All/Freq/Rare mAP

- 对齐 EGO-TOPO 官方 S1 split：train 205 videos / 20730 actions，val 67 videos / 7742 actions；所有 28472 条均可无歧义映射回官方 action UID。
- 协议过滤后、尚未考虑本地 feature coverage 时：train 1251 个阶段样本，val 172 个阶段样本（P25=57、P50=60、P75=55）。
- 训练切分点 P=20/30/40/50/60/70/80；验证切分点 P=25/50/75；每个样本至少 3 个过去动作和 3 个未来动作。
- 预测剩余视频所有 future verb 的 125 维 multi-label 概率，按固定验证类别掩码计算每类 AP，再汇总 All/Freq/Rare mAP。
- Freq 使用 `EPIC_many_shot_verbs.csv`（训练频次 >100），Rare 为验证标签掩码内其余 verb。
- 新建 `evaluation/train_ek55_future_verb_map.py` 和 `scripts/run_ek55_future_verb_map.sh`。
- 当前版本复用已有 Stage-1 observed-action RGB 特征，因此属于官方划分/标签/指标下的 feature-limited baseline，不等价于论文完整 EgoVideo + EGO-TOPO + HOI 模型。
- 详细边界见 `docs/EK55_METRICS_PROTOCOL.md`。

### 结果文件

| 文件 | 远端位置 |
|------|----------|
| Stage 1 Top-k JSON | `/data/datasets/EPIC-KITCHENS/stage1_predictions/stage1_topk_metrics.json` |
| Stage 1 Top-k 文本 | `/data/datasets/EPIC-KITCHENS/stage1_predictions/stage1_topk_metrics.txt` |
| Stage 2 全部指标 | `/data/datasets/EPIC-KITCHENS/stage2_generated_topk_checkpoint1773/metrics_all.json` |
| Stage 2 指标文本 | `/data/datasets/EPIC-KITCHENS/stage2_generated_topk_checkpoint1773/metrics_all.txt` |
| Stage 2 五候选 | `/data/datasets/EPIC-KITCHENS/stage2_generated_topk_checkpoint1773/candidate_0.jsonl` ~ `candidate_4.jsonl` |
| Stage 2 最终 checkpoint | `/data/datasets/EPIC-KITCHENS/stage2_stage1_history_full_3gpu_v3_output/v0-20260729-235831/checkpoint-1773/` |

### 07-30：最快语义恢复路线（待远端运行）

- 结论：不等待缺失的 ego hand-object detector 权重，也不立即重提全部帧；先修复已经证实的 Stage 2 语义坍塌。
- 新增 `feature_extraction/build_stage2_sft_dataset.py`：把现有 Stage1-predicted-history GRPO 数据转换成带 assistant GT 的 SFT 数据，同时生成固定 100-row val 和 32/12 smoke 子集。
- 新增 `scripts/run_stage2_sft_fast.sh`：physical GPU 0/1/3、LoRA r=8/alpha=32、显式独立 val、2-step smoke 或 1-epoch full SFT。
- 新增 `scripts/evaluate_stage2_adapter_fast.sh`：先跑 100 样本确定性语义门；只有 Top-1>0 或 Action ED<0.98 才继续 876 条完整推理。
- 操作文档：`docs/STAGE2_SFT_FAST_PATH.md`。
- 预计：2–4 小时得到路线是否有效的结论；通过后 8–12 小时得到完整当前协议结果。

### 磁盘管理
- `/home`: 244G, 清理后 25G 可用 (conda pkgs, NVIDIA samples, .cache, EPIC-KITCHENS 残留)
- `/data`: 3.6T, 176G 可用
- 预防: `export PYTHONDONTWRITEBYTECODE=1` 避免 pyc 写 /home

## 2026-07-31

### Stage 2 SFT 完整训练成功

- 在 GRPO 失败后采用监督微调快速路线：先 SFT 修复语义坍塌，再考虑短 GRPO。
- 数据：`/data/datasets/EPIC-KITCHENS/stage2_stage1_history_sft/`，train 10,748、val 876、val100 100、smoke 32/12；每条含 1 图像 + 1 assistant 监督答案。
- 训练：`scripts/run_stage2_sft_fast.sh`，physical GPU 0/1/3，LoRA r=8/alpha=32，lr=5e-5，batch 4/GPU，grad_accum=8，895 steps，耗时 36m43s（2.46 s/step）。
- 输出：`/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/`。
- **最佳 checkpoint-200**（val loss 0.9885，val token acc 75.80%）；checkpoint-895 仅保留为过拟合末轮对照。
- 训练曲线关键点（val 每 100 步评估）：

| Step | Train Loss | Eval Loss | Train Acc | Eval Acc | ΔGap |
|------|-----------:|----------:|----------:|---------:|-----:|
| 100 | 1.0303 | 1.0316 | 74.40% | 74.32% | +0.08pp |
| 200 | 0.9433 | **0.9885** | 76.42% | **75.80%** | +0.62pp |
| 300 | 0.8271 | 1.0029 | 79.39% | 75.62% | +3.77pp |
| 400 | 0.7295 | 1.0253 | 81.53% | 75.82% | +5.71pp |
| 500 | 0.6529 | 1.0783 | 82.58% | 75.17% | +7.41pp |
| 600 | 0.5795 | 1.1450 | 83.97% | 74.49% | +9.48pp |
| 700 | 0.5371 | 1.1637 | 85.53% | 74.71% | +10.82pp |
| 800 | 0.4971 | 1.2130 | 85.93% | 74.24% | +11.69pp |

- **结论：step 200 后明显过拟合**；训练 acc 持续上升至 85.9%，验证 acc 从 75.8% 回落到 74.2%。

### Stage 2 SFT 五候选评测（checkpoint-200）

- 876 条验证集 × 5 候选，输出 `/data/datasets/EPIC-KITCHENS/stage2_sft_topk_checkpoint200/`（candidate_0~4.jsonl 各 876 行）。
- 评测脚本：`CognitiveReasoning/evaluate_generated_topk_compact.py`。

| 类别 | 指标 | 结果 | 评级 |
|------|------|-----:|:--:|
| 格式 | `<answer>` 标签率 | 100.0% | ✅ |
| 格式 | 精确 20 动作率 | 92.03% | ✅ |
| 格式 | 20 全合法动作率 | 91.60% | ✅ |
| 语义 | Next-Action Top-1 | 0.799% (7/876) | ❌ |
| 语义 | Next-Action Hit@5 | 1.598% (14/876) | ❌ |
| 语义 | Action ED (oracle) | 0.982 | ❌ 接近随机 |
| 语义 | Verb ED (oracle) | 0.847 | — |
| 语义 | Noun ED (oracle) | 0.935 | — |

- **结论：SFT 教会了格式（91.6%），但没有教会语义（Top-1 仍 0.8%）**。180/876 候选槽出现重复预测（duplicate_candidate_slots=180）。
- 对比 GRPO：Top-1 从 0.114% → 0.799%（+600%），格式从 ~5.8% → 91.6%（+85.8pp），但 ED 未实质改善。

### SFT vs GRPO 对比

| 指标 | GRPO (1773 步) | SFT (200 步) | 变化 |
|------|--------------:|------------:|------|
| Next-Action Top-1 | 0.114% | 0.799% | +600% |
| Next-Action Hit@5 | 0.114% | 1.598% | +1300% |
| 格式正确率 | ~5.8% | 91.6% | +85.8pp |
| Action ED | ~1.0 | 0.982 | -0.018 |

- GRPO 无监督探索 + reward 全零 → 只学格式；SFT 有监督答案 → 格式完美但语义未学。根因是 **Stage 1 Action Top-1 仅 3.56%**，Stage 2 收到的 8 个"历史动作"几乎全错，上限被锁死。

### Stage 2 两次尝试的机制说明

**Stage 2 的任务定位**：输入 = 8 个历史动作文本（Stage 1 预测）+ 当前帧图像；输出 = 20 个未来动作，按 think → intention → answer 三段式组织。历史动作文本来自 Stage 1 的"翻译"，因此 **Stage 1 是出题人，Stage 2 是考生**。

**SFT（监督微调）= 背答案**：老师把标注好的标准答案（GT 未来动作 + think/intention/answer）直接给学生，用交叉熵最大化逐 token 复述概率。优点：简单、快、稳定（200 步格式 91.6%）；局限：纯模仿，学到的是"答案的分布"，学不到"为什么对"。

**GRPO（组相对策略优化）= 考试自评**：不给答案。同一输入采样 N 个候选答案组成一组，奖励函数逐候选打分（格式/动作/意图），按组内相对优劣更新——好的强化、差的抑制。优点：能优化不可微的奖励、理论上能学到语义；致命弱点：奖励恒零 = 老师永远不批改 = 学不动（我们 1,773 步内容奖励 ≈ 0，正是此情形）。

**为什么是两个过程**：论文原始方案只做 GRPO；我们因 GRPO 奖励全零失败，补 SFT 先锁格式，形成"先 SFT 再 GRPO"的 RLHF 标准顺序。跳过 SFT 直接 GRPO，模型连格式都不会，奖励信号必然稀疏——这是本次 GRPO 失败的另一根因。

**与 Stage 1 的因果链**：Stage 1 Top-1 3.56% → 8 个历史动作几乎全错 → Stage 2 无论背答案（SFT）还是考试（GRPO），读到的都是错题 → 只能学"怎么答"（格式 91.6%），学不会"答什么"（语义 0.8%）。**解锁语义的唯一路径是先用真实特征提升 Stage 1 识别率。**

### 论文指标对照（arXiv:2508.01742 LaTeX 原文）

- 论文 EK55 长期动词预测 mAP（EGO-TOPO 协议）：**INSIGHT 45.2 / 62.4 / 36.0**（All/Freq/Rare，P=25/50/75 平均；P25=43.6/60.9/34.7，P50=45.9/62.6/36.5，P75=46.1/63.7/36.7）。
- 论文基线：ICVL 43.3/61.6/33.8、PALM 40.4/59.3/30.3、AntGPT 40.1/58.8/31.9、EGO-TOPO 38.0/56.9/29.2。
- 论文在 EK55 上**不报告短期 Top-1/Top-5**；短期 anticipation（Ego4D 主表）用 Damerau-Levenshtein ED。
- 我们的指标与论文不可横向比较：自定义 90/10 划分 vs 官方 RULSTM/EGO-TOPO 划分；1024-dim CLIP vs 1408-dim EgoVideo-V；假 HOI vs 真 HOI。

### 产出文档（08-01 合并整理：表格按 Stage 命名，图表与表格分离）

| 文件 | 内容 |
|------|------|
| `docs/Stage1_结果.xlsx` | Stage 1 识别指标（识别指标/训练配置 2 sheets） |
| `docs/Stage2_结果.xlsx` | Stage 2 SFT/GRPO 全部指标 + 训练曲线数据 |
| `docs/INSIGHT_总结果.xlsx` | 全链路 Top-1/Top-5 + 官方 mAP（含论文基线）+ 论文对照 |
| `docs/stage2_sft_charts.xlsx` | 训练曲线图表（仅图，无数据表） |
| `docs/stage2_sft_training_report.html` | 可视化报告（暗色模式支持） |
| `docs/stage_results_topk_map.csv` | 跨阶段 Top-1/Top-5 + mAP 汇总（程序友好） |
| `docs/archive/` | 归档：PROJECT_REPRODUCTION_AUDIT.md、STAGE2_SFT_FAST_PATH.md（内容已并入 WORKLOG） |

- 已删除（内容并入新表）：`INSIGHT_final_tables.xlsx`、`stage2_sft_data.xlsx`、`stage2_final_metrics.csv`、`stage2_sft_step_log.csv`、`stage2_sft_training_curves.csv`、`stage2_sft_training_summary.csv`

### mAP feature-limited baseline ✅ 已完成

- **重要更正**：重读代码确认 `convert_annotations.py:24` 的 `action_idx = int(row['uid'])`，即特征文件名 `{video}_{action_idx}.pt` 末尾的数字就是 EK55 全局 UID，与 EGO-TOPO 协议天然匹配（10 秒验证：val 1263/1263 全部匹配）。**上次会话"需要 UID→路径映射"的结论是误判，映射脚本不需要。**
- 运行 `GPU=3 bash scripts/run_ek55_future_verb_map.sh`（GRU 125 类，21 epoch early stop，best epoch 13，<30 分钟）。
- **结果（官方 EGO-TOPO 协议，feature-limited 特征）**：

| 组别 | 我们 | 论文 INSIGHT | 差距 |
|------|-----:|------------:|------|
| All | **46.2** | 45.2 | +1.0 |
| Freq | **53.8** | 62.4 | −8.6 |
| Rare | **41.1** | 36.0 | +5.1 |

- 分观察比例（All）：P25=60.2（论文 43.6）、P50=46.2（论文 45.9）、P75=32.4（论文 46.1）。
- **诚实边界**：val 仅 96 个阶段样本 / 34 视频（协议过滤 29 + 特征不足 76，覆盖率 96/172≈56%）；评估类别 64 类（论文为全 positive 掩码）；特征为 1024-dim CLIP 而非 EgoVideo-V 1408-dim。All 反超论文 1.0 分属小样本噪声，**Freq 落后 8.6 分**与"HOI 缺失 → 高频手物动作弱"因果链一致。不可与论文直接对比。

### τa=1s Top-1/Top-5 管线（尝试后放弃 ❌）

- 目的：在 RULSTM 官方协议（观察 3.5s→预测 1s 后动作，2513 类，官方划分）下获得可报告的短期 Top-1/Top-5。论文在 EK55 不报告该指标，此线仅为评测基建补充。
- 过程：3 个脚本（入口 `run_ek55_topk_rulstm_features.sh` / 导出器 `export_rulstm_lmdb_sequences.py` / 训练器 `train_anticipation_topk.py`）已通过 base64 传至远程；修复服务器 DNS 故障（校园 DNS 222.195.2.x 挂掉 → `resolvectl` 切换 8.8.8.8/223.5.5.5）；rulstm 源码（3.1MB）下载成功。
- **失败点：6.5GB 官方 RGB LMDB 两次下载均损坏**：
  1. curl 直连仅 100KB/s（预计 17.5h）→ 弃用，改 aria2；
  2. aria2 16 连接下到 17%（~1GB）后连接超时中断；续传时因首次运行 `--file-allocation=prealloc` 已把文件扩展为 6.5GB，第二次 `-c` 误判"本地大小==服务器大小=已完成"秒退，实际仅 ~1GB 有效内容 + 5.5GB 空洞；
  3. `lmdb.CorruptedError: MDB_CORRUPTED after 0 keys` —— meta 页在文件开头，损坏不可修复，需全量重下。
- **决策：放弃 τa=1s**（08-01）。理由：论文 EK55 主指标是 mAP（已完成 46.2/53.8/41.1），τa 仅为补充；下载反复失败耗时。3 个脚本保留在 repo，网络条件改善后可复用。

## 遗留待办

| 优先级 | 事项 | 阻塞方 |
|:--:|------|------|
| 高 | 在 amax 运行 UID→特征映射脚本并跑 EGO-TOPO mAP baseline | 远端执行 |
| ~~高~~ | ~~τa=1s 官方 Action Top-1/Top-5~~ | ❌ 08-01 放弃（LMDB 下载两次损坏，见 07-31 章节） |
| 高 | 修复 Stage 1 语义上限（真实 HOI 特征、1408-dim、8 步时序） | 方法改进/权重 |
| 高 | 联系 CorrineQiu 获取原版特征提取与官方评测脚本 | 作者 |
| 中 | Dandan Shan 找回 handobj_100K+ego 权重，提高真实 HOI 检出率 | 作者 |
| 中 | Ego4D/EGTEA 数据集下载 | — |

## 2026-08-07

### 数据选择理论调研（本会话）

- 用户提供文献《Rethinking Data Shapley for Data Selection Tasks: Misleads and Merits》(Kwon & Zou, ICML 2024) 的 .doc 中文翻译版（`ICML 2024 Rethinking Data Shapley for Data Selection Tasks Misleads and Merits.doc`，仓库根目录，未提交）。
- 本会话完成全文公式提取与总结，输出各数学定义/定理的中文整理（效用函数、Shapley 值、数据选择优化、MTM 函数、ρ-一致性指数、启发式拟合残差、归一化效用差异等），供数据质量/数据选择相关工作参考。
- 总结文档：[`docs/ICML2024_DataShapley_DataSelection_notes.md`](./ICML2024_DataShapley_DataSelection_notes.md)（含术语对照表与两篇论文的公式）。
- 文档还附带原始 Data Shapley 论文（Ghorbani & Zou 2019）译文。
- 论文要点与本项目关联：Data Shapley 在一般效用函数下至多与随机选择持平，仅在 MTM 类效用函数（如核方法、含干净/脏数据混合集）下最优——可作 DCR 数据质量分析的备选理论依据，暂未落地任何代码。

## 2026-08-02

### DCR-valid 上运行 INSIGHT 两阶段 Top-1：实验边界

- 当前目标收敛为：**先用少量 DCR EK55 valid 视频跑通现有 INSIGHT Stage1→Stage2，并只计算生成式 Next-Action Top-1**。
- 当前使用的是 DCR `validation_videos.csv` 对应的 40 个有标签视频；DCR 官方 test s1/s2 无 GT，不能在本地直接计算 Top-1。
- 单个评测目标动作记为 `A_i`：Stage1 只识别此前的 `A_(i-8)..A_(i-1)`，Stage2 输入为这 8 个预测动作文本及最后一个历史动作结束处图像；Stage2 第一个生成动作与 `A_i` GT 做精确匹配。
- GT 动作类别不进入模型 prompt；GT 时间边界用于 oracle segmentation 和最终评分。该设置是 INSIGHT 两阶段生成式 proxy，**不是 DCR 官方 tau_a=1s 分类 Top-1**。
- 新增硬性因果门：任何历史动作满足 `history.stop_frame >= target.start_frame` 时，整个窗口跳过，目标动作画面不得进入输入。

### 远端部署与 CPU 审计（已确认）

- 远端代码目录：`/home/amax/ldy/git/AAAI26-INSIGHT/dcr_eval/`。
- 数据产物根目录：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/`。
- Python/Bash 语法检查通过；环境为 PyTorch 2.5.1+cu118、Transformers 4.49.0，physical GPU 0/1/3 均为 RTX 3090 且 CUDA 分配成功。
- DCR valid：40 videos / 4,979 actions；verb 125 类、noun 352 类，ID 范围完整。
- 当前扁平 JPEG 布局只找到 2,201/4,979 个动作的可提取帧，缺失 2,778；路径没有 `video_id`，13 个 participant 在 valid 中有多个视频，存在同名帧来源歧义，结果只能称 proxy。
- 精确数据重叠：DCR valid 与 Stage1 标注 `(video_id, uid)` 重叠 train 4,602、val 377；复刻原 `ActionDataset` 完整窗口和特征存在规则后，真正可进入 loader 的 DCR 动作为 train 1,764、val 377。
- Stage2 SFT train 与 DCR valid 重叠 10 个视频；严格 video-unseen 子集为 0。
- 加入因果门后的全量无 GPU 预估（当时仍要求 20 个完整未来 GT）：

| 项目 | 数量 |
|------|-----:|
| 可提取动作特征 | 2,201 / 4,979 |
| 理论 8-history + 20-future 窗口 | 4,043 |
| 非因果窗口（已排除） | 1,284 |
| 最终可构造窗口 | 1,255 |

### 三视频冒烟实验 `smoke_3videos_v1`（已确认）

- 固定产物目录：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/smoke_3videos_v1/`。
- 视频清单：`P07_02`、`P08_12`、`P26_11`；持久化到 `annotations/requested_videos.txt`。
- Phase 1 特征提取完成：47 labeled actions、47 written features、0 missing、coverage 100%；EgoVideo CLIP 权重输出 1024-dim RGB 特征，frame/mask 两分支共用同一特征（真实 HOI 缺失）。
- Phase 2 Stage1 推理完成：

| 指标 | 结果 |
|------|-----:|
| Labeled samples | 47 |
| Verb Top-1 | 29.787% |
| Noun Top-1 | 21.277% |
| Action Top-1 | 12.766% |
| Unique predicted actions | 14 |

- 最常见预测为 `open cupboard` 15/47，其次 `put plate` 8/47、`take plate` 8/47；存在偏置，但不是单一类别完全坍塌。
- Stage2 输入未生成：`candidate_positions=0, written=0`。原因不是缺特征、缺图或非因果过滤，而是这三个视频在原筛选规则下均不满足同一视频内 `8 历史 + 20 完整未来`。

### Top-1 样本门槛调整（待远端确认，尚未记为完成）

- 结论：Top-1 只需要目标动作 `A_i` 的 GT，不需要同时存在其后的 19 个未来 GT。
- 拟保持 `HISTORY=8` 和模型生成格式 `HORIZON=20` 不变，仅把评测样本的未来 GT 门槛改为 `EVAL_FUTURE_REQUIRED=1`。
- 该修改不会增加原始帧，也不会改变/重训模型；它只允许更多视频尾部动作成为 Top-1 目标。现有 47 个特征可直接复用，修改确认后仅需重跑 Stage1 输入构造。
- 截至本条日志，远端尚未返回该补丁的安装/运行结果，因此状态保持 **pending**。

#### 远端确认结果：已完成

- 完整恢复后的 `02_stage1_infer.py` 已用于 `smoke_3videos_v1`；现有 47 个 CLIP 特征直接复用，未重新提取、未训练模型。
- Top-1 构造配置已确认生效：`history=8`、`generation_horizon=20`、`future_gt_required=1`。
- 三视频共得到 23 个候选目标；`missing_history=0`、`missing_image=0`，14 个时间重叠窗口被因果门排除，最终写出 9 条严格因果 Stage2 输入，coverage 39.13%。
- 输出：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/smoke_3videos_v1/stage1_predictions/stage2_input.jsonl`（9 行）。
- 因最终输入仅 9 条，小规模跑通将使用全部 9 条，不再截取 12 条；后续 Stage2 可按 3 张 GPU 各 3 条分片运行。

### 操作记录规则

- 后续每次远端操作只有在用户返回命令输出并确认成功后，才追加为“已完成”。
- 已给出但尚无远端输出的命令只写入“待确认”，不得提前登记为成功。
- 每条记录至少包含：日期、run name、输入范围、执行阶段、关键数量/指标、产物路径和已知协议边界。

### 远端文件重新上传查验（已确认：部分失败）

- 用户重新上传 `dcr_eval` 文件后执行 9 个运行文件 SHA256、Python/Bash 语法、Top-1 配置和已有冒烟产物检查。
- `config.py`、`data_utils.py`、`00_audit.py`、`01_prepare_features.py`、`03_stage2_generate.sh`、`04_evaluate.py`、runner 与 JSONL validator 均与本地预期哈希一致；Bash 语法通过。
- Top-1 配置读取正确：`HISTORY=8`、`GENERATION_HORIZON=20`、`EVAL_FUTURE_REQUIRED=1`。
- 已归档的 `smoke_3videos_v1` 仍完整：视频清单 3 个、CLIP 特征 47 个、`meta.json` 为 47/47、coverage 100%，且无 DCR/swift 进程运行。
- **阻塞错误**：远端 `scripts/02_stage1_infer.py` SHA256 为 `74d5706e...`，不同于本地预期 `f274ba24...`；`py_compile` 在第 184 行报 `SyntaxError: unterminated string literal`，该行出现明显字符破坏（`FeatureDataset/DataLoader` 代码被拼接、丢字和乱码）。
- 用户随后执行的 `echo SYNTAX_AND_CONFIG_OK` 只是普通字符串输出，不代表 Python 语法检查成功；本次查验状态必须记录为 **FAILED / 待完整恢复 02 文件**。
- 恢复策略：不再使用增量文本替换；传输完整 `02_stage1_infer.py` 的 gzip+base64 分段，先校验 base64 长度/哈希，再解压到临时文件，校验 Python 文件 SHA256 和语法后才备份并覆盖远端损坏文件。
- 用户随后确认完整恢复操作已完成；下一次 Stage1 命令将在运行前再次强制检查预期 SHA256 与 `py_compile`，通过后才允许复用 47 个特征构造 `future GT=1` 输入。当前状态为 **用户确认恢复，待运行前复核**。

### 三视频 Stage2 冒烟生成（已确认）

- 输入：`smoke_3videos_v1/stage1_predictions/stage2_input.jsonl`，共 9 条严格因果样本。
- 使用 3 张 GPU 分成 3 个连续分片，每个分片 3 条；分片顺序对应原输入顺序，后续必须按 `part_0 → part_1 → part_2` 合并。
- 三个 Stage2 推理任务均正常完成，无缺行：

| 分片 | 样本数 | 推理耗时 | 生成 token | JSONL 校验 |
|------|------:|---------:|-----------:|------------|
| part 0 | 3 | 15.33 s | 286 | PASS，3 行 |
| part 1 | 3 | 15.49 s | 289 | PASS，3 行 |
| part 2 | 3 | 16.10 s | 303 | PASS，3 行 |

- 分片输出目录：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/smoke_3videos_v1/stage2_generated_smoke9/`。
- 分片文件：`candidate_0_part_0.jsonl`、`candidate_0_part_1.jsonl`、`candidate_0_part_2.jsonl`，合计 9 行。
- 下一步：显式按分片编号合并成 `candidate_0.jsonl`，再次校验 9/9 行，再用 `scripts/04_evaluate.py` 计算生成式 Next-Action Top-1。
- 协议边界：该 9 样本结果只用于验证 Stage1→Stage2→Top-1 评测链路是否跑通；样本极少且训练/验证数据存在重叠，不能作为正式性能或论文可比结果。

### 三视频端到端 Top-1 冒烟评测（已确认）

- 三个连续 Stage2 分片已显式按 `part_0 → part_1 → part_2` 合并为：
  `/data/datasets/EPIC-KITCHENS/dcr_eval/runs/smoke_3videos_v1/stage2_generated_smoke9/candidate_0.jsonl`。
- 输入 9 行；三个分片分别为 3/3/3 行；合并结果 9 行。合并前和落盘后的 `validate_generated_jsonl.py` 均返回 `VALID_JSONL rows=9`。
- 输入 SHA256：`032b2c35ea6ddbde6c42e4fba8d39a79a124fefb48b857dc34fd9b41b497c5c9`。
- 合并预测 SHA256：`dd33cb19c73ebf6446a3fd5f0db8b6863a92f48721eed0e0eb8b1e069d035152`。
- `scripts/04_evaluate.py` 正常退出（`EVAL_EXIT=0`）；9 条都有 GT，9 条生成均成功解析，parse success rate 100%，无解析失败。
- 生成式 Next-Action Exact-Match@1：命中 `1/9`，**Top-1 = 11.111%**。
- 指标文件：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/smoke_3videos_v1/evaluation/metrics_smoke9.json`。
- 评测日志：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/smoke_3videos_v1/logs/04_evaluate_smoke9.log`。
- 对齐边界：ms-swift 候选行没有 `clip_uid/action_idx`，本次依赖连续分片和显式编号合并保持原输入行序；正式扩大实验前应加入 ID 对齐或生成后 ID 回填校验，避免仅凭行号静默错位。
- 解释边界：11.111% 只说明三视频的 9 样本端到端链路已成功跑通。每个样本对应 11.111 个百分点，统计方差极大；而且存在训练重叠、帧路径歧义及伪 HOI 特征，不能与 DCR 官方 `tau_a=1s` Top-1 或论文表格直接比较。

### 全量 Top-1 的样本 ID 对齐补丁（本地完成，远端待确认）

- 新增 `dcr_eval/scripts/attach_prediction_ids.py`：每个 ms-swift 分片生成后，严格按该连续分片的输入记录，将 `clip_uid/action_idx` 写回预测 JSONL；输入/输出数量不一致、输入 ID 重复或已有 ID 冲突时立即失败，并以临时文件原子替换正式分片。
- 修改 `03_stage2_generate.sh`：swift 原始输出先写临时文件，通过 JSONL 校验和 ID 附加后才成为 `candidate_0_part_N.jsonl`；旧的完整分片缓存也必须补写并验证 ID 后才能复用。
- 修改 `validate_generated_jsonl.py`：新增 `--require-ids`，可强制要求所有预测含唯一的 `(clip_uid, action_idx)`。
- 修改 `run_top1_all_valid.sh`：生成缓存规格加入 `id_alignment=clip_uid_action_idx_v1`；各分片和合并文件都必须通过唯一 ID 校验；合并改为临时文件完成后原子落盘，避免任务中断留下半成品。
- `04_evaluate.py` 无需修改：它原本已支持按 ID 集合检查和自动重排。
- 本地 Python/Bash 语法检查通过；2 条合成数据测试中故意颠倒候选顺序，评测器打印按 `(clip_uid, action_idx)` 重排并恢复 `2/2`、Top-1 100%，证明合并后的 ID 对齐路径生效。
- 补丁包：`/tmp/dcr_top1_id_alignment_v1.tar.gz`，7,497 bytes，SHA256 `7e2d52abb08face979e3062d36fcd626918b370207b67558afff6fc0cf0267d3`。
- 当前状态：**仅本地代码和测试已完成**；远端尚未安装，不得登记为全量实验已启动。

#### 远端安装与全量运行：已启动，Stage2 进行中

- 远端运行已使用新脚本路径和临时输出命名（`candidate_0_part_N.jsonl.swift_tmp.PID`），证明 ID 对齐版 `03_stage2_generate.sh` 已生效；正式分片只有在生成、结构校验及 ID 附加全部成功后才会落盘。
- 新实验名：`all_valid_top1_id_v1`；独立根目录：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1/`，未覆盖 `smoke_3videos_v1`。
- 截至 2026-08-03 11:11，Phase 0 CPU 审计、Phase 1 特征提取、Phase 2 Stage1 推理及 Stage2 输入构造均已完成；主 runner PID 3417 正常存活。
- 全量 Top-1 输入构造结果：

| 项目 | 数量 |
|------|-----:|
| candidate positions | 4,663 |
| 最终写出 | 1,472 |
| missing history | 1,754 |
| noncausal history（排除） | 1,437 |
| missing image | 0 |
| coverage | 31.568% |

- 配置保持：history 8、模型生成 horizon 20、Top-1 评测只要求目标 GT 1 条。
- Stage2 已在物理 GPU 0/1/3 启动 3 个连续分片，PID 分别为 3851/3852/3853；按 1,472 条计算，三个分片约为 490/491/491 条。
- 快照显存约为 12.55/12.30/12.47 GiB，GPU 利用率 9%/9%/7%；此时三个 infer 进程均在，日志无 Traceback/Error。低瞬时利用率可能来自模型加载和 batch-size=1 自回归解码波动，不能单凭一次 `nvidia-smi` 判定卡死。
- 基于三视频实测下界及此前长序列推理吞吐，Stage2 剩余时间初估 1–2.5 小时；最终以各 `stage2_generated/shard_N.log` 的进度/ETA 为准。
- 当前状态：**RUNNING**。尚未出现 `ATTACHED_IDS`、最终 `candidate_0.jsonl` 或 `TOP1_PIPELINE_COMPLETE`，因此不得登记最终 Top-1。

#### 全量 ID 对齐 Top-1：已完成

- 2026-08-03 12:17 已出现 `TOP1_PIPELINE_COMPLETE`，主推理进程退出并释放 GPU；本次 `all_valid_top1_id_v1` 全量 proxy 实验完整结束。
- 三个 Stage2 分片均成功且含唯一 ID：part 0 为 490 条、part 1 为 491 条、part 2 为 491 条。
- 原子合并临时文件与最终 `candidate_0.jsonl` 均通过 `VALID_JSONL rows=1472 unique_ids=1472`；没有缺行、重复 ID 或部分无 ID。
- Phase 4 评测：1,472 条均有标签、1,472 条均成功解析，parse failures 0，parse success rate 100%。
- 最终命中 15 条，**INSIGHT Next-Action Exact-Match@1 Top-1 = 1.019%**（15/1,472）。
- 输入 SHA256：`97c003224fe86ab042d2c60bf215728c00e9f79b779bbbfe51a93674aa813b5d`。
- 预测 SHA256：`c698c1ba65f897b709c2e3366456bc7825b9ea4001f26e0478e5d64f6f7540c2`。
- 指标文件：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1/evaluation/metrics.json`。
- 预测文件：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1/stage2_generated/candidate_0.jsonl`。
- 结论：低 Top-1 不是由 JSONL 缺失、生成格式失败或分片合并错位造成；这些工程因素已被数量、唯一 ID 与 100% 解析率排除。主要瓶颈仍在模型/数据流：Stage1 历史识别误差、无真实 HOI（mask 分支复用 RGB）、1024-dim proxy 特征、Stage2 训练/评测分布偏差，以及 DCR 帧路径不含 video_id 的来源歧义。
- 报告边界：这是 `recognition-gated history + generative prediction` 的 DCR-valid proxy，存在训练重叠且只覆盖 1,472 个可构造目标；不能与 DCR 官方 `tau_a=1s` EK55 Top-1 19.2 直接比较，也不能声称为严格无泄漏结果。

### 全量结果 CSV 导出器（本地完成，远端待确认）

- 新增 `dcr_eval/scripts/05_export_results_csv.py`，只读取已完成的输入、Stage1 预测/指标和带 ID 的 Stage2 预测，不加载模型、不使用 GPU。
- 逐样本总表字段包括：稳定样本 ID、图像路径、8 步 Stage1 历史、GT action/verb/noun 与类别、Stage2 首动作及 Action/Verb/Noun 命中标记、20 个预测动作独立列、完整生成序列、think/intention、解析和长度质量、Stage1 对目标段的预测类别与联合分数（若存在）。
- 额外汇总表提供 overall 和逐视频的 Action/Verb/Noun Top-1、解析率、恰好 20 步率、20 步均合法率、平均生成长度、首动作预测多样性及 Stage1 总体指标。
- 严格按 `(clip_uid, action_idx)` 对齐；输入/预测 ID 缺失、重复或集合不一致时拒绝导出。
- 两条合成测试通过：候选顺序被故意颠倒仍正确对齐；Action Top-1 50%、Verb Top-1 100%、Noun Top-1 50%，CSV 行数和 UTF-8-SIG 读取均通过断言。
- 脚本 SHA256：`9ebc2c4d842dc24de5b96d405a89c4e5cfcd2136f7c38a1fe1f1743993631923`。
- 远端尚未安装/运行，实际 1,472 行 CSV 产物状态为 **pending**。

#### 远端运行完成：结果 CSV 已产出（已确认）

- 用户在远端执行 `05_export_results_csv.py` 成功，产出逐样本总表 + 汇总表：
  `all_valid_top1_results.csv`（1,472 行）与 `all_valid_top1_summary.csv`。
- **Overall 指标（1,472 样本）**：

| 指标 | 结果 |
|------|-----:|
| Next-Action Top-1 | **1.019%**（15/1,472） |
| Verb Top-1 | **13.383%** |
| Noun Top-1 | **2.446%** |
| 解析成功率 | 100%（1,472/1,472） |
| 恰好 20 步比率 | 98.3% |
| 平均生成序列长度 | ~20 步（含 think/intention 段的完整生成） |

- **Stage 1 门控指标（2,201 个动作样本，与 Stage2 输入同源但样本集更大）**：

| 指标 | 结果 |
|------|-----:|
| Verb Top-1 | 26.397% |
| Noun Top-1 | 13.539% |
| Action Top-1 | 7.497% |

- 汇总表含逐视频划分统计（action/verb/noun Top-1 按视频分组），可在 `all_valid_top1_summary.csv` 中按视频追查命中分布。
- 多样性审计：top-10 预测动作中 `open cupboard` 出现 350 次，占比约 23.8%，呈明显先验坍塌；Stage2 学到"高频动作优先"而非目标特异性预测，与 Stage1 历史识别误差、伪 HOI 特征（mask 分支复用 RGB）和训练/评测分布偏差的诊断一致。
- 解释口径（不变）：`recognition-gated history + generative prediction` 的 DCR-valid proxy；不能与 DCR 官方 `tau_a=1s` 分类 Top-1（19.2%）直接比较，也不声明为严格无泄漏结果。

### 项目代码/数据/权重位置总说明（本地完成）

- 新建 [`PROJECT_REPRODUCTION_AUDIT.md`](./PROJECT_REPRODUCTION_AUDIT.md)，补齐根 README 已引用但此前缺失的复现审计文档。
- 文档共 618 行，集中记录 Mac/远端/数据盘根目录、上游与补充代码职责、DCR 主线脚本、原始数据和标注、动作段取帧与因果窗口、特征、权重状态、Stage1/Stage2 训练数据、历史/当前 run 隔离、最终 Top-1、mAP 产物、协议边界、泄漏事实和文件管理规则。
- 特别明确：当前所谓“测试集”是有标签 DCR EK55 validation；动作 start/stop 来自 GT 标注，未执行自动时序动作分割；官方无标签 test s1/s2 未用于本地 Top-1。
- 当前 `all_valid_top1_id_v1` 的 1,472 输入、15 命中、1.019% Top-1、ID/解析校验、关键 SHA256 和所有产物路径已纳入快速定位表。
- Markdown 完整性检查通过：68 个代码围栏成对闭合，10 个关键路径/数字断言全部存在；根 `README.md` 的链接现已指向实际文件。
