# INSIGHT 项目交接说明（环境配置 + 项目现状）

> **论文**: Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation (AAAI 2026)
> **官方仓库**: https://github.com/CorrineQiu/INSIGHT
> **本项目仓库**: https://github.com/EsunLee/AAAI26-INSIGHT
> **最后更新**: 2026-10-09
>
> 本文档是本仓库中**唯一**的说明文档，同时承载"环境配置"与"项目现状"两部分内容。
> 内容依据服务器工作日志（最后记录 2026-08-14）与仓库代码整理；**2026-08-14 之后服务器上的新进展本文档未记录**。
>
> ⚠️ 每次配置或实验状态变更后请同步更新此文档。
>
> **仓库内容约定**（`.gitignore`）：只提交代码 (`*.py`/`*.sh`)、根目录 `README.md`、本文档和 `assets/` 配图。
> 数据集、模型权重、训练输出、日志、办公文档，以及所有其它说明性文档（`docs/`、各子目录 `README.md`）均不入库，但**本地保留**，见第 12 节。

---

## 1. 远程机器

| 项目 | 值 |
|------|-----|
| 访问方式 | 远程桌面 (TeamViewer) |
| 系统 | Linux (Ubuntu) |
| 主机名 / 用户名 | amax / amax |
| 项目路径 | `~/ldy/git/AAAI26-INSIGHT`（即 `/home/amax/ldy/git/AAAI26-INSIGHT`） |

### GPU 硬件

| 项目 | 值 |
|------|-----|
| GPU | 4× NVIDIA GeForce **RTX 3090** (24 GB 显存/卡) |
| Driver | 495.29.05 |
| CUDA | 11.5 (nvidia-smi) |
| **物理 GPU 2** | ⛔ **硬件故障**（Xid 79 fallen off bus、Xid 48 L2 DBE），禁止使用 |
| 可用卡 | **固定使用物理 GPU 0 / 1 / 3** |

> ⚠️ ms-swift 不识别 UUID 形式的 `CUDA_VISIBLE_DEVICES`，必须用数字索引（如 `CUDA_VISIBLE_DEVICES=0,1,3`）。

### 磁盘

| 挂载点 | 容量 | 状态 |
|--------|------|------|
| `/data` | 3.6 T | 剩余约 176 G |
| `/home` | 244 G | 曾满盘（已清理，约 25 G 可用） |

---

## 2. Conda 环境 `/data/conda_envs/insight_env`

| 项目 | 值 |
|------|-----|
| 激活命令 | `conda activate /data/conda_envs/insight_env` |
| Python | 3.10（`/data/conda_envs/insight_env/bin/python3`） |
| PyTorch | 2.5.1+cu118 |
| Transformers | 4.49.0 |
| **ms-swift** | **3.4.1**（Stage 2 训练/推理框架） |
| trl | 0.17.0（与 ms-swift 3.4.1 配套） |
| qwen-vl-utils | **0.0.8**（新版本有 IMAGE_FACTOR 兼容问题，已降级） |
| vllm | 已 `pip uninstall`（新版依赖 CUDA 13，495 驱动不可用） |
| swift CLI | `/data/conda_envs/insight_env/bin/swift` |
| pip 镜像 | `https://pypi.tuna.tsinghua.edu.cn/simple` |

> **注意**: 系统 CUDA 是 11.5，PyTorch 用 cu118 编译。两者兼容，但编译 CUDA C 扩展时可能有 nvcc 版本问题（见第 3 节 Hand Object Detector）。

### ms-swift 版本演进（踩坑记录）

1. 最初 GitHub editable 安装 **4.2.0.dev0** → 失败。
2. 原因：`ms-swift v4.2` → `trl>=0.20` → `vllm` → `PyTorch 2.6` → 要求 CUDA Driver ≥ 525，而本机是 495，依赖链无法满足。
3. 中途曾新建独立环境 `/data/conda_envs/stage2_env`（Python 3.10 + PyTorch 2.6）尝试，同样失败；该环境**是否仍存在文档未记录**，使用前请先确认。
4. 最终方案：回退在 `insight_env` 内安装 **ms-swift 3.4.1 + trl 0.17.0**。
   - 3.4.1 保留了 `swift.llm`，因此仓库内的插件代码可直接使用；
   - GRPO/SFT 训练**不需要 vllm**，推理用 `--infer_backend pt` 即可。

### 奖励插件部署位置（重要）

仓库 `CognitiveReasoning/` 下的插件需**复制**到框架目录才能生效：

```bash
# 目标目录
/data/conda_envs/insight_env/lib/python3.10/site-packages/swift/plugin/
# 复制 my_rewards_intention.py 与 plugin_in_log_intention.py 过去，并修改 3 处：
#   1) 去掉 assert 语句末尾的逗号（assert ..., 语法）
#   2) 导入路径改为 from swift.plugin.my_rewards_intention import ...
#   3) MiniLM 模型路径改为 /data/pretrain_model/all-MiniLM-L6-v2
```

### 常用环境变量（脚本中已设置）

```bash
export HF_HUB_OFFLINE=1                    # 离线，避免联网卡住
export MAX_PIXELS=50176                    # 控制 Qwen2.5-VL 图像 token 数
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export PYTHONDONTWRITEBYTECODE=1
```

### 训练配置约定

| 项目 | 值 |
|------|-----|
| 分布式 | DeepSpeed **ZeRO-2** |
| 精度 | **float16**（bf16 被 trl 的 GRPOConfig 拒绝） |
| 微调 | LoRA r=8 / alpha=32 / dropout 0.05，`--target_modules all-linear` |

> 仓库根目录的 `requirements.txt` 是上游原始大列表（含 `deepspeed==0.14.5`），与实际安装版本不完全一致；实际环境以上表为准。

---

## 3. 预训练权重 `/data/pretrain_model/`

### 目录总览

```
/data/pretrain_model/
├── all-MiniLM-L6-v2/              ✅ 文本嵌入模型（reward 里的意图相似度）
│   ├── model.safetensors          (~90 MB)
│   ├── tokenizer.json / vocab.txt
│   └── config.json
│
├── EgoVideo/                      ✅ 视频帧特征提取器
│   ├── checkpoints/
│   │   ├── base_best.pt
│   │   └── large_best.pt
│   ├── eccv-2022/                 # 原始代码
│   └── EgoVideo.zip               # 原始压缩包
│
├── Hand_Object_Detector/          ⚠️ 只有 handobj_100K，缺少 ego 优化版
│   ├── clone 自 larsrpe/hand_object_detector_pytorch (pure-pytorch fork)
│   ├── handobdet/models/res101_handobj_100K/pascal_voc/
│   │   └── faster_rcnn_1_8_89999.pth    # 手-物体检测器权重
│   └── data/pretrained_model/
│       ├── resnet101_caffe.pth           # Caffe 格式 ResNet-101 backbone
│       └── resnet101-5d3b4d8f.pth       # PyTorch 官方 ResNet-101 (备用)
│
├── Qwen2.5-VL-7B-Instruct/        ✅ Stage 2 大模型
│   ├── model-00001 ~ 00005-of-00005.safetensors
│   ├── config.json
│   ├── tokenizer.json / merges.txt
│   ├── chat_template.json
│   └── generation_config.json
│
└── SAM2/                          ✅ 分割模型
    └── sam2.1_hiera_large.pt      (~850 MB)
```

### 缺失权重

| 文件 | 用途 | 影响 | 状态 |
|------|------|------|:--:|
| `faster_rcnn_1_8_132028.pth` | handobj_100K+ego 检测器 | ego 数据上物体检测 AP 41→67 | 等 Dandan Shan，截至最新日志无新回复 |

### Hand Object Detector 细节

- **代码来源**: `larsrpe/hand_object_detector_pytorch`，非原版 `ddshan/hand_object_detector`
- **切换原因**: 原版需编译 CUDA C 扩展，系统 CUDA 11.5 的 nvcc 与 PyTorch 2.5 的 C++17 代码不兼容（`TypeList.h: 'tuple' was not declared` 编译错误）
- **pure-pytorch fork**: 所有 CUDA 算子用 `torchvision.ops` 替换，无需编译，权重完全兼容
- **代码修改**: `handobdet/hand_object_detector.py` 第 44 行，`faster_rcnn_1_8_132028.pth` → `faster_rcnn_1_8_89999.pth`
- **验证**: 模型加载成功，demo.py 可运行（缺测试图片但非 bug）

### Dandan Shan 邮件往来（获取 ego 版检测器权重）

| 项目 | 值 |
|------|-----|
| 作者邮箱 | dandans@umich.edu |
| 我方邮箱 | esun12191@gmail.com |
| 首次发信 | ~2026-07-13 |
| 回复 | 学校账号过期导致 Google Drive 失效，正在联系系里找回 |
| 状态 | 等对方主动回复；截至最新日志**无新进展** |

---

## 4. 数据集

### 4.1 EPIC-Kitchens-55 (EK55) ✅ 下载与解压均已完成

| 项目 | 值 |
|------|-----|
| 状态 | ✅ 完成（2026-07-16 启动，07-19 完成） |
| 结果 | **432/434 个 tar，共 228 GB，约 1,251,563 张 JPEG 帧** |
| 未下载 | 2 个 P32 test 文件（多次断连，永久失败），导致 P22/P24/P31 部分帧缺失（仅 warning，不影响训练） |
| 路径 | `/data/datasets/EPIC-KITCHENS/{Pxx}/rgb_frames/frame_{frame:010d}.jpg` |

> ⚠️ **重要陷阱**：解压后所有帧**平铺在 participant 级 `rgb_frames/` 目录下，没有 video_id 层级**。
> 一个 participant 含多个视频，同号帧来源歧义 → 所有结果必须标注为 **frame-path-ambiguous proxy**。

**标注数据**：

| 路径 | 说明 |
|------|------|
| `/data/datasets/EPIC-KITCHENS/annotations/EPIC_train_action_labels.csv` | 官方原始标注 |
| `/data/datasets/EPIC-KITCHENS/insight_annotations/{train.json,val.json}` | `feature_extraction/convert_annotations.py` 转换：**28,472 actions / 272 clips**，随机 10% 作 val |

> verb 0–124 共 **125 类**，noun 1–351 共 **352 类**；`action_idx` 即 EK55 全局 uid（因此**不需要**额外的 UID→特征映射步骤）。

### 4.2 DCR 项目数据（当前评测数据源，只读）

`/data/sp/projects/csj/DCR-main/`（老师的 DCR 项目，**只读不写**）：

| 文件 | 用途 |
|------|------|
| `data/EK55/validation_videos.csv` | 40 个有标签的 valid 视频 |
| `data/EK55/EPIC_train_action_labels.pkl` | 28,472 条标注 |
| `data/EK55/EPIC_verb_classes.csv` / `EPIC_noun_classes.csv` | 类别表 |
| `data/EK55/EPIC_test_s1/s2_timestamps.pkl` | 无标签，未使用 |
| `data/feature/EK55_RGB_TSM/data.mdb` | TSM 2048 维特征，19.3 GB，**未使用**（与 1024 维权重的维度不兼容） |

### 4.3 Ego4D / EGTEA — ⚠️ 均未开始

> 二者仍是待办（中优先级）。Ego4D 是论文 ED 主表指标所需，本地无数据。截至最新日志无进展。

---

## 5. 特征提取产物

均位于 `/data/datasets/EPIC-KITCHENS/features/`。

| 产物 | 路径 | 格式 |
|------|------|------|
| frame_features | `frame_features/{train,val}/{clip_uid}_{action_idx}.pt` | 每 action 采样 4 帧经 **CLIP ViT-L/14** 编码后平均的 **1024 维 float 向量**；train 14,217 + val 1,263 = **15,480 个 .pt** |
| mask_features | `mask_features/{train,val}/...` | 同结构同数量 |
| cooccurrence | `cooccurrence_matrix.pt` (690 KB) | dict：`P_n_given_v` / `P_v_given_n` / `O` / `raw_cooccurrence` |

> ⚠️ **mask 流已退化**：2026-07-30 逐文件审计发现 mask_features 与 frame_features 几乎完全相同
> （train 14,179 对中 **13,891 对完全一致**，仅 288 对不同；val 1,263 对全部相同）。
> 早期"91 条真实 HOI"不再作为最终口径 —— **mask 实际不携带独立 HOI 信号**，双流结构等同于单流。

> ⚠️ **LMDB 线路已放弃**：RULSTM 官方 6.5 GB TSN-RGB LMDB 两次下载均损坏（`MDB_CORRUPTED`），
> 2026-08-01 决定放弃 τa=1s 管线，仓库中 `export_rulstm_lmdb_sequences.py` 等脚本仅为留档。

**DCR run 独立特征**：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1/features/val/{video_id}_{uid}.pt`（同为 1024 维 CLIP，mask=frame 副本）。全量 4,979 个 valid 动作中仅 **2,201 个可提取**（覆盖率低源于帧缺失）。

---

## 6. 项目代码结构

```
~/ldy/git/AAAI26-INSIGHT/
├── HandObject/               # Stage 1: 手-物体语义动作识别
├── CognitiveReasoning/       # Stage 2: 认知推理 (Qwen2.5-VL + SFT/GRPO)
├── feature_extraction/       # 标注转换与特征提取
├── evaluation/               # 评测脚本（mAP / 审计 / 曲线）
├── dcr_eval/                 # ★ 当前主线：端到端 DCR 评测封装
├── scripts/                  # 各实验的启动 shell
├── assets/                   # README 图片
├── README.md / requirements.txt / LICENSE
└── .claude/config.md         # 本文件
```

### 6.1 `HandObject/`（Stage 1，8 个 .py）

| 文件 | 作用 | 入口 |
|------|------|------|
| `main.py` | **Stage 1 训练入口**。argparse 无参数，路径硬编码在 `main()` 内（frame/mask/annotation/checkpoint/cooccurrence + batch 8、epoch 40、lr 8e-5、input_dim 1024、verb 125/noun 352）。流程：窗口采样 → 双流 Transformer → verb/noun 头 → 共现矩阵重排 → `best_model.pth` + `test_predictions.csv` | `python main.py` |
| `model.py` | `ActionRecognitionModel`：frame/mask 双流 gated MLP + 4 层 Transformer + verb/noun 头（`input_dim` 已参数化） | — |
| `dataset.py` | `ActionDataset`：按 clip 窗口读取 `{clip_uid}_{action_idx}.pt`（window_size=8、stride 1）；test split 需 `fho_lta_test_unannotated.json` | — |
| `train.py` | `train_one_epoch` + `validate`（无共现重排的朴素验证） | — |
| `rebuild_stage1.py` | **checkpoint 重建脚本**（原 40-epoch 权重损坏后的补救），路径硬编码。实际 6 epoch 早停，产出 `stage1_rebuilt_checkpoint/best_model.pth`（89 tensors，可加载），日志 `/tmp/stage1_rebuild_training.log` | `python rebuild_stage1.py` |
| `export_stage1_predictions.py` | 导出每段唯一 Top-5 联合动作（含论文 Eq.5 语义先验）→ `stage1_{split}_predictions.jsonl` | `--frame-features --mask-features --annotations --checkpoint --cooccurrence --output-dir --splits train val` |
| `evaluate_topk.py` | **Stage 1 Top-k 评测**：verb/noun/action raw/action+先验 共 8 项 micro accuracy | 同上参数 |
| `train_anticipation_topk.py` | 独立的 **2513 类** EK55 短期 anticipation 分类器（RULSTM 协议、τa=1s、14/11 点序列）。**该线路已暂停** | `--manifest-dir --features-dir --output-dir --input-dim 1024 --epochs 40` |

### 6.2 `CognitiveReasoning/`（Stage 2）

| 文件 | 作用 |
|------|------|
| `plugin_in_log_intention.py` (31 KB) | ms-swift 外部 reward 插件合集（MathAccuracy、Code 系、`ActionIntentContLin/Pow`、`ExternalIntentScoreORM/RAW`、`CustomizedRMPlugin`/`QwenLongPlugin`），已适配 swift 3.4.1 的 `swift.plugin` 导入 |
| `my_rewards_intention.py` | 核心 `ActionIntentReward(ORM)`：`R = s_len*(0.85*s_cont + 0.05*s_int + 0.05*s_lang + 0.05*s_fmt)`（长度 / 动作 Damerau 距离连续分 / MiniLM 意图相似度 / 标签顺序 / 纯英文）。action pair 正则已修为 `^[a-z0-9_:-]+ [a-z0-9_:-]+$`（修复旧的"必须带 *" bug） |
| `cal_ED.py` | 归一化 Damerau-Levenshtein 编辑距离（action/verb/noun），oracle 取多候选中最好；历史脚本 |
| `evaluate_generated_topk.py` / `_compact.py` | **生成式 5 候选评测**。Top-1 = candidate_0（温度 0）首动作精确匹配；Top-5 实为 5 个候选首动作的 Hit@5（采样口径，**非 ranked Top-5**）。另输出 oracle-min ED 与格式指标 | 
| `run_external_reward_func_7B_qwen_intention.sh` | 上游原始 GRPO 启动脚本（**历史留档**，路径为占位符，已被服务器上的参数化命令替代） |

生成式评测入口示例：
```bash
python evaluate_generated_topk_compact.py \
  --ground-truth <val.jsonl> --predictions cand*.jsonl \
  --horizon 20 --output metrics.json
```

### 6.3 `feature_extraction/`

| 文件 | 作用 | 状态 |
|------|------|------|
| `convert_annotations.py` | EK55 CSV → insight JSON（→ `/data/datasets/EPIC-KITCHENS/insight_annotations`） | ✅ |
| `extract_features.py` | JPEG → HandObjDet + SAM2 + CLIP → `.pt`（DATA_DIR 等硬编码，`SAMPLES_PER_ACTION=4`、`SKIP_EXISTING=True`；含 EgoVideo checkpoint 键名映射到 HF `CLIPVisionModel` 的核心逻辑） | ✅（但 HOI 检出仅 0.6%） |
| `build_stage2_from_stage1.py` | Stage1 预测历史 → Stage2 数据（history 8 / horizon 20），产出 train 10,748 / val 876（含 missing_history 计数） | ✅ |
| `build_stage2_lta_dataset.py` | GT-history 长时预测对照数据（`stage2_val_lta_context.jsonl` 等） | ✅ |
| `build_stage2_sft_dataset.py` | GRPO 数据 → 带 assistant GT 的 SFT 数据（train 10748 / val 876 / val100 / 32 / 12） | ✅ |
| `extract_ek55_anticipation_rgb.py` | 无泄漏 τa=1s RGB 序列特征 | ⏸ 未跑完（随 τa 线路暂停） |
| `export_rulstm_lmdb_sequences.py` | RULSTM LMDB → 3.5s→1.0s 11 点序列 | ❌ 停用（LMDB 损坏） |

### 6.4 `evaluation/`

| 文件 | 作用 |
|------|------|
| `prepare_ek55_anticipation.py` | RULSTM 官方视频划分（train 232 / val 40 视频；23,431 / 4,970 条过滤后样本）+ 14 点 0.25 s 间隔 manifest |
| `train_ek55_future_verb_map.py` | **EGO-TOPO 协议 mAP**（P=25/50/75 × All/Freq/Rare，GRU 时序模型，125 类多标签） |
| `audit_pipeline_exact.py` | 权重/数据/哈希/Stage2 产物审计（SHA256 + JSONL 统计） |
| `plot_sft_training_curves.py` | 从 `trainer_state.json` 导出 SFT 曲线（CSV/PNG） |

### 6.5 `dcr_eval/`（★ 当前主线评测封装）

把 Stage1 → Stage2 串成端到端评测器，在 DCR 的 EK55 官方 valid（40 视频）上测生成式 Next-Action Exact-Match@1。

| 文件 | 作用 |
|------|------|
| `config.py` | 集中配置（DCR_ROOT、EK55_FRAMES、CLIP/Stage1/共现/Qwen/SFT-adapter 路径；HISTORY=8、HORIZON=20、`EVAL_FUTURE_REQUIRED=1`、TAU_A=1.0）。用环境变量 `DCR_RUN_NAME` 隔离 run，输出到 `/data/datasets/EPIC-KITCHENS/dcr_eval/runs/{RUN_NAME}/{features,annotations,stage1_predictions,stage2_generated,evaluation}` |
| `data_utils.py` | DCR 标注严格读取（video_id 正则、必需列校验） |
| `scripts/00_audit.py` | CPU 审计 + 泄漏检查 |
| `scripts/01_prepare_features.py` | 4 帧 CLIP → 1024 维 `features/val/{video}_{uid}.pt`（mask=frame） |
| `scripts/02_stage1_infer.py` | Stage1 推理 + 8 段历史构造 + 因果门（排除非因果样本） |
| `scripts/03_stage2_generate.sh` | swift infer 温度 0，三 GPU 分片；原始输出先落临时文件 → 校验 → 回填 ID |
| `scripts/attach_prediction_ids.py` / `validate_generated_jsonl.py` | ID 对齐与校验（`--require-ids`） |
| `scripts/04_evaluate.py` | Next-Action Exact-Match@1 |
| `scripts/05_export_results_csv.py` | 逐样本 + 汇总 CSV |
| `scripts/run_top1_all_valid.sh` | **一键 Phase 0–4**，含 generation_spec.txt 缓存校验与断点续跑 |
| `tests/` | 合成数据单测（`test_evaluate.py`、`test_stage2_shell.sh`，均通过） |

**一键命令（推荐唯一入口）**：

```bash
cd /home/amax/ldy/git/AAAI26-INSIGHT/dcr_eval
DCR_RUN_NAME=all_valid_top1_id_v1 DCR_EVAL_FUTURE_REQUIRED=1 RUN_MODE=full \
STAGE1_GPU=0 STAGE2_GPUS=0,1,3 bash scripts/run_top1_all_valid.sh
```

### 6.6 `scripts/`

| 脚本 | 作用 | 状态 |
|------|------|------|
| `run_stage2_sft_fast.sh` | **当前 Stage2 adapter 的来源**。`RUN_MODE=smoke\|full`，GPU 0,1,3，LoRA r=8/alpha=32/`all-linear`、lr 5e-5、batch 1×accum 4、1 epoch、ZeRO-2 | ✅ |
| `evaluate_stage2_adapter_fast.sh` | 单 adapter 确定性语义门（swift infer 温度 0 + compact 评测，SAMPLES=100 或 876；gate = Top-1>0 或 Action ED<0.98） | ✅ |
| `run_stage2_diagnostic_100.sh` | GT/pred history × base/adapter 四组对照（100 样本） | ✅ |
| `run_stage2_generated_topk.sh` | 历史 GRPO checkpoint-1773 的 5 候选生成（candidate_0 温度 0，1–4 温度 0.9 不同 seed） | 留档 |
| `run_ek55_future_verb_map.sh` | `GPU=3 bash ...`（GRU 125 类，21 epoch 早停，best epoch 13，<30 分钟） | ✅ 已完成 mAP |
| `run_ek55_anticipation_topk.sh` | MODE=prepare/probe/extract/train/all（官方短期 Top-k 线路） | ⏸ 暂停 |
| `run_ek55_topk_rulstm_features.sh` | RULSTM LMDB 替代链 | ❌ 放弃 |
| `monitor_ek55.sh` | EK55 下载监控 TUI（依赖 `scripts/epic-kitchens-download-scripts/`，**该目录不在本仓库**，仅存在于远端历史环境） | 留档 |

SFT 训练启动样例：
```bash
RUN_MODE=full nohup bash scripts/run_stage2_sft_fast.sh > /tmp/stage2_sft_fast.log 2>&1 &
```

---

## 7. 完整 Pipeline 现状

```
原始视频 (EK55)
    │
    ├─→ Hand Object Detector → 手/物体 bbox + 接触状态   ⚠️ 真实 HOI 检出仅 0.6%
    ├─→ SAM2                 → 手-物体区域 mask        ⚠️ 无独立信号（≈ frame 副本）
    ├─→ EgoVideo/CLIP        → 帧级特征 (1024 维，4 帧平均)  ⚠️ 非论文的 1408 维视频特征
    │
    ▼
特征文件 (.pt)：frame_features_dir/{train,val}/{clip_uid}_{action_idx}.pt
    │
    ▼
Stage 1: HandObject → verb/noun 分类 + 共现矩阵重排 (论文 Eq.5)
    │  当前权重: /data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth
    │  ⛔ 原 /data/datasets/EPIC-KITCHENS/checkpoint/best_model.pth 已损坏，禁止使用
    │
    ▼
Stage 2: CognitiveReasoning (Qwen2.5-VL-7B)
        fmt: 历史 8 段 → think/intention/answer → 预测未来 20 个动作
        当前方案: ms-swift 3.4.1 + LoRA SFT (checkpoint-200)
        （原论文 GRPO 方案已跑通但失效，见下）
```

### Stage 1 状态

- 训练 + 评测全流程已跑通。
- **当前权重为重建版**：`stage1_rebuilt_checkpoint/best_model.pth`。原 `checkpoint/best_model.pth`（56,840,192 bytes）zip central directory 损坏，**禁止使用**；重建版仅训练 6 epoch（早停）、89 tensors。
- 输入为每动作 4 帧平均的 1024 维 CLIP 特征；frame/mask 双流但 mask≈frame 副本。
- 输出 125 verb + 352 noun logits，乘共现先验（论文 Eq.5）取联合 action。
- 在 DCR 主线中的角色：根据 oracle GT 时间边界（非自动分割）识别目标动作前的 8 个历史动作。

### Stage 2 状态：GRPO 失败 → SFT 为当前正式方案

| 方案 | 设置 | 结果 | 结论 |
|------|------|------|------|
| **GRPO**（论文原方案） | Qwen2.5-VL-7B + LoRA r=8/alpha=32/dropout 0.05，1,773 步，6 h 19 m，ZeRO-2，3 GPU（07-29 全量跑完） | 内容奖励≈0，只学会格式（5.8%） | ❌ 失效 |
| **SFT**（当前方案） | 895 步（1 epoch，脚本 batch 1×accum 4×3 GPU），36 m 43 s，最佳 **checkpoint-200**（val loss 0.9885 / token acc 75.80%），checkpoint-895 为过拟合对照 | 见第 8 节 | ✅ 采用 |

- 框架：ms-swift 3.4.1 的 `swift sft` / `swift infer`（`--infer_backend pt`，float16，LoRA adapter）。
- 数据：10,748 train + 876 val（8 段 Stage1 预测历史 + 最后历史段 stop_frame 图像 → 20 个未来动作，think/intention/answer 格式）；训练期用 val100 子集做 eval。

> **注意（文档与脚本不一致）**：WORKLOG 2026-07-31 记录 SFT 为 "batch 4/GPU, grad_accum=8"，但脚本 `scripts/run_stage2_sft_fast.sh` 实为 `per_device_train_batch_size 1` + `gradient_accumulation_steps 4`（与 895 步 ≈ 1 epoch 吻合）。**复现时以脚本为准。**

### "top-k 评估"的三层含义（互不相同，勿混淆）

1. **Stage 1** `HandObject/evaluate_topk.py`：唯一动作段的 verb/noun/action Top-1/Top-5 micro accuracy（44,000 联合类上取 top-5）。
2. **Stage 2** `CognitiveReasoning/evaluate_generated_topk*.py`：5 候选生成 —— Top-1 = candidate_0 确定性首动作精确匹配；"Top-5" = 5 个候选首动作任一命中（sampled Hit@5，**无类别 logits，不是标准 ranked Top-5**）。
3. `train_anticipation_topk.py`：RULSTM 2513 类直接分类 Top-1/Top-5（τa=1s，**已放弃**）。

### dcr_eval 在流程中的位置

把 Stage1 → Stage2 串成端到端评测器，在 DCR 项目 EK55 官方 valid（40 视频）上测生成式 Next-Action Exact-Match@1。
这是**代理口径**，不等价于 DCR 官方 τa=1s 分类 Top-1（19.2）。它是当前唯一"全量跑完"的主线实验。

### 其他协议线路

| 线路 | 状态 |
|------|------|
| EGO-TOPO mAP | ✅ 已完成（feature-limited baseline，复用 Stage1 RGB 特征） |
| DCR 端到端 Top-1（dcr_eval） | ✅ 已完成全量（08-03） |
| τa=1s Top-1/Top-5 | ❌ 08-01 放弃（LMDB 两次下载损坏），3 个相关脚本留档 |

---

## 8. 当前结果指标

### 8.1 DCR 主线全量 `all_valid_top1_id_v1`（2026-08-03 完成，**当前最重要数字**）

**输入构造**：candidate positions 4,663 → 实际写出 **1,472** 条
（missing history 1,754、noncausal 排除 1,437、missing image 0、**覆盖率 31.568%**）；
Stage2 三分片 490/491/491，全部唯一 ID。

| 指标 | 值 |
|------|-----|
| **Next-Action Exact-Match@1 Top-1** | **1.019%（15/1,472）** |
| Verb Top-1 | 13.383% |
| Noun Top-1 | 2.446% |
| 解析率 | 100% |
| 恰好 20 步 | 98.3% |
| Stage1 门控（2,201 样本） | Verb 26.397% / Noun 13.539% / Action 7.497% |

**多样性审计**：top-10 预测中 `open cupboard` 出现 **350 次（约 23.8%）** → 明显的先验坍塌。

**产物**：`/data/datasets/EPIC-KITCHENS/dcr_eval/runs/all_valid_top1_id_v1/evaluation/metrics.json`、
`all_valid_top1_results.csv`（1,472 行）、`all_valid_top1_summary.csv`
（输入 SHA256 `97c00322...`、预测 SHA256 `c698c1ba...`）。

> ⚠️ **口径边界（不可直接对外比较）**：与 Stage1 train 精确重叠 4,602、val 377；与 Stage2 SFT train 重叠 10 个视频；**严格 video-unseen = 0**；不可与 DCR τa=1s Top-1 = 19.2 对比。

### 8.2 DCR 冒烟 `smoke_3videos_v1`（P07_02 / P08_12 / P26_11）

- Stage1：47 个样本，Verb 29.787% / Noun 21.277% / Action 12.766%
- Stage2：9 条严格因果样本，Top-1 = **11.111%（1/9）**

### 8.3 原 EK55 self-split 链路

**Stage 1**（val 1,263 可用 / 1,227 缺失，诊断口径）：

| 指标 | Top-1 / Top-5 |
|------|------|
| Verb | 18.69 / 59.62 |
| Noun | 8.08 / 22.41 |
| Action raw | 3.33 / 7.21 |
| Action + Eq.5 先验 | **3.56 / 11.16** |

**Stage 2**（876 样本）：

| 方案 | Top-1 | Hit@5 | oracle Action ED | 格式 |
|------|-------|-------|------------------|------|
| GRPO checkpoint-1773 | 0.114%（1/876） | 0.114% | 0.9985 | ~5.8% |
| **SFT checkpoint-200** | **0.799%（7/876）** | **1.598%（14/876）** | 0.982 | `<answer>` 100%、恰好 20 动作 92.03%、20 全合法 91.60% |

> SFT 结果含 180 个重复候选槽。

**Stage 2 语义诊断**（100 样本，GT/pred history × base/adapter 四组）：

| 组合 | Top-1 | Action ED |
|------|-------|-----------|
| GT history + base Qwen | 3.0% | 0.9825 |
| GT history + adapter | 0.0% | 1.0 |
| pred history + base | 0.0% | — |
| pred history + adapter | 0.0% | — |

> 结论：adapter 只把"输出 20 个合法动作"提到 100%，**语义能力未突破**。

### 8.4 EGO-TOPO mAP（feature-limited baseline）

| 指标 | 本项目 | 论文 INSIGHT |
|------|--------|--------------|
| All | **46.2** | 45.2 |
| Freq | **53.8** | 62.4 |
| Rare | **41.1** | 36.0 |

分观察比例 All：P25 **60.2** / P50 **46.2** / P75 **32.4**。

> 边界：val 仅 96 阶段样本 / 34 视频（覆盖 96/172 ≈ 56%）、64 个评估类、1024 维 CLIP 特征。
> All 反超论文属小样本噪声；Freq 落后 8.6 与 HOI 特征缺失的因果一致。
> 产物：`/data/datasets/EPIC-KITCHENS/ek55_future_verb_map_output/metrics.json`、`final_metrics_20260730/`。

### 8.5 论文参考值

- 论文 EK55 长期 mAP：45.2 / 62.4 / 36.0（EGO-TOPO P25/50/75 平均）
- DCR 官方 EK55 valid τa=1s Top-1：19.2

---

## 9. 已知问题与解决方案

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | HandObjDet CUDA 编译失败 | CUDA 11.5 nvcc 不兼容 PyTorch 2.5 C++17 代码 | 换用 larsrpe pure-pytorch fork |
| 2 | 真实 HOI 检出仅 0.6% | 只有 handobj_100K 权重，未在 ego 数据上训练 | 等 Dandan 提供 handobj_100K+ego；或自行微调 |
| 3 | mask_features 无独立信号 | 检测失败 → mask 由 frame 复制而来 | 双流退化为单流；需先解决 #2 |
| 4 | Stage1 原 checkpoint 损坏 | zip central directory 损坏 | 用 `rebuild_stage1.py` 重建（仅 6 epoch，语义上限受限） |
| 5 | GRPO 奖励恒接近 0 | 只学会格式（5.8%），内容奖励无梯度信号 | 改用 SFT 快速修复路线 |
| 6 | SFT 只学会格式 | 数据/特征上限受限 | 尚未突破（见 8.3 诊断），属当前主要瓶颈 |
| 7 | EK55 帧路径不含 video_id | 解压后平铺在 participant 级目录 | 所有结果标注 frame-path-ambiguous proxy |
| 8 | LMDB 两次下载损坏 | `MDB_CORRUPTED` | 放弃 τa=1s 线路 |
| 9 | ms-swift 4.2 装不上 | 依赖链要求 CUDA Driver ≥ 525 | 回退 ms-swift 3.4.1 + trl 0.17.0 |
| 10 | 物理 GPU 2 硬件故障 | Xid 79 / Xid 48 | 固定使用 GPU 0/1/3 |
| 11 | 评测集泄漏 | 训练/验证重叠，strict video-unseen = 0 | 当前 Top-1 仅作 proxy，不可对外宣称 |
| 12 | 远程桌面断连 | TeamViewer 不稳定 | 长任务用 nohup 后台跑 |

### 数据流层面缺陷（详见 `docs/PROJECT_REPRODUCTION_AUDIT.md`，该文档不入库）

1. 帧路径不含 video_id（来源歧义）
2. 使用 GT 时间边界而非自动分割
3. 1024 维图像 CLIP 替代论文 1408 维视频特征
4. 无真实 HOI 特征（双流退化）
5. Stage1 为损坏后重建的权重（仅 6 epoch）
6. Stage1 误差级联进 8 段历史
7. Stage2 SFT 只学格式
8. 训练/验证重叠（泄漏）
9. 当前 Top-1 协议非 DCR 官方 τa=1s

---

## 10. 遗留待办

| 优先级 | 事项 | 状态 |
|:--:|------|------|
| 高 | 修复 Stage1 语义上限：真实 HOI 特征、1408 维 EgoVideo-V 特征、8 步时序 | 未解决 |
| 高 | 联系 CorrineQiu 获取原版特征提取与官方评测脚本 | 无回复记录 |
| 高 | 运行 EGO-TOPO mAP 的 UID→特征映射 | ✅ 已完成（`action_idx` 即 uid，无需映射；mAP 已跑出，见 8.4） |
| — | τa=1s Top-1/Top-5 | ❌ 08-01 放弃 |
| 中 | 等 Dandan Shan 提供 handobj_100K+ego 权重 | 无新回复 |
| 中 | Ego4D / EGTEA 数据集下载 | 未开始 |

**其他未闭环项**

- 单样本 ID 对齐已解决（`attach_prediction_ids.py` + `--require-ids`），但 `04_evaluate.py` 仍保留"无 ID 时按行号"的警告路径。
- 2026-08-07 的 Data Shapley（ICML 2024）调研**未落地任何代码**，仅作为 DCR 数据质量分析的备选理论。
- 2026-08-14 之后服务器上的任何新进展，本文档未记录。

---

## 11. 环境激活速查 & 后台任务

```bash
# SSH / 远程桌面登录后
conda activate /data/conda_envs/insight_env
cd ~/ldy/git/AAAI26-INSIGHT
export CUDA_VISIBLE_DEVICES=0,1,3        # 物理 GPU 2 已损坏，注意必须用数字索引
```

### 后台任务

截至最新日志（2026-08-14），**没有仍在运行的后台任务**。最后一批已结束的任务：

| 任务 | PID | 完成时间 | 日志/标记 |
|------|-----|----------|-----------|
| DCR 全量 run 主 runner | 3417 | 2026-08-03 12:17 | `TOP1_PIPELINE_COMPLETE` |
| DCR Stage2 三分片（GPU 0/1/3） | 3851 / 3852 / 3853 | 同上 | — |
| EK55 下载 | 31625 | 2026-07-19 | `/tmp/ek55_download.log` |

> 历史日志示例：SFT 训练 `/tmp/stage2_sft_fast.log`、Stage1 重建 `/tmp/stage1_rebuild_training.log`。

---

## 12. 本地参考资料（**未入库**，另附）

以下文档在本地工作区保留、但**不在本仓库中**（`.gitignore` 已排除）。交接时请一并向接手人提供：

| 文件 | 内容 |
|------|------|
| `docs/WORKLOG.md` | 800 行主工作日志（最完整的过程记录） |
| `docs/PROJECT_REPRODUCTION_AUDIT.md` | 618 行复现审计（最新最全的位置与缺陷说明，08-03） |
| `docs/TECHNICAL_EXPLANATION.md` | 技术方案说明 |
| `docs/EK55_METRICS_PROTOCOL.md` | 公开可报告口径说明 |
| `docs/REPORT_TO_PROFESSOR.md` | 汇报稿 |
| `docs/ICML2024_DataShapley_DataSelection_notes.md` | Data Shapley 调研笔记（08-07，未落地代码） |
| `docs/archive/` | 旧版审计与 STAGE2_SFT_FAST_PATH |
| `docs/*.xlsx`、`stage_results_topk_map.csv`、`stage2_sft_training_report.html` | 结果表格与可视化报告 |
| `dcr_eval/README.md`、`dcr_eval/docs/{PIPELINE,REQUIREMENTS_AND_SOLUTION}.md` | dcr_eval 的详细设计与需求说明 |
| `CognitiveReasoning/README.md`、`HandObject/README.md` | 上游原始 README（路径仍为占位符，未适配本机） |

> 注：本仓库 `.gitignore` 同时排除了数据集、权重、训练输出与办公文档，接手人从仓库只拿到**代码 + 根 README + 本文档**。
