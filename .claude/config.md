# INSIGHT 项目环境配置

> **论文**: Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation (AAAI 2026)
> **官方仓库**: https://github.com/CorrineQiu/INSIGHT
> **最后更新**: 2026-07-16
>
> ⚠️ 每次配置变更后必须同步更新此文档。项目最终会交接给其他人使用。

---

## 1. 远程机器

| 项目 | 值 |
|------|-----|
| 访问方式 | 远程桌面 (TeamViewer) |
| 系统 | Linux (Ubuntu) |
| 主机名 | amax |
| 用户名 | amax |
| 项目路径 | `~/ldy/git/AAAI26-INSIGHT` |

### GPU 硬件

| 项目 | 值 |
|------|-----|
| GPU | 4× NVIDIA GeForce RTX (24 GB 显存/卡) |
| Driver | 495.29.05 |
| CUDA | 11.5 |
| GPU 0-2 | 空闲 (6 MiB) |
| GPU 3 | 桌面显示 (269 MiB) |

---

## 2. Conda 环境

| 项目 | 值 |
|------|-----|
| 环境路径 | `/data/conda_envs/insight_env` |
| 激活命令 | `conda activate /data/conda_envs/insight_env` |
| Python | 3.10 |
| PyTorch | 2.5.1+cu118 |
| pip 镜像 | `https://pypi.tuna.tsinghua.edu.cn/simple` |

> **注意**: 系统 CUDA 是 11.5 (nvidia-smi)，PyTorch 用 cu118 编译。两者兼容，但编译 CUDA C 扩展时可能有 nvcc 版本问题（见第 7 节）。

---

## 3. 预训练权重 `/data/pretrain_model/`

### 目录总览

```
/data/pretrain_model/
├── all-MiniLM-L6-v2/              ✅ 文本嵌入模型
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
| `faster_rcnn_1_8_132028.pth` | handobj_100K+ego 检测器 | ego 数据上物体检测 AP 41→67 | 等 Dandan Shan |

### Hand Object Detector 细节

- **代码来源**: `larsrpe/hand_object_detector_pytorch`，非原版 `ddshan/hand_object_detector`
- **切换原因**: 原版需编译 CUDA C 扩展，系统 CUDA 11.5 的 nvcc 与 PyTorch 2.5 的 C++17 代码不兼容（`TypeList.h: 'tuple' was not declared` 编译错误）
- **pure-pytorch fork**: 所有 CUDA 算子用 `torchvision.ops` 替换，无需编译，权重完全兼容
- **代码修改**:
  - 文件: `handobdet/hand_object_detector.py` 第 44 行
  - 改动: `faster_rcnn_1_8_132028.pth` → `faster_rcnn_1_8_89999.pth`
- **验证**: 模型加载成功，demo.py 可运行（缺测试图片但非 bug）

### Dandan Shan 邮件往来

| 项目 | 值 |
|------|-----|
| 作者邮箱 | dandans@umich.edu |
| 我方邮箱 | esun12191@gmail.com |
| 我方单位 | 合肥工业大学 |
| 首次发信 | ~2026-07-13 |
| 回复 | 学校账号过期导致 Google Drive 失效，正在联系系里找回 |
| 下次跟进 | 等 Dandan 主动回复；如一周无消息可再问 |

---

## 4. 数据集

### EPIC-Kitchens-55 ⬇️ 下载中

| 项目 | 值 |
|------|-----|
| 下载方式 | 官方脚本 `epic-kitchens-download-scripts` |
| 下载内容 | RGB 帧 (`.tar`) |
| 目标路径 | `/data/datasets/EPIC-KITCHENS/` |
| 预计大小 | ~220 GB (tar), 解压后 ~220-240 GB |
| 脚本路径 | `~/ldy/git/AAAI26-INSIGHT/scripts/epic-kitchens-download-scripts/` |
| 启动命令 | `nohup python3 epic_downloader.py --rgb-frames --epic55-only --output-path /data/datasets` |
| 启动时间 | 2026-07-16 |
| PID | 31625 |
| 日志 | `/tmp/ek55_download.log` |

#### 监控命令

```bash
tail -f /tmp/ek55_download.log           # 实时进度
jobs                                       # 后台任务状态
du -sh /data/datasets/EPIC-KITCHENS/       # 已下载大小
```

#### 下载后数据结构

```
/data/datasets/EPIC-KITCHENS/
├── P01/rgb_frames/
│   ├── P01_01.tar → P01_01/ (解压后 ~1150 万帧)
│   ├── P01_02.tar → P01_02/
│   └── ...
├── P02/rgb_frames/
└── ...
└── P08/rgb_frames/
```

#### 下载完成后需要做的事

```bash
# 逐 tar 解压（官方要求解压成文件夹）
cd /data/datasets/EPIC-KITCHENS
for p in P*/; do
  cd "$p/rgb_frames"
  for tar in *.tar; do
    tar -xf "$tar" && rm "$tar"   # 边解压边删，节省空间
  done
  cd /data/datasets/EPIC-KITCHENS
done
```

> ⚠️ 磁盘只剩 396 GB，必须边解压边删 tar，否则空间不够。

### Ego4D

> TODO: 尚未开始

### EGTEA Gaze+

> TODO: 尚未开始

---

## 5. 项目代码结构

```
~/ldy/git/AAAI26-INSIGHT/
├── HandObject/                    # Stage 1: 手-物体语义动作识别
│   ├── main.py                    # 入口：训练+测试，先改路径
│   ├── model.py                   # ActionRecognitionModel (Transformer)
│   ├── dataset.py                 # ActionDataset (窗口采样)
│   ├── train.py                   # train_one_epoch + validate
│   └── README.md
│
├── CognitiveReasoning/            # Stage 2: 认知推理 (GRPO + Qwen2.5-VL)
│   ├── README.md
│   ├── cal_ED.py                  # 计算编辑距离
│   ├── my_rewards_intention.py    # 自定义 reward 函数
│   ├── plugin_in_log_intention.py # 日志插件
│   └── run_external_reward_func_7B_qwen_intention.sh  # 启动脚本
│
├── scripts/
│   └── epic-kitchens-download-scripts/  # EK55 官方下载器
│
├── assets/                        # README 图片
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 6. 完整 Pipeline 流程

```
原始视频 (EK55/Ego4D/EGTEA)
    │
    ├─→ Hand Object Detector  ──→ 手/物体 bbox + 接触状态
    ├─→ SAM2                   ──→ 手-物体区域 mask
    ├─→ EgoVideo               ──→ 帧级视频特征
    │
    ▼
特征文件 (.pt) 按 clip_uid + action_idx 组织
    │
    ├── frame_features_dir/train|val|test/<clip_uid>_<action_idx>.pt
    └── mask_features_dir/train|val|test/<clip_uid>_<action_idx>.pt
    │
    ▼
Stage 1: HandObject ──→ verb/noun 分类 + re-ranking
    │  输出: predictions.csv + best_model.pth
    │
    ▼
Stage 2: CognitiveReasoning ──→ think → reason → answer
        (Qwen2.5-VL-7B + GRPO RL)
       输出: 最终长时动作预测
```

---

## 7. 已知问题与解决方案

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | HandObjDet CUDA 编译失败 | CUDA 11.5 nvcc 不兼容 PyTorch 2.5 C++17 代码 | 换用 larsrpe pure-pytorch fork |
| 2 | handobj_100K 物体检测弱 | 未在 ego 数据上训练 (41.7 vs 67.2 AP) | 等 Dandan 提供 handobj_100K+ego；或自行微调 |
| 3 | Google Drive/Dropbox 不可达 | 国内网络限制 | resnet101: Zenodo 已解决；检测器: 等邮件 |
| 4 | 磁盘空间紧张 | 396G 剩余，EK55 需 ~220G+ | 边解压边删 tar，预留余量
| 5 | 远程桌面断连 | TeamViewer 不稳定 | 长任务用 nohup 后台跑 |

---

## 8. 环境激活速查

```bash
# SSH 登录后
conda activate /data/conda_envs/insight_env
cd ~/ldy/git/AAAI26-INSIGHT
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 9. 后台任务

| 任务 | PID | 启动时间 | 日志 | 状态 |
|------|-----|----------|------|:--:|
| EK55 下载 | 31625 | 2026-07-16 | `/tmp/ek55_download.log` | 运行中 |
