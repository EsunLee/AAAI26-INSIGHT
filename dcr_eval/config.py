"""DCR 对接评测项目 — 集中路径配置。

真实数据与权重在远程服务器（amax），本项目提供完整代码；
运行前把下列路径与实际部署位置对齐即可。

实验隔离：每个评测口径一个独立输出目录，由环境变量 DCR_RUN_NAME 决定：
  DCR_RUN_NAME=all_valid    （默认，全量 DCR valid）
  DCR_RUN_NAME=unseen_only  （训练未见视频，无泄漏）
  DCR_RUN_NAME=smoke        （100 条冒烟测试）
不同口径的输出互不覆盖。
"""

import os
from pathlib import Path

# ────────────────────────────────────────────────────────
# 远程数据（DCR 项目自带）
# ────────────────────────────────────────────────────────
DCR_ROOT = Path("/data/sp/projects/csj/DCR-main")
DCR_VALID_CSV = DCR_ROOT / "data/EK55/validation_videos.csv"       # DCR 官方 valid 划分（有标签）
DCR_TRAIN_PKL = DCR_ROOT / "data/EK55/EPIC_train_action_labels.pkl"  # EK55 训练标注（28472 条，含 verb/noun 类）
DCR_TEST_S1 = DCR_ROOT / "data/EK55/EPIC_test_s1_timestamps.pkl"    # 官方 test s1（无标签，challenge）
DCR_TEST_S2 = DCR_ROOT / "data/EK55/EPIC_test_s2_timestamps.pkl"    # 官方 test s2（无标签）
DCR_VERB_CSV = DCR_ROOT / "data/EK55/EPIC_verb_classes.csv"         # verb id → 名称
DCR_NOUN_CSV = DCR_ROOT / "data/EK55/EPIC_noun_classes.csv"         # noun id → 名称
DCR_TSM_LMDB = DCR_ROOT / "data/feature/EK55_RGB_TSM"               # RULSTM TSM 特征（2048 维，本项目未使用，仅记录）

# ────────────────────────────────────────────────────────
# 我们的 EK55 原始帧（60fps JPEG，参与特征提取）
# ────────────────────────────────────────────────────────
EK55_FRAMES = Path("/data/datasets/EPIC-KITCHENS")   # 结构: {Pxx}/rgb_frames/frame_{帧号:010d}.jpg

# ────────────────────────────────────────────────────────
# 权重
# ────────────────────────────────────────────────────────
CLIP_CKPT = "/data/pretrain_model/EgoVideo/checkpoints/large_best.pt"          # CLIP ViT-L/14（1024 维）
STAGE1_CKPT = "/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint/best_model.pth"  # Stage 1 重建权重
COOCCURRENCE = "/data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt"  # verb-noun 共现矩阵
SFT_MODEL = "/data/pretrain_model/Qwen2.5-VL-7B-Instruct"                      # Stage 2 基座
SFT_ADAPTER = "/data/datasets/EPIC-KITCHENS/stage2_sft_fast_output/v0-20260730-225536/checkpoint-200"  # SFT LoRA adapter

# ────────────────────────────────────────────────────────
# 输出目录（按实验隔离，脚本自动创建）
# ────────────────────────────────────────────────────────
RUN_NAME = os.environ.get("DCR_RUN_NAME", "all_valid")
RUN_ROOT = Path("/data/datasets/EPIC-KITCHENS/dcr_eval/runs") / RUN_NAME
FEAT_DIR = RUN_ROOT / "features"           # 01 输出: features/val/{video}_{uid}.pt
ANN_DIR = RUN_ROOT / "annotations"         # 01 输出: annotations/val.json（insight 格式）+ 子集文件
PRED_DIR = RUN_ROOT / "stage1_predictions" # 02 输出: stage1_metrics.json / stage1_val_predictions.jsonl / stage2_input.jsonl
GEN_DIR = RUN_ROOT / "stage2_generated"    # 03 输出: candidate_0_part_*.jsonl → candidate_0.jsonl
EVAL_DIR = RUN_ROOT / "evaluation"         # 04 输出: metrics.json

# ────────────────────────────────────────────────────────
# 任务参数（训练时定死，不可改动）
# ────────────────────────────────────────────────────────
SAMPLES_PER_ACTION = 4    # Stage 1 每片段采样帧数
MIN_FRAMES_PER_ACTION = 2 # 对齐 extract_features.py 的最少可用帧门槛
FEATURE_COLOR_ORDER = "rgb"  # 对齐 extract_features.py 在 CLIP 前的 BGR→RGB 转换
INPUT_DIM = 1024          # CLIP 特征维度
VERB_CLASSES = 125        # EK55 动词类数
NOUN_CLASSES = 352        # EK55 名词类数
HISTORY = 8               # Stage 2 历史动作数量
HORIZON = 20              # Stage 2 未来动作数量
EVAL_FUTURE_REQUIRED = int(os.environ.get("DCR_EVAL_FUTURE_REQUIRED", "1"))
# Top-1 只需要目标动作本身有 GT；模型仍按训练格式生成 HORIZON=20 步。
if not 1 <= EVAL_FUTURE_REQUIRED <= HORIZON:
    raise ValueError(
        f"DCR_EVAL_FUTURE_REQUIRED must be in 1..{HORIZON}, "
        f"got {EVAL_FUTURE_REQUIRED}"
    )
TAU_A = 1.0               # DCR 协议的预测锚点（目标动作开始前 1 秒）
