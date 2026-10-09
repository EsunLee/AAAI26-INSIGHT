"""01 — 数据准备：DCR valid 标注 → 4 帧 CLIP 特征 + insight 格式标注。

输入：
  - DCR valid 划分（validation_videos.csv 过滤 EPIC_train_action_labels.pkl，有标签）
  - 我们本地 EK55 原始帧（60fps JPEG）

输出：
  - features/val/{video_id}_{uid}.pt     （每段 4 帧平均的 CLIP 1024 维特征；mask = 帧副本）
  - annotations/val.json                 （insight 格式，供 02 的 Stage 1 推理与评测复用）
  - annotations/meta.json                （统计信息）

运行：python scripts/01_prepare_features.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from data_utils import load_action_labels, load_video_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos-file", type=Path, default=None,
        help="只处理该文件列出的视频（00_audit 导出的 unseen_valid_videos.txt）；"
             "缺省处理全部 DCR valid。",
    )
    return parser.parse_args()

# ────────────────────────────────────────────────────────
# CLIP（EgoVideo large_best → HF CLIPVisionModel，与 extract_features.py 一致）
# 显式 float32：避免 NumPy 把 float32 输入提升为 float64（模型权重是 float32）
# ────────────────────────────────────────────────────────
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def build_clip():
    from transformers import CLIPVisionModel, CLIPVisionConfig

    cfg = CLIPVisionConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        image_size=224, patch_size=14, intermediate_size=4096,
    )
    model = CLIPVisionModel(cfg)
    ckpt = torch.load(C.CLIP_CKPT, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    clip_state = {}
    for k, v in state.items():
        if not k.startswith("module.visual."):
            continue
        if any(x in k for x in ["temporal", "image_projection", "logit_scale"]):
            continue
        if "positional_embedding" in k:
            old_size = int((v.shape[0] - 1) ** 0.5)
            new_size = 224 // 14
            cls_t = v[0:1]
            spatial = v[1:].reshape(old_size, old_size, 1024).permute(2, 0, 1).unsqueeze(0)
            spatial = F.interpolate(spatial, size=(new_size, new_size), mode="bicubic", align_corners=False)
            spatial = spatial.squeeze(0).permute(1, 2, 0).reshape(-1, 1024)
            clip_state["vision_model.embeddings.position_embedding.weight"] = torch.cat([cls_t, spatial], 0)
            continue
        if "attn.Wqkv" in k:
            layer = k.split(".resblocks.")[1].split(".")[0]
            parts = v.chunk(3, 0)
            suf = "weight" if ".weight" in k else "bias"
            clip_state[f"vision_model.encoder.layers.{layer}.self_attn.q_proj.{suf}"] = parts[0]
            clip_state[f"vision_model.encoder.layers.{layer}.self_attn.k_proj.{suf}"] = parts[1]
            clip_state[f"vision_model.encoder.layers.{layer}.self_attn.v_proj.{suf}"] = parts[2]
            continue
        nk = k.replace("module.visual.", "vision_model.")
        nk = nk.replace("transformer.resblocks.", "encoder.layers.")
        nk = nk.replace("ln_1.", "layer_norm1.").replace("ln_2.", "layer_norm2.")
        nk = nk.replace("ln_pre.", "pre_layrnorm.").replace("ln_post.", "post_layernorm.")
        nk = nk.replace("attn.out_proj.", "self_attn.out_proj.")
        nk = nk.replace("conv1.", "embeddings.patch_embedding.")
        nk = nk.replace("class_embedding", "embeddings.class_embedding")
        if "conv1.weight" in k:
            v = v.reshape(1024, 3, 1, 14, 14).squeeze(2)
        clip_state[nk] = v
    model.load_state_dict(clip_state)
    return model.to("cuda").eval()


def load_frame(participant: str, frame_idx: int):
    """读取单帧（BGR）。帧号为 60fps 原始帧号（与 DCR 标注 start_frame/stop_frame 一致）。"""
    path = C.EK55_FRAMES / participant / "rgb_frames" / f"frame_{frame_idx:010d}.jpg"
    return cv2.imread(str(path))  # 不存在时返回 None


def clip_feat(clip, frames_bgr):
    """多帧 → CLIP 平均特征 [1024]（跳过缺失帧；全缺返回 None）。

    兼容性约束：Stage1 重建权重的训练特征来自
    feature_extraction/extract_features.py。该脚本在 process_split() 中将
    cv2.imread 的 BGR 显式转为 RGB，然后才调用 CLIP；评测必须复刻这个顺序。
    """
    tensors = []
    for img in frames_bgr:
        if img is None:
            continue
        if C.FEATURE_COLOR_ORDER == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif C.FEATURE_COLOR_ORDER != "bgr":
            raise ValueError(
                f"FEATURE_COLOR_ORDER must be 'bgr' or 'rgb', got {C.FEATURE_COLOR_ORDER!r}"
            )
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = (img - MEAN) / STD  # float32 - float32 → 保持 float32
        tensors.append(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to("cuda"))
    if len(tensors) < C.MIN_FRAMES_PER_ACTION:
        return None
    with torch.no_grad():
        feats = clip(torch.cat(tensors)).pooler_output
    return feats.mean(0).cpu()


def sample_4_frames(start: int, stop: int) -> list[int]:
    """与 extract_features.py 一致的 4 帧均匀采样（DCR 标注帧号为 60fps 原始帧号）。"""
    n = stop - start
    if n <= C.SAMPLES_PER_ACTION:
        return list(range(start, stop))
    step = n / C.SAMPLES_PER_ACTION
    return [int(start + i * step) for i in range(C.SAMPLES_PER_ACTION)]


def main() -> None:
    args = parse_args()
    (C.FEAT_DIR / "val").mkdir(parents=True, exist_ok=True)
    C.ANN_DIR.mkdir(parents=True, exist_ok=True)

    # 断点续跑前锁定特征规格，防止将不同提取配置的 .pt 混在同一目录。
    feature_spec = {
        "format_version": 1,
        "clip_checkpoint": str(C.CLIP_CKPT),
        "clip_checkpoint_size": Path(C.CLIP_CKPT).stat().st_size,
        "clip_checkpoint_mtime_ns": Path(C.CLIP_CKPT).stat().st_mtime_ns,
        "samples_per_action": C.SAMPLES_PER_ACTION,
        "min_frames_per_action": C.MIN_FRAMES_PER_ACTION,
        "color_order": C.FEATURE_COLOR_ORDER,
        "feature_dim": C.INPUT_DIM,
        "frame_root": str(C.EK55_FRAMES),
    }
    spec_path = C.ANN_DIR / "feature_spec.json"
    existing_features = list((C.FEAT_DIR / "val").glob("*.pt"))
    if spec_path.exists():
        previous_spec = json.loads(spec_path.read_text())
        if previous_spec != feature_spec:
            raise RuntimeError(
                f"特征缓存规格与当前配置不一致：{spec_path}\n"
                f"cached={previous_spec}\ncurrent={feature_spec}\n"
                "请换一个新 DCR_RUN_NAME，不要混用旧 .pt。"
            )
    elif existing_features:
        raise RuntimeError(
            f"{C.FEAT_DIR / 'val'} 已有 {len(existing_features)} 个 .pt，但缺少 {spec_path}；"
            "无法证明缓存来源。请换一个新 DCR_RUN_NAME。"
        )
    else:
        spec_path.write_text(json.dumps(feature_spec, ensure_ascii=False, indent=2) + "\n")

    # 1. DCR valid 标注
    # DCR pkl 的 uid 在【索引】里而不是普通列：显式转成列，两处脚本用同一方式访问
    valid_videos = load_video_ids(C.DCR_VALID_CSV)
    df = load_action_labels(C.DCR_TRAIN_PKL)
    df = df[df["video_id"].isin(valid_videos)]
    if df.empty:
        raise RuntimeError("DCR valid 过滤后没有任何 action")

    # 子集过滤（00_audit 导出 unseen_valid_videos.txt → 无泄漏评测）
    if args.videos_file is not None:
        subset = {line.strip() for line in args.videos_file.read_text().splitlines() if line.strip()}
        if not subset:
            raise SystemExit(f"[FATAL] {args.videos_file} 为空 —— 该口径没有可评测视频，终止")
        df = df[df["video_id"].isin(subset)]
        print(f"[subset] 仅处理 {args.videos_file.name}: {df['video_id'].nunique()} 个视频 / {len(df)} 条动作")

    print(f"[valid] actions={len(df)} videos={df['video_id'].nunique()}")

    # 2. CLIP
    clip = build_clip()
    print("[clip] loaded:", C.CLIP_CKPT)

    # 3. 逐段提特征（断点续跑：.pt 已存在则跳过提取，但仍写入标注）
    clips = []
    written = missing = skipped_existing = 0
    for row in df.itertuples(index=False):
        action_id = f"{row.video_id}_{row.uid}"
        out_path = C.FEAT_DIR / "val" / f"{action_id}.pt"
        if not out_path.exists():
            idxs = sample_4_frames(int(row.start_frame), int(row.stop_frame))
            frames = [load_frame(row.participant_id, fi) for fi in idxs]
            feat = clip_feat(clip, frames)
            if feat is None:
                missing += 1
                continue
            if tuple(feat.shape) != (C.INPUT_DIM,) or not torch.isfinite(feat).all():
                raise RuntimeError(
                    f"{action_id}: CLIP 特征异常 shape={tuple(feat.shape)} "
                    f"finite={bool(torch.isfinite(feat).all())}"
                )
            # 只保存一份特征文件；02 推理时 frame/mask 两个分支输入同一文件
            # （无真实 HOI：mask 分支与 frame 分支共用 CLIP 特征，双流退化为单流）
            torch.save(feat, out_path)
            written += 1
            if written % 500 == 0:
                print(f"  {written} done...")
        else:
            skipped_existing += 1
        clips.append({
            "clip_uid": row.video_id,
            "action_idx": int(row.uid),
            "verb_label": int(row.verb_class),
            "noun_label": int(row.noun_class),
        })

    (C.ANN_DIR / "val.json").write_text(json.dumps({"clips": clips}, ensure_ascii=False))
    meta = {
        "source": "DCR EK55 valid split (validation_videos.csv filter)",
        "run_name": C.RUN_NAME,
        "labeled_actions": len(df),
        "written_features": written,
        "skipped_existing": skipped_existing,
        "missing_features": missing,
        "coverage": round(100.0 * len(clips) / max(len(df), 1), 2),
        "feature_color_order": C.FEATURE_COLOR_ORDER,
        "feature_color_reason": "matches Stage1 process_split BGR-to-RGB behavior",
        "feature_spec": feature_spec,
        "frame_layout": "participant/rgb_frames/frame_N.jpg (video_id absent; potentially ambiguous)",
        "note": "single CLIP feature file; frame/mask branches share it (no real HOI)",
    }
    (C.ANN_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[done] written={written} skipped_existing={skipped_existing} missing={missing} "
          f"annotations={len(clips)} → {C.FEAT_DIR} + {C.ANN_DIR}/val.json")


if __name__ == "__main__":
    main()
