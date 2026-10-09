"""00 — CPU 审计：运行任何 GPU 流程之前必须先通过。

检查项：
  1. DCR valid 标注结构（uid 位置、列名、条数）
  2. 类别 id 范围与名称映射（verb 0-124 / noun 0-351）
  3. 原始帧覆盖率（DCR valid 视频的帧是否齐全）
  4. 【核心】数据划分泄漏：DCR valid 视频 ∩ 我们 Stage1/Stage2 训练视频
  5. 已有特征复用率：features/frame_features/{train,val} 覆盖 DCR valid 的比例

输出：打印各项结论；泄漏检查结果决定后续评测口径。

运行：python scripts/00_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from data_utils import load_action_labels, load_video_ids  # noqa: E402


def _unwrap_state(value):
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    elif isinstance(value, dict) and "model" in value and isinstance(value["model"], dict):
        value = value["model"]
    if not isinstance(value, dict):
        raise TypeError(f"Stage1 checkpoint 不是 state dict: {type(value).__name__}")
    return {str(key).removeprefix("module."): tensor for key, tensor in value.items()}


def preflight_paths_and_weights() -> None:
    """在任何昂贵特征提取前校验所有必需输入。"""
    required = {
        "DCR valid split": C.DCR_VALID_CSV,
        "DCR action labels": C.DCR_TRAIN_PKL,
        "DCR verb classes": C.DCR_VERB_CSV,
        "DCR noun classes": C.DCR_NOUN_CSV,
        "EK55 frame root": C.EK55_FRAMES,
        "CLIP checkpoint": Path(C.CLIP_CKPT),
        "Stage1 checkpoint": Path(C.STAGE1_CKPT),
        "co-occurrence matrix": Path(C.COOCCURRENCE),
        "Qwen base model": Path(C.SFT_MODEL),
        "Stage2 SFT adapter": Path(C.SFT_ADAPTER),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("[FATAL] 缺少必需输入:\n  " + "\n  ".join(missing))

    clip_path = Path(C.CLIP_CKPT)
    if not clip_path.is_file() or clip_path.stat().st_size < 10 * 1024 * 1024:
        raise SystemExit(f"[FATAL] CLIP checkpoint 不像完整权重: {clip_path}")
    if not (Path(C.SFT_MODEL) / "config.json").exists():
        raise SystemExit(f"[FATAL] Qwen 目录缺 config.json: {C.SFT_MODEL}")
    adapter_files = list(Path(C.SFT_ADAPTER).glob("adapter_model.*"))
    if not (Path(C.SFT_ADAPTER) / "adapter_config.json").exists() or not adapter_files:
        raise SystemExit(f"[FATAL] LoRA adapter 缺少 adapter_config.json/adapter_model.*: {C.SFT_ADAPTER}")

    try:
        state = _unwrap_state(torch.load(C.STAGE1_CKPT, map_location="cpu", weights_only=True))
    except Exception as exc:
        raise SystemExit(f"[FATAL] Stage1 checkpoint 无法加载: {C.STAGE1_CKPT}\n{exc}") from exc
    expected_state_shapes = {
        "frame_stage1.weight": (2048, C.INPUT_DIM),
        "mask_stage1.weight": (2048, C.INPUT_DIM),
        "verb_head.weight": (C.VERB_CLASSES, 512),
        "noun_head.weight": (C.NOUN_CLASSES, 512),
    }
    for key, shape in expected_state_shapes.items():
        if key not in state or tuple(state[key].shape) != shape:
            actual = None if key not in state else tuple(state[key].shape)
            raise SystemExit(f"[FATAL] Stage1 {key} shape={actual}, expected={shape}")

    try:
        co = torch.load(C.COOCCURRENCE, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(f"[FATAL] 共现矩阵无法加载: {C.COOCCURRENCE}\n{exc}") from exc
    expected_co_shapes = {
        "P_n_given_v": (C.VERB_CLASSES, C.NOUN_CLASSES),
        "P_v_given_n": (C.NOUN_CLASSES, C.VERB_CLASSES),
    }
    for key, shape in expected_co_shapes.items():
        if key not in co or tuple(co[key].shape) != shape:
            actual = None if key not in co else tuple(co[key].shape)
            raise SystemExit(f"[FATAL] co-occurrence {key} shape={actual}, expected={shape}")
    print("[0] 路径/权重预检: PASS")


def load_dcr_valid_df():
    valid_videos = load_video_ids(C.DCR_VALID_CSV)
    df = load_action_labels(C.DCR_TRAIN_PKL)
    df = df[df["video_id"].isin(valid_videos)]
    if df.empty:
        raise SystemExit("[FATAL] DCR valid 过滤后没有任何 action")
    return valid_videos, df


def load_our_split_samples():
    """我们的 90/10 划分：insight_annotations/{train,val}.json 的 clip_uid。

    泄漏判断的必需输入：文件缺失 → fail-fast（缺失 = 无法证明无泄漏，禁止继续）。
    """
    split_videos = {"train": set(), "val": set()}
    split_actions = {"train": set(), "val": set()}
    split_rows = {"train": [], "val": []}
    for split in ("train", "val"):
        p = Path("/data/datasets/EPIC-KITCHENS/insight_annotations") / f"{split}.json"
        if not p.exists():
            raise SystemExit(f"[FATAL] 找不到 Stage1 训练划分 {p} —— 无法做泄漏判断，审计终止")
        data = json.loads(p.read_text())
        split_rows[split] = data.get("clips", [])
        split_videos[split].update(row["clip_uid"] for row in split_rows[split])
        split_actions[split].update(
            (row["clip_uid"], int(row["action_idx"])) for row in split_rows[split]
        )
    return split_videos, split_actions, split_rows


def stage1_loader_action_ids(split: str, rows: list[dict]) -> tuple[set[tuple[str, int]], int]:
    """复制 HandObject/dataset.py 的 8 动作完整窗口筛选，返回真正进入 loader 的动作 ID。"""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["clip_uid"], []).append(row)
    used: set[tuple[str, int]] = set()
    accepted_windows = 0
    frame_root = Path("/data/datasets/EPIC-KITCHENS/features/frame_features") / split
    mask_root = Path("/data/datasets/EPIC-KITCHENS/features/mask_features") / split
    for video, clips in groups.items():
        clips.sort(key=lambda row: int(row["action_idx"]))
        windows = (
            [clips[index:index + C.HISTORY] for index in range(len(clips) - C.HISTORY + 1)]
            if len(clips) >= C.HISTORY
            else [[clip] for clip in clips]
        )
        for window in windows:
            keys = [(video, int(clip["action_idx"])) for clip in window]
            if all(
                (frame_root / f"{key_video}_{uid}.pt").exists()
                and (mask_root / f"{key_video}_{uid}.pt").exists()
                for key_video, uid in keys
            ):
                accepted_windows += 1
                used.update(keys)
    return used, accepted_windows


def load_sft_train_videos():
    """Stage 2 SFT 训练数据的视频（stage2_stage1_history_sft 下的训练 jsonl）。"""
    root = Path("/data/datasets/EPIC-KITCHENS/stage2_stage1_history_sft")
    if not root.exists():
        raise SystemExit(f"[FATAL] 找不到 Stage2 SFT 训练数据 {root} —— 无法做泄漏判断，审计终止")
    vids = set()
    found_train = False
    for p in sorted(root.glob("*.jsonl")):
        if "train" not in p.name:
            continue
        found_train = True
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "clip_uid" in obj:
                    vids.add(obj["clip_uid"])
        print(f"[sft] 训练文件 {p.name}: 视频数 {len(vids)}（累计）")
    if not found_train:
        raise SystemExit(f"[FATAL] {root} 下没有 *train*.jsonl —— 无法做泄漏判断，审计终止")
    return vids


def frame_coverage_of_valid(df, valid_videos):
    """原始帧覆盖率：每条动作的 start/stop 帧是否存在。"""
    missing = 0
    total = len(df)
    for row in df.itertuples(index=False):
        p = C.EK55_FRAMES / row.participant_id / "rgb_frames" / f"frame_{int(row.stop_frame):010d}.jpg"
        if not p.exists():
            missing += 1
    return total - missing, missing, total


def sampled_frame_indices(start: int, stop: int) -> list[int]:
    """复制 01_prepare_features.py 的采样规则。"""
    n = stop - start
    if n <= C.SAMPLES_PER_ACTION:
        return list(range(start, stop))
    step = n / C.SAMPLES_PER_ACTION
    return [int(start + i * step) for i in range(C.SAMPLES_PER_ACTION)]


def frame_path(participant: str, frame_index: int) -> Path:
    return C.EK55_FRAMES / participant / "rgb_frames" / f"frame_{frame_index:010d}.jpg"


def predict_pipeline_coverage(df):
    """仅查文件存在性，预估 01 可提特征数和 02 可构造的 Stage2 窗口数。"""
    extractable = set()
    for row in df.itertuples(index=False):
        sampled = sampled_frame_indices(int(row.start_frame), int(row.stop_frame))
        available = sum(frame_path(row.participant_id, index).exists() for index in sampled)
        if available >= C.MIN_FRAMES_PER_ACTION:
            extractable.add((row.video_id, int(row.uid)))

    candidate_windows = eligible_windows = noncausal_windows = 0
    for video, group in df.sort_values(["video_id", "start_frame", "uid"]).groupby(
        "video_id", sort=False
    ):
        rows = list(group.itertuples(index=False))
        for index in range(C.HISTORY, len(rows) - C.EVAL_FUTURE_REQUIRED + 1):
            candidate_windows += 1
            history = rows[index - C.HISTORY:index]
            target = rows[index]
            if any(int(row.stop_frame) >= int(target.start_frame) for row in history):
                noncausal_windows += 1
                continue
            if not all((video, int(row.uid)) in extractable for row in history):
                continue
            observed = history[-1]
            image_ok = any(
                frame_path(observed.participant_id, frame).exists()
                for frame in (
                    int(observed.stop_frame),
                    int(observed.stop_frame) - 1,
                    int(observed.start_frame),
                )
            )
            if image_ok:
                eligible_windows += 1
    return {
        "extractable_actions": len(extractable),
        "candidate_stage2_windows": candidate_windows,
        "noncausal_stage2_windows": noncausal_windows,
        "eligible_stage2_windows": eligible_windows,
        "history_actions_required": C.HISTORY,
        "generated_future_actions": C.HORIZON,
        "future_gt_actions_required_for_evaluation": C.EVAL_FUTURE_REQUIRED,
    }


def existing_feature_coverage(valid_videos):
    """已有 frame_features/{train,val} 中属于 DCR valid 视频的特征数（复用潜力）。"""
    result = {}
    for split in ("train", "val"):
        root = Path("/data/datasets/EPIC-KITCHENS/features/frame_features") / split
        n = 0
        if root.exists():
            for p in root.glob("*.pt"):
                video = p.stem.rsplit("_", 1)[0]
                if video in valid_videos:
                    n += 1
        result[split] = n
    return result


def main() -> None:
    print("=" * 70)
    print("DCR 对接评测 — 阶段 0 CPU 审计")
    print("=" * 70)
    preflight_paths_and_weights()

    # 1. DCR valid 结构
    valid_videos, df = load_dcr_valid_df()
    print(f"\n[1] DCR valid: videos={len(valid_videos)} actions={len(df)}")
    print(f"    columns={list(df.columns)}")
    print(f"    uid 唯一: {df['uid'].is_unique}  范围: {df['uid'].min()}–{df['uid'].max()}")

    # 2. 类别映射
    print("\n[2] 类别映射（范围校验）:")
    vdf = pd.read_csv(C.DCR_VERB_CSV)
    ndf = pd.read_csv(C.DCR_NOUN_CSV)
    id_col_v = "verb_id" if "verb_id" in vdf.columns else vdf.columns[0]
    id_col_n = "noun_id" if "noun_id" in ndf.columns else ndf.columns[0]
    v_ids = set(vdf[id_col_v])
    n_ids = set(ndf[id_col_n])
    print(f"    verb: {len(v_ids)} 类, id 范围 {min(v_ids)}–{max(v_ids)}, 预期 0–124: {v_ids == set(range(125))}")
    print(f"    noun: {len(n_ids)} 类, id 范围 {min(n_ids)}–{max(n_ids)}, 预期 0–351: {n_ids == set(range(352))}")

    # 3. 帧覆盖率
    ok, missing, total = frame_coverage_of_valid(df, valid_videos)
    print(f"\n[3] 原始帧覆盖率（以 stop_frame 抽查）: {ok}/{total} 缺失={missing}")
    videos_per_participant = df.groupby("participant_id")["video_id"].nunique()
    ambiguous_participants = videos_per_participant[videos_per_participant > 1]
    if not ambiguous_participants.empty:
        print(
            "    ⚠️ 帧路径不含 video_id："
            f"{len(ambiguous_participants)} 个 participant 在 valid 中包含多个视频，"
            "存在同名帧冲突/来源歧义；本结果只能作 proxy。"
        )
    predicted_coverage = predict_pipeline_coverage(df)
    print(
        "    无 GPU 精确预估: "
        f"01可提特征={predicted_coverage['extractable_actions']}/{len(df)}, "
        f"评测要求={C.HISTORY}历史+{C.EVAL_FUTURE_REQUIRED}目标GT, "
        f"Stage2候选窗口={predicted_coverage['candidate_stage2_windows']}, "
        f"非因果窗口={predicted_coverage['noncausal_stage2_windows']}, "
        f"最终可构造={predicted_coverage['eligible_stage2_windows']}"
    )

    # 4. 泄漏检查（核心）
    print("\n[4] 数据划分与精确动作重叠检查:")
    split_videos, split_actions, split_rows = load_our_split_samples()
    train_vids, val_vids = split_videos["train"], split_videos["val"]
    sft_vids = load_sft_train_videos()
    leak_s1 = valid_videos & train_vids
    leak_val = valid_videos & val_vids
    leak_sft = valid_videos & sft_vids
    print(f"    DCR valid ∩ Stage1 训练视频: {len(leak_s1)} 个 {sorted(leak_s1) if leak_s1 else ''}")
    print(f"    DCR valid ∩ Stage1 验证视频: {len(leak_val)} 个 {sorted(leak_val) if leak_val else ''}")
    print(f"    DCR valid ∩ Stage2 SFT 训练: {len(leak_sft)} 个 {sorted(leak_sft) if leak_sft else ''}")
    dcr_action_ids = {(row.video_id, int(row.uid)) for row in df.itertuples(index=False)}
    annotated_train_overlap = dcr_action_ids & split_actions["train"]
    annotated_val_overlap = dcr_action_ids & split_actions["val"]
    loader_train_ids, train_windows = stage1_loader_action_ids("train", split_rows["train"])
    loader_val_ids, val_windows = stage1_loader_action_ids("val", split_rows["val"])
    loader_train_overlap = dcr_action_ids & loader_train_ids
    loader_val_overlap = dcr_action_ids & loader_val_ids
    print(
        "    精确 (video_id, uid) 标注重叠: "
        f"Stage1 train={len(annotated_train_overlap)}, val={len(annotated_val_overlap)}"
    )
    print(
        "    按原 ActionDataset 完整窗口实际可进入 loader: "
        f"train={len(loader_train_overlap)} 个 DCR 动作/{train_windows} 窗口, "
        f"val={len(loader_val_overlap)} 个 DCR 动作/{val_windows} 窗口"
    )
    if loader_train_overlap or loader_val_overlap or leak_sft:
        print(
            "    ⚠️ 结论: 不只是视频重叠；上面的非零 loader 动作是同一 (video_id, uid)，"
            "确实参与过 Stage1 训练/模型选择。结果只能作为当前流程 proxy。"
        )
    elif leak_s1 or leak_val:
        print("    ⚠️ 仅发现视频级重叠，未证明相同动作进入 loader；需按视频划分规则说明。")
    else:
        print("    ✅ 结论: 无泄漏 —— DCR valid 全部视频未参与训练，可作验证性能。")

    # 5. 导出评测子集（代理实验模式）：后续 01/02 用 --videos-file 过滤
    print("\n[5] 导出评测子集:")
    # Stage1 val 参与了 early stopping / best checkpoint 选择，也必须算“已见”。
    unseen = valid_videos - train_vids - val_vids - sft_vids
    out_dir = C.ANN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "all_valid_videos.txt").write_text("\n".join(sorted(valid_videos)) + "\n")
    (out_dir / "unseen_valid_videos.txt").write_text("\n".join(sorted(unseen)) + "\n")
    audit_summary = {
        "run_name": C.RUN_NAME,
        "valid_videos": len(valid_videos),
        "valid_actions": len(df),
        "frames_found_at_stop": ok,
        "frames_missing_at_stop": missing,
        **predicted_coverage,
        "frame_layout": "participant/rgb_frames/frame_N.jpg (video_id absent)",
        "ambiguous_multi_video_participants": {
            str(participant): int(count)
            for participant, count in ambiguous_participants.items()
        },
        "stage1_train_overlap_videos": sorted(leak_s1),
        "stage1_validation_overlap_videos": sorted(leak_val),
        "stage2_train_overlap_videos": sorted(leak_sft),
        "stage1_train_annotation_action_overlap": len(annotated_train_overlap),
        "stage1_validation_annotation_action_overlap": len(annotated_val_overlap),
        "stage1_train_loader_action_overlap": len(loader_train_overlap),
        "stage1_validation_loader_action_overlap": len(loader_val_overlap),
        "stage1_train_accepted_windows": train_windows,
        "stage1_validation_accepted_windows": val_windows,
        "unseen_valid_videos": len(unseen),
    }
    (out_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"    all_valid_videos.txt:   {len(valid_videos)} 个视频（全量 DCR valid）")
    print(f"    unseen_valid_videos.txt: {len(unseen)} 个视频（Stage1 train/val 与 Stage2 train 均未见）")
    print(f"    audit_summary.json:     {out_dir / 'audit_summary.json'}")
    print("    用法: python scripts/01_prepare_features.py --videos-file annotations/unseen_valid_videos.txt")
    if not unseen:
        print("    ⚠️ 严格 unseen 子集为空：只能报告 all-valid proxy，不能声称无泄漏性能。")
    elif len(unseen) < len(valid_videos):
        print("    建议同时报告两个口径: all-valid proxy 与 unseen-only proxy")

    # 6. 已有特征复用率（修正统计口径：动作特征数 + 覆盖率 + 视频数）
    print("\n[6] 已有特征复用潜力（features/frame_features）:")
    n_actions = len(df)
    for split, n in existing_feature_coverage(valid_videos).items():
        pct = 100.0 * n / max(n_actions, 1)
        print(f"    {split}: {n} 个动作特征（覆盖 DCR valid 动作 {pct:.1f}%）")


if __name__ == "__main__":
    main()
