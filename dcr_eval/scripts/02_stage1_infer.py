"""02 — Stage 1 推理 + 8 历史构造 → Stage 2 输入。

两步：
  A. 对 features/val/ 全部片段跑 Stage 1（best_model.pth + 共现矩阵）
     → stage1_predictions/stage1_val_predictions.jsonl（每段的 Top-1 pred_verb/pred_noun）
  B. 仅对“已有 HISTORY 段历史、且后面仍有 HORIZON 段完整未来”的位置
     构造 Stage 2 输入：
     历史 = 前 8 段的 Stage 1 预测文本（从旧到新）
     图像 = 第 index-1 段（历史最后一段）的 stop_frame 帧
     GT   = 该段的 verb/noun 标签（valid 有标签；test 无标签时 GT 字段留空）
     → stage1_predictions/stage2_input.jsonl（ms-swift RLHF JSONL 格式）

运行：python scripts/02_stage1_infer.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from data_utils import load_action_labels, load_video_ids  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "HandObject"))
from model import ActionRecognitionModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos-file", type=Path, default=None,
        help="只处理该文件列出的视频（00_audit 导出的 unseen_valid_videos.txt）；"
             "缺省处理全部 DCR valid。",
    )
    return parser.parse_args()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return {key.removeprefix("module."): value for key, value in state.items()}


class FeatureDataset(Dataset):
    """读 features/val/{video}_{uid}.pt（frame 与 mask 同文件）。"""

    def __init__(self, annotations: Path, feature_root: Path, split: str):
        data = json.loads((annotations / f"{split}.json").read_text())
        self.samples = []
        self.rows = []
        for clip in data.get("clips", []):
            action_id = f'{clip["clip_uid"]}_{clip["action_idx"]}'
            path = feature_root / split / f"{action_id}.pt"
            if not path.exists():
                continue
            self.samples.append((path, clip["clip_uid"], int(clip["action_idx"])))
            self.rows.append(clip)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, video, uid = self.samples[i]
        feat = torch.load(path, map_location="cpu", weights_only=True).float()
        if tuple(feat.shape) != (C.INPUT_DIM,):
            raise ValueError(
                f"{path}: 期望特征 shape=({C.INPUT_DIM},)，实际={tuple(feat.shape)}"
            )
        if not torch.isfinite(feat).all():
            raise ValueError(f"{path}: 特征包含 NaN/Inf")
        return feat, feat, video, uid  # frame, mask(=frame), video, uid


# ────────────────────────────────────────────────────────
# 类名映射（verb/noun id → 文本，供历史动作文本与 GT 文本）
# 按列名读取（不依赖 CSV 行序）；范围断言在运行前暴露映射错误
# ────────────────────────────────────────────────────────
def _id_name_maps(csv_path: Path, id_candidates, name_candidates, expected_ids):
    df = pd.read_csv(csv_path)
    print(f"[labels] {csv_path.name} columns={list(df.columns)}")
    id_col = next((c for c in id_candidates if c in df.columns), df.columns[0])
    name_col = next((c for c in name_candidates if c in df.columns),
                    [c for c in df.columns if c != id_col][0])
    mapping = {}
    for _, r in df.iterrows():
        mapping[int(r[id_col])] = str(r[name_col]).lower().strip()
    if set(mapping) != set(expected_ids):
        raise ValueError(
            f"{csv_path.name}: id 集合 {sorted(mapping)[:5]}... 与预期 {len(expected_ids)} 类不符"
        )
    return mapping


def load_label_maps():
    verbs = _id_name_maps(
        C.DCR_VERB_CSV, ["verb_id", "id"], ["verb", "name"], set(range(125))
    )
    nouns = _id_name_maps(
        C.DCR_NOUN_CSV, ["noun_id", "id"], ["noun", "name"], set(range(352))
    )
    return verbs, nouns


def action_text(pred, verbs, nouns):
    return f"{verbs[pred[0]]} {nouns[pred[1]]}"


def observation_image(participant: str, start_frame: int, stop_frame: int):
    """复刻 Stage2 训练构造器的图像回退顺序：stop, stop-1, start。"""
    for fi in (stop_frame, stop_frame - 1, start_frame):
        p = C.EK55_FRAMES / participant / "rgb_frames" / f"frame_{fi:010d}.jpg"
        if p.exists():
            return str(p)
    return None


# ────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    feature_spec_path = C.ANN_DIR / "feature_spec.json"
    if not feature_spec_path.exists():
        raise RuntimeError(
            f"缺少 {feature_spec_path}；不能证明 Stage1 输入特征的生成配置。"
            "请先运行 01_prepare_features.py。"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActionRecognitionModel(
        input_dim=C.INPUT_DIM, mlp_hidden_dim=2048, mlp_output_dim=256,
        transformer_layers=4, n_heads=8, transformer_hidden_dim=2048,
        verb_num_classes=C.VERB_CLASSES, noun_num_classes=C.NOUN_CLASSES,
    )
    model.load_state_dict(load_state(C.STAGE1_CKPT), strict=True)
    model.to(device).eval()

    co = torch.load(C.COOCCURRENCE, map_location="cpu", weights_only=True)
    expected_n_given_v = (C.VERB_CLASSES, C.NOUN_CLASSES)
    expected_v_given_n = (C.NOUN_CLASSES, C.VERB_CLASSES)
    if tuple(co["P_n_given_v"].shape) != expected_n_given_v:
        raise ValueError(
            f"P_n_given_v shape={tuple(co['P_n_given_v'].shape)}, expected={expected_n_given_v}"
        )
    if tuple(co["P_v_given_n"].shape) != expected_v_given_n:
        raise ValueError(
            f"P_v_given_n shape={tuple(co['P_v_given_n'].shape)}, expected={expected_v_given_n}"
        )
    semantic_prior = 0.5 * (co["P_n_given_v"].float() + co["P_v_given_n"].float().t()).to(device)

    verbs, nouns = load_label_maps()

    # 标注（含 GT）：A 段门控指标与 B 段历史构造共用；UID 处理与 01 一致
    valid_videos = load_video_ids(C.DCR_VALID_CSV)
    df = load_action_labels(C.DCR_TRAIN_PKL)
    df = df[df["video_id"].isin(valid_videos)]
    if df.empty:
        raise RuntimeError("DCR valid 过滤后没有任何 action")
    if args.videos_file is not None:
        subset = {line.strip() for line in args.videos_file.read_text().splitlines() if line.strip()}
        if not subset:
            raise SystemExit(f"[FATAL] {args.videos_file} 为空 —— 该口径没有可评测视频，终止")
        df = df[df["video_id"].isin(subset)]
        print(f"[subset] {args.videos_file.name}: {df['video_id'].nunique()} 视频 / {len(df)} 动作")
    gt_map = {
        (row.video_id, int(row.uid)): (int(row.verb_class), int(row.noun_class))
        for row in df.itertuples(index=False)
    }
    # 与 build_stage2_from_stage1.py 一致：用整数帧号排序，UID 作稳定次键。
    df = df.sort_values(["video_id", "start_frame", "uid"])

    ds = FeatureDataset(C.ANN_DIR, C.FEAT_DIR, "val")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"[dataset] usable={len(ds)}")

    # A. Stage 1 推理 + 门控指标（先看识别质量，再决定是否跑 Stage 2）
    predictions = {}    # (video, uid) -> (pred_verb, pred_noun)
    action_scores = {}  # (video, uid) -> Top-1 联合分数
    correct = {k: 0 for k in ("verb_top1", "noun_top1", "action_top1")}
    total = 0
    pred_action_counter = Counter()
    with torch.inference_mode():
        for frame, mask, videos, uids in tqdm(loader, desc="Stage1 infer"):
            verb_logits, noun_logits = model(frame.to(device), mask.to(device))
            verb_prob = verb_logits.softmax(-1)
            noun_prob = noun_logits.softmax(-1)
            verb_top1 = verb_prob.argmax(-1)
            noun_top1 = noun_prob.argmax(-1)
            joint = verb_prob.unsqueeze(2) * noun_prob.unsqueeze(1) * semantic_prior.unsqueeze(0)
            action_score, action_top1 = joint.flatten(1).max(-1)
            for video, uid, pv_cls, pn_cls, action_idx, score in zip(
                    videos, uids.tolist(), verb_top1, noun_top1, action_top1, action_score):
                key = (video, int(uid))
                pv = int(action_idx // C.NOUN_CLASSES)
                pn = int(action_idx % C.NOUN_CLASSES)
                predictions[key] = (pv, pn)
                action_scores[key] = float(score)
                gt = gt_map.get(key)
                if gt is None:
                    continue
                tv, tn = gt
                total += 1
                true_action = tv * C.NOUN_CLASSES + tn
                correct["verb_top1"] += int(pv_cls == tv)
                correct["noun_top1"] += int(pn_cls == tn)
                correct["action_top1"] += int(action_idx == true_action)
                pred_action_counter[f"{verbs[pv]} {nouns[pn]}"] += 1

    if total == 0:
        raise RuntimeError("没有带 GT 的样本参与门控指标（检查 GT 映射与子集过滤）")
    stage1_metrics = {
        "split": "val",
        "checkpoint": str(C.STAGE1_CKPT),
        "cooccurrence": str(C.COOCCURRENCE),
        "feature_spec": json.loads(feature_spec_path.read_text()),
        "labeled_samples": total,
        "verb_top1": round(100.0 * correct["verb_top1"] / total, 3),
        "noun_top1": round(100.0 * correct["noun_top1"] / total, 3),
        "action_top1": round(100.0 * correct["action_top1"] / total, 3),
        "prediction_diversity": {
            "unique_predicted_actions": len(pred_action_counter),
            "top_common": pred_action_counter.most_common(10),
        },
    }
    (C.PRED_DIR / "stage1_metrics.json").write_text(
        json.dumps(stage1_metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stage1_metrics, ensure_ascii=False, indent=2))
    print("[gate] 门控判定: 若 action_top1 接近随机或 top_common 占比过高（预测坍缩），"
          "先修复识别，不必立即跑 Stage 2")

    pred_path = C.PRED_DIR / "stage1_val_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as out:
        for (video, uid), (v, n) in sorted(predictions.items()):
            out.write(json.dumps({
                "clip_uid": video, "action_idx": uid,
                "pred_verb": v, "pred_noun": n,
                "action_score": action_scores[(video, uid)],
            }, ensure_ascii=False) + "\n")
    print(f"[stage1] predictions → {pred_path}")

    # B. 8 历史构造 → Stage 2 输入
    # df 已在 A 段加载（含子集过滤），且已按 [video_id, start_frame, uid] 排序


    # PROMPT 逐字复刻训练原文（build_stage2_from_stage1.py），保证推理分布与训练一致
    PROMPT = """The Stage 1 model predicted the following observed egocentric actions, from oldest to newest:
{history}

Using the image and predicted action history, predict exactly 20 future actions in chronological order. Every action must contain exactly two whitespace-separated EPIC-KITCHENS tokens: one verb and one noun. Compound tokens may contain a hyphen or colon. Keep <think> under 20 words and <intention> under 12 words. Return only:
<think>brief reasoning</think><intention>likely task intention</intention><answer>verb noun, verb noun, ... exactly 20 actions</answer>"""

    written = missing_history = missing_image = noncausal_history = candidate_positions = 0
    stage2_path = C.PRED_DIR / "stage2_input.jsonl"
    with stage2_path.open("w", encoding="utf-8") as out:
        for video, group in df.groupby("video_id", sort=False):
            rows = list(group.itertuples(index=False))
            # Top-1 只需 target 自身有 GT，不要求其后另有 19 个 GT。
            # 模型输出格式仍保持训练时的 HORIZON=20，不改变提示词与解码分布。
            for i in range(C.HISTORY, len(rows) - C.EVAL_FUTURE_REQUIRED + 1):
                candidate_positions += 1
                target = rows[i]
                hist_rows = rows[i - C.HISTORY:i]
                # 目标动作的任何帧都不得进入输入。按 start_frame 排序仍不足以排除
                # 重叠标注，因此显式要求所有历史段在目标开始前结束。
                if any(int(h.stop_frame) >= int(target.start_frame) for h in hist_rows):
                    noncausal_history += 1
                    continue
                hist_text = []
                for h in hist_rows:
                    pred = predictions.get((video, int(h.uid)))
                    if pred is None:
                        break
                    hist_text.append(action_text(pred, verbs, nouns))
                if len(hist_text) != C.HISTORY:
                    missing_history += 1
                    continue
                image = observation_image(
                    target.participant_id,
                    int(hist_rows[-1].start_frame),
                    int(hist_rows[-1].stop_frame),
                )
                if image is None:
                    missing_image += 1
                    continue
                gt = f"{verbs[int(target.verb_class)]} {nouns[int(target.noun_class)]}"
                # 历史填充与训练一致：编号换行（"1. take knife\n2. open fridge"）
                history_text = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(hist_text))
                record = {
                    "images": [image],
                    # 推理输入只保留 user 消息（训练数据即如此，GT 单独存字段）
                    "messages": [{"role": "user", "content": PROMPT.format(history=history_text)}],
                    "gt_action_text": gt,
                    "gt_verb_class": int(target.verb_class),
                    "gt_noun_class": int(target.noun_class),
                    "clip_uid": video,
                    "action_idx": int(target.uid),
                    "is_challenge": False,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    build_metrics = {
        "run_name": C.RUN_NAME,
        "history": C.HISTORY,
        "generation_horizon": C.HORIZON,
        "future_gt_required": C.EVAL_FUTURE_REQUIRED,
        "candidate_positions": candidate_positions,
        "written": written,
        "missing_history": missing_history,
        "missing_image": missing_image,
        "noncausal_history": noncausal_history,
        "coverage_pct": round(100.0 * written / max(candidate_positions, 1), 3),
    }
    (C.PRED_DIR / "stage2_input_metrics.json").write_text(
        json.dumps(build_metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"[stage2 input] {json.dumps(build_metrics, ensure_ascii=False)} → {stage2_path}")


if __name__ == "__main__":
    main()
