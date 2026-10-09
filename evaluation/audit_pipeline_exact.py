#!/usr/bin/env python3
"""Produce a compact, machine-readable audit of the reproduced INSIGHT pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_stats(path: Path) -> dict[str, Any]:
    rows = 0
    videos: set[str] = set()
    history_lengths: Counter[int] = Counter()
    future_lengths: Counter[int] = Counter()
    existing_images = 0
    keys: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            if not keys:
                keys = sorted(row)
            clip_uid = row.get("clip_uid")
            if clip_uid is not None:
                videos.add(str(clip_uid))
            if "stage1_history" in row:
                history_lengths[len(row["stage1_history"])] += 1
            if "future_actions" in row:
                future_lengths[len(row["future_actions"])] += 1
            images = row.get("images") or []
            existing_images += int(bool(images) and Path(images[0]).is_file())

    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": rows,
        "videos": len(videos),
        "keys": keys,
        "history_lengths": dict(history_lengths),
        "future_lengths": dict(future_lengths),
        "existing_images": existing_images,
    }


def audit_features(data_root: Path) -> dict[str, Any]:
    frame_root = data_root / "features/frame_features"
    mask_root = data_root / "features/mask_features"
    frames = {p.relative_to(frame_root).as_posix(): p for p in frame_root.rglob("*.pt")}
    masks = {p.relative_to(mask_root).as_posix(): p for p in mask_root.rglob("*.pt")}
    common = sorted(frames.keys() & masks.keys())

    frame_shapes: Counter[str] = Counter()
    mask_shapes: Counter[str] = Counter()
    frame_dtypes: Counter[str] = Counter()
    mask_dtypes: Counter[str] = Counter()
    equal = nonfinite_frame = nonfinite_mask = zero_frame = zero_mask = 0

    for key in common:
        frame = torch.load(frames[key], map_location="cpu", weights_only=True)
        mask = torch.load(masks[key], map_location="cpu", weights_only=True)
        frame_shapes[str(tuple(frame.shape))] += 1
        mask_shapes[str(tuple(mask.shape))] += 1
        frame_dtypes[str(frame.dtype)] += 1
        mask_dtypes[str(mask.dtype)] += 1
        equal += int(torch.equal(frame, mask))
        nonfinite_frame += int(not torch.isfinite(frame).all())
        nonfinite_mask += int(not torch.isfinite(mask).all())
        zero_frame += int(torch.count_nonzero(frame) == 0)
        zero_mask += int(torch.count_nonzero(mask) == 0)

    return {
        "frame_files": len(frames),
        "mask_files": len(masks),
        "frame_by_split": dict(Counter(k.split("/")[0] for k in frames)),
        "mask_by_split": dict(Counter(k.split("/")[0] for k in masks)),
        "matched_pairs": len(common),
        "frame_only": len(frames.keys() - masks.keys()),
        "mask_only": len(masks.keys() - frames.keys()),
        "frame_shapes": dict(frame_shapes),
        "mask_shapes": dict(mask_shapes),
        "frame_dtypes": dict(frame_dtypes),
        "mask_dtypes": dict(mask_dtypes),
        "frame_equals_mask": equal,
        "frame_differs_mask": len(common) - equal,
        "nonfinite_frame": nonfinite_frame,
        "nonfinite_mask": nonfinite_mask,
        "all_zero_frame": zero_frame,
        "all_zero_mask": zero_mask,
    }


def audit_stage1(data_root: Path) -> dict[str, Any]:
    checkpoint = data_root / "stage1_rebuilt_checkpoint/best_model.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {
        "checkpoint": str(checkpoint),
        "sha256": sha256(checkpoint),
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "frame_input_shape": list(state["frame_stage1.weight"].shape),
        "mask_input_shape": list(state["mask_stage1.weight"].shape),
        "verb_head_shape": list(state["verb_head.weight"].shape),
        "noun_head_shape": list(state["noun_head.weight"].shape),
        "train_predictions": jsonl_stats(
            data_root / "stage1_predictions/stage1_train_predictions.jsonl"
        ),
        "val_predictions": jsonl_stats(
            data_root / "stage1_predictions/stage1_val_predictions.jsonl"
        ),
    }


def audit_stage2(data_root: Path, repo_root: Path, installed_plugin_root: Path) -> dict[str, Any]:
    run = (
        data_root
        / "stage2_stage1_history_full_3gpu_v3_output/v0-20260729-235831"
    )
    final_checkpoint = run / "checkpoint-1773"
    args = json.loads((run / "args.json").read_text(encoding="utf-8"))
    state = json.loads(
        (final_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    adapter = json.loads(
        (final_checkpoint / "adapter_config.json").read_text(encoding="utf-8")
    )
    wanted = [
        "model", "model_type", "dataset", "val_dataset", "rlhf_type",
        "train_type", "tuner_type", "external_plugins", "reward_funcs",
        "reward_weights", "num_generations", "temperature", "max_length",
        "max_completion_length", "soft_cache_length", "num_train_epochs",
        "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "learning_rate", "torch_dtype",
        "deepspeed", "save_steps", "eval_steps", "logging_steps",
        "lora_rank", "lora_alpha", "target_modules", "seed",
    ]
    plugins = {}
    for name in ("plugin_in_log_intention.py", "my_rewards_intention.py"):
        repo_file = repo_root / "CognitiveReasoning" / name
        installed_file = installed_plugin_root / name
        plugins[name] = {
            "repo_sha256": sha256(repo_file) if repo_file.is_file() else None,
            "installed_sha256": sha256(installed_file) if installed_file.is_file() else None,
            "identical": (
                repo_file.is_file()
                and installed_file.is_file()
                and sha256(repo_file) == sha256(installed_file)
            ),
        }

    candidates_root = data_root / "stage2_generated_topk_checkpoint1773"
    candidates = []
    for path in sorted(candidates_root.glob("candidate_*.jsonl")):
        candidates.append(jsonl_stats(path))

    return {
        "run": str(run),
        "args": {key: args[key] for key in wanted if key in args},
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "best_metric": state.get("best_metric"),
        "global_step": state.get("global_step"),
        "final_adapter_sha256": sha256(final_checkpoint / "adapter_model.safetensors"),
        "adapter_config": adapter,
        "plugins": plugins,
        "train_dataset": jsonl_stats(
            data_root / "stage2_stage1_history/stage2_train_stage1_history.jsonl"
        ),
        "val_dataset": jsonl_stats(
            data_root / "stage2_stage1_history/stage2_val_stage1_history.jsonl"
        ),
        "candidates": candidates,
        "metrics": json.loads(
            (candidates_root / "metrics_all.json").read_text(encoding="utf-8")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/datasets/EPIC-KITCHENS")
    )
    parser.add_argument(
        "--installed-plugin-root",
        type=Path,
        default=Path(
            "/data/conda_envs/insight_env/lib/python3.10/site-packages/swift/plugin"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "features": audit_features(args.data_root),
        "stage1": audit_stage1(args.data_root),
        "stage2": audit_stage2(
            args.data_root, args.repo_root, args.installed_plugin_root
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
