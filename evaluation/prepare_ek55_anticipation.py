"""Prepare the official RULSTM EK55 train/validation anticipation manifests.

The public INSIGHT repository does not include the EK55 Top-k evaluation
pipeline.  This script follows the video split released by RULSTM and creates
one sample per labeled action.  Frame indices are sampled strictly before the
target action, with the newest observation at ``anticipation_time`` seconds
before the action starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


# fpv-iplab/rulstm, RULSTM/data/ek55/validation_videos.csv
RULSTM_EK55_VALIDATION_VIDEOS = {
    "P01_01", "P01_10", "P02_03", "P02_05", "P03_06", "P03_11",
    "P04_09", "P06_05", "P07_02", "P07_08", "P07_10", "P08_01",
    "P08_05", "P08_12", "P10_01", "P13_04", "P13_06", "P13_09",
    "P14_01", "P14_02", "P20_03", "P20_04", "P22_08", "P22_10",
    "P22_11", "P22_13", "P23_03", "P24_08", "P25_11", "P26_02",
    "P26_11", "P26_16", "P27_03", "P28_05", "P28_12", "P28_13",
    "P30_01", "P30_03", "P31_01", "P31_08",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--anticipation-time", type=float, default=1.0)
    parser.add_argument("--expected-actions", type=int, default=2513)
    return parser.parse_args()


def normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def sampled_frames(
    start_frame: int,
    fps: float,
    time_step: float,
    sequence_length: int,
    anticipation_time: float,
) -> list[int]:
    """Return chronological frames, ending at or before start - tau_a."""
    offsets = [
        anticipation_time + time_step * index
        for index in range(sequence_length - 1, -1, -1)
    ]
    return [math.floor(start_frame - offset * fps) for offset in offsets]


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.time_step <= 0 or args.sequence_length <= 0:
        raise ValueError("fps, time-step and sequence-length must be positive")
    if args.anticipation_time <= 0:
        raise ValueError("anticipation-time must be positive")

    with args.annotations_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "uid", "video_id", "start_frame", "stop_frame", "verb", "noun",
        "verb_class", "noun_class",
    }
    missing_columns = required.difference(rows[0] if rows else {})
    if missing_columns:
        raise ValueError(f"Missing annotation columns: {sorted(missing_columns)}")

    pairs = sorted({
        (int(row["verb_class"]), int(row["noun_class"])) for row in rows
    })
    if args.expected_actions and len(pairs) != args.expected_actions:
        raise ValueError(
            f"Found {len(pairs)} unique verb-noun pairs; expected "
            f"{args.expected_actions}. Check that this is the complete EK55 "
            "EPIC_train_action_labels.csv."
        )
    pair_to_action = {pair: index for index, pair in enumerate(pairs)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "num_actions": len(pairs),
        "num_verbs": max(int(row["verb_class"]) for row in rows) + 1,
        "num_nouns": max(int(row["noun_class"]) for row in rows) + 1,
        "actions": [
            {"action_class": index, "verb_class": verb, "noun_class": noun}
            for index, (verb, noun) in enumerate(pairs)
        ],
    }
    (args.output_dir / "action_vocab.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n"
    )

    outputs = {
        split: (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val")
    }
    counts = Counter()
    video_sets = {"train": set(), "val": set()}
    try:
        for row in rows:
            video = row["video_id"].strip()
            split = "val" if video in RULSTM_EK55_VALIDATION_VIDEOS else "train"
            counts[f"{split}_raw"] += 1
            video_sets[split].add(video)
            start = int(row["start_frame"])
            frames = sampled_frames(
                start,
                args.fps,
                args.time_step,
                args.sequence_length,
                args.anticipation_time,
            )
            # Match RULSTM: pad early timestamps with the first valid past
            # frame, and discard only samples with no past frame at all.
            positive_frames = [frame for frame in frames if frame >= 1]
            if not positive_frames:
                counts[f"{split}_insufficient_history"] += 1
                continue
            first_valid = min(positive_frames)
            frames = [frame if frame >= 1 else first_valid for frame in frames]
            verb = int(row["verb_class"])
            noun = int(row["noun_class"])
            record = {
                "uid": int(row["uid"]),
                "video_id": video,
                "participant_id": row.get("participant_id") or video.split("_")[0],
                "start_frame": start,
                "stop_frame": int(row["stop_frame"]),
                "sample_frames": frames,
                "anticipation_time": args.anticipation_time,
                "time_step": args.time_step,
                "verb": normalize_token(row["verb"]),
                "noun": normalize_token(row["noun"]),
                "verb_class": verb,
                "noun_class": noun,
                "action_class": pair_to_action[(verb, noun)],
            }
            outputs[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for output in outputs.values():
            output.close()

    metadata = {
        "source": "fpv-iplab/rulstm EK55 video split",
        "annotations": str(args.annotations_csv),
        "fps": args.fps,
        "time_step": args.time_step,
        "sequence_length": args.sequence_length,
        "anticipation_time": args.anticipation_time,
        "train_samples": counts["train"],
        "val_samples": counts["val"],
        "train_raw_actions": counts["train_raw"],
        "val_raw_actions": counts["val_raw"],
        "train_videos": len(video_sets["train"]),
        "val_videos": len(video_sets["val"]),
        "train_insufficient_history": counts["train_insufficient_history"],
        "val_insufficient_history": counts["val_insufficient_history"],
        **{key: vocabulary[key] for key in ("num_actions", "num_verbs", "num_nouns")},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
