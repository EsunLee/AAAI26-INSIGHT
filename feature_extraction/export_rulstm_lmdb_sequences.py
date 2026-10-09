"""Export official RULSTM EK55 RGB LMDB features as tau=1s sequences.

The released LMDB uses video frames resampled to 30 FPS and stores a
1024-dimensional TSN-RGB vector at the timestamps needed by RULSTM.  This
adapter selects observations from 3.5s through 1.0s before each target action,
so no target-action frame is consumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rulstm-data-dir", type=Path, required=True)
    parser.add_argument("--lmdb-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--anticipation-time", type=float, default=1.0)
    parser.add_argument("--earliest-time", type=float, default=3.5)
    parser.add_argument("--input-dim", type=int, default=1024)
    return parser.parse_args()


def read_annotations(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        for values in reader:
            if not values:
                continue
            uid, video, start, stop, verb, noun, action = values
            rows.append({
                "uid": int(uid), "video_id": video.strip(),
                "start_frame": int(start), "stop_frame": int(stop),
                "verb_class": int(verb), "noun_class": int(noun),
                "action_class": int(action),
            })
    return rows


def read_actions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actions = [{
        "action_class": int(row["id"]),
        "verb_class": int(row["verb"]),
        "noun_class": int(row["noun"]),
        "action": row["action"],
    } for row in rows]
    actions.sort(key=lambda row: row["action_class"])
    expected = list(range(len(actions)))
    if [row["action_class"] for row in actions] != expected:
        raise ValueError("RULSTM actions.csv IDs are not contiguous")
    return actions


def sample_frames(
    start: int, fps: float, earliest: float, latest: float, step: float
) -> list[int]:
    offsets = np.arange(earliest, latest - step / 2, -step)
    frames = [math.floor(start - float(offset) * fps) for offset in offsets]
    positive = [frame for frame in frames if frame >= 1]
    if not positive:
        return []
    first = min(positive)
    return [frame if frame >= 1 else first for frame in frames]


def main() -> None:
    args = parse_args()
    if not (0 < args.anticipation_time <= args.earliest_time):
        raise ValueError("Require 0 < anticipation-time <= earliest-time")
    if args.time_step <= 0 or args.fps <= 0:
        raise ValueError("fps and time-step must be positive")
    actions = read_actions(args.rulstm_data_dir / "actions.csv")
    if len(actions) != 2513:
        raise ValueError(f"Expected 2513 EK55 actions, found {len(actions)}")
    vocabulary = {
        "num_actions": len(actions), "num_verbs": 125, "num_nouns": 352,
        "actions": actions,
    }
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    args.features_dir.mkdir(parents=True, exist_ok=True)
    (args.manifest_dir / "action_vocab.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n"
    )

    env = lmdb.open(
        str(args.lmdb_dir), readonly=True, lock=False, readahead=False,
        max_readers=1, subdir=True,
    )
    stats = Counter()
    split_counts = {}
    try:
        with env.begin(buffers=True) as transaction:
            for split, filename in (("train", "training.csv"), ("val", "validation.csv")):
                rows = read_annotations(args.rulstm_data_dir / filename)
                output_dir = args.features_dir / split
                output_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = args.manifest_dir / f"{split}.jsonl"
                with manifest_path.open("w", encoding="utf-8") as manifest:
                    for row in tqdm(rows, desc=f"RULSTM LMDB {split}"):
                        frames = sample_frames(
                            row["start_frame"], args.fps, args.earliest_time,
                            args.anticipation_time, args.time_step,
                        )
                        if not frames:
                            stats[f"{split}_insufficient_history"] += 1
                            continue
                        features = []
                        for frame in frames:
                            key = f'{row["video_id"]}_frame_{frame:010d}.jpg'.encode()
                            value = transaction.get(key)
                            if value is None:
                                features = []
                                break
                            vector = np.frombuffer(value, dtype=np.float32)
                            if vector.size != args.input_dim:
                                raise ValueError(
                                    f"LMDB key {key!r} has dim {vector.size}, "
                                    f"expected {args.input_dim}"
                                )
                            features.append(torch.from_numpy(vector.copy()))
                        if not features:
                            stats[f"{split}_missing_sequence"] += 1
                            continue
                        feature_path = output_dir / f'{row["uid"]}.pt'
                        if not feature_path.exists():
                            torch.save(torch.stack(features).half(), feature_path)
                        record = {
                            **row,
                            "participant_id": row["video_id"].split("_")[0],
                            "sample_frames": frames,
                            "anticipation_time": args.anticipation_time,
                            "time_step": args.time_step,
                        }
                        manifest.write(json.dumps(record) + "\n")
                        stats[split] += 1
                split_counts[f"{split}_raw_actions"] = len(rows)
    finally:
        env.close()

    metadata = {
        "source": "fpv-iplab/rulstm official EK55 TSN-RGB LMDB",
        "fps": args.fps,
        "time_step": args.time_step,
        "sequence_length": int(round(
            (args.earliest_time - args.anticipation_time) / args.time_step
        )) + 1,
        "anticipation_time": args.anticipation_time,
        "earliest_time": args.earliest_time,
        "train_samples": stats["train"], "val_samples": stats["val"],
        **split_counts,
        "train_insufficient_history": stats["train_insufficient_history"],
        "val_insufficient_history": stats["val_insufficient_history"],
        "train_missing_sequence": stats["train_missing_sequence"],
        "val_missing_sequence": stats["val_missing_sequence"],
        "num_actions": 2513, "num_verbs": 125, "num_nouns": 352,
    }
    (args.manifest_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

