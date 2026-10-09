"""Build EPIC-KITCHENS Stage 2 samples for long-term anticipation.

Each sample observes the final frame of one annotated action and uses the next
20 actions from the same video as the target.  This is a deterministic EK55
proxy for the missing official INSIGHT Stage 2 data-preparation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


USER_PROMPT = (
    "Analyze the observed egocentric frame and predict exactly {horizon} future "
    "actions in chronological order. Use lowercase EPIC-KITCHENS labels. "
    "Return exactly this structure: "
    "<think>brief visual reasoning</think>"
    "<intention>likely task intention</intention>"
    "<answer>verb noun, verb noun, ...</answer>. "
    "The answer must contain exactly {horizon} comma-separated actions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/datasets/EPIC-KITCHENS"),
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output-prefix", default="stage2")
    return parser.parse_args()


def action_pair(row: dict[str, str]) -> str:
    return f'{row["verb"]} {row["noun"]}'.lower().strip()


def infer_intention(future: list[dict[str, str]]) -> str:
    """Create a varying, reproducible proxy intention from future objects."""
    nouns = [row["noun"].lower().strip() for row in future]
    first_position = {noun: i for i, noun in enumerate(nouns)}
    ranked = sorted(
        Counter(nouns).items(),
        key=lambda item: (-item[1], first_position[item[0]], item[0]),
    )
    salient = [noun for noun, _ in ranked[:3]]
    return "continue the kitchen task involving " + ", ".join(salient)


def load_actions(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_video[row["video_id"]].append(row)
    for rows in by_video.values():
        rows.sort(key=lambda row: (int(row["start_frame"]), int(row["uid"])))
    return by_video


def load_split_videos(annotation_path: Path) -> set[str]:
    with annotation_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)["clips"]
    return {item["clip_uid"] for item in annotations}


def find_observation_frame(data_root: Path, row: dict[str, str]) -> Path | None:
    participant = row["video_id"].split("_")[0]
    frame_dir = data_root / participant / "rgb_frames"
    start = int(row["start_frame"])
    stop = int(row["stop_frame"])

    # Prefer the last observed frame, then tolerate annotation/frame indexing
    # differences without crossing into the future action.
    candidates = [stop, stop - 1, start, start + 1]
    for frame_index in candidates:
        if frame_index < 0:
            continue
        path = frame_dir / f"frame_{frame_index:010d}.jpg"
        if path.exists():
            return path
    return None


def build_split(
    data_root: Path,
    actions_by_video: dict[str, list[dict[str, str]]],
    split: str,
    horizon: int,
    output_prefix: str,
) -> tuple[int, int]:
    annotation_path = data_root / "insight_annotations" / f"{split}.json"
    split_videos = load_split_videos(annotation_path)
    output_path = data_root / f"{output_prefix}_{split}_lta.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_frames = 0
    with output_path.open("w", encoding="utf-8") as output:
        for video_id in sorted(split_videos):
            rows = actions_by_video.get(video_id, [])
            for index in range(max(0, len(rows) - horizon)):
                current = rows[index]
                future = rows[index + 1 : index + 1 + horizon]
                if len(future) != horizon:
                    continue

                image_path = find_observation_frame(data_root, current)
                if image_path is None:
                    missing_frames += 1
                    continue

                current_action = action_pair(current)
                future_actions = [action_pair(row) for row in future]
                intention = infer_intention(future)
                reference = (
                    f"<think>The observed action is {current_action}; infer the "
                    f"likely continuation from the visible context.</think>"
                    f"<intention>{intention}</intention>"
                    f"<answer>{', '.join(future_actions)}</answer>"
                )
                record = {
                    "images": [str(image_path)],
                    "messages": [
                        {
                            "role": "user",
                            "content": USER_PROMPT.format(horizon=horizon),
                        },
                        {"role": "assistant", "content": reference},
                    ],
                    # Explicit columns make the custom reward independent of
                    # how a particular ms-swift version strips assistant turns.
                    "gt_answer": reference,
                    "gt_intention": intention,
                    "solution": reference,
                    "clip_uid": video_id,
                    "observation_action_idx": int(current["uid"]),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(
        f"[{split}] wrote {written} samples to {output_path}; "
        f"missing observation frames: {missing_frames}"
    )
    return written, missing_frames


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")

    csv_path = args.data_root / "annotations" / "EPIC_train_action_labels.csv"
    actions_by_video = load_actions(csv_path)
    for split in args.splits:
        build_split(
            data_root=args.data_root,
            actions_by_video=actions_by_video,
            split=split,
            horizon=args.horizon,
            output_prefix=args.output_prefix,
        )


if __name__ == "__main__":
    main()
