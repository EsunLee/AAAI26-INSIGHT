"""Build future-20 Stage-2 data whose observed action history comes from Stage 1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROMPT = """The Stage 1 model predicted the following observed egocentric actions, from oldest to newest:
{history}

Using the image and predicted action history, predict exactly 20 future actions in chronological order. Every action must contain exactly two whitespace-separated EPIC-KITCHENS tokens: one verb and one noun. Compound tokens may contain a hyphen or colon. Keep <think> under 20 words and <intention> under 12 words. Return only:
<think>brief reasoning</think><intention>likely task intention</intention><answer>verb noun, verb noun, ... exactly 20 actions</answer>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data/datasets/EPIC-KITCHENS"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    return parser.parse_args()


def load_actions(path: Path):
    by_video = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_video[row["video_id"]].append(row)
    for rows in by_video.values():
        rows.sort(key=lambda row: (int(row["start_frame"]), int(row["uid"])))
    return by_video


def load_predictions(path: Path):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[(row["clip_uid"], int(row["action_idx"]))] = row
    return result


def load_split_videos(path: Path):
    data = json.loads(path.read_text())
    return {row["clip_uid"] for row in data.get("clips", [])}


def label_maps(path: Path):
    verbs, nouns = {}, {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            verbs[int(row["verb_class"])] = row["verb"].lower().strip()
            nouns[int(row["noun_class"])] = row["noun"].lower().strip()
    return verbs, nouns


def action_text(row):
    return f'{row["verb"]} {row["noun"]}'.lower().strip()


def intention(future):
    nouns = [row["noun"].lower().strip() for row in future]
    first = {noun: i for i, noun in enumerate(nouns)}
    ranked = sorted(Counter(nouns).items(), key=lambda x: (-x[1], first[x[0]], x[0]))
    return "continue task with " + ", ".join(noun for noun, _ in ranked[:3])


def observation_image(data_root: Path, row):
    participant = row["video_id"].split("_")[0]
    root = data_root / participant / "rgb_frames"
    for index in (int(row["stop_frame"]), int(row["stop_frame"]) - 1, int(row["start_frame"])):
        path = root / f"frame_{index:010d}.jpg"
        if path.exists():
            return str(path)
    return None


def main() -> None:
    args = parse_args()
    if args.history <= 0 or args.horizon <= 0:
        raise ValueError("--history and --horizon must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = load_actions(args.data_root / "annotations" / "EPIC_train_action_labels.csv")
    verbs, nouns = label_maps(args.data_root / "annotations" / "EPIC_train_action_labels.csv")

    for split in args.splits:
        predictions = load_predictions(args.predictions_dir / f"stage1_{split}_predictions.jsonl")
        videos = load_split_videos(args.annotations / f"{split}.json")
        output_path = args.output_dir / f"stage2_{split}_stage1_history.jsonl"
        written = missing_history = missing_image = 0
        with output_path.open("w", encoding="utf-8") as output:
            for video in sorted(videos):
                rows = actions.get(video, [])
                for index in range(args.history - 1, len(rows) - args.horizon):
                    observed = rows[index - args.history + 1:index + 1]
                    future = rows[index + 1:index + 1 + args.horizon]
                    pred_rows = [predictions.get((video, int(row["uid"]))) for row in observed]
                    if any(row is None for row in pred_rows):
                        missing_history += 1
                        continue
                    image = observation_image(args.data_root, observed[-1])
                    if image is None:
                        missing_image += 1
                        continue
                    predicted_history = [
                        f'{verbs[row["pred_verb"]]} {nouns[row["pred_noun"]]}' for row in pred_rows
                    ]
                    future_actions = [action_text(row) for row in future]
                    intent = intention(future)
                    reference = (
                        f"<think>Infer the continuation from predicted history and visual context.</think>"
                        f"<intention>{intent}</intention>"
                        f"<answer>{', '.join(future_actions)}</answer>"
                    )
                    record = {
                        "images": [image],
                        "messages": [{"role": "user", "content": PROMPT.format(
                            history="\n".join(f"{i + 1}. {text}" for i, text in enumerate(predicted_history))
                        )}],
                        "gt_answer": reference,
                        "gt_intention": intent,
                        "solution": reference,
                        "clip_uid": video,
                        "observation_action_idx": int(observed[-1]["uid"]),
                        "next_action": future_actions[0],
                        "future_actions": future_actions,
                        "stage1_history": predicted_history,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
        print(
            f"[{split}] wrote={written}, missing_history={missing_history}, "
            f"missing_image={missing_image}: {output_path}"
        )


if __name__ == "__main__":
    main()
