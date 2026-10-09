"""Convert Stage-2 GRPO records into supervised ms-swift chat records.

The existing Stage-2 files contain a user message plus the reference completion in
``gt_answer``/``solution``. SFT needs that reference as an assistant message. This
script performs only that conversion; it does not replace Stage-1 predicted
history with ground truth history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TAG_PATTERN = re.compile(
    r"^\s*<think>.*?</think>\s*<intention>.*?</intention>\s*"
    r"<answer>.*?</answer>\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-subset-size", type=int, default=100)
    parser.add_argument("--smoke-train-size", type=int, default=32)
    parser.add_argument("--smoke-val-size", type=int, default=12)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def reference_answer(row: dict[str, Any]) -> str:
    for key in ("gt_answer", "solution"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"Missing gt_answer/solution for {row.get('clip_uid')}:{row.get('observation_action_idx')}"
    )


def convert(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Every row must contain at least one message")

    # Source inference rows should contain only the prompt. Removing any stale
    # assistant message makes the conversion idempotent and prevents label leak.
    prompt_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") != "assistant"
    ]
    if not prompt_messages or prompt_messages[-1].get("role") != "user":
        raise ValueError("The last non-assistant message must be the user prompt")

    answer = reference_answer(row)
    if not TAG_PATTERN.fullmatch(answer):
        raise ValueError(
            f"Malformed tagged answer for {row.get('clip_uid')}:{row.get('observation_action_idx')}"
        )

    result = dict(row)
    result["messages"] = prompt_messages + [{"role": "assistant", "content": answer}]
    return result


def stable_key(row: dict[str, Any]) -> str:
    identity = f"{row.get('clip_uid', '')}:{row.get('observation_action_idx', '')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(name: str, rows: list[dict[str, Any]]) -> None:
    videos = Counter(str(row.get("clip_uid", "")) for row in rows)
    image_count = sum(
        isinstance(row.get("images"), list) and len(row["images"]) == 1 for row in rows
    )
    assistant_count = sum(
        sum(message.get("role") == "assistant" for message in row["messages"]) == 1
        for row in rows
    )
    print(
        f"{name}: rows={len(rows)} videos={len(videos)} "
        f"one_image={image_count} one_assistant={assistant_count}"
    )


def main() -> None:
    args = parse_args()
    if args.val_subset_size <= 0 or args.smoke_train_size <= 0 or args.smoke_val_size <= 0:
        raise ValueError("Subset sizes must be positive")

    train = [convert(row) for row in read_jsonl(args.train)]
    val = [convert(row) for row in read_jsonl(args.val)]
    if len(val) < args.val_subset_size:
        raise ValueError(
            f"Validation dataset has {len(val)} rows, fewer than {args.val_subset_size}"
        )

    # A hash-selected validation subset is deterministic, spans the source file,
    # and remains disjoint from the training split.
    val_subset = sorted(val, key=stable_key)[: args.val_subset_size]
    smoke_train = sorted(train, key=stable_key)[: min(args.smoke_train_size, len(train))]
    smoke_val = sorted(val, key=stable_key)[: min(args.smoke_val_size, len(val))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": (args.output_dir / "stage2_train_sft.jsonl", train),
        "val": (args.output_dir / "stage2_val_sft.jsonl", val),
        "val100": (args.output_dir / "stage2_val_sft_100.jsonl", val_subset),
        "smoke_train": (args.output_dir / "stage2_train_sft_smoke.jsonl", smoke_train),
        "smoke_val": (args.output_dir / "stage2_val_sft_smoke.jsonl", smoke_val),
    }
    for name, (path, rows) in outputs.items():
        write_jsonl(path, rows)
        summarize(name, rows)
        print(f"  output={path}")


if __name__ == "__main__":
    main()
