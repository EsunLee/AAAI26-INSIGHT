"""验证 ms-swift 生成 JSONL 的行数和最小结构完整性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("expected_lines", type=int)
    parser.add_argument(
        "--require-ids", action="store_true",
        help="要求每行包含唯一的 clip_uid/action_idx",
    )
    return parser.parse_args()


def response_value(row: dict):
    if "response" in row:
        return row["response"]
    if "generated_text" in row:
        return row["generated_text"]
    if "output" in row:
        return row["output"]
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return message.get("content")
    raise ValueError("缺少 response/generated_text/output/choices 生成字段")


def main() -> None:
    args = parse_args()
    if args.expected_lines <= 0:
        raise ValueError("expected_lines must be positive")
    count = 0
    sample_ids: set[tuple[str, int]] = set()
    with args.path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{args.path}:{line_number}: JSON 损坏: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{args.path}:{line_number}: 期望 JSON object")
            response = response_value(row)
            if not isinstance(response, str):
                raise ValueError(f"{args.path}:{line_number}: 生成内容不是字符串")
            if args.require_ids:
                clip_uid = row.get("clip_uid")
                action_idx = row.get("action_idx")
                if clip_uid is None or action_idx is None:
                    raise ValueError(
                        f"{args.path}:{line_number}: 缺少 clip_uid/action_idx"
                    )
                try:
                    sid = (str(clip_uid), int(action_idx))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{args.path}:{line_number}: action_idx 不是整数"
                    ) from exc
                if sid in sample_ids:
                    raise ValueError(f"{args.path}:{line_number}: 重复样本 ID {sid}")
                sample_ids.add(sid)
            count += 1
    if count != args.expected_lines:
        raise ValueError(f"{args.path}: rows={count}, expected={args.expected_lines}")
    suffix = f" unique_ids={len(sample_ids)}" if args.require_ids else ""
    print(f"VALID_JSONL rows={count}{suffix} path={args.path}")


if __name__ == "__main__":
    main()
