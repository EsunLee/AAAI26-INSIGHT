"""把 Stage2 输入中的稳定样本 ID 写回 ms-swift 生成结果。

ms-swift 的 result_path 默认不保证保留数据集自定义字段。本脚本在每个连续
分片生成完成后，严格检查输入/输出行数与输入 ID 唯一性，然后为每条预测加入
``clip_uid`` 和 ``action_idx``。之后即使合并顺序变化，04_evaluate.py 也能按 ID
恢复正确的 GT 配对；缺失、重复或冲突时直接失败。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="本分片 Stage2 输入 JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="ms-swift 生成 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="带样本 ID 的输出 JSONL")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 损坏: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: 期望 JSON object")
            rows.append(row)
    return rows


def sample_id(row: dict, *, path: Path, line_number: int) -> tuple[str, int]:
    clip_uid = row.get("clip_uid")
    action_idx = row.get("action_idx")
    if clip_uid is None or action_idx is None:
        raise ValueError(
            f"{path}:{line_number}: 缺少 clip_uid/action_idx，无法建立可靠对齐"
        )
    try:
        return str(clip_uid), int(action_idx)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: action_idx 不是整数: {action_idx!r}") from exc


def main() -> None:
    args = parse_args()
    inputs = load_jsonl(args.input)
    predictions = load_jsonl(args.predictions)
    if not inputs:
        raise ValueError(f"输入分片为空: {args.input}")
    if len(inputs) != len(predictions):
        raise ValueError(
            f"输入/预测行数不一致: input={len(inputs)} predictions={len(predictions)}"
        )

    ids = [
        sample_id(row, path=args.input, line_number=index)
        for index, row in enumerate(inputs, start=1)
    ]
    if len(set(ids)) != len(ids):
        raise ValueError(f"输入分片存在重复 (clip_uid, action_idx): {args.input}")

    enriched: list[dict] = []
    for index, (prediction, (clip_uid, action_idx)) in enumerate(
        zip(predictions, ids), start=1
    ):
        # 若推理框架未来开始保留 ID，则必须与输入完全一致，禁止静默覆盖。
        existing_clip = prediction.get("clip_uid")
        existing_action = prediction.get("action_idx")
        if existing_clip is not None and str(existing_clip) != clip_uid:
            raise ValueError(
                f"{args.predictions}:{index}: clip_uid 冲突: "
                f"prediction={existing_clip!r} input={clip_uid!r}"
            )
        if existing_action is not None:
            try:
                existing_action_int = int(existing_action)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{args.predictions}:{index}: 预测 action_idx 非整数"
                ) from exc
            if existing_action_int != action_idx:
                raise ValueError(
                    f"{args.predictions}:{index}: action_idx 冲突: "
                    f"prediction={existing_action_int} input={action_idx}"
                )
        row = dict(prediction)
        row["clip_uid"] = clip_uid
        row["action_idx"] = action_idx
        enriched.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in enriched:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"ATTACHED_IDS rows={len(enriched)} unique_ids={len(set(ids))} "
        f"input={args.input} output={args.output}"
    )


if __name__ == "__main__":
    main()
