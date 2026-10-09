"""导出 DCR-valid 两阶段 Top-1 的逐样本总表和汇总表。

本脚本只读取已完成的 Stage1/Stage2 产物，不运行模型、不使用 GPU。
预测与 GT 严格通过 (clip_uid, action_idx) 对齐；缺失、重复或集合不一致时失败。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
INTENTION_RE = re.compile(r"<intention>(.*?)</intention>", re.S | re.I)
ACTION_RE = re.compile(r"^[a-z0-9_:-]+ [a-z0-9_:-]+$", re.I)
HISTORY_RE = re.compile(r"^\s*\d+\.\s*(.*?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=C.PRED_DIR / "stage2_input.jsonl")
    parser.add_argument(
        "--predictions", type=Path, default=C.GEN_DIR / "candidate_0.jsonl"
    )
    parser.add_argument(
        "--stage1-predictions",
        type=Path,
        default=C.PRED_DIR / "stage1_val_predictions.jsonl",
    )
    parser.add_argument(
        "--stage1-metrics", type=Path, default=C.PRED_DIR / "stage1_metrics.json"
    )
    parser.add_argument(
        "--output", type=Path, default=C.EVAL_DIR / "all_valid_top1_results.csv"
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=C.EVAL_DIR / "all_valid_top1_summary.csv",
    )
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


def sid(row: dict, *, path: Path, line_number: int) -> tuple[str, int]:
    clip_uid = row.get("clip_uid")
    action_idx = row.get("action_idx")
    if clip_uid is None or action_idx is None:
        raise ValueError(f"{path}:{line_number}: 缺少 clip_uid/action_idx")
    try:
        return str(clip_uid), int(action_idx)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: action_idx 非整数") from exc


def unique_by_id(rows: list[dict], path: Path) -> dict[tuple[str, int], dict]:
    result: dict[tuple[str, int], dict] = {}
    for index, row in enumerate(rows, start=1):
        key = sid(row, path=path, line_number=index)
        if key in result:
            raise ValueError(f"{path}:{index}: 重复样本 ID {key}")
        result[key] = row
    return result


def response_value(row: dict) -> str:
    if isinstance(row.get("response"), str):
        return row["response"]
    if isinstance(row.get("generated_text"), str):
        return row["generated_text"]
    if isinstance(row.get("output"), str):
        return row["output"]
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def normalize(value: str | None) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()) if value else ""


def tag_value(pattern: re.Pattern[str], response: str) -> str:
    match = pattern.search(response)
    return re.sub(r"\s+", " ", match.group(1).strip()) if match else ""


def split_action(action: str) -> tuple[str, str]:
    tokens = normalize(action).split()
    return (tokens[0], tokens[1]) if len(tokens) == 2 else ("", "")


def prompt_text(record: dict) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def history_actions(record: dict) -> list[str]:
    history: list[str] = []
    for line in prompt_text(record).splitlines():
        match = HISTORY_RE.match(line)
        if match:
            history.append(normalize(match.group(1)))
    return history[: C.HISTORY]


def image_path(record: dict) -> str:
    images = record.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    return ""


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 便于 Excel 直接显示 UTF-8 文本。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_rows = load_jsonl(args.input)
    prediction_rows = load_jsonl(args.predictions)
    input_map = unique_by_id(input_rows, args.input)
    prediction_map = unique_by_id(prediction_rows, args.predictions)
    if set(input_map) != set(prediction_map):
        missing = sorted(set(input_map) - set(prediction_map))[:5]
        extra = sorted(set(prediction_map) - set(input_map))[:5]
        raise ValueError(f"输入/预测 ID 集合不一致: missing={missing} extra={extra}")

    stage1_map: dict[tuple[str, int], dict] = {}
    if args.stage1_predictions.exists():
        stage1_map = unique_by_id(load_jsonl(args.stage1_predictions), args.stage1_predictions)
    stage1_metrics = {}
    if args.stage1_metrics.exists():
        stage1_metrics = json.loads(args.stage1_metrics.read_text(encoding="utf-8"))

    rows: list[dict] = []
    video_stats: dict[str, Counter] = defaultdict(Counter)
    first_action_counter: Counter[str] = Counter()
    for sample_index, record in enumerate(input_rows, start=1):
        key = sid(record, path=args.input, line_number=sample_index)
        prediction = prediction_map[key]
        response = response_value(prediction)
        answer = tag_value(ANSWER_RE, response)
        actions = [normalize(item) for item in answer.split(",") if normalize(item)]
        pred_first = actions[0] if actions else ""
        gt_action = normalize(str(record.get("gt_action_text", "")))
        gt_verb, gt_noun = split_action(gt_action)
        pred_verb, pred_noun = split_action(pred_first)
        parse_success = int(bool(answer and pred_first))
        valid_actions = sum(bool(ACTION_RE.fullmatch(action)) for action in actions)
        action_correct = int(bool(gt_action and pred_first == gt_action))
        verb_correct = int(bool(gt_verb and pred_verb == gt_verb))
        noun_correct = int(bool(gt_noun and pred_noun == gt_noun))
        histories = history_actions(record)
        stage1 = stage1_map.get(key, {})
        clip_uid, action_idx = key
        first_action_counter[pred_first or "<parse_failure>"] += 1
        video_stats[clip_uid].update(
            samples=1,
            action_correct=action_correct,
            verb_correct=verb_correct,
            noun_correct=noun_correct,
            parse_success=parse_success,
            exactly_20=int(len(actions) == C.HORIZON),
            exactly_20_valid=int(len(actions) == C.HORIZON and valid_actions == C.HORIZON),
        )
        row = {
            "sample_index": sample_index,
            "clip_uid": clip_uid,
            "action_idx": action_idx,
            "image_path": image_path(record),
            "gt_action": gt_action,
            "gt_verb": gt_verb,
            "gt_noun": gt_noun,
            "gt_verb_class": record.get("gt_verb_class", ""),
            "gt_noun_class": record.get("gt_noun_class", ""),
            "pred_first_action": pred_first,
            "pred_verb": pred_verb,
            "pred_noun": pred_noun,
            "action_top1_correct": action_correct,
            "verb_top1_correct": verb_correct,
            "noun_top1_correct": noun_correct,
            "parse_success": parse_success,
            "generated_item_count": len(actions),
            "valid_action_count": valid_actions,
            "exactly_20_items": int(len(actions) == C.HORIZON),
            "exactly_20_valid_actions": int(
                len(actions) == C.HORIZON and valid_actions == C.HORIZON
            ),
            "think": tag_value(THINK_RE, response),
            "intention": tag_value(INTENTION_RE, response),
            "predicted_actions": " | ".join(actions),
            "raw_response": response,
            "stage1_target_pred_verb_class": stage1.get("pred_verb", ""),
            "stage1_target_pred_noun_class": stage1.get("pred_noun", ""),
            "stage1_target_action_score": stage1.get("action_score", ""),
        }
        for index in range(C.HISTORY):
            row[f"history_{index + 1}"] = histories[index] if index < len(histories) else ""
        for index in range(C.HORIZON):
            row[f"pred_action_{index + 1:02d}"] = actions[index] if index < len(actions) else ""
        rows.append(row)

    total = len(rows)
    action_hits = sum(row["action_top1_correct"] for row in rows)
    verb_hits = sum(row["verb_top1_correct"] for row in rows)
    noun_hits = sum(row["noun_top1_correct"] for row in rows)
    parse_hits = sum(row["parse_success"] for row in rows)
    exact20 = sum(row["exactly_20_items"] for row in rows)
    valid20 = sum(row["exactly_20_valid_actions"] for row in rows)
    generated_counts = [int(row["generated_item_count"]) for row in rows]
    overall = {
        "scope": "overall",
        "clip_uid": "ALL",
        "samples": total,
        "action_top1_correct": action_hits,
        "action_top1_pct": percentage(action_hits, total),
        "verb_top1_correct": verb_hits,
        "verb_top1_pct": percentage(verb_hits, total),
        "noun_top1_correct": noun_hits,
        "noun_top1_pct": percentage(noun_hits, total),
        "parse_successes": parse_hits,
        "parse_success_rate_pct": percentage(parse_hits, total),
        "exactly_20_items": exact20,
        "exactly_20_items_rate_pct": percentage(exact20, total),
        "exactly_20_valid_actions": valid20,
        "exactly_20_valid_actions_rate_pct": percentage(valid20, total),
        "mean_generated_items": round(statistics.mean(generated_counts), 3) if total else 0.0,
        "unique_predicted_first_actions": len(first_action_counter),
        "top10_predicted_first_actions": json.dumps(
            first_action_counter.most_common(10), ensure_ascii=False
        ),
        "stage1_labeled_samples": stage1_metrics.get("labeled_samples", ""),
        "stage1_verb_top1_pct": stage1_metrics.get("verb_top1", ""),
        "stage1_noun_top1_pct": stage1_metrics.get("noun_top1", ""),
        "stage1_action_top1_pct": stage1_metrics.get("action_top1", ""),
    }
    summary_rows = [overall]
    for clip_uid in sorted(video_stats):
        stats = video_stats[clip_uid]
        count = stats["samples"]
        summary_rows.append(
            {
                "scope": "video",
                "clip_uid": clip_uid,
                "samples": count,
                "action_top1_correct": stats["action_correct"],
                "action_top1_pct": percentage(stats["action_correct"], count),
                "verb_top1_correct": stats["verb_correct"],
                "verb_top1_pct": percentage(stats["verb_correct"], count),
                "noun_top1_correct": stats["noun_correct"],
                "noun_top1_pct": percentage(stats["noun_correct"], count),
                "parse_successes": stats["parse_success"],
                "parse_success_rate_pct": percentage(stats["parse_success"], count),
                "exactly_20_items": stats["exactly_20"],
                "exactly_20_items_rate_pct": percentage(stats["exactly_20"], count),
                "exactly_20_valid_actions": stats["exactly_20_valid"],
                "exactly_20_valid_actions_rate_pct": percentage(
                    stats["exactly_20_valid"], count
                ),
            }
        )

    detail_fields = [
        "sample_index", "clip_uid", "action_idx", "image_path",
        "gt_action", "gt_verb", "gt_noun", "gt_verb_class", "gt_noun_class",
        "pred_first_action", "pred_verb", "pred_noun",
        "action_top1_correct", "verb_top1_correct", "noun_top1_correct",
        "parse_success", "generated_item_count", "valid_action_count",
        "exactly_20_items", "exactly_20_valid_actions",
        *[f"history_{index}" for index in range(1, C.HISTORY + 1)],
        "think", "intention",
        *[f"pred_action_{index:02d}" for index in range(1, C.HORIZON + 1)],
        "predicted_actions", "raw_response",
        "stage1_target_pred_verb_class", "stage1_target_pred_noun_class",
        "stage1_target_action_score",
    ]
    summary_fields = [
        "scope", "clip_uid", "samples",
        "action_top1_correct", "action_top1_pct",
        "verb_top1_correct", "verb_top1_pct",
        "noun_top1_correct", "noun_top1_pct",
        "parse_successes", "parse_success_rate_pct",
        "exactly_20_items", "exactly_20_items_rate_pct",
        "exactly_20_valid_actions", "exactly_20_valid_actions_rate_pct",
        "mean_generated_items", "unique_predicted_first_actions",
        "top10_predicted_first_actions", "stage1_labeled_samples",
        "stage1_verb_top1_pct", "stage1_noun_top1_pct", "stage1_action_top1_pct",
    ]
    write_csv(args.output, detail_fields, rows)
    write_csv(args.summary_output, summary_fields, summary_rows)
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"DETAIL_CSV rows={len(rows)} path={args.output}")
    print(f"SUMMARY_CSV rows={len(summary_rows)} path={args.summary_output}")


if __name__ == "__main__":
    main()
