"""Evaluate Stage-2 candidate generations with Top-k and sequence metrics.

The next-action Top-k metric is a project-defined metric: candidate rank is the
order of ``--predictions`` files and only the first action of each generated
sequence is used.  The sequence edit-distance metrics follow the Ego4D LTA
protocol described in the INSIGHT paper: normalize every candidate to the
ground-truth horizon, compute action/verb/noun Damerau-Levenshtein distance,
and take the best (minimum) distance over the first k candidates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.I | re.S)
ACTION_RE = re.compile(r"^[a-z0-9_:-]+\s+[a-z0-9_:-]+$", flags=re.I)
PAD_ACTION = "<pad> <pad>"
INVALID_ACTION = "<invalid> <invalid>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=20)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    return rows


def text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(text_values(item))
        return result
    if isinstance(value, dict):
        for key in ("content", "text", "response", "message"):
            if key in value:
                values = text_values(value[key])
                if values:
                    return values
    return []


def generated_responses(row: dict[str, Any]) -> list[str]:
    # Do not inspect solution/gt_answer: those are references, not predictions.
    for key in (
        "responses",
        "response",
        "prediction",
        "predictions",
        "generated_text",
        "output",
        "completion",
    ):
        if key in row:
            values = text_values(row[key])
            if values:
                return values

    messages = row.get("messages", [])
    assistants = [
        message.get("content", "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    return [assistants[-1]] if assistants else []


def normalize_action(text: str) -> str | None:
    text = text.strip().lower()
    text = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .;\n\t")
    return text if ACTION_RE.fullmatch(text) else None


def answer_items(response: str) -> tuple[list[str], bool]:
    match = ANSWER_RE.search(response)
    body = match.group(1) if match else response
    return [item.strip() for item in body.split(",") if item.strip()], match is not None


def parsed_actions(response: str) -> tuple[list[str], int, bool]:
    items, has_answer_tags = answer_items(response)
    actions = []
    valid = 0
    for item in items:
        action = normalize_action(item)
        if action is None:
            actions.append(INVALID_ACTION)
        else:
            actions.append(action)
            valid += 1
    return actions, valid, has_answer_tags


def first_action(response: str) -> str | None:
    items, _ = answer_items(response)
    return normalize_action(items[0]) if items else None


def ground_truth_sequence(row: dict[str, Any], horizon: int) -> list[str]:
    future = row.get("future_actions")
    if isinstance(future, list):
        actions = [normalize_action(item) for item in future if isinstance(item, str)]
        if len(actions) >= horizon and all(action is not None for action in actions[:horizon]):
            return [action for action in actions[:horizon] if action is not None]

    for key in ("gt_answer", "solution"):
        value = row.get(key)
        if isinstance(value, str):
            actions, valid, _ = parsed_actions(value)
            if len(actions) >= horizon and valid >= horizon:
                return actions[:horizon]
    raise ValueError("Ground-truth row has no valid future action sequence")


def normalize_horizon(actions: list[str], horizon: int) -> list[str]:
    return (actions[:horizon] + [PAD_ACTION] * horizon)[:horizon]


def split_actions(actions: list[str]) -> tuple[list[str], list[str]]:
    verbs, nouns = [], []
    for action in actions:
        verb, noun = action.split(" ", 1)
        verbs.append(verb)
        nouns.append(noun)
    return verbs, nouns


def damerau_levenshtein(left: list[str], right: list[str]) -> int:
    rows, columns = len(left), len(right)
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        distance[i][0] = i
    for j in range(columns + 1):
        distance[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and left[i - 1] == right[j - 2]
                and left[i - 2] == right[j - 1]
            ):
                distance[i][j] = min(distance[i][j], distance[i - 2][j - 2] + cost)
    return distance[rows][columns]


def sequence_distances(predicted: list[str], target: list[str]) -> tuple[float, float, float]:
    pred_verbs, pred_nouns = split_actions(predicted)
    true_verbs, true_nouns = split_actions(target)
    denominator = len(target)
    return (
        damerau_levenshtein(predicted, target) / denominator,
        damerau_levenshtein(pred_verbs, true_verbs) / denominator,
        damerau_levenshtein(pred_nouns, true_nouns) / denominator,
    )


def main() -> None:
    args = parse_args()
    if args.k <= 0 or args.horizon <= 0:
        raise ValueError("--k and --horizon must be positive")

    ground_truth = read_jsonl(args.ground_truth)
    prediction_sets = [read_jsonl(path) for path in args.predictions]
    for path, rows in zip(args.predictions, prediction_sets):
        if len(rows) != len(ground_truth):
            raise ValueError(
                f"Row-count mismatch: {path} has {len(rows)}, "
                f"ground truth has {len(ground_truth)}"
            )

    top1_correct = topk_correct = 0
    parse_failures = duplicate_candidates = 0
    per_file_correct = [0] * len(prediction_sets)
    candidate1_ed = [0.0, 0.0, 0.0]
    oracle_ed = [0.0, 0.0, 0.0]
    tagged = exact_length = all_valid = 0
    total_raw_actions = total_valid_actions = 0
    candidate_slots = 0

    for index, gt_row in enumerate(ground_truth):
        target_sequence = ground_truth_sequence(gt_row, args.horizon)
        target_first = target_sequence[0]
        first_candidates: list[str | None] = []
        sequences: list[list[str]] = []

        for file_index, rows in enumerate(prediction_sets):
            responses = generated_responses(rows[index])
            response = responses[0] if responses else ""
            first = first_action(response)
            actions, valid_count, has_tags = parsed_actions(response)
            if first == target_first:
                per_file_correct[file_index] += 1
            first_candidates.append(first)
            sequences.append(normalize_horizon(actions, args.horizon))

            if file_index < args.k:
                candidate_slots += 1
                tagged += int(has_tags)
                exact_length += int(len(actions) == args.horizon)
                all_valid += int(len(actions) == args.horizon and valid_count == args.horizon)
                total_raw_actions += len(actions)
                total_valid_actions += valid_count

        first_candidates = first_candidates[: args.k]
        sequences = sequences[: args.k]
        parse_failures += sum(candidate is None for candidate in first_candidates)
        valid_first = [candidate for candidate in first_candidates if candidate is not None]
        duplicate_candidates += len(valid_first) - len(set(valid_first))
        top1_correct += int(bool(first_candidates) and first_candidates[0] == target_first)
        topk_correct += int(target_first in valid_first)

        distances = [sequence_distances(sequence, target_sequence) for sequence in sequences]
        if not distances:
            distances = [(1.0, 1.0, 1.0)]
        for metric_index in range(3):
            candidate1_ed[metric_index] += distances[0][metric_index]
            oracle_ed[metric_index] += min(item[metric_index] for item in distances)

    total = len(ground_truth)
    if total == 0:
        raise ValueError("Ground-truth file is empty")
    metric_names = ("action", "verb", "noun")
    result = {
        "samples": total,
        "candidate_files_used": min(args.k, len(prediction_sets)),
        "next_action_topk": {
            "definition": (
                "Project-defined generative accuracy: first action of candidate 1 "
                f"for Top-1; target present in the first {args.k} candidates for Top-{args.k}."
            ),
            "top1": 100.0 * top1_correct / total,
            f"top{args.k}": 100.0 * topk_correct / total,
            "top1_correct": top1_correct,
            f"top{args.k}_correct": topk_correct,
            "parse_failures_in_candidate_slots": parse_failures,
            "duplicate_candidate_slots": duplicate_candidates,
            "per_prediction_file_top1": {
                str(path): 100.0 * correct / total
                for path, correct in zip(args.predictions, per_file_correct)
            },
        },
        "ego4d_style_normalized_edit_distance": {
            "definition": (
                f"Damerau-Levenshtein distance after truncating/padding to {args.horizon}; "
                f"oracle is the separate minimum over the first {args.k} candidates. Lower is better."
            ),
            "candidate1": {
                name: candidate1_ed[i] / total for i, name in enumerate(metric_names)
            },
            f"oracle_min_over_{args.k}": {
                name: oracle_ed[i] / total for i, name in enumerate(metric_names)
            },
        },
        "generation_quality": {
            "answer_tag_rate": tagged / candidate_slots if candidate_slots else 0.0,
            f"exactly_{args.horizon}_items_rate": exact_length / candidate_slots if candidate_slots else 0.0,
            f"exactly_{args.horizon}_valid_actions_rate": all_valid / candidate_slots if candidate_slots else 0.0,
            "mean_items_per_candidate": total_raw_actions / candidate_slots if candidate_slots else 0.0,
            "mean_valid_actions_per_candidate": total_valid_actions / candidate_slots if candidate_slots else 0.0,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
