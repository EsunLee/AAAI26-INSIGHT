"""Export loss/token-accuracy curves from an ms-swift trainer state.

The script writes a machine-readable CSV and, when matplotlib is available, a
PNG with separate train and validation curves. Action-level Hit@k is not a
token-level Trainer metric and must be obtained by generating from saved
checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.parent.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def find_trainer_state(run_dir: Path) -> Path:
    root_state = run_dir / "trainer_state.json"
    if root_state.is_file():
        return root_state
    states = list(run_dir.glob("checkpoint-*/trainer_state.json"))
    if not states:
        raise FileNotFoundError(f"No trainer_state.json under {run_dir}")
    return max(states, key=checkpoint_step)


def normalize(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history:
        step = item.get("step", item.get("global_step"))
        if step is None:
            continue
        if "eval_loss" in item:
            rows.append(
                {
                    "step": int(step),
                    "epoch": item.get("epoch"),
                    "split": "val",
                    "loss": item.get("eval_loss"),
                    "token_acc": item.get("eval_token_acc"),
                    "learning_rate": item.get("learning_rate"),
                }
            )
        elif "loss" in item:
            rows.append(
                {
                    "step": int(step),
                    "epoch": item.get("epoch"),
                    "split": "train",
                    "loss": item.get("loss"),
                    "token_acc": item.get("token_acc"),
                    "learning_rate": item.get("learning_rate"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "epoch", "split", "loss", "token_acc", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split, color in (("train", "tab:blue"), ("val", "tab:orange")):
        selected = [row for row in rows if row["split"] == split]
        steps = [row["step"] for row in selected]
        losses = [row["loss"] for row in selected]
        accuracies = [row["token_acc"] for row in selected]
        axes[0].plot(steps, losses, marker="o", markersize=3, label=split, color=color)
        valid_accuracy = [(step, acc) for step, acc in zip(steps, accuracies) if acc is not None]
        if valid_accuracy:
            axes[1].plot(
                [item[0] for item in valid_accuracy],
                [item[1] for item in valid_accuracy],
                marker="o",
                markersize=3,
                label=split,
                color=color,
            )

    axes[0].set_title("SFT loss")
    axes[0].set_xlabel("global step")
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].set_title("SFT token accuracy")
    axes[1].set_xlabel("global step")
    axes[1].set_ylabel("accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    state_path = find_trainer_state(args.run_dir)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = normalize(state.get("log_history", []))
    if not rows:
        raise ValueError(f"No train/eval curve entries in {state_path}")

    output_dir = args.output_dir or args.run_dir / "curve_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sft_loss_token_accuracy.csv"
    png_path = output_dir / "sft_loss_token_accuracy.png"
    write_csv(csv_path, rows)
    plotted = write_plot(png_path, rows)

    print(f"trainer_state={state_path}")
    print(f"curve_rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"png={png_path if plotted else 'not_written_matplotlib_missing'}")


if __name__ == "__main__":
    main()
