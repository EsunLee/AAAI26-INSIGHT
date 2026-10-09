"""Evaluate unique EK55 actions with verb, noun, and joint-action Top-k accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import ActionRecognitionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-features", type=Path, required=True)
    parser.add_argument("--mask-features", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cooccurrence", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--verb-classes", type=int, default=125)
    parser.add_argument("--noun-classes", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return {key.removeprefix("module."): value for key, value in state.items()}


class UniqueActionDataset(Dataset):
    """Load every labeled action segment once, without overlapping-window duplication."""

    def __init__(self, annotations: Path, frame_root: Path, mask_root: Path, split: str):
        data = json.loads((annotations / f"{split}.json").read_text())
        self.samples = []
        seen = set()
        skipped = 0
        for clip in data.get("clips", []):
            action_id = f'{clip["clip_uid"]}_{clip["action_idx"]}'
            if action_id in seen:
                continue
            seen.add(action_id)
            frame_path = frame_root / split / f"{action_id}.pt"
            mask_path = mask_root / split / f"{action_id}.pt"
            if not frame_path.exists() or not mask_path.exists():
                skipped += 1
                continue
            self.samples.append((
                frame_path,
                mask_path,
                int(clip["verb_label"]),
                int(clip["noun_label"]),
                action_id,
            ))
        print(f"Found {len(self.samples)} unique labeled actions; skipped_missing={skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        frame_path, mask_path, verb, noun, action_id = self.samples[index]
        frame = torch.load(frame_path, map_location="cpu", weights_only=True).float()
        mask = torch.load(mask_path, map_location="cpu", weights_only=True).float()
        return frame, mask, verb, noun, action_id


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UniqueActionDataset(
        annotations=args.annotations,
        frame_root=args.frame_features,
        mask_root=args.mask_features,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ActionRecognitionModel(
        input_dim=args.input_dim,
        mlp_hidden_dim=2048,
        mlp_output_dim=256,
        transformer_layers=4,
        n_heads=8,
        transformer_hidden_dim=2048,
        verb_num_classes=args.verb_classes,
        noun_num_classes=args.noun_classes,
    )
    model.load_state_dict(load_state(args.checkpoint), strict=True)
    model.to(device).eval()

    cooccurrence = torch.load(args.cooccurrence, map_location="cpu", weights_only=True)
    p_n_given_v = cooccurrence["P_n_given_v"].float().to(device)
    p_v_given_n = cooccurrence["P_v_given_n"].float().t().to(device)
    expected_shape = (args.verb_classes, args.noun_classes)
    if tuple(p_n_given_v.shape) != expected_shape or tuple(p_v_given_n.shape) != expected_shape:
        raise ValueError(
            f"Co-occurrence shape mismatch: {tuple(p_n_given_v.shape)}, "
            f"{tuple(p_v_given_n.shape)}; expected {expected_shape}"
        )
    semantic_prior = 0.5 * (p_n_given_v + p_v_given_n)

    correct = {name: 0 for name in (
        "verb_top1", "verb_top5", "noun_top1", "noun_top5",
        "action_raw_top1", "action_raw_top5",
        "action_top1", "action_top5",
    )}
    total = 0

    with torch.inference_mode():
        for frame, mask, verb_gt, noun_gt, action_ids in tqdm(loader, desc="EK55 Top-k"):
            frame = frame.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            verb_gt = verb_gt.to(device, non_blocking=True)
            noun_gt = noun_gt.to(device, non_blocking=True)
            verb_logits, noun_logits = model(frame, mask)

            verb_prob = verb_logits.softmax(dim=-1)
            noun_prob = noun_logits.softmax(dim=-1)
            verb_top5 = verb_prob.topk(5, dim=-1).indices
            noun_top5 = noun_prob.topk(5, dim=-1).indices

            raw_joint = verb_prob.unsqueeze(2) * noun_prob.unsqueeze(1)
            raw_action_top5 = raw_joint.flatten(1).topk(5, dim=-1).indices

            # Equation (5) in the paper, ranked over all V x N action pairs.
            joint = (
                verb_prob.unsqueeze(2)
                * noun_prob.unsqueeze(1)
                * semantic_prior.unsqueeze(0)
            )
            action_top5 = joint.flatten(1).topk(5, dim=-1).indices

            for i, action_id in enumerate(action_ids):
                if verb_gt[i] < 0 or noun_gt[i] < 0:
                    continue
                total += 1

                true_v = int(verb_gt[i])
                true_n = int(noun_gt[i])
                true_action = true_v * args.noun_classes + true_n
                verbs = verb_top5[i].tolist()
                nouns = noun_top5[i].tolist()
                raw_actions = raw_action_top5[i].tolist()
                actions = action_top5[i].tolist()

                correct["verb_top1"] += int(verbs[0] == true_v)
                correct["verb_top5"] += int(true_v in verbs)
                correct["noun_top1"] += int(nouns[0] == true_n)
                correct["noun_top5"] += int(true_n in nouns)
                correct["action_raw_top1"] += int(raw_actions[0] == true_action)
                correct["action_raw_top5"] += int(true_action in raw_actions)
                correct["action_top1"] += int(actions[0] == true_action)
                correct["action_top5"] += int(true_action in actions)

    if total == 0:
        raise RuntimeError("No labeled actions were evaluated")

    result = {
        "split": args.split,
        "unique_actions": total,
        "checkpoint": str(args.checkpoint),
        "metric_definition": "micro accuracy over unique labeled action segments",
        **{name: 100.0 * value / total for name, value in correct.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
