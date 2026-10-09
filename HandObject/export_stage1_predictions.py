"""Export one deduplicated Stage-1 prediction per EK55 action segment."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--verb-classes", type=int, default=125)
    parser.add_argument("--noun-classes", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class ActionFeatureDataset(Dataset):
    def __init__(self, annotation_file: Path, frame_root: Path, mask_root: Path, split: str):
        rows = json.loads(annotation_file.read_text()).get("clips", [])
        self.samples = []
        seen = set()
        missing = 0
        for row in rows:
            uid = int(row["action_idx"])
            video = row["clip_uid"]
            action_id = f"{video}_{uid}"
            if action_id in seen:
                continue
            seen.add(action_id)
            frame_path = frame_root / split / f"{action_id}.pt"
            mask_path = mask_root / split / f"{action_id}.pt"
            if not frame_path.exists() or not mask_path.exists():
                missing += 1
                continue
            self.samples.append((frame_path, mask_path, video, uid))
        print(f"[{split}] usable={len(self.samples)}, missing={missing}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        frame_path, mask_path, video, uid = self.samples[index]
        frame = torch.load(frame_path, map_location="cpu", weights_only=True).float()
        mask = torch.load(mask_path, map_location="cpu", weights_only=True).float()
        return frame, mask, video, uid


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    co = torch.load(args.cooccurrence, map_location="cpu", weights_only=True)
    semantic_prior = 0.5 * (
        co["P_n_given_v"].float() + co["P_v_given_n"].float().t()
    )
    expected_shape = (args.verb_classes, args.noun_classes)
    if tuple(semantic_prior.shape) != expected_shape:
        raise ValueError(
            f"Co-occurrence shape is {tuple(semantic_prior.shape)}; expected {expected_shape}"
        )
    semantic_prior = semantic_prior.to(device)

    for split in args.splits:
        dataset = ActionFeatureDataset(
            args.annotations / f"{split}.json",
            args.frame_features,
            args.mask_features,
            split,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        output_path = args.output_dir / f"stage1_{split}_predictions.jsonl"
        written = 0
        with output_path.open("w", encoding="utf-8") as output, torch.inference_mode():
            for frame, mask, videos, uids in tqdm(loader, desc=f"Stage1 {split}"):
                frame = frame.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                verb_logits, noun_logits = model(frame, mask)
                verb_prob = verb_logits.softmax(-1)
                noun_prob = noun_logits.softmax(-1)
                joint = (
                    verb_prob.unsqueeze(2)
                    * noun_prob.unsqueeze(1)
                    * semantic_prior.unsqueeze(0)
                )
                top_scores, top_indices = joint.flatten(1).topk(5, dim=-1)
                for i, (video, uid) in enumerate(zip(videos, uids.tolist())):
                    candidates = []
                    for score, flat_index in zip(top_scores[i].tolist(), top_indices[i].tolist()):
                        verb = flat_index // args.noun_classes
                        noun = flat_index % args.noun_classes
                        candidates.append({"verb": verb, "noun": noun, "score": score})
                    record = {
                        "clip_uid": video,
                        "action_idx": uid,
                        "pred_verb": candidates[0]["verb"],
                        "pred_noun": candidates[0]["noun"],
                        "action_candidates": candidates,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
        print(f"[{split}] wrote={written}: {output_path}")


if __name__ == "__main__":
    main()
