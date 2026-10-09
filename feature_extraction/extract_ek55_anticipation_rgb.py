"""Extract RGB sequence features for EK55 1-second action anticipation.

Unlike ``extract_features.py``, this script never reads frames from inside the
target action.  Every sequence ends at least ``anticipation_time`` seconds
before the annotated action start, as specified in the prepared manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPVisionConfig, CLIPVisionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch-actions", type=int, default=2)
    parser.add_argument("--nearest-radius", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-template", help=(
        "Optional path template containing {data_root}, {participant}, "
        "{video_id}, and {frame}, e.g. "
        "'{data_root}/{participant}/{video_id}/rgb_frames/frame_{frame:010d}.jpg'"
    ))
    parser.add_argument("--allow-ambiguous-flat-layout", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def build_clip(checkpoint: Path, device: torch.device) -> CLIPVisionModel:
    config = CLIPVisionConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        intermediate_size=4096,
    )
    model = CLIPVisionModel(config)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data.get("state_dict", checkpoint_data)
    clip_state = {}
    for key, value in state.items():
        if not key.startswith("module.visual."):
            continue
        if any(token in key for token in ("temporal", "image_projection", "logit_scale")):
            continue
        if "positional_embedding" in key:
            old_size = int((value.shape[0] - 1) ** 0.5)
            cls_token = value[:1]
            spatial = value[1:].reshape(old_size, old_size, 1024)
            spatial = spatial.permute(2, 0, 1).unsqueeze(0)
            spatial = F.interpolate(
                spatial, size=(16, 16), mode="bicubic", align_corners=False
            )
            spatial = spatial.squeeze(0).permute(1, 2, 0).reshape(-1, 1024)
            clip_state["vision_model.embeddings.position_embedding.weight"] = torch.cat(
                [cls_token, spatial], dim=0
            )
            continue
        if "attn.Wqkv" in key:
            layer = key.split(".resblocks.")[1].split(".")[0]
            suffix = "weight" if key.endswith("weight") else "bias"
            query, key_value, value_value = value.chunk(3, dim=0)
            prefix = f"vision_model.encoder.layers.{layer}.self_attn"
            clip_state[f"{prefix}.q_proj.{suffix}"] = query
            clip_state[f"{prefix}.k_proj.{suffix}"] = key_value
            clip_state[f"{prefix}.v_proj.{suffix}"] = value_value
            continue
        new_key = key.replace("module.visual.", "vision_model.")
        new_key = new_key.replace("transformer.resblocks.", "encoder.layers.")
        new_key = new_key.replace("ln_1.", "layer_norm1.")
        new_key = new_key.replace("ln_2.", "layer_norm2.")
        new_key = new_key.replace("ln_pre.", "pre_layrnorm.")
        new_key = new_key.replace("ln_post.", "post_layernorm.")
        new_key = new_key.replace("attn.out_proj.", "self_attn.out_proj.")
        new_key = new_key.replace("conv1.", "embeddings.patch_embedding.")
        new_key = new_key.replace("class_embedding", "embeddings.class_embedding")
        if "conv1.weight" in key:
            value = value.reshape(1024, 3, 1, 14, 14).squeeze(2)
        clip_state[new_key] = value
    model.load_state_dict(clip_state, strict=True)
    return model.to(device).eval()


class FrameResolver:
    def __init__(
        self,
        data_root: Path,
        template: Optional[str],
        nearest_radius: int,
        allow_flat: bool,
    ) -> None:
        self.data_root = data_root
        self.template = template
        self.nearest_radius = nearest_radius
        self.allow_flat = allow_flat
        self.layout_cache: dict[str, str] = {}
        self.used_flat = False

    def _format(self, template: str, participant: str, video: str, frame: int) -> Path:
        return Path(template.format(
            data_root=self.data_root,
            participant=participant,
            video_id=video,
            frame=frame,
        ))

    def _templates(self, participant: str, video: str) -> list[tuple[str, bool]]:
        if self.template:
            return [(self.template, "{video_id}" not in self.template)]
        root = "{data_root}"
        return [
            (root + "/{participant}/rgb_frames/{video_id}/frame_{frame:010d}.jpg", False),
            (root + "/{participant}/{video_id}/rgb_frames/frame_{frame:010d}.jpg", False),
            (root + "/{video_id}/rgb_frames/frame_{frame:010d}.jpg", False),
            (root + "/rgb/train/{participant}/{video_id}/frame_{frame:010d}.jpg", False),
            (root + "/frames_rgb_flow/rgb/train/{participant}/{video_id}/frame_{frame:010d}.jpg", False),
            (root + "/{participant}/rgb_frames/frame_{frame:010d}.jpg", True),
        ]

    def resolve(
        self,
        participant: str,
        video: str,
        frame: int,
        latest_allowed: int,
    ) -> Optional[Path]:
        cached = self.layout_cache.get(video)
        templates = [(cached, "{video_id}" not in cached)] if cached else self._templates(participant, video)
        candidates = [0]
        for delta in range(1, self.nearest_radius + 1):
            candidates.extend((-delta, delta))
        for template, is_flat in templates:
            if is_flat and not self.allow_flat:
                continue
            for delta in candidates:
                candidate_frame = frame + delta
                if candidate_frame < 1 or candidate_frame > latest_allowed:
                    continue
                path = self._format(template, participant, video, candidate_frame)
                if path.exists():
                    self.layout_cache[video] = template
                    self.used_flat = self.used_flat or is_flat
                    return path
        return None


MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def load_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV failed to read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return (tensor - MEAN) / STD


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.batch_actions <= 0 or args.nearest_radius < 0:
        raise ValueError("batch-actions must be positive and nearest-radius non-negative")
    resolver = FrameResolver(
        args.data_root,
        args.frame_template,
        args.nearest_radius,
        args.allow_ambiguous_flat_layout,
    )

    prepared: dict[str, list[tuple[dict, list[Path]]]] = {}
    metadata_path = args.manifest_dir / "metadata.json"
    fps = float(json.loads(metadata_path.read_text())["fps"]) if metadata_path.exists() else 60.0
    for split in args.splits:
        rows = read_jsonl(args.manifest_dir / f"{split}.jsonl")
        valid = []
        missing = 0
        for row in tqdm(rows, desc=f"Resolve {split}"):
            latest = int(np.floor(row["start_frame"] - row["anticipation_time"] * fps))
            paths = [
                resolver.resolve(
                    row["participant_id"], row["video_id"], int(frame), latest
                )
                for frame in row["sample_frames"]
            ]
            if any(path is None for path in paths):
                missing += 1
                continue
            valid.append((row, paths))
        prepared[split] = valid
        print(f"[{split}] resolvable={len(valid)} missing_sequences={missing}")

    if resolver.used_flat:
        print(
            "WARNING: using ambiguous participant-level flat rgb_frames paths. "
            "Frame names can collide across videos; results are not official-comparable.",
            file=sys.stderr,
        )
    if not any(prepared.values()):
        raise RuntimeError(
            "No complete sequence could be resolved. Use the canonical per-video "
            "frame layout or pass --frame-template. The flat layout is refused by "
            "default because it can mix videos."
        )
    if args.probe_only:
        return

    device = torch.device(args.device)
    model = build_clip(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in prepared.items():
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        written = skipped = 0
        for start in tqdm(range(0, len(samples), args.batch_actions), desc=f"Extract {split}"):
            batch = samples[start:start + args.batch_actions]
            todo = []
            for row, paths in batch:
                output_path = split_dir / f'{row["uid"]}.pt'
                if output_path.exists():
                    skipped += 1
                else:
                    todo.append((row, paths, output_path))
            if not todo:
                continue
            pixels = torch.stack([
                load_image(path) for _, paths, _ in todo for path in paths
            ]).to(device, non_blocking=True)
            with torch.inference_mode():
                features = model(pixels).pooler_output
            sequence_length = len(todo[0][1])
            features = features.reshape(len(todo), sequence_length, -1).half().cpu()
            for feature, (_, _, output_path) in zip(features, todo):
                torch.save(feature, output_path)
                written += 1
        print(f"[{split}] wrote={written} skipped_existing={skipped} output={split_dir}")


if __name__ == "__main__":
    main()
