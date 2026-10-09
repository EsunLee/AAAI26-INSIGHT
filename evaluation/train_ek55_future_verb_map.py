"""Train and evaluate EK55 future-verb mAP under the EGO-TOPO protocol.

The model consumes available Stage-1 frame features from actions completed in
the observed prefix and predicts the set of verbs beginning in the remaining
video.  Validation uses P={25, 50, 75}; All/Freq/Rare class groups follow the
official EK55 many-shot verb definition (>100 training instances).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


TRAIN_RATIOS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
VAL_RATIOS = (0.25, 0.5, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--frame-features", type=Path, required=True)
    parser.add_argument("--many-shot-verbs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-verbs", type=int, default=125)
    parser.add_argument("--max-observed-actions", type=int, default=64)
    parser.add_argument("--min-observed-features", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_epic_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_egotopo_rows(path: Path) -> dict[str, list[dict]]:
    by_video = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            video_path, start, end, verb, noun = line.rstrip().split("\t")
            video = video_path.split("/")[-1]
            by_video[video].append({
                "video_id": video,
                "start_frame": int(start),
                "stop_frame": int(end),
                "verb_class": int(verb),
                "noun_class": int(noun),
            })
    for rows in by_video.values():
        rows.sort(key=lambda row: (row["start_frame"], row["stop_frame"]))
    return by_video


def load_video_lengths(path: Path) -> dict[str, int]:
    lengths = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            video_path, length = line.rstrip().split("\t")
            lengths[video_path.split("/")[-1]] = int(length)
    return lengths


def index_features(root: Path) -> dict[int, Path]:
    result = {}
    for split in ("train", "val"):
        folder = root / split
        if not folder.exists():
            continue
        for path in folder.glob("*.pt"):
            try:
                uid = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if uid in result and result[uid] != path:
                raise ValueError(f"Duplicate feature UID {uid}: {result[uid]} and {path}")
            result[uid] = path
    return result


def action_uid_index(rows: list[dict]) -> dict[tuple, int]:
    index = {}
    for row in rows:
        key = (
            row["video_id"], int(row["start_frame"]), int(row["stop_frame"]),
            int(row["verb_class"]), int(row["noun_class"]),
        )
        index[key] = int(row["uid"])
    return index


def uniformly_limit(items: list, maximum: int) -> list:
    if len(items) <= maximum:
        return items
    indices = np.round(np.linspace(0, len(items) - 1, maximum)).astype(int)
    return [items[index] for index in indices]


def build_samples(
    rows_by_video: dict[str, list[dict]],
    video_lengths: dict[str, int],
    ratios: tuple[float, ...],
    uid_index: dict[tuple, int],
    feature_index: dict[int, Path],
    num_verbs: int,
    maximum: int,
    minimum: int,
) -> tuple[list[dict], dict]:
    samples = []
    stats = Counter()
    for video, rows in rows_by_video.items():
        if video not in video_lengths:
            stats["missing_video_length"] += 1
            continue
        for ratio_index, ratio in enumerate(ratios):
            cutoff = int(ratio * video_lengths[video])
            past = [row for row in rows if row["stop_frame"] <= cutoff]
            future = [row for row in rows if row["start_frame"] > cutoff]
            if len(past) < 3 or len(future) < 3:
                stats["protocol_filtered"] += 1
                continue
            observed = []
            for row in past:
                key = (
                    video, row["start_frame"], row["stop_frame"],
                    row["verb_class"], row["noun_class"],
                )
                uid = uid_index.get(key)
                path = feature_index.get(uid) if uid is not None else None
                if path is not None:
                    observed.append(path)
            observed = uniformly_limit(observed, maximum)
            if len(observed) < minimum:
                stats["insufficient_observed_features"] += 1
                continue
            labels = torch.zeros(num_verbs, dtype=torch.float32)
            for verb in {row["verb_class"] for row in future}:
                labels[verb] = 1.0
            samples.append({
                "video_id": video,
                "ratio": ratio,
                "ratio_index": ratio_index,
                "feature_paths": observed,
                "labels": labels,
                "past_actions": len(past),
                "future_actions": len(future),
            })
    stats["samples"] = len(samples)
    stats["videos"] = len({sample["video_id"] for sample in samples})
    return samples, dict(stats)


class FutureVerbDataset(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples
        # The same observed action features are reused at seven training
        # cutoffs and over many epochs.  Keeping them in host memory avoids
        # repeatedly opening tens of thousands of tiny .pt files.
        self.feature_cache = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        features = []
        for path in sample["feature_paths"]:
            feature = self.feature_cache.get(path)
            if feature is None:
                feature = torch.load(path, map_location="cpu", weights_only=True).float()
                if feature.ndim == 2:
                    feature = feature.mean(0)
                if feature.ndim != 1:
                    raise ValueError(f"Expected vector feature at {path}, got {feature.shape}")
                self.feature_cache[path] = feature
            features.append(feature)
        return (
            torch.stack(features), sample["labels"], sample["ratio_index"],
            sample["video_id"], sample["ratio"],
        )


def collate(batch):
    lengths = torch.tensor([item[0].shape[0] for item in batch], dtype=torch.long)
    dimension = batch[0][0].shape[1]
    padded = torch.zeros(len(batch), int(lengths.max()), dimension)
    for index, item in enumerate(batch):
        padded[index, :item[0].shape[0]] = item[0]
    labels = torch.stack([item[1] for item in batch])
    ratio_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    videos = [item[3] for item in batch]
    ratios = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    return padded, lengths, labels, ratio_indices, videos, ratios


class FutureVerbGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_verbs: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(0.2), nn.Linear(hidden_dim, num_verbs)
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sequence = self.input_norm(sequence)
        packed = pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def grouped_map(
    labels: np.ndarray,
    scores: np.ndarray,
    frequent: set[int],
    evaluation_classes: set[int],
) -> dict[str, float]:
    class_ap = {
        class_id: average_precision(labels[:, class_id], scores[:, class_id])
        if labels[:, class_id].sum() > 0 else 0.0
        for class_id in sorted(evaluation_classes)
    }
    groups = {
        "all": sorted(class_ap),
        "freq": sorted(set(class_ap).intersection(frequent)),
        "rare": sorted(set(class_ap).difference(frequent)),
    }
    result = {}
    for name, classes in groups.items():
        result[name] = 100.0 * float(np.mean([class_ap[c] for c in classes])) if classes else float("nan")
        result[f"{name}_classes"] = len(classes)
    return result


@torch.no_grad()
def evaluate(model, loader, device, frequent):
    model.eval()
    outputs, targets, ratio_indices = [], [], []
    for sequence, lengths, labels, ratio_index, _, _ in tqdm(loader, desc="mAP val", leave=False):
        logits = model(sequence.to(device), lengths)
        outputs.append(logits.sigmoid().cpu().numpy())
        targets.append(labels.numpy())
        ratio_indices.append(ratio_index.numpy())
    scores = np.concatenate(outputs)
    labels = np.concatenate(targets)
    ratio_indices = np.concatenate(ratio_indices)
    # EGO-TOPO constructs one label mask from all validation records and
    # reuses it for every observation ratio.  A class that is positive at a
    # different ratio but absent at this ratio therefore contributes AP=0,
    # rather than being silently removed from the denominator.
    evaluation_classes = set(np.flatnonzero(labels.sum(0) > 0).tolist())
    by_ratio = {}
    for index, ratio in enumerate(VAL_RATIOS):
        mask = ratio_indices == index
        by_ratio[str(int(ratio * 100))] = {
            "samples": int(mask.sum()),
            **grouped_map(
                labels[mask], scores[mask], frequent, evaluation_classes
            ),
        }
    average = {
        name: float(np.mean([by_ratio[key][name] for key in by_ratio]))
        for name in ("all", "freq", "rare")
    }
    return {"by_observation_percent": by_ratio, "average": average}


def load_frequent_verbs(
    path: Optional[Path], epic_rows: list[dict]
) -> set[int]:
    if path and path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames:
                raise ValueError(f"Empty many-shot verb CSV: {path}")
            key = "verb_class" if "verb_class" in rows.fieldnames else rows.fieldnames[0]
            return {int(row[key]) for row in rows}
    # EPIC_many_shot_verbs.csv is derived from the complete public EK55
    # training annotations, not from EGO-TOPO's 205-video model-training split.
    counts = Counter(int(row["verb_class"]) for row in epic_rows)
    return {verb for verb, count in counts.items() if count > 100}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epic_rows = load_epic_rows(args.annotations_csv)
    uid_index = action_uid_index(epic_rows)
    feature_index = index_features(args.frame_features)

    train_rows = load_egotopo_rows(args.split_dir / "train_S1.csv")
    val_rows = load_egotopo_rows(args.split_dir / "val_S1.csv")
    train_lengths = load_video_lengths(args.split_dir / "train_S1_nframes.csv")
    val_lengths = load_video_lengths(args.split_dir / "val_S1_nframes.csv")
    train_samples, train_stats = build_samples(
        train_rows, train_lengths, TRAIN_RATIOS, uid_index, feature_index,
        args.num_verbs, args.max_observed_actions, args.min_observed_features,
    )
    val_samples, val_stats = build_samples(
        val_rows, val_lengths, VAL_RATIOS, uid_index, feature_index,
        args.num_verbs, args.max_observed_actions, args.min_observed_features,
    )
    print("train_stats", json.dumps(train_stats))
    print("val_stats", json.dumps(val_stats))
    if not train_samples or not val_samples:
        raise RuntimeError("No usable EGO-TOPO samples; check feature coverage and split paths")

    frequent = load_frequent_verbs(args.many_shot_verbs, epic_rows)
    train_dataset = FutureVerbDataset(train_samples)
    val_dataset = FutureVerbDataset(val_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate,
    )
    device = torch.device(args.device)
    model = FutureVerbGRU(args.input_dim, args.hidden_dim, args.num_verbs).to(device)

    label_matrix = torch.stack([sample["labels"] for sample in train_samples])
    positives = label_matrix.sum(0)
    negatives = len(label_matrix) - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(max=50).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=3
    )
    best_score = -1.0
    stale = 0
    checkpoint = args.output_dir / "best_model.pth"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = samples = 0
        loop = tqdm(train_loader, desc=f"mAP train {epoch}/{args.epochs}")
        for sequence, lengths, labels, _, _, _ in loop:
            sequence = sequence.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequence, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch = labels.shape[0]
            running += loss.item() * batch
            samples += batch
            loop.set_postfix(loss=f"{running / samples:.4f}")
        metrics = evaluate(model, val_loader, device, frequent)
        score = metrics["average"]["all"]
        record = {"epoch": epoch, "train_loss": running / samples, **metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        scheduler.step(score)
        if score > best_score:
            best_score = score
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    best = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    final = evaluate(model, val_loader, device, frequent)
    result = {
        "protocol": "EGO-TOPO EK55 future-verb multi-label anticipation",
        "observation_percent": [25, 50, 75],
        "groups": "All / official many-shot verbs (>100) / remaining Rare verbs",
        "feature_source": "available Stage-1 observed-action RGB features",
        "train_stats": train_stats,
        "val_stats": val_stats,
        "frequent_verb_classes": sorted(frequent),
        "best_epoch": int(best["epoch"]),
        **final,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
