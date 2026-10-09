"""Train/evaluate an EK55 RGB action-anticipation classifier.

Primary metrics are micro Action Top-1 and Top-5 accuracy at the anticipation
time encoded by the manifest.  Verb and noun Top-k values are reported as
auxiliary diagnostics.  A direct head ranks the 2513 observed EK55 actions;
factorized verb/noun scores are also included for ablation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class AnticipationDataset(Dataset):
    def __init__(self, manifest: Path, feature_root: Path, split: str) -> None:
        rows = read_jsonl(manifest)
        self.samples = []
        missing = 0
        for row in rows:
            path = feature_root / split / f'{row["uid"]}.pt'
            if not path.exists():
                missing += 1
                continue
            self.samples.append((
                path,
                int(row["verb_class"]),
                int(row["noun_class"]),
                int(row["action_class"]),
                int(row["uid"]),
            ))
        self.manifest_samples = len(rows)
        self.missing = missing
        print(
            f"[{split}] usable={len(self.samples)} manifest={len(rows)} "
            f"missing_features={missing} coverage="
            f"{100.0 * len(self.samples) / max(len(rows), 1):.2f}%"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, verb, noun, action, uid = self.samples[index]
        sequence = torch.load(path, map_location="cpu", weights_only=True).float()
        if sequence.ndim != 2:
            raise ValueError(f"Expected [T,D] feature at {path}, got {tuple(sequence.shape)}")
        return sequence, verb, noun, action, uid


class AnticipationClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sequence_length: int,
        layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        num_verbs: int,
        num_nouns: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = nn.Parameter(torch.zeros(1, sequence_length + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.verb_head = nn.Linear(hidden_dim, num_verbs)
        self.noun_head = nn.Linear(hidden_dim, num_nouns)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, sequence: torch.Tensor):
        batch, length, _ = sequence.shape
        if length + 1 > self.position.shape[1]:
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum "
                f"{self.position.shape[1] - 1}"
            )
        sequence = self.input_projection(sequence)
        cls = self.cls_token.expand(batch, -1, -1)
        hidden = torch.cat([cls, sequence], dim=1)
        hidden = hidden + self.position[:, :length + 1]
        hidden = self.output_norm(self.transformer(hidden)[:, 0])
        return self.verb_head(hidden), self.noun_head(hidden), self.action_head(hidden)


def factorized_action_scores(
    verb_logits: torch.Tensor,
    noun_logits: torch.Tensor,
    action_pairs: torch.Tensor,
) -> torch.Tensor:
    verb_log_prob = verb_logits.log_softmax(-1)
    noun_log_prob = noun_logits.log_softmax(-1)
    return (
        verb_log_prob[:, action_pairs[:, 0]]
        + noun_log_prob[:, action_pairs[:, 1]]
    )


def empty_counts() -> dict[str, int]:
    return {key: 0 for key in (
        "verb_top1", "verb_top5", "noun_top1", "noun_top5",
        "action_top1", "action_top5",
        "action_factorized_top1", "action_factorized_top5",
    )}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_pairs: torch.Tensor,
    criterion: nn.Module,
    aux_weight: float,
) -> dict[str, float]:
    model.eval()
    counts = empty_counts()
    total = 0
    total_loss = 0.0
    for sequence, verb, noun, action, _ in tqdm(loader, desc="Validate", leave=False):
        sequence = sequence.to(device, non_blocking=True)
        verb = verb.to(device, non_blocking=True)
        noun = noun.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        verb_logits, noun_logits, action_logits = model(sequence)
        loss = (
            criterion(action_logits, action)
            + aux_weight * criterion(verb_logits, verb)
            + aux_weight * criterion(noun_logits, noun)
        )
        batch = sequence.shape[0]
        total_loss += loss.item() * batch
        total += batch

        predictions = {
            "verb": verb_logits.topk(5, -1).indices,
            "noun": noun_logits.topk(5, -1).indices,
            "action": action_logits.topk(5, -1).indices,
            "action_factorized": factorized_action_scores(
                verb_logits, noun_logits, action_pairs
            ).topk(5, -1).indices,
        }
        labels = {"verb": verb, "noun": noun, "action": action,
                  "action_factorized": action}
        for name, top5 in predictions.items():
            label = labels[name]
            counts[f"{name}_top1"] += (top5[:, 0] == label).sum().item()
            counts[f"{name}_top5"] += (top5 == label[:, None]).any(-1).sum().item()

    if total == 0:
        raise RuntimeError("Validation dataset has no usable features")
    return {
        "loss": total_loss / total,
        "samples": total,
        **{key: 100.0 * value / total for key, value in counts.items()},
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = json.loads((args.manifest_dir / "action_vocab.json").read_text())
    metadata = json.loads((args.manifest_dir / "metadata.json").read_text())
    action_pairs = torch.tensor([
        [row["verb_class"], row["noun_class"]] for row in vocabulary["actions"]
    ], dtype=torch.long)

    train_set = AnticipationDataset(
        args.manifest_dir / "train.jsonl", args.features_dir, "train"
    )
    val_set = AnticipationDataset(
        args.manifest_dir / "val.jsonl", args.features_dir, "val"
    )
    if not train_set.samples or not val_set.samples:
        raise RuntimeError("Both train and val require extracted anticipation features")
    sequence_length = train_set[0][0].shape[0]
    if val_set[0][0].shape[0] != sequence_length:
        raise ValueError("Train and validation sequence lengths differ")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device)
    action_pairs = action_pairs.to(device)
    model = AnticipationClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        sequence_length=sequence_length,
        layers=args.layers,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        num_verbs=vocabulary["num_verbs"],
        num_nouns=vocabulary["num_nouns"],
        num_actions=vocabulary["num_actions"],
    ).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=True)

    criterion = nn.CrossEntropyLoss()
    checkpoint_path = args.output_dir / "best_model.pth"
    history = []
    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.3, patience=2
        )
        best_score = -1.0
        stale = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = samples = 0
            loop = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
            for sequence, verb, noun, action, _ in loop:
                sequence = sequence.to(device, non_blocking=True)
                verb = verb.to(device, non_blocking=True)
                noun = noun.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                verb_logits, noun_logits, action_logits = model(sequence)
                loss = (
                    criterion(action_logits, action)
                    + args.aux_weight * criterion(verb_logits, verb)
                    + args.aux_weight * criterion(noun_logits, noun)
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batch = sequence.shape[0]
                running_loss += loss.item() * batch
                samples += batch
                loop.set_postfix(loss=f"{running_loss / samples:.4f}")
            metrics = evaluate(
                model, val_loader, device, action_pairs, criterion, args.aux_weight
            )
            metrics.update({
                "epoch": epoch,
                "train_loss": running_loss / samples,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            history.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False))
            score = metrics["action_top1"] + metrics["action_top5"]
            scheduler.step(score)
            if score > best_score:
                best_score = score
                stale = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, checkpoint_path)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping after {epoch} epochs")
                    break
        best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["model"], strict=True)
    elif not args.checkpoint:
        if not checkpoint_path.exists():
            raise ValueError("--eval-only requires --checkpoint or output-dir/best_model.pth")
        best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best.get("model", best), strict=True)

    final_metrics = evaluate(
        model, val_loader, device, action_pairs, criterion, args.aux_weight
    )
    result = {
        "protocol": "EK55 action anticipation, tau_a=1.0s, RULSTM video split",
        "metric_definition": "micro accuracy over direct ranked joint-action scores",
        "manifest_metadata": metadata,
        "train_usable": len(train_set),
        "train_manifest": train_set.manifest_samples,
        "val_usable": len(val_set),
        "val_manifest": val_set.manifest_samples,
        **final_metrics,
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
