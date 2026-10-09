"""Rebuild the existing Stage-1 checkpoint with the original configuration."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from main import get_dataloader, load_cooccurrence_matrix, train
from model import ActionRecognitionModel


FRAME_DIR = "/data/datasets/EPIC-KITCHENS/features/frame_features"
MASK_DIR = "/data/datasets/EPIC-KITCHENS/features/mask_features"
ANNOTATION_DIR = "/data/datasets/EPIC-KITCHENS/insight_annotations"
COOCCURRENCE = "/data/datasets/EPIC-KITCHENS/features/cooccurrence_matrix.pt"
CHECKPOINT_DIR = Path("/data/datasets/EPIC-KITCHENS/stage1_rebuilt_checkpoint")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("/tmp/stage1_rebuild_training.log"), logging.StreamHandler()],
        force=True,
    )
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / "best_model.pth"
    if checkpoint.exists():
        # Refuse to load or overwrite a partial result from an earlier attempt.
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        raise FileExistsError(f"Usable checkpoint already exists: {checkpoint}")

    train_loader = get_dataloader(
        splits=["train"],
        frame_features_dir=FRAME_DIR,
        mask_features_dir=MASK_DIR,
        annotation_dir=ANNOTATION_DIR,
        batch_size=8,
        shuffle=True,
        num_workers=4,
    )
    val_loader = get_dataloader(
        splits=["val"],
        frame_features_dir=FRAME_DIR,
        mask_features_dir=MASK_DIR,
        annotation_dir=ANNOTATION_DIR,
        batch_size=8,
        shuffle=False,
        num_workers=4,
    )
    logging.info("Stage1 loaders: train_batches=%d val_batches=%d", len(train_loader), len(val_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionModel(
        input_dim=1024,
        mlp_hidden_dim=2048,
        mlp_output_dim=256,
        transformer_layers=4,
        n_heads=8,
        transformer_hidden_dim=2048,
        verb_num_classes=125,
        noun_num_classes=352,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=2e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2, min_lr=1e-6
    )
    cooccurrence = load_cooccurrence_matrix(COOCCURRENCE)

    logging.info("STAGE1_TRAIN_BEGIN device=%s", device)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_plateau=scheduler,
        device=device,
        epochs=40,
        checkpoint_dir=str(CHECKPOINT_DIR),
        cooccurrence_matrix=cooccurrence,
        patience=5,
        max_grad_norm=1.0,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    logging.info("STAGE1_TRAIN_DONE tensors=%d checkpoint=%s", len(state), checkpoint)


if __name__ == "__main__":
    main()
