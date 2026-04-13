"""
src/fusion/train_fusion.py
──────────────────────────
Training script for the FusionMLP.

Since we typically don't have ground-truth "fused emotion" labels,
we generate a synthetic training set from the individual model outputs
on a held-out labelled dataset (or from saved predictions).

Strategy:
  1. Run all three inference models over a common labelled split.
  2. Use the majority-vote of the three dominant labels as synthetic target.
  3. Train FusionMLP to reproduce that label from the concatenated prob vectors.

This allows the fusion layer to learn modality weighting without requiring
expensive end-to-end joint training across all three backbones.

Usage:
    python src/fusion/train_fusion.py

Prerequisites:
    All three per-modality models must be trained first.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, classification_report
from src.fusion.fusion_model import FusionMLP
from config.emotions import UNIFIED_EMOTIONS
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYNTHETIC_DATA_PATH = "data/processed/fusion_training_data.json"
OUTPUT_PATH         = "models/fusion/fusion_mlp.pth"
LABEL2ID            = {e: i for i, e in enumerate(UNIFIED_EMOTIONS)}


class FusionDataset(Dataset):
    """
    Dataset of (vision_probs, audio_probs, text_probs, label) tuples.
    Loaded from a pre-generated JSON file.
    """
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        v = torch.tensor([r["vision"].get(e, 0.0) for e in UNIFIED_EMOTIONS[:7]], dtype=torch.float32)
        a = torch.tensor([r["audio"].get(e,  0.0) for e in UNIFIED_EMOTIONS],     dtype=torch.float32)
        t = torch.tensor([r["text"].get(e,   0.0) for e in UNIFIED_EMOTIONS],     dtype=torch.float32)
        label = torch.tensor(LABEL2ID.get(r["label"], 0), dtype=torch.long)
        return v, a, t, label


def majority_vote(v_dom: str, a_dom: str, t_dom: str) -> str:
    """Return the emotion that appears most often across three modalities."""
    votes = [v_dom, a_dom, t_dom]
    return max(set(votes), key=votes.count)


def generate_synthetic_data():
    """
    Placeholder that generates dummy training records for the fusion model.
    In a real setup, this function would run all three inference models
    over a labelled multi-modal dataset and serialise predictions to JSON.

    For demonstration purposes we generate N=2000 random probability records
    with noisy majority-vote labels so training doesn't error out.
    """
    logger.warning(
        "Generating SYNTHETIC fusion training data. "
        "For production: replace with real multi-modal predictions."
    )
    os.makedirs("data/processed", exist_ok=True)
    rng     = np.random.default_rng(42)
    records = []

    for _ in range(2000):
        def rand_probs(n):
            p = rng.dirichlet(np.ones(n))
            return {UNIFIED_EMOTIONS[i]: float(p[i]) for i in range(n)}

        v_probs  = rand_probs(7)   # vision: 7-class
        a_probs  = rand_probs(8)
        t_probs  = rand_probs(8)

        # Map vision's 7-class to UNIFIED_EMOTIONS by taking max-keyed entry
        # (simplified for synthetic generation)
        v_dom = max(v_probs, key=v_probs.get)
        a_dom = max(a_probs, key=a_probs.get)
        t_dom = max(t_probs, key=t_probs.get)
        label = majority_vote(v_dom, a_dom, t_dom)

        # Pad vision probs to 8-class (calm=0) for storage consistency
        v_full = {**{e: 0.0 for e in UNIFIED_EMOTIONS}, **v_probs}

        records.append({"vision": v_full, "audio": a_probs, "text": t_probs, "label": label})

    with open(SYNTHETIC_DATA_PATH, "w") as f:
        json.dump(records, f)
    logger.info(f"Saved {len(records)} synthetic records → {SYNTHETIC_DATA_PATH}")
    return records


def train():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)["training"]["fusion"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training FusionMLP on: {device}")

    # Load or generate training data
    if os.path.exists(SYNTHETIC_DATA_PATH):
        with open(SYNTHETIC_DATA_PATH) as f:
            records = json.load(f)
        logger.info(f"Loaded {len(records)} fusion training records.")
    else:
        records = generate_synthetic_data()

    dataset = FusionDataset(records)
    val_size   = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    model     = FusionMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs("models/fusion", exist_ok=True)
    best_f1 = 0.0

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0

        for v, a, t, labels in train_loader:
            v, a, t, labels = v.to(device), a.to(device), t.to(device), labels.to(device)
            optimizer.zero_grad()
            # FusionMLP expects vision as 7-dim; slice first 7 dims
            probs  = model(v[:, :7], a, t)
            # Compute loss from logits before final softmax (re-use net internals)
            loss   = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for v, a, t, labels in val_loader:
                v, a, t = v.to(device), a.to(device), t.to(device)
                probs   = model(v[:, :7], a, t)
                preds   = probs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        scheduler.step()

        logger.info(f"Epoch [{epoch+1:02d}/{cfg['epochs']}] Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), OUTPUT_PATH)
            logger.info(f"  ✓ Saved FusionMLP (F1: {val_f1:.4f})")

    logger.info(f"Training complete. Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
