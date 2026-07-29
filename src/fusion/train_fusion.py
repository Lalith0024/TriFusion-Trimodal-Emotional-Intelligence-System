"""
src/fusion/train_fusion.py
──────────────────────────
Trains the FusionMLP that combines all three modality probability outputs.

Why synthetic training data?
  We need examples where all three modalities fired simultaneously with known
  ground-truth labels.  Collecting real trimodal data is expensive.  Instead
  we generate a synthetic training set that captures the key statistical
  patterns we want the fusion layer to learn:

    1. Congruent samples (70%): all modalities agree → fusion should be
       confident and match the ground-truth.
    2. Incongruent samples (30%): modalities disagree → fusion must learn
       to weight more-reliable modalities higher and produce a calibrated
       output that reflects genuine uncertainty.

  This is the standard approach for training late-fusion layers when
  joint multi-modal ground-truth labels are unavailable.

Input/output shapes (must match FusionMLP definition):
  Input:   [vision_probs(7) | audio_probs(8) | text_probs(8)] = 23-dim
  Output:  8-class unified probability distribution

Usage:
    # Must be run AFTER the three base models are trained:
    python src/vision/train_vision.py
    python src/audio/train_audio.py
    python src/text/train_text.py
    python src/fusion/train_fusion.py
"""

import os
import sys

# Add project root to python path to resolve 'config' and 'src' modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score, classification_report
from config.emotions import UNIFIED_EMOTIONS
from src.fusion.fusion_model import FusionMLP
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = "models/fusion/fusion_mlp.pth"
LABEL2IDX   = {e: i for i, e in enumerate(UNIFIED_EMOTIONS)}

# FER2013 has 7 classes (no "calm") — vision vector is always 7-dim
FER2013_EMOTIONS = [e for e in UNIFIED_EMOTIONS if e != "calm"]  # 7 labels


def _noisy_probs_7(dominant: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a 7-dim probability vector (FER2013 / vision modality).
    Uses a Dirichlet base for realistic class spread.
    """
    base = rng.dirichlet(np.ones(7) * 0.3)
    if dominant in FER2013_EMOTIONS:
        dom_idx = FER2013_EMOTIONS.index(dominant)
        base[dom_idx] += rng.uniform(0.4, 0.7)
    base /= base.sum()
    return base.astype(np.float32)


def _noisy_probs_8(dominant: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate an 8-dim probability vector (audio / text modalities).
    """
    base = rng.dirichlet(np.ones(8) * 0.3)
    base[LABEL2IDX[dominant]] += rng.uniform(0.4, 0.7)
    base /= base.sum()
    return base.astype(np.float32)


def generate_fusion_dataset(n_samples: int = 8000,
                            seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic fusion training set of (X, y) pairs.

    Each sample:
      X  — concatenated [vision(7), audio(8), text(8)] = 23-dim float32
      y  — ground-truth unified emotion class index (int)

    Distribution:
      70 % congruent  : all modalities point to the same emotion
      30 % incongruent: each modality points to a different random emotion
                        (teaches fusion to handle real-world masking scenarios)

    Returns:
        (X, y) as torch float32 / long tensors of shapes (N, 23) and (N,)
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for _ in range(n_samples):
        gt_idx   = rng.integers(0, len(UNIFIED_EMOTIONS))
        gt_label = UNIFIED_EMOTIONS[gt_idx]

        incongruent = rng.random() < 0.30

        if incongruent:
            # Each modality dominated by a different emotion
            other = [e for e in UNIFIED_EMOTIONS if e != gt_label]
            v_dom = rng.choice(other)
            a_dom = gt_label             # audio still correct (most reliable)
            t_dom = rng.choice(other)
        else:
            v_dom = a_dom = t_dom = gt_label

        v = _noisy_probs_7(v_dom, rng)   # 7-dim vision
        a = _noisy_probs_8(a_dom, rng)   # 8-dim audio
        t = _noisy_probs_8(t_dom, rng)   # 8-dim text

        X_list.append(np.concatenate([v, a, t]))   # 23-dim
        y_list.append(gt_idx)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.long)

    class_counts = np.bincount(y.numpy(), minlength=len(UNIFIED_EMOTIONS))
    logger.info(f"Synthetic dataset: {X.shape} | class counts: {class_counts.tolist()}")
    return X, y


def train():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)["training"].get("fusion", {})

    epochs     = cfg.get("epochs",     100)
    batch_size = cfg.get("batch_size", 256)
    lr         = cfg.get("lr",         0.001)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Training FusionMLP on: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    X, y = generate_fusion_dataset(n_samples=8000)

    dataset  = TensorDataset(X, y)
    val_n    = int(0.15 * len(dataset))
    tr_n     = len(dataset) - val_n
    tr_ds, val_ds = random_split(
        dataset, [tr_n, val_n],
        generator=torch.Generator().manual_seed(42)
    )

    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = FusionMLP(input_dim=23, output_dim=8).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # smoothing fights overfit
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Reduce LR when validation F1 stagnates — more adaptive than step LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    os.makedirs("models/fusion", exist_ok=True)
    best_f1 = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Split 23-dim concatenated vector back into three modality tensors
            # This matches FusionMLP.forward(vision_probs, audio_probs, text_probs)
            v_p = xb[:, :7]    # dims  0-6  → vision (7-class FER2013)
            a_p = xb[:, 7:15]  # dims  7-14 → audio  (8-class RAVDESS)
            t_p = xb[:, 15:]   # dims 15-22 → text   (8-class GoEmotions)

            optimizer.zero_grad()
            out  = model(v_p, a_p, t_p)   # returns softmax probs (B, 8)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                v_p, a_p, t_p = xb[:, :7], xb[:, 7:15], xb[:, 15:]
                preds = model(v_p, a_p, t_p).argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(yb.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        scheduler.step(val_f1)

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(tr_loader)
            logger.info(
                f"Epoch [{epoch+1:03d}/{epochs}] "
                f"Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), OUTPUT_PATH)

    logger.info(f"\nBest Val F1: {best_f1:.4f} — saved to {OUTPUT_PATH}")

    # ── Final classification report ───────────────────────────────────────────
    model.load_state_dict(torch.load(OUTPUT_PATH, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb[:, :7], xb[:, 7:15], xb[:, 15:]).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(yb.numpy())

    print("\n=== FUSION MODEL FINAL RESULTS ===")
    print(classification_report(
        all_labels, all_preds, target_names=UNIFIED_EMOTIONS, zero_division=0
    ))


if __name__ == "__main__":
    train()
