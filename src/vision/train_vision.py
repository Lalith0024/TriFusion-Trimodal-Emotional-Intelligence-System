"""
src/vision/train_vision.py
──────────────────────────
Training script for FacialEmotionNet on FER2013.

Strategy:
  • Weighted cross-entropy + label smoothing to handle class imbalance.
  • CosineAnnealingLR scheduler for smoother convergence.
  • Best model saved by weighted-F1 on held-out validation split.

Usage:
    python src/vision/train_vision.py

Prerequisites:
    python data/download_datasets.py   (produces data/raw/fer2013/)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, f1_score
from src.vision.emotion_model import FacialEmotionNet, get_transforms
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_class_weights(dataset: ImageFolder, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency class weights to counter FER2013 imbalance.
    'disgusted' is heavily under-represented (~600 samples vs ~8000 for happy).
    """
    counts = [0] * num_classes
    for _, label in dataset:
        counts[label] += 1
    # Guard against zero counts
    weights = [1.0 / max(c, 1) for c in counts]
    return torch.tensor(weights, dtype=torch.float).to(device)


def train():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)["training"]["vision"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Training device: {device}")

    # ------------------------------------------------------------------ data
    fer_train_root = "data/raw/fer2013/train"
    fer_test_root  = "data/raw/fer2013/test"

    if not os.path.exists(fer_train_root):
        logger.error("FER2013 not found. Run: python data/download_datasets.py")
        sys.exit(1)

    train_dataset = ImageFolder(fer_train_root, transform=get_transforms(train=True))
    test_dataset  = ImageFolder(fer_test_root,  transform=get_transforms(train=False))

    # 90 / 10 train / val split — reproducible with fixed generator seed
    val_size   = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # num_workers=0 is safer on macOS to prevent multiprocessing hangs
    train_loader = DataLoader(train_ds,    batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,      batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_dataset)}")
    logger.info(f"Classes: {train_dataset.classes}")

    # --------------------------------------------------------------- model
    model = FacialEmotionNet(pretrained=True).to(device)

    # Weighted loss + label smoothing (0.1) for noisy FER2013 labels
    class_weights = compute_class_weights(train_dataset, len(train_dataset.classes), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    os.makedirs("models/vision", exist_ok=True)
    best_val_f1 = 0.0

    # --------------------------------------------------------- training loop
    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0

        # Wrap train_loader with tqdm for a visible progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ------------------------------------------------------ validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                preds = model(images.to(device)).argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        scheduler.step()

        logger.info(
            f"Epoch [{epoch+1:02d}/{cfg['epochs']}] "
            f"Loss: {running_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "models/vision/efficientnet_fer2013.pth")
            logger.info(f"  ✓ Best model saved (F1: {val_f1:.4f})")

    # --------------------------------------------------------- final test
    logger.info("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load("models/vision/efficientnet_fer2013.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            all_preds.extend(model(images.to(device)).argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "="*50)
    print("FINAL TEST RESULTS — FER2013")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes, zero_division=0))


if __name__ == "__main__":
    train()
