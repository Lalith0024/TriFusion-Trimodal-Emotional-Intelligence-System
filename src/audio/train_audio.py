"""
src/audio/train_audio.py
────────────────────────
Fine-tunes Wav2Vec2 on RAVDESS for speech emotion recognition.

Strategy:
  • Staged fine-tuning: freeze feature encoder for first N epochs,
    then unfreeze for full end-to-end training.
  • HuggingFace Trainer API handles mixed-precision, logging, and
    best-model checkpoint saving automatically.
  • Custom data collator pads variable-length waveforms per batch.

Usage:
    python src/audio/train_audio.py

Prerequisites:
    python data/download_datasets.py   (produces data/raw/ravdess/)
"""

import os
import sys
import re
import numpy as np
import torch
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from transformers import (
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.audio.emotion_model import build_wav2vec2_model, freeze_feature_encoder, unfreeze_feature_encoder
from config.emotions import RAVDESS_LABELS, RAVDESS_TO_UNIFIED
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAVDESS_DIR = Path("data/raw/ravdess")
OUTPUT_DIR  = Path("models/audio/wav2vec2_ravdess")
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# RAVDESS filename parser
# Filename format: 03-01-04-01-02-01-12.wav
#   field[2] (index 2) → emotion code 01-08
# ---------------------------------------------------------------------------
def parse_ravdess_label(filepath: Path) -> int:
    """Extract 0-indexed emotion label from RAVDESS filename."""
    parts = filepath.stem.split("-")
    # emotion code is 1-indexed (01..08) → convert to 0-indexed
    return int(parts[2]) - 1


def load_ravdess_dataset() -> DatasetDict:
    """
    Walk RAVDESS actor directories, load waveforms, and build a DatasetDict.
    Applies 80/10/10 stratified split by label.
    """
    audio_files = sorted(RAVDESS_DIR.rglob("*.wav"))
    if not audio_files:
        logger.error(f"No .wav files found in {RAVDESS_DIR}. Run download_datasets.py first.")
        sys.exit(1)

    paths, labels = [], []
    for f in audio_files:
        try:
            label = parse_ravdess_label(f)
            paths.append(str(f))
            labels.append(label)
        except (IndexError, ValueError):
            logger.warning(f"Skipping malformed filename: {f.name}")

    logger.info(f"Loaded {len(paths)} audio files | Label dist: {np.bincount(labels).tolist()}")

    # Stratified split → train/val/test
    tr_p, te_p, tr_l, te_l = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=42)
    va_p, te_p, va_l, te_l = train_test_split(te_p,  te_l,  test_size=0.5, stratify=te_l,  random_state=42)

    def make_dataset(ps, ls):
        return Dataset.from_dict({"path": ps, "label": ls})

    return DatasetDict({
        "train": make_dataset(tr_p, tr_l),
        "val":   make_dataset(va_p, va_l),
        "test":  make_dataset(te_p, te_l),
    })


def preprocess(batch: dict, processor: Wav2Vec2Processor) -> dict:
    """
    Load waveform from disk and extract input_values for Wav2Vec2Processor.
    Resamples to 16 kHz if source rate differs.
    """
    waveform, sr = librosa.load(batch["path"], sr=SAMPLE_RATE, mono=True)
    inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False)
    batch["input_values"] = inputs.input_values[0].numpy()
    return batch


@dataclass
class SpeechCollator:
    """
    Pads variable-length input_values to the longest sequence in the batch.
    Wav2Vec2Processor.pad() handles attention mask creation automatically.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [f["label"] for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        batch["labels"] = torch.tensor(label_features, dtype=torch.long)
        return batch


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"weighted_f1": f1}


def train():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)["training"]["audio"]

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model     = build_wav2vec2_model(pretrained=True)

    # Phase 1: freeze feature encoder for stability
    freeze_feature_encoder(model)
    logger.info("Feature encoder frozen for initial epochs.")

    dataset = load_ravdess_dataset()

    # Pre-process all splits (loads waveforms in-memory — fine for RAVDESS ~2 GB)
    logger.info("Preprocessing audio files...")
    fn = lambda b: preprocess(b, processor)
    dataset = dataset.map(fn, remove_columns=["path"])

    collator = SpeechCollator(processor=processor)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),   # mixed precision on CUDA GPU
        logging_steps=20,
        report_to="none",                 # disable wandb/tb by default
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Starting Phase 1 training (frozen encoder)...")
    trainer.train()

    # Phase 2: unfreeze encoder for full fine-tuning
    logger.info(f"Unfreezing encoder after epoch {cfg['freeze_base_epochs']} — full fine-tune.")
    unfreeze_feature_encoder(model)
    trainer.train()

    # Save final model + processor together (loadable with from_pretrained)
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # Final test evaluation
    results = trainer.evaluate(dataset["test"])
    logger.info(f"Test results: {results}")


if __name__ == "__main__":
    train()
