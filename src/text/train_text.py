"""
src/text/train_text.py
──────────────────────
Fine-tunes RoBERTa on GoEmotions (remapped to 8 unified classes).

Strategy:
  • Load GoEmotions "simplified" split (single-label, 28 classes including
    "neutral") from HuggingFace datasets.
  • Remap 27 fine-grained labels to 8 unified emotion classes using
    GOEMOTIONS_TO_UNIFIED mapping from config/emotions.py.
  • Use HuggingFace Trainer with linear warmup + weight decay.
  • Best model saved by weighted-F1 on validation set.

Usage:
    python src/text/train_text.py

Prerequisites:
    python data/download_datasets.py   (produces data/raw/goemotions/)
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers import (
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score

from src.text.emotion_model import build_roberta_model
from config.emotions import GOEMOTIONS_TO_UNIFIED, UNIFIED_EMOTIONS
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOEMOTIONS_PATH = Path("data/raw/goemotions")
OUTPUT_DIR      = Path("models/text/roberta_goemotions")

# Build label-index mapping for the 8 unified classes
LABEL2ID = {e: i for i, e in enumerate(UNIFIED_EMOTIONS)}
ID2LABEL = {i: e for i, e in enumerate(UNIFIED_EMOTIONS)}


def remap_goemotions_label(example: dict) -> dict:
    """
    GoEmotions simplified has a 'labels' list (multi-label, but simplified
    split is effectively single-label).  We take the first label, look up
    its string name, then remap to our 8-class unified schema.
    """
    # GoEmotions 'simplified' label names are already strings in 'label' field
    # but the HF dataset uses integer IDs — we use the dataset's label feature
    original_label = example.get("label", 26)  # 26 = 'neutral' fallback
    # Convert int → string using dataset's feature names (set in map call)
    label_name   = example.get("label_str", "neutral")
    unified      = GOEMOTIONS_TO_UNIFIED.get(label_name, "neutral")
    example["label"] = LABEL2ID[unified]
    return example


def tokenize(example: dict, tokenizer: RobertaTokenizerFast) -> dict:
    return tokenizer(
        example["text"],
        max_length=128,
        truncation=True,
        padding=False
    )


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"weighted_f1": f1}


def train():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)["training"]["text"]

    if not GOEMOTIONS_PATH.exists():
        logger.error(f"GoEmotions not found at {GOEMOTIONS_PATH}. Run download_datasets.py first.")
        sys.exit(1)

    logger.info("Loading GoEmotions dataset...")
    dataset = load_from_disk(str(GOEMOTIONS_PATH))

    # GoEmotions simplified has 28 labels; extract label names per split
    label_feature = dataset["train"].features["labels"]  # Sequence of ClassLabel
    # Simplified: first label in list = dominant label
    def extract_single_label(ex):
        label_id  = ex["labels"][0] if ex["labels"] else 27  # 27 = neutral
        label_str = label_feature.feature.int2str(label_id)
        unified   = GOEMOTIONS_TO_UNIFIED.get(label_str, "neutral")
        return {"label": LABEL2ID[unified], "text": ex["text"]}

    logger.info("Remapping GoEmotions labels to unified 8-class schema...")
    dataset = dataset.map(extract_single_label, remove_columns=[c for c in dataset["train"].column_names if c not in ["text"]])

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    logger.info("Tokenising dataset...")
    tokenized = dataset.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True, remove_columns=["text"]
    )

    model   = build_roberta_model(pretrained=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("Starting RoBERTa fine-tuning on GoEmotions...")
    trainer.train()

    # Save model + tokenizer for inference (loadable with from_pretrained)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"Model saved to {OUTPUT_DIR}")

    results = trainer.evaluate(tokenized["test"])
    logger.info(f"Test results: {results}")


if __name__ == "__main__":
    train()
