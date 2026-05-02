"""
src/text/train_text.py
──────────────────────
Fine-tunes RoBERTa-base on GoEmotions for text-based emotion classification.

Strategy:
  • GoEmotions "simplified" has 28 fine-grained categories.
  • We remap them to our unified 8-class schema (config/emotions.py) before training.
  • HuggingFace Trainer handles mixed precision, gradient accumulation,
    best-model checkpointing, and early stopping automatically.
  • Tokenizer + model are saved together so TextInference can reload both
    with a single from_pretrained() call.

Label mapping rationale:
  GoEmotions is multi-label but the "simplified" split provides effectively
  single-label examples.  We take the first label ID per example (most
  prominent) and map its string name through GOEMOTIONS_TO_UNIFIED.
  Examples with empty label lists default to "neutral".

Usage:
    python src/text/train_text.py

Prerequisites:
    pip install datasets transformers  (already in requirements.txt)
    GoEmotions is downloaded automatically from HuggingFace Hub — no manual
    download step required.
"""

import os
import sys
import numpy as np
import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from config.emotions import UNIFIED_EMOTIONS, GOEMOTIONS_TO_UNIFIED
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "models/text/roberta_goemotions"
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128

# Unified 8-class label ↔ integer index mappings
LABEL2IDX = {e: i for i, e in enumerate(UNIFIED_EMOTIONS)}
IDX2LABEL = {i: e for i, e in enumerate(UNIFIED_EMOTIONS)}


def build_label_mapper(dataset):
    """
    Extract the actual label names from the dataset's ClassLabel feature.
    This is safer than hardcoding the list since dataset schema may vary.

    Returns:
        A function (example → example) that writes an integer 'label' field.
    """
    # GoEmotions simplified stores labels as a Sequence of ClassLabel ints
    try:
        label_names = dataset["train"].features["labels"].feature.names
    except AttributeError:
        # Fallback: the canonical 28 go_emotions label order
        label_names = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral",
        ]
        logger.warning("Using hardcoded GoEmotions label list (feature names not found).")

    def map_label(example: dict) -> dict:
        """Take first (most prominent) label; remap to unified 8-class int."""
        label_ids = example.get("labels", [])
        if label_ids:
            go_name = label_names[label_ids[0]]
        else:
            go_name = "neutral"

        unified          = GOEMOTIONS_TO_UNIFIED.get(go_name, "neutral")
        example["label"] = LABEL2IDX[unified]
        return example

    return map_label


def tokenize_batch(batch: dict, tokenizer: RobertaTokenizerFast) -> dict:
    """Tokenise a batch of texts.  Dynamic padding is applied by the collator."""
    return tokenizer(
        batch["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,   # DataCollatorWithPadding handles per-batch padding
    )


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0)
    }


def train():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)["training"]["text"]

    # ── Load dataset directly from HuggingFace Hub ───────────────────────────
    logger.info("Downloading GoEmotions from HuggingFace Hub (cached after first run)...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    # Expected splits: train, validation, test

    # ── Remap 28-class GoEmotions → unified 8-class integer labels ───────────
    map_fn = build_label_mapper(dataset)
    logger.info("Remapping GoEmotions labels to unified 8-class schema...")
    # remove_columns drops 'labels' (old multi-label field) and keeps 'text'
    cols_to_remove = [c for c in dataset["train"].column_names if c != "text"]
    dataset = dataset.map(map_fn, remove_columns=cols_to_remove)

    # ── Tokenise all splits ───────────────────────────────────────────────────
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    logger.info("Tokenising dataset...")
    tok_fn  = lambda batch: tokenize_batch(batch, tokenizer)
    dataset = dataset.map(tok_fn, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(UNIFIED_EMOTIONS),
        problem_type="single_label_classification",
        id2label=IDX2LABEL,
        label2id=LABEL2IDX,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        warmup_ratio=0.06,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),   # mixed precision on GPU
        logging_steps=50,
        report_to="none",                 # disable wandb / tensorboard by default
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Training RoBERTa on GoEmotions (unified 8-class)...")
    trainer.train()

    # ── Test evaluation ───────────────────────────────────────────────────────
    results = trainer.evaluate(dataset["test"])
    logger.info(f"Test results: {results}")

    # Full per-class report
    test_pred = trainer.predict(dataset["test"])
    preds     = np.argmax(test_pred.predictions, axis=-1)
    print("\n=== ROBERTA FINAL TEST RESULTS ===")
    print(classification_report(
        test_pred.label_ids, preds,
        target_names=UNIFIED_EMOTIONS, zero_division=0
    ))

    # ── Save model + tokenizer together (TextInference uses from_pretrained) ──
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model + tokenizer saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
