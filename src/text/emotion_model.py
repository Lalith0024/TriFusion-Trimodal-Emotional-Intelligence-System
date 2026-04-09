"""
src/text/emotion_model.py
─────────────────────────
RoBERTa-base fine-tuned for text emotion classification.

Why RoBERTa over BERT?
  • RoBERTa removes Next Sentence Prediction and trains on more data
    with larger batches — it consistently outperforms BERT on
    sentiment/emotion classification benchmarks.
  • We use the "base" variant (125 M params) for a good speed/accuracy
    trade-off at inference time.

The GoEmotions dataset has 27 fine-grained emotion classes.
We remap them to the 8 unified classes at the dataset level
(see train_text.py) so the model head outputs 8 logits directly.
"""

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig
from config.emotions import TEXT_NUM_CLASSES


def build_roberta_model(
    num_labels: int = TEXT_NUM_CLASSES,
    pretrained: bool = True
) -> RobertaForSequenceClassification:
    """
    Build RoBERTa for sequence classification.

    Args:
        num_labels: Number of unified emotion classes (default 8).
        pretrained: If True, load roberta-base HuggingFace weights.
                    Set False when loading from a local fine-tuned checkpoint.

    Returns:
        RobertaForSequenceClassification instance ready for training/inference.
    """
    if pretrained:
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=num_labels,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True  # replaces original 2-class head safely
        )
    else:
        config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_labels)
        model = RobertaForSequenceClassification(config)

    return model
