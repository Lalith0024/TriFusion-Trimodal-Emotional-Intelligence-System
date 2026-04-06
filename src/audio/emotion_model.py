"""
src/audio/emotion_model.py
──────────────────────────
Wav2Vec2 model configured for speech emotion recognition.

Design notes:
  • We use facebook/wav2vec2-base (95 M params) rather than the larger
    variants — it fits in 4 GB VRAM and runs inference in ~120 ms on CPU.
  • Wav2Vec2ForSequenceClassification pools the transformer output across
    the time axis (mean pooling) before the classification head.
  • We set ignore_mismatched_sizes=True so the pre-trained 32-class
    LM head is silently replaced with our num_labels-class head.
  • During training we freeze the feature encoder for the first
    freeze_base_epochs epochs (see train_audio.py) to stabilise gradients.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config
from config.emotions import RAVDESS_NUM_CLASSES


def build_wav2vec2_model(
    num_labels: int = RAVDESS_NUM_CLASSES,
    pretrained: bool = True
) -> Wav2Vec2ForSequenceClassification:
    """
    Build and return a Wav2Vec2ForSequenceClassification instance.

    Args:
        num_labels: Number of output emotion classes (default 8 for RAVDESS).
        pretrained: If True, initialise with facebook/wav2vec2-base weights.
                    Set False when loading a full fine-tuned checkpoint.

    Returns:
        HuggingFace Wav2Vec2ForSequenceClassification model.
    """
    if pretrained:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_labels,
            problem_type="single_label_classification",
            # Suppress size mismatch warning for the classification head
            ignore_mismatched_sizes=True
        )
    else:
        config = Wav2Vec2Config.from_pretrained(
            "facebook/wav2vec2-base", num_labels=num_labels
        )
        model = Wav2Vec2ForSequenceClassification(config)

    return model


def freeze_feature_encoder(model: Wav2Vec2ForSequenceClassification) -> None:
    """
    Freeze the convolutional feature encoder layers.
    Call this for the first N epochs so only the transformer + head are trained.
    Significantly reduces memory and stabilises early training.
    """
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False


def unfreeze_feature_encoder(model: Wav2Vec2ForSequenceClassification) -> None:
    """Unfreeze feature encoder for full fine-tuning in later epochs."""
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = True
