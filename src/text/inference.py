"""
src/text/inference.py
─────────────────────
Text emotion inference using fine-tuned RoBERTa.

Flow:
  1. Tokenise input text with RobertaTokenizerFast (max 128 tokens).
  2. Model returns logits → softmax → unified 8-class probability dict.
  3. Short/empty text (< 2 chars) bypasses the model and returns a
     uniform distribution to avoid over-confident neutral predictions.

The probability dict keys follow UNIFIED_EMOTIONS order so FusionInference
can concat them with vision and audio tensors without any re-indexing.
"""

import os
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from config.emotions import UNIFIED_EMOTIONS
import logging

logger = logging.getLogger(__name__)


class TextInference:
    """
    Stateful text emotion inference engine.
    Instantiate once; call .predict(text) per transcription.
    """

    def __init__(self, model_path: str = None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        fallback_model = "SamLowe/roberta-base-go_emotions"

        is_valid_local = (
            model_path
            and os.path.exists(model_path)
            and os.path.isdir(model_path)
            and os.path.exists(os.path.join(model_path, "config.json"))
            and (
                os.path.exists(os.path.join(model_path, "model.safetensors"))
                or os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
            )
        )
        model_src = model_path if is_valid_local else fallback_model

        logger.info(f"Loading text model from: {model_src}")

        try:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_src)
        except Exception as e:
            logger.warning(f"Failed to load text tokenizer from {model_src} ({e}). Falling back to {fallback_model}.")
            self.tokenizer = RobertaTokenizerFast.from_pretrained(fallback_model)

        try:
            self.model = RobertaForSequenceClassification.from_pretrained(model_src).to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load text model from {model_src} ({e}). Falling back to {fallback_model}.")
            self.model = RobertaForSequenceClassification.from_pretrained(fallback_model).to(self.device)

        self.model.eval()


    def predict(self, text: str) -> dict:
        """
        Classify the emotion expressed in `text`.

        Args:
            text: Transcribed speech string (or any freeform text).

        Returns:
            {
              "probabilities": dict[emotion → float],
              "dominant":      str,
              "confidence":    float,
              "text":          str   (echoed back for logging/display)
            }
        """
        # Short text → uniform distribution to avoid noise-driven predictions
        if not text or len(text.strip()) < 2:
            return {
                "probabilities": {e: 1.0 / 8 for e in UNIFIED_EMOTIONS},
                "dominant":      "neutral",
                "confidence":    0.125,
                "text":          text
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Initialize base probabilities
        probabilities = {e: 0.0 for e in UNIFIED_EMOTIONS}
        num_labels = len(self.model.config.id2label)

        if num_labels == len(UNIFIED_EMOTIONS):
            # Custom 8-class model (trained locally)
            for i, e in enumerate(UNIFIED_EMOTIONS):
                probabilities[e] = float(probs[i] if probs.ndim > 0 else probs)
        else:
            # Community 28-class model (e.g. SamLowe/roberta-base-go_emotions)
            from config.emotions import GOEMOTIONS_TO_UNIFIED
            for i in range(num_labels):
                label = self.model.config.id2label[i].lower()
                unified = GOEMOTIONS_TO_UNIFIED.get(label, "neutral")
                probabilities[unified] += float(probs[i] if probs.ndim > 0 else probs)

        dominant = max(probabilities, key=probabilities.get)

        return {
            "probabilities": probabilities,
            "dominant":      dominant,
            "confidence":    probabilities[dominant],
            "text":          text
        }
