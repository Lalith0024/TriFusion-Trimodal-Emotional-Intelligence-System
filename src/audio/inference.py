"""
src/audio/inference.py
──────────────────────
Audio emotion inference from raw 16 kHz waveform.

Flow:
  1. Wav2Vec2Processor normalises the waveform and creates attention masks.
  2. Model returns logits → softmax → RAVDESS-ordered probability dict.
  3. Probabilities are returned using RAVDESS label names, which already
     align with the unified 8-class schema (no remapping needed here).
  4. FusionInference.fuse() handles final alignment.

Graceful degradation:
  If no checkpoint is found, loads the pretrained base model
  (random classification head) so the system still starts and
  the dashboard can display placeholder values.
"""

import os
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from config.emotions import RAVDESS_LABELS, UNIFIED_EMOTIONS
import logging

logger = logging.getLogger(__name__)


class AudioInference:
    """
    Stateful audio inference engine.
    Instantiate once; call .predict(waveform) per audio chunk.
    """

    def __init__(self, model_path: str = None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        model_src      = model_path if (model_path and os.path.exists(model_path)) else "facebook/wav2vec2-base"

        logger.info(f"Loading audio model from: {model_src}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_src)
        self.model     = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_src,
            num_labels=len(RAVDESS_LABELS),
            ignore_mismatched_sizes=True   # safe even if loading fine-tuned ckpt
        ).to(self.device)
        self.model.eval()

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Run emotion inference on a single waveform chunk.

        Args:
            waveform:    1-D float32 numpy array at 16 kHz.
            sample_rate: Sampling rate of the waveform (should always be 16000).

        Returns:
            {
              "probabilities": dict[emotion → float],  # RAVDESS 8-class labels
              "dominant":      str,
              "confidence":    float
            }
        """
        # Processor normalises the raw waveform (zero-mean, unit-var)
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Build probability dict in RAVDESS label order
        probabilities = {label: float(probs[i]) for i, label in enumerate(RAVDESS_LABELS)}
        dominant      = max(probabilities, key=probabilities.get)

        return {
            "probabilities": probabilities,
            "dominant":      dominant,
            "confidence":    probabilities[dominant]
        }
