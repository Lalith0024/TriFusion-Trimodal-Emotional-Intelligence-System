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
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
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
        fallback_model = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

        # Resolve relative model_path to absolute path relative to project root
        target_path = model_path
        if model_path and not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            target_path = os.path.join(project_root, model_path)

        # Check if local model directory is complete (must have config + weights)
        is_valid_local = (
            target_path
            and os.path.exists(target_path)
            and os.path.isdir(target_path)
            and (
                os.path.exists(os.path.join(target_path, "preprocessor_config.json"))
                or os.path.exists(os.path.join(target_path, "processor_config.json"))
                or os.path.exists(os.path.join(target_path, "config.json"))
            )
            and (
                os.path.exists(os.path.join(target_path, "model.safetensors"))
                or os.path.exists(os.path.join(target_path, "pytorch_model.bin"))
            )
        )



        model_src = target_path if is_valid_local else fallback_model
        logger.info(f"Loading audio model from: {model_src} (absolute path: {os.path.abspath(model_src)})")

        # Load processor (auto-populating target_path if local directory was missing files)
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(model_src)
        except Exception as e:
            logger.warning(f"Failed to load processor from {model_src} ({e}). Falling back to {fallback_model}.")
            self.processor = AutoFeatureExtractor.from_pretrained(fallback_model)
            if target_path and os.path.isdir(target_path):
                try:
                    self.processor.save_pretrained(target_path)
                    logger.info(f"Auto-populated preprocessor_config.json in {target_path}")
                except Exception as save_err:
                    logger.debug(f"Could not auto-save processor: {save_err}")

        # Load model (auto-populating target_path if local directory was missing files)
        try:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_src).to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load model from {model_src} ({e}). Falling back to {fallback_model}.")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(fallback_model).to(self.device)

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

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Initialize base probabilities
        probabilities = {label: 0.0 for label in RAVDESS_LABELS}
        num_labels = len(self.model.config.id2label)

        # Check if it's the locally trained model by inspecting the label map order
        is_local_model = (
            num_labels == len(RAVDESS_LABELS) 
            and 0 in self.model.config.id2label 
            and str(self.model.config.id2label[0]).lower() == RAVDESS_LABELS[0].lower()
        )

        if is_local_model:
            # Custom 8-class RAVDESS model (trained locally)
            for i, label in enumerate(RAVDESS_LABELS):
                probabilities[label] = float(probs[i] if probs.ndim > 0 else probs)
        else:
            # Community model (e.g. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
            for i in range(num_labels):
                label_str = str(self.model.config.id2label.get(i, "")).lower()
                # Map to our standard RAVDESS/Unified string
                if label_str in RAVDESS_LABELS:
                    probabilities[label_str] += float(probs[i] if probs.ndim > 0 else probs)
                else:
                    probabilities["neutral"] += float(probs[i] if probs.ndim > 0 else probs)

        dominant = max(probabilities, key=probabilities.get)

        return {
            "probabilities": probabilities,
            "dominant":      dominant,
            "confidence":    probabilities[dominant]
        }
