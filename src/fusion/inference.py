"""
src/fusion/inference.py
───────────────────────
Full trimodal fusion inference pipeline.

This is the central coordinator — it:
  1. Extracts probability vectors from each modality result dict.
  2. Computes incongruence score BEFORE fusion (uses raw modality signals).
  3. Runs FusionMLP to produce the final weighted probability distribution.
  4. Returns a comprehensive result dict that feeds into AgentState.

Graceful degradation:
  If no fusion checkpoint exists, FusionMLP uses its random-init weights.
  The output will still be a valid probability distribution — just not
  particularly meaningful until the model is trained.
"""

import os
import torch
import numpy as np
from typing import Optional

from src.fusion.fusion_model import FusionMLP
from src.fusion.incongruence import compute_incongruence, get_incongruence_label
from config.emotions import UNIFIED_EMOTIONS
import logging

logger = logging.getLogger(__name__)

# Canonical 7-dim ordering for vision (FER2013, no "calm")
_VISION_EMOTIONS_7 = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]


class FusionInference:
    """
    Trimodal fusion inference engine.
    Instantiate once; call .fuse(vision_result, audio_result, text_result) per frame.
    """

    def __init__(self, fusion_model_path: str = None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model  = FusionMLP().to(self.device)

        if fusion_model_path and os.path.exists(fusion_model_path):
            self.model.load_state_dict(torch.load(fusion_model_path, map_location=self.device))
            logger.info(f"FusionMLP loaded from {fusion_model_path}")
        else:
            logger.warning("No fusion checkpoint found — using random weights (demo mode).")

        self.model.eval()

    def _probs_to_tensor(self, probs_dict: dict, keys: list) -> torch.Tensor:
        """Convert a probability dict → ordered float32 tensor."""
        return torch.tensor(
            [probs_dict.get(e, 0.0) for e in keys], dtype=torch.float32
        ).unsqueeze(0).to(self.device)

    def fuse(
        self,
        vision_result: dict,
        audio_result:  dict,
        text_result:   dict
    ) -> dict:
        """
        Combine three modality outputs into the final fused emotion state.

        Args:
            vision_result: Output dict from VisionInference.predict().
            audio_result:  Output dict from AudioInference.predict().
            text_result:   Output dict from TextInference.predict().

        Returns:
            Comprehensive fusion result dict including incongruence score,
            fused probabilities, dominant emotion, and per-modality results.
        """
        # Extract probability dicts (fall back to uniform if missing)
        uniform8 = {e: 1.0 / 8 for e in UNIFIED_EMOTIONS}
        v_probs  = vision_result.get("probabilities", uniform8)
        a_probs  = audio_result.get("probabilities",  uniform8)
        t_probs  = text_result.get("probabilities",   uniform8)

        # ---- Incongruence computed on raw modality signals (pre-fusion) ----
        incongruence_score  = compute_incongruence(v_probs, a_probs, t_probs)
        incongruence_label, incongruence_color = get_incongruence_label(incongruence_score)

        # ---- FusionMLP forward pass ----------------------------------------
        # Vision tensor uses 7-dim FER2013 ordering (no "calm")
        v_tensor = self._probs_to_tensor(v_probs, _VISION_EMOTIONS_7)
        a_tensor = self._probs_to_tensor(a_probs, UNIFIED_EMOTIONS)
        t_tensor = self._probs_to_tensor(t_probs, UNIFIED_EMOTIONS)

        with torch.no_grad():
            fused_probs = self.model(v_tensor, a_tensor, t_tensor)  # (1, 8)
        fused_np = fused_probs.squeeze().cpu().numpy()

        fused_dict = {e: float(fused_np[i]) for i, e in enumerate(UNIFIED_EMOTIONS)}
        dominant   = max(fused_dict, key=fused_dict.get)

        return {
            "fused_probabilities": fused_dict,
            "dominant_emotion":    dominant,
            "confidence":          fused_dict[dominant],
            "incongruence_score":  incongruence_score,
            "incongruence_label":  incongruence_label,
            "incongruence_color":  incongruence_color,
            "modality_results": {
                "vision": vision_result,
                "audio":  audio_result,
                "text":   text_result
            }
        }
