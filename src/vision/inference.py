"""
src/vision/inference.py
───────────────────────
Real-time vision inference pipeline.

Flow per frame:
  1. FaceDetector detects + crops face region from BGR frame.
  2. Crop is preprocessed (BGR→RGB PIL → ImageNet normalise → tensor).
  3. FacialEmotionNet produces 7-class FER2013 probabilities.
  4. Probabilities are remapped to the unified 8-class schema
     ("calm" will always be 0 from vision since it's not a FER2013 class).
  5. Returns a dict compatible with FusionInference.fuse().

Graceful degradation:
  If no face is detected, returns uniform 1/8 distribution so downstream
  fusion still receives a valid tensor and the system keeps running.
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from src.vision.emotion_model import FacialEmotionNet
from src.vision.face_detector import FaceDetector
from config.emotions import FER2013_LABELS, UNIFIED_EMOTIONS
import logging

logger = logging.getLogger(__name__)

# Pre-bake the val-time transform for speed (no re-instantiation per call)
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class VisionInference:
    """
    Thread-safe vision emotion inference.
    Instantiate once and call .predict(frame) repeatedly.
    """

    def __init__(self, model_path: str = None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = FacialEmotionNet(pretrained=True).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Vision model loaded from {model_path}")
        else:
            logger.warning("No vision checkpoint found — using random weights (UI demo mode).")

        self.model.eval()
        self.face_detector = FaceDetector()

    def predict(self, frame: np.ndarray) -> dict:
        """
        Run full vision pipeline on a single BGR frame.

        Args:
            frame: (H, W, 3) uint8 BGR array from cv2.VideoCapture.

        Returns:
            {
              "probabilities":  dict[emotion → float],  # unified 8-class
              "dominant":       str,
              "confidence":     float,
              "face_detected":  bool,
              "bbox":           dict | None
            }
        """
        face_crop, bbox_info = self.face_detector.detect_and_crop(frame)

        # No face → return uniform distribution so fusion can still run
        if face_crop is None:
            return {
                "probabilities": {e: 1.0 / 8 for e in UNIFIED_EMOTIONS},
                "dominant": "neutral",
                "confidence": 0.0,
                "face_detected": False,
                "bbox": None
            }

        # BGR (cv2) → RGB PIL → ImageNet-normalised tensor
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        tensor   = _TRANSFORM(face_pil).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probs = self.model.get_probabilities(tensor).squeeze().cpu().numpy()

        # Map FER2013 7-class → unified 8-class
        # FER2013 has no "calm" class so it remains 0.0
        unified_probs = {e: 0.0 for e in UNIFIED_EMOTIONS}
        for i, label in enumerate(FER2013_LABELS):
            unified_probs[label] += float(probs[i])

        # Re-normalise in case of floating point drift
        total = sum(unified_probs.values())
        if total > 0:
            unified_probs = {k: v / total for k, v in unified_probs.items()}

        dominant   = max(unified_probs, key=unified_probs.get)
        confidence = unified_probs[dominant]

        return {
            "probabilities": unified_probs,
            "dominant":      dominant,
            "confidence":    confidence,
            "face_detected": True,
            "bbox":          bbox_info
        }

    def close(self):
        """Release MediaPipe detector resources."""
        self.face_detector.close()
