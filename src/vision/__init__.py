# src/vision/__init__.py
# Exposes the two main vision interfaces: detector and inference runner.
from src.vision.face_detector import FaceDetector
from src.vision.inference import VisionInference

__all__ = ["FaceDetector", "VisionInference"]
