# src/text/__init__.py
# Exposes transcriber and text emotion inference pipeline.
from src.text.transcriber import Transcriber
from src.text.inference import TextInference

__all__ = ["Transcriber", "TextInference"]
