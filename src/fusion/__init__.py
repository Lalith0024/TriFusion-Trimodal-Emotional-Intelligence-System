# src/fusion/__init__.py
# Exposes fusion model and incongruence utilities.
from src.fusion.fusion_model import FusionMLP
from src.fusion.inference import FusionInference
from src.fusion.incongruence import compute_incongruence, get_incongruence_label

__all__ = ["FusionMLP", "FusionInference", "compute_incongruence", "get_incongruence_label"]
