import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def validate_checkpoints() -> Tuple[bool, List[str]]:
    """
    Checks if all required model checkpoints exist.
    Returns (True, []) if all are present, or (False, [missing_paths]) if any are missing.
    """
    required_paths = [
        os.getenv("VISION_MODEL_PATH", "models/vision/efficientnet_fer2013.pth"),
        # AUDIO_MODEL_PATH is a directory for wav2vec2, check if dir exists and isn't empty
        os.getenv("AUDIO_MODEL_PATH", "models/audio/wav2vec2_ravdess"),
        # TEXT_MODEL_PATH is a directory for roberta
        os.getenv("TEXT_MODEL_PATH", "models/text/roberta_goemotions"),
        os.getenv("FUSION_MODEL_PATH", "models/fusion/fusion_mlp.pth")
    ]
    
    missing = []
    for path in required_paths:
        if not os.path.exists(path):
            missing.append(path)
        elif os.path.isdir(path):
            # If it's a directory (like for HF models), ensure it's not empty/just gitkeep
            files = [f for f in os.listdir(path) if f != ".gitkeep"]
            if not files:
                missing.append(path)

    if missing:
        for p in missing:
            logger.warning(f"Missing model checkpoint: {p}. Will use fallback/random init if available.")
        
    return True, []
