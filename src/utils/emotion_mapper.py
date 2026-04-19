"""
src/utils/emotion_mapper.py
───────────────────────────
Utility functions for converting between emotion label spaces.

All three modalities output probabilities using their own label schema.
This module provides a single map_to_unified() function that converts
any modality-specific dict into the canonical UNIFIED_EMOTIONS schema.

Example:
    # FER2013 output: {"angry": 0.4, "happy": 0.6, ...}  (7 keys)
    unified = map_to_unified(fer_dict, mapping=FER2013_TO_UNIFIED_STR)
    # Returns: {"angry": 0.4, "happy": 0.6, ..., "calm": 0.0}  (8 keys)
"""

from typing import Dict
from config.emotions import UNIFIED_EMOTIONS, FER2013_LABELS, RAVDESS_LABELS


def map_to_unified(
    probs: Dict[str, float],
    source_schema: str = "unified"
) -> Dict[str, float]:
    """
    Map a modality-specific probability dict to the unified 8-class schema.

    Args:
        probs:         Input probability dict.
        source_schema: One of "fer2013" | "ravdess" | "unified".
                       "fer2013" adds "calm"=0.0 and re-normalises.
                       "ravdess" labels are already in unified space.
                       "unified" passes through with validation.

    Returns:
        Dict with exactly the 8 UNIFIED_EMOTIONS as keys, summing to ~1.0.
    """
    result = {e: 0.0 for e in UNIFIED_EMOTIONS}

    if source_schema == "fer2013":
        # FER2013 has 7 classes; "calm" is absent → stays 0.0
        for label, prob in probs.items():
            if label in result:
                result[label] += prob
        # calm defaults to 0.0 (already set above)

    elif source_schema in ("ravdess", "unified"):
        # Labels are already in unified space — direct copy
        for label, prob in probs.items():
            if label in result:
                result[label] = prob

    else:
        raise ValueError(f"Unknown source_schema: '{source_schema}'. Expected fer2013|ravdess|unified.")

    # Re-normalise to sum to 1.0 (guards against floating-point drift)
    total = sum(result.values())
    if total > 1e-8:
        result = {k: v / total for k, v in result.items()}

    return result


def dominant_emotion(probs: Dict[str, float]) -> tuple:
    """
    Return the (emotion, probability) pair with the highest probability.

    Args:
        probs: Probability dict (any schema).

    Returns:
        (emotion_str, probability_float)
    """
    emotion = max(probs, key=probs.get)
    return emotion, probs[emotion]
