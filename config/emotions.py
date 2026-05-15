# =============================================================================
# config/emotions.py
# Unified emotion label system across all three modalities.
# All modalities are ultimately mapped to this shared 8-class vocabulary
# so the fusion model always receives consistent, aligned inputs.
# =============================================================================

UNIFIED_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "calm"
]

# ---------------------------------------------------------------------------
# FER2013 Vision mapping
# FER2013 has 7 classes — "calm" is absent so its probability from vision
# will be 0 unless added via weighted rounding.
# ---------------------------------------------------------------------------
FER2013_TO_UNIFIED = {
    0: "angry",      # Angry
    1: "disgusted",  # Disgust
    2: "fearful",    # Fear
    3: "happy",      # Happy
    4: "sad",        # Sad
    5: "surprised",  # Surprise
    6: "neutral",    # Neutral
    # NOTE: "calm" has near-zero prob from vision (not a FER2013 class)
}
FER2013_LABELS = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
FER2013_NUM_CLASSES = 7

# ---------------------------------------------------------------------------
# RAVDESS Audio mapping
# RAVDESS 8 classes map cleanly onto the unified schema.
# ---------------------------------------------------------------------------
RAVDESS_TO_UNIFIED = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgusted",
    7: "surprised"
}
RAVDESS_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
RAVDESS_NUM_CLASSES = 8

# ---------------------------------------------------------------------------
# GoEmotions Text mapping
# GoEmotions has 27 fine-grained categories that collapse into the 8 unified
# emotions. Multiple GoEmotion labels can map to the same unified class.
# ---------------------------------------------------------------------------
GOEMOTIONS_TO_UNIFIED = {
    "admiration": "happy", "amusement": "happy", "approval": "happy",
    "caring": "calm",      "curiosity": "neutral", "desire": "happy",
    "excitement": "happy", "gratitude": "happy",   "joy": "happy",
    "love": "happy",       "optimism": "happy",    "pride": "happy",
    "relief": "calm",      "anger": "angry",        "annoyance": "angry",
    "disapproval": "angry","disgust": "disgusted",  "embarrassment": "sad",
    "fear": "fearful",     "grief": "sad",          "nervousness": "fearful",
    "remorse": "sad",      "sadness": "sad",        "confusion": "neutral",
    "realization": "surprised", "surprise": "surprised", "neutral": "neutral",
    "disappointment": "sad",
}
TEXT_NUM_CLASSES = 8

# ---------------------------------------------------------------------------
# Wellness routing — maps dominant emotion → intervention tool name
# ---------------------------------------------------------------------------
EMOTION_TO_INTERVENTION = {
    "angry":     "grounding_technique",
    "fearful":   "breathing_exercise",
    "sad":       "affirmation_generator",
    "disgusted": "cognitive_reframe",
    "surprised": "grounding_technique",
    "neutral":   "affirmation_generator",
    "calm":      "positive_reinforcement",
    "happy":     "positive_reinforcement"
}

# Severity score used for escalation logic:
# 3 = highest urgency, 0 = no urgency
EMOTION_SEVERITY = {
    "fearful":   3,
    "sad":       2,
    "angry":     2,
    "disgusted": 1,
    "surprised": 1,
    "neutral":   0,
    "calm":      0,
    "happy":     0
}
