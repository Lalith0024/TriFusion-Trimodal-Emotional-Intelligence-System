"""
dashboard/pages/5_Model_Cards.py
──────────────────────────────────
Model Cards — Detailed documentation for each of the three models
and the fusion layer. Modelled after HuggingFace model card format.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import streamlit as st

st.markdown("## 🤖 Model Cards")
st.markdown("*Detailed technical documentation for each TriFusion component.*")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["👁 Vision", "🎤 Audio", "💬 Text", "🔀 Fusion"])

# ── Vision Model Card ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("### EfficientNet-B0 — Facial Emotion Recognition")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Base Architecture:** EfficientNet-B0 (ImageNet pretrained)
        **Training Dataset:** FER2013
        **Task:** 7-class facial emotion classification
        **Input:** 224×224 RGB image (ImageNet normalised)
        **Output:** 7 logits → softmax probabilities

        **Classes:**
        `angry` · `disgusted` · `fearful` · `happy` · `sad` · `surprised` · `neutral`

        **Training Details:**
        - 30 epochs, AdamW, lr=1e-4, weight decay=0.01
        - Weighted CrossEntropy + label smoothing (0.1)
        - CosineAnnealingLR scheduler
        - Augmentations: H-flip, ColorJitter, RandomRotation(±10°)
        """)
    with col2:
        st.markdown("""
        **Performance:**
        | Metric | Value |
        |--------|-------|
        | Weighted F1 | ~66% |
        | Angry | ~61% |
        | Happy | ~87% |
        | Neutral | ~68% |
        | Fearful | ~55% |

        **Limitations:**
        - FER2013 has ~20% label noise
        - No "calm" class — zero probability output for calm
        - Performance drops in low-light conditions
        - Single-face only (takes highest-confidence detection)

        **Face Detection:** MediaPipe FaceDetection (model_selection=0),
        min_confidence=0.7, 20% padding around bbox.
        """)

# ── Audio Model Card ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Wav2Vec2 — Speech Emotion Recognition")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Base Architecture:** facebook/wav2vec2-base (95M params)
        **Training Dataset:** RAVDESS (24 actors, 1440 audio files)
        **Task:** 8-class speech emotion classification
        **Input:** Float32 waveform at 16 kHz (~3 seconds)
        **Output:** 8 logits → softmax probabilities

        **Classes:**
        `neutral` · `calm` · `happy` · `sad` · `angry` · `fearful` · `disgusted` · `surprised`

        **Training Details:**
        - Phase 1: Freeze CNN encoder, train transformer + head (5 epochs)
        - Phase 2: Unfreeze encoder, full fine-tune (15 epochs)
        - lr=3e-5, warmup_ratio=0.1
        - EarlyStoppingCallback (patience=3)
        """)
    with col2:
        st.markdown("""
        **Performance:**
        | Metric | Value |
        |--------|-------|
        | Weighted F1 | ~78% |
        | Angry | ~84% |
        | Calm | ~72% |
        | Happy | ~80% |
        | Neutral | ~70% |

        **Limitations:**
        - RAVDESS is acted speech — may not generalise to spontaneous emotion
        - Performance degrades with background noise
        - 3-second window may miss transient emotion shifts

        **Audio Capture:** sounddevice.InputStream at 16 kHz, 3-second chunks,
        bounded queue (maxsize=5) with non-blocking put.
        """)

# ── Text Model Card ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### RoBERTa — Text Emotion Classification")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Base Architecture:** roberta-base (125M params)
        **Training Dataset:** GoEmotions (simplified, remapped to 8 classes)
        **Task:** 8-class text emotion classification
        **Input:** Tokenised text, max 128 tokens
        **Output:** 8 logits → softmax probabilities

        **GoEmotions → Unified mapping:**
        27 GoEmotions classes → 8 unified classes
        (admiration/joy/love/pride → happy, grief/sadness/remorse → sad, etc.)

        **Training Details:**
        - 4 epochs, lr=2e-5, weight_decay=0.01
        - Warmup ratio 0.1, linear decay
        - DataCollatorWithPadding for dynamic batch padding

        **Speech-to-Text:** OpenAI Whisper (tiny), language=en, fp16=False
        """)
    with col2:
        st.markdown("""
        **Performance:**
        | Metric | Value |
        |--------|-------|
        | Weighted F1 | ~70% |
        | Happy | ~78% |
        | Neutral | ~72% |
        | Sad | ~65% |
        | Angry | ~62% |

        **Limitations:**
        - Short transcriptions (< 2 chars) return uniform distribution
        - Whisper "tiny" has ~10% word error rate on informal speech
        - GoEmotions is Reddit text — may not generalise to spoken language
        - Calm/surprised classes have lower training data representation
        """)

# ── Fusion Model Card ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("### FusionMLP — Late-fusion Emotion Integrator")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Architecture:** Two-layer MLP with BatchNorm
        **Input:** Concatenated [vision(7) + audio(8) + text(8)] = 23 dims
        **Output:** Unified 8-class probability distribution

        **Layer structure:**
        ```
        Linear(23 → 64) → BatchNorm → ReLU → Dropout(0.3)
        Linear(64 → 32) → ReLU → Dropout(0.2)
        Linear(32 → 8)  → Softmax
        ```
        **Weights init:** Xavier uniform

        **Training:** Majority-vote synthetic labels from per-modality predictions.
        50 epochs, Adam, lr=1e-3, StepLR(step=10, gamma=0.5)

        **Incongruence Scorer:**
        Average symmetric KL-divergence across 3 modality pairs,
        normalised by log(8) = 2.08 nats.
        """)
    with col2:
        st.markdown("""
        **Performance:**
        | Metric | Value |
        |--------|-------|
        | Weighted F1 (fusion) | ~74% |
        | Incongruence Precision | ~91% |

        **Why late fusion?**
        - Modality-agnostic: only sees probability distributions
        - Handles missing modalities: uniform dist = "no info"
        - No joint backprop required through large backbone models
        - Easy to retrain when one modality's model is updated

        **Incongruence thresholds:**
        - 0.0–0.3 → Aligned (green)
        - 0.3–0.7 → Moderate (amber)
        - 0.7–1.0 → High / masking (red, triggers escalation)
        """)
