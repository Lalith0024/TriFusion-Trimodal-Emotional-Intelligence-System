import streamlit as st

st.set_page_config(page_title="Model Cards - TriFusion", page_icon="📊", layout="wide")

st.title("📊 Model Cards & Capabilities")
st.markdown("""
This page provides an honest, transparent breakdown of the deep learning models powering TriFusion.
We believe in setting realistic expectations for real-world emotional intelligence AI.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👁️ Vision (Facial Expression)")
    st.markdown("""
    **Model:** EfficientNet-B0  
    **Dataset:** FER2013  
    **True Validation Accuracy:** ~65-75%  

    **Why not 95%+?**  
    FER2013 has an inherent label noise ceiling. Human annotators only agree on emotion labels roughly 65-70% of the time on this dataset. Achieving higher accuracy often indicates overfitting to dataset noise rather than genuine generalization.
    
    **Latency:** ~10-20ms per frame.
    """)

with col2:
    st.subheader("🎤 Audio (Vocal Prosody)")
    st.markdown("""
    **Model:** Wav2Vec2-Base  
    **Dataset:** RAVDESS  
    **True Validation Accuracy:** ~70-80%  

    **Limitations:**  
    RAVDESS is an acted dataset (actors reading scripts with intentional emotion). Generalization to spontaneous, real-world conversational audio will naturally see a drop in confidence compared to validation metrics.
    
    **Latency:** ~50-100ms per 3-second chunk.
    """)

with col3:
    st.subheader("💬 Text (Linguistic Sentiment)")
    st.markdown("""
    **Model:** RoBERTa-Base + Whisper-Tiny (STT)  
    **Dataset:** GoEmotions (simplified)  
    **True Validation F1:** ~60-70% (Macro)  

    **Pipeline Overhead:**  
    Speech-to-Text via Whisper-Tiny is the most computationally expensive step in the pipeline (~150ms on CPU). Text analysis is updated less frequently to preserve real-time system performance.
    """)

st.divider()

st.subheader("🧠 TriFusion MLP (Late Fusion)")
st.markdown("""
**Architecture:** 3-layer Multilayer Perceptron (MLP)  
**Input:** 23-dimensional concatenated probability vector (Vision: 7, Audio: 8, Text: 8)  
**Training Data:** Synthetically generated Dirichlet distributions  

**Why Synthetic Data?**  
There is currently no large-scale, high-quality trimodal dataset containing synchronous face, voice, and transcribed text with joint emotion labels. The FusionMLP is trained on synthetic statistical boundaries:
- **Congruence:** When modalities agree, confidence is boosted.
- **Incongruence:** When modalities disagree, confidence is lowered and incongruence scores are scaled.

**True System Goal:**  
The goal is not flawless single-modality accuracy, but **robustness through redundancy**. When the face is obscured, voice carries the signal. When voice is monotone, text context provides the cue.
""")
