"""
dashboard/pages/2_About.py
──────────────────────────
About page — explains the project, architecture, and the science behind it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import streamlit as st
from dashboard.components.sidebar import render_sidebar

render_sidebar()

st.markdown("## 📖 About TriFusion")
st.markdown("*The system that understands what you feel, not just what you say.*")

st.markdown("---")
st.markdown("### The Problem")
st.markdown("""
Today's emotion AI is **unimodal and naive**. A chatbot reads your text. A sentiment analyzer
reads your words. Nobody builds systems that understand what humans have always known intuitively:

> **People lie with their words, but rarely with their face and voice simultaneously.**

75% of people experiencing acute stress report feeling "fine" when asked directly (APA, 2023).
Crisis intervention systems miss early distress signals because they rely on self-reporting.
Customer service AI cannot detect frustrated users until they escalate verbally — losing trust
and retention.

TriFusion solves this.
""")

st.markdown("---")
st.markdown("### The Three Modalities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">👁 VISION</div>
        <div style="font-size:1.3rem;font-weight:700;color:#6366f1;">EfficientNet-B0</div>
        <div style="color:#64748b;font-size:0.83rem;margin-top:8px;line-height:1.7;">
            <b>Dataset:</b> FER2013 — 35,887 facial images<br>
            <b>Classes:</b> 7 (angry, disgust, fear, happy, sad, surprise, neutral)<br>
            <b>Architecture:</b> CNN with custom 2-layer MLP head<br>
            <b>Detection:</b> MediaPipe face detection + crop<br>
            <b>Expected F1:</b> ~66% weighted
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">🎤 AUDIO</div>
        <div style="font-size:1.3rem;font-weight:700;color:#22d3ee;">Wav2Vec2</div>
        <div style="color:#64748b;font-size:0.83rem;margin-top:8px;line-height:1.7;">
            <b>Dataset:</b> RAVDESS — 7,356 recordings<br>
            <b>Classes:</b> 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised)<br>
            <b>Architecture:</b> Self-supervised audio transformer<br>
            <b>Strategy:</b> Staged fine-tuning (freeze → unfreeze encoder)<br>
            <b>Expected F1:</b> ~78% weighted
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">💬 TEXT</div>
        <div style="font-size:1.3rem;font-weight:700;color:#f59e0b;">RoBERTa</div>
        <div style="color:#64748b;font-size:0.83rem;margin-top:8px;line-height:1.7;">
            <b>Dataset:</b> GoEmotions — 58,009 comments<br>
            <b>Classes:</b> 27 → remapped to unified 8<br>
            <b>Architecture:</b> Robustly trained BERT variant<br>
            <b>STT:</b> Whisper (tiny) for real-time transcription<br>
            <b>Expected F1:</b> ~70% weighted
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### The Incongruence Science")
st.markdown("""
The core innovation is the **KL-divergence incongruence scorer**.

**KL divergence** (Kullback-Leibler divergence) is a measure from information theory that
quantifies how different two probability distributions are from each other. In TriFusion,
we compute the average **symmetric KL divergence** across all three pairs of modality distributions:

```
incongruence = average(KL(vision||audio), KL(vision||text), KL(audio||text))
```

When this score is **high** (>0.7), the three signals are statistically incompatible — one
modality is saying something very different from the others. This is the mathematical
signature of emotional masking.

**Example — masked fear:**
- Face: fear=0.68, neutral=0.22 (face can't lie)
- Voice: stress=0.71, angry=0.18 (voice breaks under pressure)
- Words: "I'm totally fine, no worries" → positive=0.89

Incongruence score: **0.87** → WellnessAgent activates masked distress protocol.
""")

# Interactive incongruence demo
st.markdown("#### Try it — move the sliders to see how incongruence changes")
col_a, col_b = st.columns(2)
with col_a:
    face_fear  = st.slider("Face: Fear signal",    0.0, 1.0, 0.7, 0.01)
    voice_stress = st.slider("Voice: Stress signal", 0.0, 1.0, 0.6, 0.01)
with col_b:
    text_pos   = st.slider("Text: Positive signal", 0.0, 1.0, 0.8, 0.01)

# Compute approximate incongruence for demo
import numpy as np
p1 = np.array([face_fear, 1-face_fear]) + 1e-8
p2 = np.array([voice_stress, 1-voice_stress]) + 1e-8
p3 = np.array([text_pos, 1-text_pos]) + 1e-8
for p in [p1, p2, p3]:
    p /= p.sum()

def kl(a, b): return float(np.sum(a * np.log(a / b)))
inc_demo = ((kl(p1,p2)+kl(p2,p1))/2 + (kl(p1,p3)+kl(p3,p1))/2 + (kl(p2,p3)+kl(p3,p2))/2) / 3
inc_norm = min(inc_demo / 0.7, 1.0)

color = "#22c55e" if inc_norm < 0.3 else ("#f59e0b" if inc_norm < 0.7 else "#ef4444")
label = "ALIGNED" if inc_norm < 0.3 else ("MODERATE" if inc_norm < 0.7 else "HIGH — MASKING DETECTED")
st.markdown(f"""
<div style="margin-top:1rem; padding:1rem 1.5rem; background:var(--surface);
            border-radius:12px; border:1px solid {color}44;">
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#64748b;">
        INCONGRUENCE SCORE
    </div>
    <div style="font-size:2rem; font-weight:700; color:{color}; margin:0.2rem 0;">
        {inc_norm:.2f}
    </div>
    <div style="font-size:0.85rem; color:{color}; font-weight:600;">{label}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### System Architecture")
st.code("""
Webcam (30fps) ──► CaptureThread ──► frame_queue ──► InferenceThread
                                          │                  │
                                    display_buffer     EfficientNet-B0
                                          │            (vision, 7-class)
                                     Streamlit UI           │
                                    (always fresh)    Wav2Vec2 fine-tuned
                                                      (audio, 8-class)
                                                            │
                                                     RoBERTa fine-tuned
                                                      (text, 8-class)
                                                            │
                                                      FusionMLP (23→8)
                                                  + KL Incongruence Score
                                                            │
                                                    WellnessAgent
                                                  (LangGraph, 5 tools)
                                                            │
                                                    Result buffer
                                                  (Streamlit reads here)
""", language="text")

st.markdown("### Why the Camera Runs at 30+ FPS")
st.markdown("""
The naive approach runs capture and inference in the same thread. At 50ms per inference
cycle, that caps out at 20 FPS — and inference gets slower as models load.

TriFusion uses **decoupled threading**:
- **CaptureThread**: does nothing but read frames. Zero ML. Runs at 30+ FPS natively.
- **InferenceThread**: consumes frames at whatever speed the models allow (~12–15 FPS on CPU).
- **UI**: always reads from `display_buffer` which holds the latest raw frame from CaptureThread.

Result: the camera feels smooth at 30 FPS even though ML inference runs at 12 FPS.
The face bounding box from the last inference result is overlaid on every display frame.
""")

st.markdown("---")
st.markdown("### Tech Stack")
import pandas as pd
tech_data = {
    "Layer": ["Vision DL", "Audio DL", "Text DL", "Fusion", "Agent", "Serving", "Dashboard"],
    "Technology": ["EfficientNet-B0 + MediaPipe + OpenCV", "Wav2Vec2 + librosa + sounddevice",
                   "RoBERTa + Whisper STT", "Custom PyTorch MLP + KL divergence",
                   "LangGraph + Groq LLaMA-3.3-70B + Pydantic V2",
                   "FastAPI + Docker + Redis", "Streamlit + Plotly"],
    "Dataset / Source": ["FER2013 (35K images)", "RAVDESS (7,356 recordings)",
                         "GoEmotions (58K comments)", "Synthetic (8K samples, 30% incongruent)",
                         "LLM inference via Groq API", "—", "—"]
}
st.dataframe(pd.DataFrame(tech_data), hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown("""
Built by **Lalithendra Kasula** — [GitHub](https://github.com/Lalith0024)
""")
