"""
dashboard/pages/2_About.py
──────────────────────────
About page — Architecture overview, team info, and tech stack.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st

st.markdown("## 📖 About TriFusion")
st.markdown("*The system that listens to what you don't say.*")

st.markdown("---")

# ── What is TriFusion? ────────────────────────────────────────────────────────
col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    st.markdown("### Core Concept")
    st.markdown("""
    TriFusion is a **real-time trimodal emotional intelligence system** that
    simultaneously analyses three channels of human communication:

    | Channel | Technology          | Dataset     | Output |
    |---------|---------------------|-------------|--------|
    | 👁 Face  | EfficientNet-B0     | FER2013     | 7 emotion probabilities |
    | 🎤 Voice | Wav2Vec2            | RAVDESS     | 8 emotion probabilities |
    | 💬 Words | RoBERTa + Whisper   | GoEmotions  | 8 emotion probabilities |

    These three distributions are fused by a custom **FusionMLP** (PyTorch).
    A **KL-divergence incongruence score** then measures how much the signals
    disagree — identifying emotional masking, suppression, or ambiguity.

    A **LangGraph WellnessAgent** (LLaMA-3.3-70B via Groq) responds with
    targeted micro-interventions chosen from a library of 6 evidence-based tools.
    """)

with col_r:
    st.markdown("### Core Innovation")
    st.markdown("""
    **What is incongruence?**

    When your face shows fear but your words say "I'm fine" —
    that gap is called **emotional incongruence**.

    Traditional emotion AI reads one channel and trusts it.
    TriFusion reads three and *specifically looks for disagreement*.

    The **symmetric KL-divergence** between the three probability
    distributions is normalised to [0, 1]:

    - **0.0 – 0.3** → Aligned (signals agree)
    - **0.3 – 0.7** → Moderate (normal variation)
    - **0.7 – 1.0** → High — masking likely detected
    """)

st.markdown("---")

# ── System Architecture ───────────────────────────────────────────────────────
st.markdown("### System Architecture")
st.markdown("""
```
Webcam ─── MediaPipe Face Detect ─── EfficientNet-B0 ──► Vision Probs (7)
                                                                         │
Microphone ─── AudioRecorder ─────── Wav2Vec2 ───────────► Audio Probs (8) ─► FusionMLP
                                                                         │         │
                  └──────────────── Whisper STT ─── RoBERTa ──► Text Probs (8) ─►┘
                                                                         │
                                                          KL Incongruence Scorer
                                                                         │
                                                         LangGraph WellnessAgent
                                                         (5 tool nodes, escalation)
                                                                         │
                                                     Streamlit Real-time Dashboard
```
""")

st.markdown("---")

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("### Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">ML / Deep Learning</div>
        <ul style="color:#94a3b8;font-size:0.88rem;margin-top:8px;padding-left:18px;">
            <li>PyTorch 2.2</li>
            <li>HuggingFace Transformers 4.40</li>
            <li>EfficientNet-B0 (torchvision)</li>
            <li>Wav2Vec2 (facebook/wav2vec2-base)</li>
            <li>RoBERTa-base</li>
            <li>OpenAI Whisper (tiny)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">Computer Vision / Audio</div>
        <ul style="color:#94a3b8;font-size:0.88rem;margin-top:8px;padding-left:18px;">
            <li>MediaPipe Face Detection</li>
            <li>OpenCV 4.9</li>
            <li>sounddevice 0.4.6</li>
            <li>librosa 0.10</li>
        </ul>
        <div class="modality-label" style="margin-top:12px;">Agent Framework</div>
        <ul style="color:#94a3b8;font-size:0.88rem;margin-top:8px;padding-left:18px;">
            <li>LangGraph 0.1</li>
            <li>LangChain 0.2</li>
            <li>Groq API (LLaMA-3.3-70B)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">Infrastructure</div>
        <ul style="color:#94a3b8;font-size:0.88rem;margin-top:8px;padding-left:18px;">
            <li>FastAPI 0.111 (REST API)</li>
            <li>Streamlit 1.34 (Dashboard)</li>
            <li>Redis 7 (Session history)</li>
            <li>Docker + docker-compose</li>
            <li>Pydantic v2</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
Built by **Lalithendra Kasula**
— [GitHub](https://github.com/Lalith0024)
""")
