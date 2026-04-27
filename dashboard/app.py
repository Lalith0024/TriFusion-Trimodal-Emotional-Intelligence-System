"""
dashboard/app.py
────────────────
TriFusion Streamlit Dashboard — Main Entry Point (Home Page).

This file configures global page settings, injects the dark-theme CSS,
and renders the home/landing page. All other pages live in dashboard/pages/.

Dark theme design system:
  --primary   #6366f1  (Indigo)
  --accent    #22d3ee  (Cyan)
  --danger    #ef4444  (Red)
  --warning   #f59e0b  (Amber)
  --success   #22c55e  (Green)
  --bg        #0a0a0f  (Near-black)
  --surface   #13131a  (Dark surface)
  --surface2  #1a1a24  (Slightly lighter surface)
  --border    #2a2a3a  (Subtle border)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Must be the very first Streamlit call ────────────────────────────────────
st.set_page_config(
    page_title="TriFusion — Emotional Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "TriFusion: Trimodal Emotional Intelligence by Lalithendra Kasula"}
)

# ── Global CSS (injected on every page because app.py is always executed) ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap');

:root {
    --primary:      #6366f1;
    --primary-glow: rgba(99,102,241,0.25);
    --accent:       #22d3ee;
    --danger:       #ef4444;
    --warning:      #f59e0b;
    --success:      #22c55e;
    --bg:           #0a0a0f;
    --surface:      #13131a;
    --surface2:     #1a1a24;
    --border:       #2a2a3a;
    --text:         #e2e8f0;
    --text-muted:   #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

/* ── Sidebar ───────────────────────────────────────────────── */
div[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
div[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Typography helpers ────────────────────────────────────── */
.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.sub-header {
    color: var(--text-muted);
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.03em;
}

/* ── Cards ─────────────────────────────────────────────────── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.metric-card:hover { border-color: var(--primary); }

/* ── Badges ────────────────────────────────────────────────── */
.emotion-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-aligned  { background: rgba(34,197,94,0.15);  color:#22c55e; border:1px solid #22c55e; }
.badge-moderate { background: rgba(245,158,11,0.15); color:#f59e0b; border:1px solid #f59e0b; }
.badge-high     { background: rgba(239,68,68,0.15);  color:#ef4444; border:1px solid #ef4444; }

/* ── Agent response box ────────────────────────────────────── */
.agent-response {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(34,211,238,0.04));
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-size: 0.95rem;
    line-height: 1.75;
    margin-top: 1rem;
}

/* ── Modality label ────────────────────────────────────────── */
.modality-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 5px;
}

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.6rem !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Plotly chart background transparency ──────────────────── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Progress bars ─────────────────────────────────────────── */
.stProgress > div > div > div { background: var(--primary) !important; }

/* ── Info / success / warning boxes ───────────────────────── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-family:Space Mono,monospace;font-size:1.15rem;font-weight:700;
                    background:linear-gradient(135deg,#6366f1,#22d3ee);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            🧠 TriFusion
        </div>
        <div style='color:#64748b;font-size:0.75rem;margin-top:3px;'>
            Trimodal Emotional Intelligence
        </div>
    </div>
    <hr style='border:0;border-top:1px solid #2a2a3a;margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    st.page_link("app.py",                          label="🏠 Home")
    st.page_link("pages/1_Live_Dashboard.py",        label="🎥 Live Dashboard")
    st.page_link("pages/2_About.py",                 label="📖 About")
    st.page_link("pages/3_Demo_Scenarios.py",        label="🎭 Demo Scenarios")
    st.page_link("pages/4_Session_History.py",       label="📊 Session History")
    st.page_link("pages/5_Model_Cards.py",           label="🤖 Model Cards")

    st.markdown('<hr style="border:0;border-top:1px solid #2a2a3a;margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown("**System Status**")
    st.success("✓ Vision Module Ready")
    st.success("✓ Audio Module Ready")
    st.success("✓ Text Module Ready")
    st.success("✓ WellnessAgent Ready")

# ── Home Page Content ─────────────────────────────────────────────────────────
st.markdown('<div class="main-header">TriFusion</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Trimodal Emotional Intelligence — Face · Voice · Words</div>',
    unsafe_allow_html=True
)

# Model summary cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">👁 Vision Module</div>
        <div style="font-size:1.45rem;font-weight:700;color:#6366f1;">EfficientNet-B0</div>
        <div style="color:#64748b;font-size:0.83rem;margin-top:4px;">
            FER2013 dataset · 7 emotion classes<br>
            MediaPipe face detection · ~66% weighted F1
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">🎤 Audio Module</div>
        <div style="font-size:1.45rem;font-weight:700;color:#22d3ee;">Wav2Vec2</div>
        <div style="color:#64748b;font-size:0.83rem;margin-top:4px;">
            RAVDESS dataset · 8 emotion classes<br>
            Staged fine-tuning · ~78% weighted F1
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="modality-label">💬 Text Module</div>
        <div style="font-size:1.45rem;font-weight:700;color:#f59e0b;">RoBERTa</div>
        <div style="color:#64748b;font-size:0.83rem;margin-top:4px;">
            GoEmotions dataset · 8 emotion classes<br>
            Remapped to unified schema · ~70% weighted F1
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr style="border:0;border-top:1px solid #2a2a3a;margin:1.5rem 0;">', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### What makes TriFusion different?")
    st.markdown("""
    Most emotion AI reads **one signal** and trusts it blindly. TriFusion reads
    three simultaneously and — more importantly — detects when they **disagree**.

    That disagreement is precisely where masked stress, suppressed anxiety, and
    hidden frustration live. The **KL-divergence incongruence scorer** measures
    how statistically different your face, voice, and words are from each other.

    High incongruence triggers the WellnessAgent's masked-distress protocol —
    catching what you don't say out loud.
    """)

with col_r:
    st.markdown("### WellnessAgent Interventions")
    interventions = {
        "😮‍💨 Breathing Exercise":     "For fearful or panicked states",
        "🌍 Grounding Technique":       "For angry or dissociated states",
        "💬 Affirmation Generator":     "For sadness or low confidence",
        "🎵 Music Recommendation":      "For any dysregulated state",
        "🔄 Cognitive Reframe":         "When spoken words show distorted thinking",
        "🚨 Crisis Escalation":         "High incongruence + severe emotion × 3 frames",
    }
    for k, v in interventions.items():
        st.markdown(f"**{k}** — {v}")

st.markdown('<hr style="border:0;border-top:1px solid #2a2a3a;margin:1.5rem 0;">', unsafe_allow_html=True)

if st.button("→ Launch Live Dashboard", type="primary"):
    st.switch_page("pages/1_Live_Dashboard.py")
