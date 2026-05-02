"""
dashboard/components/sidebar.py
────────────────────────────────
Consistent sidebar rendered on every dashboard page.
Shows real-time model file presence so the user knows which weights are ready.
"""

import os
import streamlit as st

# Canonical paths for each trained model artifact
_MODEL_PATHS = {
    "Vision (EfficientNet)":  "models/vision/efficientnet_fer2013.pth",
    "Audio (Wav2Vec2)":       "models/audio/wav2vec2_ravdess",
    "Text (RoBERTa)":         "models/text/roberta_goemotions",
    "Fusion MLP":             "models/fusion/fusion_mlp.pth",
}


def _model_status(path: str) -> tuple[str, str]:
    """Return (icon, label) for the model at `path`."""
    exists = os.path.exists(path)
    return ("✓", "Ready") if exists else ("○", "Not trained")


def render_sidebar():
    """Render the sidebar navigation and live model status panel."""
    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style='padding:1rem 0 0.5rem'>
            <div style='font-family:Space Mono,monospace; font-size:1.2rem; font-weight:700;
                        background:linear-gradient(135deg,#6366f1,#22d3ee);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                🧠 TriFusion
            </div>
            <div style='color:#64748b; font-size:0.72rem; margin-top:2px; letter-spacing:0.04em;'>
                Trimodal Emotional Intelligence
            </div>
        </div>
        <hr style='border:0; border-top:1px solid #2a2a3a; margin:0.6rem 0;'>
        """, unsafe_allow_html=True)

        # ── Navigation ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='color:#64748b; font-size:0.7rem; letter-spacing:0.1em; "
            "text-transform:uppercase; margin-bottom:6px;'>Navigation</div>",
            unsafe_allow_html=True
        )
        st.page_link("app.py",                         label="🏠  Home")
        st.page_link("pages/1_Live_Dashboard.py",       label="🎥  Live Dashboard")
        st.page_link("pages/2_About.py",                label="📖  About")
        st.page_link("pages/3_Demo_Scenarios.py",       label="🎭  Demo Scenarios")
        st.page_link("pages/4_Session_History.py",      label="📊  Session History")
        st.page_link("pages/5_Model_Cards.py",          label="🤖  Model Cards")

        st.markdown(
            "<hr style='border:0; border-top:1px solid #2a2a3a; margin:0.8rem 0;'>",
            unsafe_allow_html=True
        )

        # ── Model status ───────────────────────────────────────────────────
        st.markdown(
            "<div style='color:#64748b; font-size:0.7rem; letter-spacing:0.1em; "
            "text-transform:uppercase; margin-bottom:8px;'>Model Status</div>",
            unsafe_allow_html=True
        )

        all_ready = True
        for name, path in _MODEL_PATHS.items():
            icon, label = _model_status(path)
            color        = "#22c55e" if icon == "✓" else "#64748b"
            if icon != "✓":
                all_ready = False
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"align-items:center; margin-bottom:5px; font-size:0.78rem;'>"
                f"<span style='color:#94a3b8;'>{name}</span>"
                f"<span style='color:{color}; font-family:Space Mono,monospace; "
                f"font-size:0.7rem;'>{icon} {label}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            "<hr style='border:0; border-top:1px solid #2a2a3a; margin:0.8rem 0;'>",
            unsafe_allow_html=True
        )

        # ── Mode badge ─────────────────────────────────────────────────────
        from src.pipeline.manager import SIMULATION_MODE
        if SIMULATION_MODE:
            st.markdown("""
            <div style='background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
                        border-radius:8px; padding:8px 10px; font-size:0.75rem;
                        color:#f59e0b; text-align:center;'>
                ⚡ Simulation Mode<br>
                <span style='color:#78716c; font-size:0.68rem;'>
                    Train models → set SIMULATION_MODE=False
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.3);
                        border-radius:8px; padding:8px 10px; font-size:0.75rem;
                        color:#22c55e; text-align:center;'>
                🟢 Live Inference Mode
            </div>
            """, unsafe_allow_html=True)
