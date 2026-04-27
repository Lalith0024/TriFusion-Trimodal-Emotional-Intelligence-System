import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import time
import os
from datetime import datetime
import plotly.graph_objects as go

# Component imports
from dashboard.components.radar_chart import render_radar_chart
from dashboard.components.incongruence_meter import render_incongruence_meter
from src.pipeline.manager import PipelineManager
from config.emotions import UNIFIED_EMOTIONS
from dashboard.components.sidebar import render_sidebar

load_dotenv()

# ── Singleton Pipeline Manager ───────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = PipelineManager()
    st.session_state.session_running = False
    # Fix the missing attribute error by initializing session state properly
    st.session_state.last_agent_response = "Ready to analyze. Click Start Session."

st.markdown("## 🎥 Live Dashboard")
st.markdown("*Real-time trimodal emotional analysis — Face · Voice · Words*")

# ── Shared Sidebar ────────────────────────────────────────────────────────────
render_sidebar()

# ── Beta Warning ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#451a03; border-left:4px solid #fbbf24; padding:0.6rem 1rem; border-radius:4px; margin-bottom:1.5rem;">
    <div style="color:#fbbf24; font-size:0.85rem; font-weight:600;">⚠️ Beta Version Notice</div>
    <div style="color:#fcd34d; font-size:0.75rem;">
        Hardware capture is optimized for local environments. Please check 
        <a href="https://github.com/Lalith0024/TriFusion-Trimodal-Emotional-Intelligence-System" style="color:#fbbf24; text-decoration:underline;">README</a> 
        for full system requirements.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_main, col_right = st.columns([3, 2], gap="large")

with col_main:
    st.markdown("### 📷 Live Feed")
    webcam_placeholder = st.empty()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if not st.session_state.session_running:
            if st.button("▶ Start Session", type="primary", use_container_width=True):
                st.session_state.pipeline.start()
                st.session_state.session_running = True
                st.session_state.start_time = time.time()
                st.rerun()
        else:
            if st.button("⏹ Stop Session", use_container_width=True):
                st.session_state.pipeline.stop()
                st.session_state.session_running = False
                st.rerun()

    # ── Modality signal badges ────────────────────────────────────────────
    st.markdown("### 📡 Modality Signals")
    badge_col1, badge_col2, badge_col3 = st.columns(3)
    face_badge  = badge_col1.empty()
    voice_badge = badge_col2.empty()
    text_badge  = badge_col3.empty()

with col_right:
    st.markdown("### 🕸 Emotion Radar")
    radar_placeholder = st.empty()

    st.markdown("### 📊 Incongruence Meter")
    incongruence_placeholder = st.empty()

    st.markdown("### 💬 WellnessAgent")
    agent_placeholder = st.empty()

# ── Timeline ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Session Emotion Timeline")
timeline_placeholder = st.empty()

# ── FRAGMENT: Real-time update logic (Flicker-free) ──────────────────────────
@st.fragment(run_every=0.1)
def sync_dashboard():
    if not st.session_state.session_running:
        # Idle state
        webcam_placeholder.markdown("""
        <div style="background:#13131a; border:2px dashed #2a2a3a; border-radius:12px;
                    height:380px; display:flex; align-items:center; justify-content:center;
                    flex-direction:column; gap:12px;">
            <div style="font-size:3rem; opacity:0.3;">📷</div>
            <div style="color:#64748b; font-size:0.9rem;">
                Camera feed will appear here during an active session.
            </div>
            <div style="color:#6366f1; font-size:0.8rem; margin-top:10px;">
                Click <b>Start Session</b> to grant permissions.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        agent_placeholder.markdown(f"""
        <div class="agent-response">
            {st.session_state.last_agent_response}
        </div>
        """, unsafe_allow_html=True)
        return

    # Active state - read from background thread
    data = st.session_state.pipeline.get_latest()
    
    # 1. Update Camera
    if data["frame"] is not None:
        webcam_placeholder.image(data["frame"], channels="RGB")
    
    # 2. Update Charts
    v_res = data["vision"]
    a_res = data["audio"]
    t_res = data["text"]
    f_res = data["fusion"]

    if v_res:
        with radar_placeholder:
            # We update the chart every frame but use a stable key within the fragment
            render_radar_chart({
                "FACE":  v_res["probabilities"],
                "VOICE": a_res["probabilities"],
                "TEXT":  t_res["probabilities"]
            }, key="live_radar_main")
        
        with incongruence_placeholder:
            render_incongruence_meter(f_res.get("incongruence_score", 0.0))

        # 3. Update Badges
        face_badge.markdown(f"""
        <div class="metric-card">
            <div class="modality-label">👁 FACE</div>
            <div style="font-weight:700;color:#6366f1;font-size:1.05rem;">{v_res['dominant'].upper()}</div>
            <div style="color:#64748b;font-size:0.78rem;margin-top:3px;">{v_res['confidence']:.0%} confidence</div>
        </div>""", unsafe_allow_html=True)

        voice_badge.markdown(f"""
        <div class="metric-card">
            <div class="modality-label">🎤 VOICE</div>
            <div style="font-weight:700;color:#22d3ee;font-size:1.05rem;">{a_res['dominant'].upper()}</div>
            <div style="color:#64748b;font-size:0.78rem;margin-top:3px;">CALIBRATING...</div>
        </div>""", unsafe_allow_html=True)

        text_badge.markdown(f"""
        <div class="metric-card">
            <div class="modality-label">💬 TEXT</div>
            <div style="font-weight:700;color:#f59e0b;font-size:1.05rem;">{t_res['dominant'].upper()}</div>
            <div style="color:#64748b;font-size:0.78rem;margin-top:3px;">LISTENING...</div>
        </div>""", unsafe_allow_html=True)

        # Update persistent state for when session stops
        st.session_state.last_agent_response = data['agent_response']
        
        agent_placeholder.markdown(f"""
        <div class="agent-response">
            {st.session_state.last_agent_response}
        </div>
        """, unsafe_allow_html=True)

# Run the fragment
sync_dashboard()
