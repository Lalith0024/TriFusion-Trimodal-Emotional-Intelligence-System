"""
dashboard/pages/1_Live_Dashboard.py
─────────────────────────────────────
Real-time trimodal emotional intelligence dashboard.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv

from dashboard.components.radar_chart      import render_radar_chart
from dashboard.components.incongruence_meter import render_incongruence_meter
from dashboard.components.sidebar          import render_sidebar
from src.pipeline.manager                  import PipelineManager
from config.emotions                       import UNIFIED_EMOTIONS
from streamlit_webrtc                      import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import cv2

load_dotenv()

# ── Emotion → colour map for consistent badge/card theming ───────────────────
EMOTION_COLORS = {
    "happy":     "#22c55e",
    "calm":      "#06b6d4",
    "neutral":   "#6366f1",
    "surprised": "#f59e0b",
    "fearful":   "#8b5cf6",
    "sad":       "#3b82f6",
    "angry":     "#ef4444",
    "disgusted": "#f97316",
}

def _color(emotion: str) -> str:
    return EMOTION_COLORS.get(emotion.lower(), "#6366f1")


# ── Singleton pipeline — one per Streamlit session ───────────────────────────
@st.cache_resource
def get_pipeline():
    return PipelineManager()

pipeline = get_pipeline()

if "session_running" not in st.session_state:
    st.session_state.session_running = False
    st.session_state.start_time      = None
    st.session_state.last_agent_resp = "Ready to analyze. Click START on the camera."
    st.session_state.timeline_data   = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## 🎥 Live Dashboard")
st.markdown("*Real-time trimodal emotional analysis — Face · Voice · Words*")

_is_simulation = os.environ.get("SIMULATION_MODE", "false").lower() in ("true", "1", "t")
if _is_simulation:
    st.markdown("""
    <div style="background:#1a1025; border-left:4px solid #a855f7; padding:0.55rem 1rem;
                border-radius:6px; margin-bottom:1.2rem;">
      <span style="color:#d8b4fe; font-size:0.82rem; font-weight:600;">✨ Simulation Mode Active</span>
      <span style="color:#e9d5ff; font-size:0.78rem; margin-left:6px;">
        Running on synthetic data (no camera/mic access required). Perfect for testing and cloud deployment.
      </span>
    </div>
    """, unsafe_allow_html=True)


col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### 📷 Live Feed")
    
    # WebRTC callbacks
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        pipeline.feed_video_frame(img)
        
        # Draw annotations
        v = pipeline.get_latest().get("vision", {})
        if v and v.get("face_detected") and v.get("bbox") and hasattr(pipeline, 'vision_inf'):
            img = pipeline.vision_inf.face_detector.draw_overlay(
                img, v["bbox"], v["dominant"], v["confidence"]
            )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        array = frame.to_ndarray()
        if array.shape[0] > 1: array = array.mean(axis=0)
        else: array = array[0]
        pipeline.feed_audio_chunk(array)
        return frame

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="trifusion-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        async_processing=True,
        media_stream_constraints={"video": True, "audio": True},
    )
    
    # Sync pipeline state with WebRTC state
    if ctx.state.playing and not pipeline.running:
        pipeline.start()
        st.session_state.session_running = True
        st.session_state.start_time = time.time()
        st.session_state.timeline_data = []
    elif not ctx.state.playing and pipeline.running:
        pipeline.stop()
        st.session_state.session_running = False
        st.session_state.show_results_popup = True

    @st.fragment(run_every=0.5)
    def sync_left():
        data    = pipeline.get_latest()
        v_res   = data.get("vision",  {})
        a_res   = data.get("audio",   {})
        t_res   = data.get("text",    {})
        f_res   = data.get("fusion",  {})
        fps     = data.get("fps",     0.0)
        elapsed = time.time() - (st.session_state.start_time or time.time())
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        # ── Session controls ──
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 3])
        with ctrl_c1:
            if not st.session_state.session_running:
                if st.session_state.get("show_results_popup"):
                    st.success("✅ Session Recorded!")
                    c1, c2 = st.columns(2)
                    if c1.button("📈 View Results", type="primary", width="stretch"):
                        st.session_state.show_results_popup = False
                        st.switch_page("pages/4_Session_History.py")
                    if c2.button("Dismiss", width="stretch"):
                        st.session_state.show_results_popup = False
                        st.rerun()
                else:
                    st.markdown("""
                    <div style="color: #6366f1; font-size: 0.85rem; font-weight: 500; margin-top: 10px;">
                        Click START above to begin
                    </div>
                    """, unsafe_allow_html=True)
                    
        with ctrl_c2:
            if st.session_state.session_running:
                fps_color = "#22c55e" if fps >= 25 else ("#f59e0b" if fps >= 15 else "#ef4444")
                st.markdown(f"""
                <div style="background:#13131a; border:1px solid #2a2a3a; border-radius:8px;
                            padding:6px 12px; text-align:center; margin-top:4px;">
                    <div style="font-family:'Space Mono',monospace; font-size:1.1rem;
                                font-weight:700; color:{fps_color};">{fps:.0f}</div>
                    <div style="color:#64748b; font-size:0.65rem; letter-spacing:0.08em;">FPS</div>
                </div>
                """, unsafe_allow_html=True)
                
        with ctrl_c3:
            if st.session_state.session_running:
                st.markdown(f"""
                <div style="color:#64748b; font-size:0.82rem; margin-top:10px; padding-left:4px;">
                    ⏱ Session: <b style="color:#e2e8f0;">{mins:02d}:{secs:02d}</b>
                </div>
                """, unsafe_allow_html=True)

        # ── Fused state card ──
        st.markdown("### 🧠 Fused Emotional State")
        if f_res and st.session_state.session_running:
            dom_emotion = f_res.get("dominant_emotion", "neutral")
            confidence  = f_res.get("confidence", 0.0)
            dom_color   = _color(dom_emotion)
            inc_score   = f_res.get("incongruence_score", 0.0)

            st.markdown(f"""
            <div class="metric-card" style="border-color:{dom_color}44; padding:1.2rem 1.6rem;">
                <div class="modality-label">🧠 FUSED OUTPUT</div>
                <div style="font-family:'Space Mono',monospace; font-size:2rem;
                            font-weight:700; color:{dom_color}; margin:0.3rem 0;
                            text-transform:uppercase; letter-spacing:0.04em;">
                    {dom_emotion}
                </div>
                <div style="display:flex; gap:1.5rem; margin-top:0.4rem;">
                    <div>
                        <span style="color:#64748b; font-size:0.72rem;">CONFIDENCE</span><br>
                        <span style="color:#e2e8f0; font-weight:600;">
                            {confidence:.0%}
                        </span>
                    </div>
                    <div>
                        <span style="color:#64748b; font-size:0.72rem;">INCONGRUENCE</span><br>
                        <span style="color:#e2e8f0; font-weight:600;">
                            {inc_score:.0%}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="text-align:center; padding:1.5rem;">
                <div style="color:#64748b; font-size:0.85rem; letter-spacing:0.08em;">
                    WAITING FOR SESSION
                </div>
                <div style="font-size:2rem; margin:0.4rem 0; opacity:0.4;">—</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Modality badges ──
        st.markdown("### 📡 Modality Signals")
        badge_c1, badge_c2, badge_c3 = st.columns(3)
        
        def _render_badge(col, icon, label, result, sub=""):
            with col:
                if not result or not st.session_state.session_running:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="modality-label">{icon} {label}</div>
                        <div style="font-weight:700; color:#2a2a3a; font-size:1rem;">—</div>
                        <div style="color:#64748b; font-size:0.75rem; margin-top:3px;">idle</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    dom  = result.get("dominant", "neutral")
                    conf = result.get("confidence", 0.0)
                    col_color  = _color(dom)
                    st.markdown(f"""
                    <div class="metric-card" style="border-color:{col_color}33;">
                        <div class="modality-label">{icon} {label}</div>
                        <div style="font-weight:700; color:{col_color}; font-size:1.05rem;
                                    text-transform:uppercase;">{dom}</div>
                        <div style="color:#64748b; font-size:0.74rem; margin-top:3px;">
                            {conf:.0%} conf · {sub}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        _render_badge(badge_c1, "👁",  "FACE",  v_res, "live")
        _render_badge(badge_c2, "🎤", "VOICE", a_res, "calibrating")
        _render_badge(badge_c3, "💬", "TEXT",  t_res, "listening")

    sync_left()

with col_right:
    @st.fragment(run_every=0.5)
    def sync_right():
        data    = pipeline.get_latest()
        v_res   = data.get("vision",  {})
        a_res   = data.get("audio",   {})
        t_res   = data.get("text",    {})
        f_res   = data.get("fusion",  {})

        st.markdown("### 🕸 Emotion Radar")
        
        if st.session_state.session_running and v_res and a_res and t_res:
            render_radar_chart({
                "FACE":  v_res.get("probabilities", {}),
                "VOICE": a_res.get("probabilities", {}),
                "TEXT":  t_res.get("probabilities", {}),
            }, key="live_radar")
        else:
            uniform = {e: 1.0/len(UNIFIED_EMOTIONS) for e in UNIFIED_EMOTIONS}
            render_radar_chart({
                "FACE": uniform, "VOICE": uniform, "TEXT": uniform
            }, key="idle_radar")

        st.markdown("### 📊 Incongruence Meter")
        if st.session_state.session_running and f_res:
            render_incongruence_meter(f_res.get("incongruence_score", 0.0))
        else:
            render_incongruence_meter(0.0)

        st.markdown("### 💬 WellnessAgent")
        resp = data.get("agent_response", st.session_state.last_agent_resp)
        if st.session_state.session_running:
            st.session_state.last_agent_resp = resp
            
        st.markdown(f"""
        <div class="agent-response">
            <span style="display:inline-block; margin-bottom:6px;
                         font-family:'Space Mono',monospace; font-size:0.65rem;
                         color:#6366f1; letter-spacing:0.1em;">
                WELLNESSAGENT
            </span><br>
            {resp}
        </div>
        """, unsafe_allow_html=True)

    sync_right()

# ── Timeline ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Session Timeline")

@st.fragment(run_every=0.5)
def sync_timeline():
    data    = pipeline.get_latest()
    f_res   = data.get("fusion",  {})
    v_res   = data.get("vision",  {})
    a_res   = data.get("audio",   {})
    t_res   = data.get("text",    {})
    elapsed = time.time() - (st.session_state.start_time or time.time())
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    if f_res and st.session_state.session_running:
        tick = int(elapsed)
        tl   = st.session_state.timeline_data
        if not tl or tl[-1].get("tick") != tick:
            entry = {
                "tick":    tick,
                "time":    f"{mins:02d}:{secs:02d}",
                "face":    v_res.get("confidence", 0.0) if v_res else 0.0,
                "voice":   a_res.get("confidence", 0.0) if a_res else 0.0,
                "text":    t_res.get("confidence", 0.0) if t_res else 0.0,
                "fusion":  f_res.get("confidence", 0.0),
                "inc":     f_res.get("incongruence_score", 0.0),
                "emotion": f_res.get("dominant_emotion", "neutral"),
            }
            tl.append(entry)
            st.session_state.timeline_data = tl[-120:]

    tl = st.session_state.timeline_data
    if st.session_state.session_running and len(tl) >= 2:
        import plotly.graph_objects as go
        times   = [d["time"]  for d in tl]
        fig = go.Figure()
        series = [
            ("FACE",        [d["face"]  for d in tl], "#6366f1", "rgba(99, 102, 241, 0.1)"),
            ("VOICE",       [d["voice"] for d in tl], "#22d3ee", "rgba(34, 211, 238, 0.1)"),
            ("TEXT",        [d["text"]  for d in tl], "#f59e0b", "rgba(245, 158, 11, 0.1)"),
            ("FUSION",      [d["fusion"]for d in tl], "#22c55e", "rgba(34, 197, 94, 0.1)"),
            ("INCONGRUENCE",[d["inc"]   for d in tl], "#ef4444", "rgba(239, 68, 68, 0.1)"),
        ]
        for name, vals, color, fill in series:
            fig.add_trace(go.Scatter(
                x=times, y=vals, name=name,
                line=dict(color=color, width=2),
                mode="lines",
                fill="tozeroy" if name == "INCONGRUENCE" else "none",
                fillcolor=fill,
            ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,26,0.5)",
            height=180,
            margin=dict(l=0, r=0, t=8, b=30),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                font=dict(color="#94a3b8", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                showgrid=False, color="#2a2a3a",
                tickfont=dict(color="#64748b", size=9),
            ),
            yaxis=dict(
                range=[0, 1], showgrid=True,
                gridcolor="#1a1a24", color="#2a2a3a",
                tickfont=dict(color="#64748b", size=9),
                tickformat=".0%",
            ),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.markdown("""
        <div style="height: 180px; display: flex; align-items: center; justify-content: center;
                    color: #64748b; font-size: 0.9rem; background: rgba(13,13,26,0.5);
                    border: 1px dashed #2a2a3a; border-radius: 8px;">
            Timeline will populate once session starts.
        </div>
        """, unsafe_allow_html=True)

sync_timeline()
