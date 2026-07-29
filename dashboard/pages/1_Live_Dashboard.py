"""
dashboard/pages/1_Live_Dashboard.py
─────────────────────────────────────
Real-time trimodal emotional intelligence dashboard.

Layout:
  ┌─────────────────────────────┬─────────────────────────┐
  │  Camera feed                │  Modality badges (3-col) │
  │  FPS · Session time         │  Emotion radar chart     │
  │  Fused state card           │  Incongruence meter      │
  │  Start / Stop               │  WellnessAgent response  │
  └─────────────────────────────┴─────────────────────────┘
  ─── Session emotion timeline ──────────────────────────────

The @st.fragment(run_every=0.1) decorator refreshes the inner UI
at 10 Hz without triggering a full page re-render, keeping the
Streamlit experience smooth.
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
if "pipeline" not in st.session_state:
    with st.spinner("🧠 Loading Trimodal Neural Networks (Vision, Audio, Text)... this takes ~30 seconds on first load."):
        st.session_state.pipeline        = PipelineManager()
        st.session_state.session_running = False
        st.session_state.start_time      = None
        st.session_state.last_agent_resp = "Ready to analyze. Click ▶ Start Session."
        st.session_state.timeline_data   = []     # list of dicts for timeline chart


# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## 🎥 Live Dashboard")
st.markdown("*Real-time trimodal emotional analysis — Face · Voice · Words*")

import os
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

# ── FRAGMENT: refreshes at up to 30 Hz for smooth UI ─────────────────────────
@st.fragment(run_every=0.033)
def sync_dashboard():
    """Reads latest data from the background pipeline and renders the entire live dashboard UI."""
    data    = st.session_state.pipeline.get_latest()
    frame   = data.get("frame")
    v_res   = data.get("vision",  {})
    a_res   = data.get("audio",   {})
    t_res   = data.get("text",    {})
    f_res   = data.get("fusion",  {})
    fps     = data.get("fps",     0.0)
    elapsed = time.time() - (st.session_state.start_time or time.time())
    
    # Pre-calculate common variables
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    
    # Update timeline data if running
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
            st.session_state.timeline_data = tl[-120:]  # keep last 120s

    # ── Two-column layout ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # ── 1. Camera feed ──
        st.markdown("### 📷 Live Feed")
        if frame is not None and st.session_state.session_running:
            import cv2
            import base64
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            st.markdown(
                f'<img src="data:image/jpeg;base64,{b64_str}" style="width:100%; border-radius:8px; border:1px solid #2a2a3a;">',
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #0a0a0f 0%, #13131a 100%);
                        border: 2px dashed #2a2a3a; border-radius: 16px;
                        height: 400px; display: flex; align-items: center;
                        justify-content: center; flex-direction: column; gap: 16px;
                        position: relative; overflow: hidden;">
                <div style="position:absolute; inset:0; background: radial-gradient(ellipse at 50% 50%, rgba(99,102,241,0.05) 0%, transparent 70%);"></div>
                <div style="font-size: 4rem; opacity: 0.2; filter: grayscale(1);">📷</div>
                <div style="color: #64748b; font-size: 0.9rem; text-align: center; line-height: 1.8; z-index: 1;">
                    Camera feed will appear here.<br>
                    <span style="color: #6366f1; font-size: 0.82rem; font-weight: 500;">
                        Click ▶ Start Session to begin analysis.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── 2. Session controls ──
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
                    if st.button("▶ Start Session", type="primary", width="stretch", key="btn_start"):
                        st.session_state.pipeline.start()
                        st.session_state.session_running = True
                        st.session_state.start_time      = time.time()
                        st.session_state.timeline_data   = []
                        st.rerun()
            else:
                if st.button("⏹ Stop Session", width="stretch", key="btn_stop"):
                    st.session_state.pipeline.stop()
                    st.session_state.session_running = False
                    st.session_state.show_results_popup = True
                    st.rerun()
                    
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

        # ── 3. Fused state card ──
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

        # ── 4. Modality badges ──
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

    with col_right:
        st.markdown("### 🕸 Emotion Radar")
        
        now_ts = time.time()
        last_chart_ts = st.session_state.get("_last_chart_ts", 0.0)
        should_update_charts = (now_ts - last_chart_ts) >= 0.15

        if st.session_state.session_running and v_res and a_res and t_res:
            render_radar_chart({
                "FACE":  v_res.get("probabilities", {}),
                "VOICE": a_res.get("probabilities", {}),
                "TEXT":  t_res.get("probabilities", {}),
            }, key="live_radar")
            if should_update_charts:
                st.session_state._last_chart_ts = now_ts
        else:
            # Empty chart for idle
            from config.emotions import UNIFIED_EMOTIONS
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

    # ── Timeline at full width ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Session Timeline")
    
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


# Run the auto-refreshing fragment
sync_dashboard()

