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

st.markdown("""
<div style="background:#1a0f00; border-left:4px solid #f59e0b; padding:0.55rem 1rem;
            border-radius:6px; margin-bottom:1.2rem;">
  <span style="color:#fbbf24; font-size:0.82rem; font-weight:600;">⚠ Beta</span>
  <span style="color:#fcd34d; font-size:0.78rem; margin-left:6px;">
    Hardware capture requires a local environment with a connected webcam.
    Audio and text channels use placeholder inputs until mic recorder is wired.
  </span>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # Camera feed placeholder
    st.markdown("### 📷 Live Feed")
    webcam_ph = st.empty()

    # Session controls
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 3])
    with ctrl_c1:
        if not st.session_state.session_running:
            if st.button("▶ Start Session", type="primary", use_container_width=True,
                         key="btn_start"):
                st.session_state.pipeline.start()
                st.session_state.session_running = True
                st.session_state.start_time      = time.time()
                st.session_state.timeline_data   = []
                st.rerun()
        else:
            if st.button("⏹ Stop Session", use_container_width=True, key="btn_stop"):
                st.session_state.pipeline.stop()
                st.session_state.session_running = False
                st.rerun()
    with ctrl_c2:
        fps_badge_ph = st.empty()
    with ctrl_c3:
        timer_ph = st.empty()

    # Fused emotion state card (most prominent result)
    st.markdown("### 🧠 Fused Emotional State")
    fused_card_ph = st.empty()

    # Modality signal badges
    st.markdown("### 📡 Modality Signals")
    badge_c1, badge_c2, badge_c3 = st.columns(3)
    face_ph  = badge_c1.empty()
    voice_ph = badge_c2.empty()
    text_ph  = badge_c3.empty()

with col_right:
    st.markdown("### 🕸 Emotion Radar")
    radar_ph = st.empty()

    st.markdown("### 📊 Incongruence Meter")
    inc_ph = st.empty()

    st.markdown("### 💬 WellnessAgent")
    agent_ph = st.empty()

# ── Timeline at full width ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Session Timeline")
timeline_ph = st.empty()


# ── HELPERS — render idle state elements once (outside fragment) ──────────────
def _render_idle_camera():
    webcam_ph.markdown("""
    <div style="background:#13131a; border:2px dashed #2a2a3a; border-radius:14px;
                height:360px; display:flex; align-items:center; justify-content:center;
                flex-direction:column; gap:14px;">
        <div style="font-size:3.5rem; opacity:0.25;">📷</div>
        <div style="color:#64748b; font-size:0.9rem; text-align:center; line-height:1.6;">
            Camera feed will appear here during an active session.<br>
            <span style="color:#6366f1; font-size:0.82rem;">
                Click <b>▶ Start Session</b> to begin.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_idle_fused():
    fused_card_ph.markdown("""
    <div class="metric-card" style="text-align:center; padding:1.5rem;">
        <div style="color:#64748b; font-size:0.85rem; letter-spacing:0.08em;">
            WAITING FOR SESSION
        </div>
        <div style="font-size:2rem; margin:0.4rem 0; opacity:0.4;">—</div>
    </div>
    """, unsafe_allow_html=True)


def _render_idle_badge(ph, icon, label):
    ph.markdown(f"""
    <div class="metric-card">
        <div class="modality-label">{icon} {label}</div>
        <div style="font-weight:700; color:#2a2a3a; font-size:1rem;">—</div>
        <div style="color:#64748b; font-size:0.75rem; margin-top:3px;">idle</div>
    </div>
    """, unsafe_allow_html=True)


# Render initial idle state
_render_idle_camera()
_render_idle_fused()
_render_idle_badge(face_ph,  "👁", "FACE")
_render_idle_badge(voice_ph, "🎤", "VOICE")
_render_idle_badge(text_ph,  "💬", "TEXT")
agent_ph.markdown(f"""
<div class="agent-response">
    {st.session_state.last_agent_resp}
</div>
""", unsafe_allow_html=True)


import PIL.Image

# ── FRAGMENT: refreshes at 5 Hz for maximum stability ────────────────────────
@st.fragment(run_every=0.2)
def sync_dashboard():
    """Reads latest data from the background pipeline and updates all UI placeholders."""

    if not st.session_state.session_running:
        # Not running — just persist last agent response
        agent_ph.markdown(f"""
        <div class="agent-response">{st.session_state.last_agent_resp}</div>
        """, unsafe_allow_html=True)
        fps_badge_ph.empty()
        timer_ph.empty()
        return

    # ── Read latest results from pipeline threads ─────────────────────────────
    data    = st.session_state.pipeline.get_latest()
    frame   = data.get("frame")
    v_res   = data.get("vision",  {})
    a_res   = data.get("audio",   {})
    t_res   = data.get("text",    {})
    f_res   = data.get("fusion",  {})
    fps     = data.get("fps",     0.0)
    elapsed = time.time() - (st.session_state.start_time or time.time())

    # ── 1. Camera feed ────────────────────────────────────────────────────────
    if frame is not None:
        try:
            # Converting to PIL is often more stable for Streamlit serialization
            pil_img = PIL.Image.fromarray(frame)
            webcam_ph.image(pil_img, use_container_width=True)
        except Exception:
            # Fallback to raw if PIL fails
            webcam_ph.image(frame, channels="RGB", use_container_width=True)
    elif not st.session_state.session_running:
        _render_idle_camera()

    # ── 2. FPS badge + session timer ──────────────────────────────────────────
    fps_color = "#22c55e" if fps >= 25 else ("#f59e0b" if fps >= 15 else "#ef4444")
    fps_badge_ph.markdown(f"""
    <div style="background:#13131a; border:1px solid #2a2a3a; border-radius:8px;
                padding:6px 12px; text-align:center; margin-top:4px;">
        <div style="font-family:'Space Mono',monospace; font-size:1.1rem;
                    font-weight:700; color:{fps_color};">{fps:.0f}</div>
        <div style="color:#64748b; font-size:0.65rem; letter-spacing:0.08em;">FPS</div>
    </div>
    """, unsafe_allow_html=True)

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    timer_ph.markdown(f"""
    <div style="color:#64748b; font-size:0.82rem; margin-top:10px; padding-left:4px;">
        ⏱ Session: <b style="color:#e2e8f0;">{mins:02d}:{secs:02d}</b>
    </div>
    """, unsafe_allow_html=True)

    # ── 3. Fused state card ───────────────────────────────────────────────────
    if f_res:
        dom_emotion = f_res.get("dominant_emotion", "neutral")
        confidence  = f_res.get("confidence", 0.0)
        dom_color   = _color(dom_emotion)
        inc_score   = f_res.get("incongruence_score", 0.0)

        fused_card_ph.markdown(f"""
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

    # ── 4. Modality badges ────────────────────────────────────────────────────
    def _badge(ph, icon, label, result, sub=""):
        if not result:
            _render_idle_badge(ph, icon, label)
            return
        dom  = result.get("dominant", "neutral")
        conf = result.get("confidence", 0.0)
        col  = _color(dom)
        ph.markdown(f"""
        <div class="metric-card" style="border-color:{col}33;">
            <div class="modality-label">{icon} {label}</div>
            <div style="font-weight:700; color:{col}; font-size:1.05rem;
                        text-transform:uppercase;">{dom}</div>
            <div style="color:#64748b; font-size:0.74rem; margin-top:3px;">
                {conf:.0%} conf · {sub}
            </div>
        </div>
        """, unsafe_allow_html=True)

    _badge(face_ph,  "👁",  "FACE",  v_res, "live")
    _badge(voice_ph, "🎤", "VOICE", a_res, "calibrating")
    _badge(text_ph,  "💬", "TEXT",  t_res, "listening")

    # ── 5. Radar chart ────────────────────────────────────────────────────────
    if v_res and a_res and t_res:
        with radar_ph:
            render_radar_chart({
                "FACE":  v_res.get("probabilities", {}),
                "VOICE": a_res.get("probabilities", {}),
                "TEXT":  t_res.get("probabilities", {}),
            }, key="live_radar")

    # ── 6. Incongruence meter ─────────────────────────────────────────────────
    if f_res:
        with inc_ph:
            render_incongruence_meter(f_res.get("incongruence_score", 0.0))

    # ── 7. WellnessAgent response ─────────────────────────────────────────────
    resp = data.get("agent_response", st.session_state.last_agent_resp)
    st.session_state.last_agent_resp = resp
    agent_ph.markdown(f"""
    <div class="agent-response">
        <span style="display:inline-block; margin-bottom:6px;
                     font-family:'Space Mono',monospace; font-size:0.65rem;
                     color:#6366f1; letter-spacing:0.1em;">
            WELLNESSAGENT
        </span><br>
        {resp}
    </div>
    """, unsafe_allow_html=True)

    # ── 8. Timeline (sampled at ~1 Hz — every 10 fragment ticks) ─────────────
    if f_res:
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
            st.session_state.timeline_data = tl[-120:] + [entry]  # keep last 120s

    # Draw the timeline
    tl = st.session_state.timeline_data
    if len(tl) >= 2:
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
        timeline_ph.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# Run the auto-refreshing fragment
sync_dashboard()
