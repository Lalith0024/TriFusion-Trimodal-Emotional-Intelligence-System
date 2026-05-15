"""
dashboard/pages/4_Session_History.py
──────────────────────────────────────
Session History — Visualises emotion and incongruence data from the current
or a past session. Uses st.session_state.timeline_data from the Live Dashboard
as the primary source; falls back to Redis when available.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import streamlit as st
import plotly.graph_objects as go
import json
from dashboard.components.sidebar import render_sidebar

render_sidebar()

st.markdown("## 📊 Session History")
st.markdown("*Review emotion signals and incongruence patterns from your session.*")
st.markdown("---")

# ── Data source: prefer in-session timeline_data, fallback to Redis ────────────
timeline_data = st.session_state.get("timeline_data", [])

# Try Redis if in-session data is empty
if not timeline_data:
    try:
        from src.utils.redis_client import get_frames
        col_id, col_btn = st.columns([3, 1])
        with col_id:
            session_id = st.text_input("Session ID", value="default",
                                       placeholder="Enter session ID to view")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh"):
                st.rerun()
        frames = get_frames(session_id)
        if frames:
            # Normalise Redis frame format to match timeline_data schema
            timeline_data = [
                {
                    "tick":    i,
                    "time":    f.get("timestamp", "")[-8:],
                    "face":    f.get("confidence", 0.0),
                    "voice":   f.get("confidence", 0.0),
                    "text":    f.get("confidence", 0.0),
                    "fusion":  f.get("confidence", 0.0),
                    "inc":     f.get("incongruence_score", 0.0),
                    "emotion": f.get("dominant_emotion", "neutral"),
                }
                for i, f in enumerate(frames)
            ]
    except Exception:
        pass  # Redis unavailable — timeline_data stays empty

# ── No data state ──────────────────────────────────────────────────────────────
if not timeline_data:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a0a0f,#13131a);
                border:2px dashed #2a2a3a; border-radius:16px;
                height:300px; display:flex; align-items:center;
                justify-content:center; flex-direction:column; gap:16px;
                position:relative; overflow:hidden; margin-top:1rem;">
        <div style="position:absolute;inset:0;background:radial-gradient(ellipse at 50% 50%,
             rgba(99,102,241,0.04) 0%,transparent 70%);"></div>
        <div style="font-size:3rem;opacity:0.2;">📊</div>
        <div style="color:#64748b;font-size:0.9rem;text-align:center;line-height:1.8;z-index:1;">
            No session data yet.<br>
            <span style="color:#6366f1;font-size:0.82rem;font-weight:500;">
                Start a session on the Live Dashboard to populate history.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Summary metrics ────────────────────────────────────────────────────────────
tl = timeline_data
avg_inc   = sum(d["inc"] for d in tl) / len(tl)
emotions  = [d["emotion"] for d in tl]
dominant  = max(set(emotions), key=emotions.count)
high_inc  = sum(1 for d in tl if d["inc"] > 0.7)
duration  = tl[-1]["time"] if tl else "00:00"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Session Duration", duration)
m2.metric("Avg Incongruence", f"{avg_inc:.2f}")
m3.metric("Dominant Emotion", dominant.capitalize())
m4.metric("High-Inc Frames",  high_inc)

st.markdown("---")

# ── Confidence over time ───────────────────────────────────────────────────────
st.markdown("### 📈 Modality Confidence Over Time")
times = [d["time"] for d in tl]
fig_conf = go.Figure()
series = [
    ("FACE",        [d["face"]   for d in tl], "#6366f1", "rgba(99,102,241,0.1)"),
    ("VOICE",       [d["voice"]  for d in tl], "#22d3ee", "rgba(34,211,238,0.1)"),
    ("TEXT",        [d["text"]   for d in tl], "#f59e0b", "rgba(245,158,11,0.1)"),
    ("FUSION",      [d["fusion"] for d in tl], "#22c55e", "rgba(34,197,94,0.1)"),
]
for name, vals, color, fill in series:
    fig_conf.add_trace(go.Scatter(
        x=times, y=vals, name=name,
        line=dict(color=color, width=2),
        mode="lines",
    ))
fig_conf.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,26,0.5)",
    height=200, margin=dict(l=0, r=0, t=8, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                font=dict(color="#94a3b8", size=10), bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=False, color="#2a2a3a", tickfont=dict(color="#64748b", size=9)),
    yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1a1a24",
               tickfont=dict(color="#64748b", size=9), tickformat=".0%"),
)
st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})

# ── Incongruence over time with threshold bands ────────────────────────────────
st.markdown("### 🔀 Incongruence Score Over Time")
inc_vals = [d["inc"] for d in tl]
fig_inc = go.Figure()
# Threshold bands
fig_inc.add_hrect(y0=0.0, y1=0.3, fillcolor="rgba(34,197,94,0.05)",  line_width=0)
fig_inc.add_hrect(y0=0.3, y1=0.7, fillcolor="rgba(245,158,11,0.05)", line_width=0)
fig_inc.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(239,68,68,0.05)",  line_width=0)
# Dashed threshold lines
for y, color, label in [(0.3, "#f59e0b", "moderate"), (0.7, "#ef4444", "high")]:
    fig_inc.add_hline(y=y, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right",
                      annotation_font_color=color, annotation_font_size=10)
fig_inc.add_trace(go.Scatter(
    x=times, y=inc_vals, name="Incongruence",
    line=dict(color="#ef4444", width=2),
    fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
    mode="lines",
))
fig_inc.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,26,0.5)",
    height=200, margin=dict(l=0, r=0, t=8, b=30), showlegend=False,
    xaxis=dict(showgrid=False, color="#2a2a3a", tickfont=dict(color="#64748b", size=9)),
    yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1a1a24",
               tickfont=dict(color="#64748b", size=9), tickformat=".0%"),
)
st.plotly_chart(fig_inc, use_container_width=True, config={"displayModeBar": False})

# ── Dominant emotion distribution ──────────────────────────────────────────────
st.markdown("### 🥧 Dominant Emotion Distribution")
EMOTION_COLORS = {
    "happy": "#22c55e", "calm": "#06b6d4", "neutral": "#6366f1",
    "surprised": "#f59e0b", "fearful": "#8b5cf6", "sad": "#3b82f6",
    "angry": "#ef4444", "disgusted": "#f97316",
}
from collections import Counter
counts = Counter(emotions)
labels_pie = list(counts.keys())
values_pie = list(counts.values())
colors_pie = [EMOTION_COLORS.get(e, "#6366f1") for e in labels_pie]

fig_pie = go.Figure(go.Pie(
    labels=[l.capitalize() for l in labels_pie],
    values=values_pie,
    marker_colors=colors_pie,
    hole=0.5,
    textfont=dict(color="#e2e8f0", size=12),
))
fig_pie.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", height=280,
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(0,0,0,0)"),
    showlegend=True,
)
st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# ── Raw data table ─────────────────────────────────────────────────────────────
with st.expander("📋 Raw Timeline Data"):
    import pandas as pd
    df = pd.DataFrame([{
        "Time":         d["time"],
        "Emotion":      d["emotion"].capitalize(),
        "Face Conf":    f"{d['face']:.1%}",
        "Voice Conf":   f"{d['voice']:.1%}",
        "Text Conf":    f"{d['text']:.1%}",
        "Fusion Conf":  f"{d['fusion']:.1%}",
        "Incongruence": f"{d['inc']:.2f}",
    } for d in tl])
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── Export session ─────────────────────────────────────────────────────────────
col_export, _ = st.columns([2, 5])
with col_export:
    json_bytes = json.dumps(tl, indent=2).encode()
    st.download_button(
        label="⬇ Export Session JSON",
        data=json_bytes,
        file_name="trifusion_session.json",
        mime="application/json",
        use_container_width=True,
    )
