"""
dashboard/pages/4_Session_History.py
──────────────────────────────────────
Session History — View past sessions stored in Redis.
Falls back to in-memory store if Redis is unavailable.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from src.utils.redis_client import get_frames, clear_session
from dashboard.components.emotion_timeline import render_emotion_timeline, render_incongruence_timeline

st.markdown("## 📊 Session History")
st.markdown("*Review past emotion sessions and incongruence patterns.*")
st.markdown("---")

col_id, col_btn = st.columns([3, 1])
with col_id:
    session_id = st.text_input("Session ID", value="default", placeholder="Enter session ID to view")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh"):
        st.rerun()

frames = get_frames(session_id)

if not frames:
    st.info(f"No session data found for session `{session_id}`. Run a live session to populate history.")
else:
    st.success(f"Found **{len(frames)} frames** for session `{session_id}`.")

    # ── Summary metrics ───────────────────────────────────────────────────
    avg_incongruence = sum(f.get("incongruence_score", 0) for f in frames) / len(frames)
    emotions = [f.get("dominant_emotion", "neutral") for f in frames]
    most_common = max(set(emotions), key=emotions.count)
    high_inc_frames = sum(1 for f in frames if f.get("incongruence_score", 0) > 0.7)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Frames", len(frames))
    m2.metric("Avg Incongruence", f"{avg_incongruence:.2f}")
    m3.metric("Dominant Emotion", most_common.capitalize())
    m4.metric("High Incongruence Frames", high_inc_frames)

    st.markdown("---")

    # ── Timelines ─────────────────────────────────────────────────────────
    render_emotion_timeline(frames, title="Fused Emotion Probabilities Over Session")
    render_incongruence_timeline(frames)

    st.markdown("---")

    # ── Raw data table ────────────────────────────────────────────────────
    with st.expander("📋 Raw Frame Data"):
        import pandas as pd
        rows = []
        for f in frames:
            rows.append({
                "Timestamp":          f.get("timestamp","")[-8:],
                "Dominant Emotion":   f.get("dominant_emotion",""),
                "Confidence":         f"{f.get('confidence',0):.1%}",
                "Incongruence Score": f"{f.get('incongruence_score',0):.2f}",
                "User Text":          f.get("user_text","")[:60]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Clear session ─────────────────────────────────────────────────────
    if st.button("🗑 Clear Session History", type="secondary"):
        clear_session(session_id)
        st.success(f"Session `{session_id}` cleared.")
        st.rerun()
