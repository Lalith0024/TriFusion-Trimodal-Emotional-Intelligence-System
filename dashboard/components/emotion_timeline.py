"""
dashboard/components/emotion_timeline.py
─────────────────────────────────────────
Emotion timeline chart for the Session History page.

Renders a stacked area chart (Plotly) showing how each unified emotion's
probability evolved over the course of a session. Each session frame
becomes one x-axis tick.

Colour palette mirrors the rest of the dashboard.
"""

import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict
from config.emotions import UNIFIED_EMOTIONS


# Assign a distinct colour to each unified emotion
EMOTION_COLORS = {
    "neutral":   "#94a3b8",
    "happy":     "#22c55e",
    "sad":       "#60a5fa",
    "angry":     "#f87171",
    "fearful":   "#a78bfa",
    "surprised": "#fbbf24",
    "disgusted": "#fb923c",
    "calm":      "#34d399",
}


def render_emotion_timeline(
    frames: List[Dict],
    height: int = 350,
    title: str = "Session Emotion Timeline",
    key: str = None
) -> None:
    """
    Render a line chart showing fused emotion probabilities over session time.

    Args:
        frames: List of frame dicts, each containing:
                  "fused_probabilities" (dict) and "timestamp" (str).
        height: Chart height in pixels.
        title:  Chart title string.
        key:    Unique Streamlit element key.
    """
    if not frames:
        st.info("No session data yet. Start a live session to see the timeline.")
        return

    timestamps = [f.get("timestamp", str(i)) for i, f in enumerate(frames)]
    # Use short timestamp labels (HH:MM:SS) for readability
    short_ts   = [t[11:19] if len(t) > 10 else str(i) for i, t in enumerate(timestamps)]

    fig = go.Figure()

    for emotion in UNIFIED_EMOTIONS:
        color  = EMOTION_COLORS.get(emotion, "#ffffff")
        values = [
            f.get("fused_probabilities", {}).get(emotion, 0.0) for f in frames
        ]
        fig.add_trace(go.Scatter(
            x=short_ts,
            y=values,
            name=emotion.capitalize(),
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color.replace("#", "rgba(") + ",0.06)",  # very subtle fill
            hovertemplate=f"<b>{emotion}</b>: %{{y:.1%}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#94a3b8", size=14)),
        xaxis=dict(
            title="Time",
            gridcolor="#2a2a3a", linecolor="#2a2a3a",
            tickfont=dict(color="#64748b", size=10)
        ),
        yaxis=dict(
            title="Probability",
            range=[0, 1],
            gridcolor="#2a2a3a", linecolor="#2a2a3a",
            tickformat=".0%",
            tickfont=dict(color="#64748b", size=10)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(color="#94a3b8", size=10),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=-0.35,
            xanchor="center", x=0.5
        ),
        margin=dict(t=40, b=60, l=50, r=20),
        height=height
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def render_incongruence_timeline(
    frames: List[Dict],
    height: int = 220,
    key: str = None
) -> None:
    """
    Separate chart showing only the incongruence score over time.
    Useful for identifying when masking events occurred.
    Args:
        key: Unique Streamlit element key.
    """
    if not frames:
        return

    short_ts = [f.get("timestamp", "")[-8:] for f in frames]
    scores   = [f.get("incongruence_score", 0.0) for f in frames]

    # Background colour bands for ALIGNED / MODERATE / HIGH zones
    fig = go.Figure()

    # Threshold bands
    fig.add_hrect(y0=0,   y1=0.3, fillcolor="rgba(34,197,94,0.06)",  line_width=0)
    fig.add_hrect(y0=0.3, y1=0.7, fillcolor="rgba(245,158,11,0.06)", line_width=0)
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(239,68,68,0.06)",  line_width=0)

    # Threshold dashed lines
    fig.add_hline(y=0.3, line_dash="dot", line_color="#22c55e", line_width=1)
    fig.add_hline(y=0.7, line_dash="dot", line_color="#ef4444", line_width=1)

    fig.add_trace(go.Scatter(
        x=short_ts, y=scores,
        mode="lines+markers",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=5, color="#6366f1"),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
        name="Incongruence",
        hovertemplate="Incongruence: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Incongruence Score Timeline", font=dict(color="#94a3b8", size=13)),
        xaxis=dict(gridcolor="#2a2a3a", tickfont=dict(color="#64748b", size=9)),
        yaxis=dict(range=[0, 1], tickformat=".0%",
                   gridcolor="#2a2a3a", tickfont=dict(color="#64748b", size=9)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=35, b=30, l=45, r=15),
        height=height
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)
