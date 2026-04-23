"""
dashboard/components/radar_chart.py
─────────────────────────────────────
Plotly trimodal emotion radar chart component.

Renders three overlapping filled polygons — one per modality —
on a polar axis where each spoke represents one unified emotion.

Colour assignment:
  FACE  → Indigo  (#6366f1)
  VOICE → Cyan    (#22d3ee)
  TEXT  → Amber   (#f59e0b)
"""

import plotly.graph_objects as go
import streamlit as st
from config.emotions import UNIFIED_EMOTIONS


# Consistent colour palette across all dashboard components
MODALITY_COLORS = {
    "FACE":  ("rgba(99,102,241,0.25)",  "#6366f1"),
    "VOICE": ("rgba(34,211,238,0.25)",  "#22d3ee"),
    "TEXT":  ("rgba(245,158,11,0.25)",  "#f59e0b"),
}


def render_radar_chart(modality_data: dict, height: int = 290, key: str = None):
    """
    Render a Plotly radar chart for trimodal emotion probabilities.

    Args:
        modality_data: {
            "FACE":  {emotion: probability, ...},
            "VOICE": {emotion: probability, ...},
            "TEXT":  {emotion: probability, ...}
        }
        height: Chart height in pixels.
        key:    Unique Streamlit element key.
    """
    emotions = UNIFIED_EMOTIONS
    fig = go.Figure()

    for modality, probs in modality_data.items():
        values = [probs.get(e, 0.0) for e in emotions]
        values.append(values[0])   # close the polygon
        fill_color, line_color = MODALITY_COLORS.get(modality, ("rgba(255,255,255,0.1)", "#fff"))

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=emotions + [emotions[0]],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            name=modality,
            hovertemplate="<b>%{theta}</b><br>%{r:.1%}<extra>" + modality + "</extra>"
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor="#2a2a3a",
                linecolor="#2a2a3a"
            ),
            angularaxis=dict(
                gridcolor="#2a2a3a",
                linecolor="#2a2a3a",
                tickfont=dict(color="#94a3b8", size=11)
            )
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="#94a3b8", size=11),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=30, l=20, r=20),
        height=height
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
