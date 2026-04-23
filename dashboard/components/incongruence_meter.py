"""
dashboard/components/incongruence_meter.py
───────────────────────────────────────────
Visual incongruence score meter component.

Renders a colour-coded progress bar with a badge label
showing ALIGNED / MODERATE / HIGH — MASKING DETECTED.

The width of the bar animates via CSS transition for a
smooth live-updating effect when scores change frame-to-frame.
"""

import streamlit as st
from src.fusion.incongruence import get_incongruence_label


def render_incongruence_meter(score: float):
    """
    Render the incongruence meter for the given score.

    Args:
        score: Float in [0.0, 1.0] from compute_incongruence().
    """
    label, color = get_incongruence_label(score)
    pct = int(score * 100)

    # Badge class maps to CSS defined in app.py
    if score < 0.3:
        badge_class = "badge-aligned"
    elif score < 0.7:
        badge_class = "badge-moderate"
    else:
        badge_class = "badge-high"

    st.markdown(f"""
    <div style="margin-bottom:10px;">
        <span class="emotion-badge {badge_class}">{label}</span>
    </div>

    <!-- Progress bar track -->
    <div style="background:#1a1a24; border-radius:8px; height:13px;
                overflow:hidden; border:1px solid #2a2a3a;">
        <div style="height:100%; width:{pct}%;
                    background: linear-gradient(90deg, {color}aa, {color});
                    border-radius:8px;
                    transition: width 0.6s ease-in-out;"></div>
    </div>

    <!-- Min / score / max labels -->
    <div style="display:flex; justify-content:space-between; margin-top:5px;">
        <span style="color:#64748b; font-size:0.72rem;">0 — Aligned</span>
        <span style="color:{color}; font-size:0.88rem; font-weight:700;
                     font-family:'Space Mono',monospace;">{pct}%</span>
        <span style="color:#64748b; font-size:0.72rem;">100 — Masking</span>
    </div>
    """, unsafe_allow_html=True)


def render_incongruence_gauge(score: float):
    """
    Alternative gauge chart using Plotly (for Session History page).
    Returns a Plotly figure — caller must call st.plotly_chart().
    """
    import plotly.graph_objects as go
    from src.fusion.incongruence import get_incongruence_label

    label, color = get_incongruence_label(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2a2a3a",
                     "tickfont": {"color": "#64748b"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#13131a",
            "bordercolor": "#2a2a3a",
            "steps": [
                {"range": [0,  30],  "color": "rgba(34,197,94,0.1)"},
                {"range": [30, 70],  "color": "rgba(245,158,11,0.1)"},
                {"range": [70, 100], "color": "rgba(239,68,68,0.1)"},
            ]
        },
        title={"text": f"Incongruence<br><b>{label}</b>",
               "font": {"color": "#94a3b8", "size": 12}}
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(t=20, b=10, l=10, r=10)
    )
    return fig
