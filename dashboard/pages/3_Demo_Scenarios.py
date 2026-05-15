"""
dashboard/pages/3_Demo_Scenarios.py
─────────────────────────────────────
Pre-built demo scenarios for when webcam isn't available.
Shows exactly what TriFusion detects and how WellnessAgent responds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import streamlit as st
import plotly.graph_objects as go
from dashboard.components.radar_chart import render_radar_chart
from dashboard.components.incongruence_meter import render_incongruence_meter
from dashboard.components.sidebar import render_sidebar
from config.emotions import UNIFIED_EMOTIONS

render_sidebar()

st.markdown("## 🎭 Demo Scenarios")
st.markdown("*Four archetypal emotional states demonstrating TriFusion's detection capabilities.*")
st.markdown("""
<div style="background:#0f172a; border-left:4px solid #6366f1; padding:0.6rem 1rem;
            border-radius:6px; margin-bottom:1.5rem;">
    <div style="color:#a5b4fc; font-size:0.83rem;">
        These scenarios use pre-computed values to illustrate what TriFusion detects
        in the wild. Run the Live Dashboard with a webcam for real-time analysis.
    </div>
</div>
""", unsafe_allow_html=True)

SCENARIOS = [
    {
        "id": 1,
        "title": "The Hidden Stress",
        "emoji": "😐",
        "description": "User appears neutral, speaks calmly, but words carry underlying anxiety. Classic suppression pattern found in high-pressure work environments.",
        "face":  {"neutral": 0.48, "fearful": 0.22, "sad": 0.12, "angry": 0.08, "happy": 0.04, "surprised": 0.03, "disgusted": 0.02, "calm": 0.01},
        "voice": {"neutral": 0.35, "calm": 0.28, "fearful": 0.18, "sad": 0.10, "happy": 0.05, "angry": 0.03, "surprised": 0.01, "disgusted": 0.00},
        "text":  {"neutral": 0.52, "happy": 0.21, "calm": 0.14, "surprised": 0.05, "sad": 0.04, "fearful": 0.02, "angry": 0.01, "disgusted": 0.01},
        "incongruence": 0.76,
        "fused_dominant": "fearful",
        "fused_confidence": 0.61,
        "user_said": "I'm doing fine, just got a lot on my plate right now.",
        "agent_intervention": "breathing_exercise",
        "agent_response": "I hear you — having a full plate takes real energy to manage. Let's try something quick:\n\nBreathe in for **4 counts**, hold for **7**, out for **8**.\n\nYou don't have to solve everything right now. Just this breath.",
    },
    {
        "id": 2,
        "title": "Genuine Happiness",
        "emoji": "😊",
        "description": "All three modalities aligned on joy. Low incongruence. This is what authentic positive emotion looks like — face, voice, and words all telling the same story.",
        "face":  {"happy": 0.72, "calm": 0.10, "surprised": 0.08, "neutral": 0.05, "fearful": 0.02, "sad": 0.01, "angry": 0.01, "disgusted": 0.01},
        "voice": {"happy": 0.68, "calm": 0.14, "neutral": 0.10, "surprised": 0.05, "fearful": 0.01, "sad": 0.01, "angry": 0.01, "disgusted": 0.00},
        "text":  {"happy": 0.75, "calm": 0.09, "neutral": 0.08, "surprised": 0.04, "fearful": 0.02, "sad": 0.01, "angry": 0.01, "disgusted": 0.00},
        "incongruence": 0.08,
        "fused_dominant": "happy",
        "fused_confidence": 0.71,
        "user_said": "That actually went really well! I'm genuinely excited about this.",
        "agent_intervention": "positive_reinforcement",
        "agent_response": "That energy is real — hold onto this feeling for a moment.\n\nNotice how your body feels right now. This is what it feels like when things click.",
    },
    {
        "id": 3,
        "title": "Masked Fear",
        "emoji": "😬",
        "description": "The most dangerous pattern. Face and voice both showing fear while words assert everything is fine. High incongruence triggers escalation protocol.",
        "face":  {"fearful": 0.61, "sad": 0.18, "neutral": 0.10, "angry": 0.05, "disgusted": 0.03, "happy": 0.01, "surprised": 0.01, "calm": 0.01},
        "voice": {"fearful": 0.52, "neutral": 0.22, "calm": 0.12, "sad": 0.08, "angry": 0.04, "happy": 0.01, "surprised": 0.01, "disgusted": 0.00},
        "text":  {"happy": 0.58, "neutral": 0.24, "calm": 0.11, "surprised": 0.04, "fearful": 0.02, "sad": 0.01, "angry": 0.00, "disgusted": 0.00},
        "incongruence": 0.89,
        "fused_dominant": "fearful",
        "fused_confidence": 0.68,
        "user_said": "I'm fine. Everything is totally fine. Please don't worry about me.",
        "agent_intervention": "escalation",
        "agent_response": "I notice you're working hard to hold things together right now — and I want you to know that takes courage.\n\nIf you're carrying something difficult, you don't have to carry it alone. These are available 24/7:\n\n📞 **iCall India:** 9152987821\n📞 **Vandrevala Foundation:** 1860-2662-345\n\nYou reached out. That matters.",
    },
    {
        "id": 4,
        "title": "Suppressed Anger",
        "emoji": "😤",
        "description": "Face and voice registering anger but words staying controlled and neutral. Common in professional settings where direct expression isn't acceptable.",
        "face":  {"angry": 0.58, "disgusted": 0.18, "neutral": 0.12, "sad": 0.06, "fearful": 0.03, "surprised": 0.02, "happy": 0.01, "calm": 0.00},
        "voice": {"angry": 0.44, "neutral": 0.28, "disgusted": 0.14, "fearful": 0.07, "sad": 0.04, "happy": 0.01, "calm": 0.01, "surprised": 0.01},
        "text":  {"neutral": 0.61, "sad": 0.14, "calm": 0.10, "happy": 0.07, "fearful": 0.04, "angry": 0.02, "disgusted": 0.01, "surprised": 0.01},
        "incongruence": 0.72,
        "fused_dominant": "angry",
        "fused_confidence": 0.56,
        "user_said": "I just think there might be a different way to approach this situation.",
        "agent_intervention": "grounding_technique",
        "agent_response": "You're navigating something frustrating with a lot of care. Let's slow down for one moment:\n\nName **5 things you can see** around you right now — just look around and notice them.\n\nTake your time. I'll be here.",
    },
]

# Scenario selector
selected_id = st.radio(
    "Select scenario",
    [f"Scenario {s['id']} — {s['emoji']} {s['title']}" for s in SCENARIOS],
    horizontal=True,
    label_visibility="collapsed"
)
scenario = SCENARIOS[int(selected_id[9]) - 1]

st.markdown(f"### {scenario['emoji']} Scenario {scenario['id']}: {scenario['title']}")
st.markdown(f"*{scenario['description']}*")

# Modality + results layout
col_l, col_r = st.columns([3, 2])

with col_l:
    st.markdown("#### Detected Signals")

    inc = scenario["incongruence"]
    inc_color = "#22c55e" if inc < 0.3 else ("#f59e0b" if inc < 0.7 else "#ef4444")
    inc_label = "ALIGNED" if inc < 0.3 else ("MODERATE" if inc < 0.7 else "HIGH — MASKING DETECTED")

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:0.8rem;">
        <div class="modality-label">🗣 USER SAID</div>
        <div style="font-style:italic; color:#a5b4fc; font-size:0.95rem; margin-top:4px;">
            "{scenario['user_said']}"
        </div>
    </div>
    """, unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    def scenario_badge(col, icon, label, probs, color):
        dom = max(probs, key=probs.get)
        conf = probs[dom]
        col.markdown(f"""
        <div class="metric-card" style="border-color:{color}33;">
            <div class="modality-label">{icon} {label}</div>
            <div style="font-weight:700;color:{color};font-size:1rem;text-transform:uppercase;">{dom}</div>
            <div style="color:#64748b;font-size:0.73rem;margin-top:3px;">{conf:.0%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

    scenario_badge(b1, "👁", "FACE",  scenario["face"],  "#6366f1")
    scenario_badge(b2, "🎤","VOICE", scenario["voice"], "#22d3ee")
    scenario_badge(b3, "💬","TEXT",  scenario["text"],  "#f59e0b")

    st.markdown(f"""
    <div class="metric-card" style="border-color:{inc_color}44; margin-top:0.5rem;">
        <div class="modality-label">📊 INCONGRUENCE SCORE</div>
        <div style="font-size:2rem;font-weight:700;color:{inc_color};font-family:'Space Mono',monospace;">
            {inc:.2f}
        </div>
        <div style="font-size:0.82rem;color:{inc_color};font-weight:600;">{inc_label}</div>
        <div style="background:#2a2a3a;border-radius:6px;height:8px;margin-top:8px;overflow:hidden;">
            <div style="height:100%;width:{inc*100:.0f}%;background:{inc_color};border-radius:6px;
                        transition:width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fused output
    dom_emotion = scenario["fused_dominant"]
    EMOTION_COLORS = {"happy":"#22c55e","calm":"#06b6d4","neutral":"#6366f1","surprised":"#f59e0b",
                      "fearful":"#8b5cf6","sad":"#3b82f6","angry":"#ef4444","disgusted":"#f97316"}
    dom_color = EMOTION_COLORS.get(dom_emotion, "#6366f1")
    st.markdown(f"""
    <div class="metric-card" style="border-color:{dom_color}44;">
        <div class="modality-label">🧠 FUSED OUTPUT</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;
                    color:{dom_color};text-transform:uppercase;">{dom_emotion}</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:4px;">
            {scenario['fused_confidence']:.0%} confidence · intervention: {scenario['agent_intervention'].replace('_',' ')}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    st.markdown("#### Emotion Radar")
    render_radar_chart({
        "FACE":  scenario["face"],
        "VOICE": scenario["voice"],
        "TEXT":  scenario["text"],
    }, key=f"scenario_radar_{scenario['id']}")

    st.markdown("#### WellnessAgent Response")
    st.markdown(f"""
    <div class="agent-response">
        <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#6366f1;
                     letter-spacing:0.1em;display:block;margin-bottom:8px;">WELLNESSAGENT</span>
        {scenario['agent_response'].replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
