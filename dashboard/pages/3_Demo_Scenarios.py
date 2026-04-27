"""
dashboard/pages/3_Demo_Scenarios.py
─────────────────────────────────────
Demo Scenarios — Simulated trimodal emotion readings with agent responses.
No camera or microphone required — great for evaluations and presentations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import streamlit as st
from dashboard.components.sidebar import render_sidebar
from dashboard.components.radar_chart import render_radar_chart
from dashboard.components.incongruence_meter import render_incongruence_meter
from src.fusion.incongruence import compute_incongruence

st.set_page_config(page_title="TriFusion | Demo", layout="wide")

render_sidebar()

st.markdown("## 🎭 Demo Scenarios")
st.markdown("*Explore how TriFusion responds to different emotional patterns — no camera required.*")
st.markdown("---")

SCENARIOS = {
    "😷 Masked Distress (High Incongruence)": {
        "description": "Face and voice signal fear/sadness, but the person says 'I'm totally fine.'",
        "vision": {"neutral":0.05,"happy":0.03,"sad":0.15,"angry":0.05,"fearful":0.60,"surprised":0.07,"disgusted":0.02,"calm":0.03},
        "audio":  {"neutral":0.10,"happy":0.04,"sad":0.20,"angry":0.06,"fearful":0.45,"surprised":0.05,"disgusted":0.04,"calm":0.06},
        "text":   {"neutral":0.20,"happy":0.62,"sad":0.04,"angry":0.03,"fearful":0.03,"surprised":0.04,"disgusted":0.01,"calm":0.03},
        "user_text": "I'm totally fine, don't worry about me.",
        "expected_route": "escalate / grounding_technique",
        "response": """🔴 **Masked Distress Protocol Active**\n\nI sense there might be more going on than what you're expressing — and that's completely okay.\n\nLet's try a quick grounding exercise: look around and name **5 things you can see**, **4 you can touch**, **3 you can hear**, **2 you can smell**, **1 you can taste**.\n\nTake your time. I'm here."""
    },
    "😊 Genuine Happiness (Low Incongruence)": {
        "description": "All three modalities agree — the person is genuinely happy.",
        "vision": {"neutral":0.08,"happy":0.72,"sad":0.03,"angry":0.02,"fearful":0.03,"surprised":0.08,"disgusted":0.01,"calm":0.03},
        "audio":  {"neutral":0.12,"happy":0.65,"sad":0.05,"angry":0.03,"fearful":0.04,"surprised":0.06,"disgusted":0.01,"calm":0.04},
        "text":   {"neutral":0.15,"happy":0.60,"sad":0.04,"angry":0.02,"fearful":0.03,"surprised":0.10,"disgusted":0.01,"calm":0.05},
        "user_text": "I just got amazing news — I'm really excited!",
        "expected_route": "positive_reinforcement",
        "response": """✨ **Genuine Joy Detected**\n\nYou're in a beautiful space right now — and that's worth pausing for.\n\nLet yourself really be here in it. These moments are worth noticing.\n\nKeep going — whatever you're doing, it's working."""
    },
    "😡 Anger with Verbal Suppression": {
        "description": "Face and voice show clear anger, but the person tries to sound neutral.",
        "vision": {"neutral":0.05,"happy":0.02,"sad":0.08,"angry":0.72,"fearful":0.05,"surprised":0.04,"disgusted":0.03,"calm":0.01},
        "audio":  {"neutral":0.10,"happy":0.03,"sad":0.07,"angry":0.60,"fearful":0.08,"surprised":0.04,"disgusted":0.05,"calm":0.03},
        "text":   {"neutral":0.55,"happy":0.10,"sad":0.08,"angry":0.12,"fearful":0.06,"surprised":0.04,"disgusted":0.03,"calm":0.02},
        "user_text": "It's fine. I'm not upset about anything.",
        "expected_route": "grounding_technique",
        "response": """🟠 **Suppressed Anger Detected**\n\nIt's okay to feel what you're feeling — anger is a signal, not a flaw.\n\nTry **box breathing**: inhale 4 counts → hold 4 → exhale 4 → hold 4. Repeat 3 times.\n\nWhenever you're ready, we can talk about what's underneath it."""
    },
    "😢 Genuine Sadness (Aligned Signals)": {
        "description": "All three modalities agree on sadness — person is openly expressing grief.",
        "vision": {"neutral":0.08,"happy":0.03,"sad":0.72,"angry":0.05,"fearful":0.06,"surprised":0.03,"disgusted":0.02,"calm":0.01},
        "audio":  {"neutral":0.10,"happy":0.04,"sad":0.68,"angry":0.06,"fearful":0.07,"surprised":0.02,"disgusted":0.02,"calm":0.01},
        "text":   {"neutral":0.10,"happy":0.05,"sad":0.65,"angry":0.05,"fearful":0.08,"surprised":0.03,"disgusted":0.02,"calm":0.02},
        "user_text": "I've been feeling really down lately, like nothing matters.",
        "expected_route": "affirmation_generator",
        "response": """💙 **Genuine Sadness Acknowledged**\n\nWhat you're feeling is real and valid. You don't have to minimise it.\n\nThree things to hold onto:\n1. *This feeling is temporary — even when it doesn't feel like it.*\n2. *You've navigated hard days before. You're still here.*\n3. *Reaching out is a form of courage.*\n\nYou are not alone in this."""
    },
}

selected = st.selectbox("Select a demo scenario:", list(SCENARIOS.keys()))
scenario = SCENARIOS[selected]

st.markdown(f"**Description:** {scenario['description']}")
st.markdown(f"**User said:** *\"{scenario['user_text']}\"*")
st.markdown(f"**Expected route:** `{scenario['expected_route']}`")

inc_score = compute_incongruence(scenario["vision"], scenario["audio"], scenario["text"])

col_viz, col_meter = st.columns([3, 2], gap="large")

with col_viz:
    st.markdown("#### Trimodal Emotion Radar")
    render_radar_chart({"FACE": scenario["vision"], "VOICE": scenario["audio"], "TEXT": scenario["text"]})

with col_meter:
    st.markdown("#### Incongruence Score")
    render_incongruence_meter(inc_score)
    st.markdown("#### Signal Summary")
    for label, key in [("👁 Face","vision"),("🎤 Voice","audio"),("💬 Text","text")]:
        probs = scenario[key]
        dom = max(probs, key=probs.get)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:6px;padding:0.7rem 1rem;">
            <div class="modality-label">{label}</div>
            <div style="font-weight:600;">{dom.upper()}
                <span style="color:#64748b;font-weight:400;font-size:0.8rem;"> — {probs[dom]:.0%}</span>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🤖 WellnessAgent Response")
st.markdown(f'<div class="agent-response">{scenario["response"]}</div>', unsafe_allow_html=True)
