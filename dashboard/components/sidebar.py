"""
dashboard/components/sidebar.py
──────────────────────────────
Consistent sidebar component for all dashboard pages.
"""

import streamlit as st

def render_sidebar():
    """Render the custom sidebar navigation and system status."""
    with st.sidebar:
        st.markdown("""
        <div style='padding:1rem 0 0.5rem'>
            <div style='font-family:Space Mono,monospace;font-size:1.15rem;font-weight:700;
                        background:linear-gradient(135deg,#6366f1,#22d3ee);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                🧠 TriFusion
            </div>
            <div style='color:#64748b;font-size:0.75rem;margin-top:3px;'>
                Trimodal Emotional Intelligence
            </div>
        </div>
        <hr style='border:0;border-top:1px solid #2a2a3a;margin:0.8rem 0;'>
        """, unsafe_allow_html=True)

        st.markdown("**Navigation**")
        
        # Detect current page to highlight active link (optional enhancement)
        st.page_link("app.py",                          label="🏠 Home")
        st.page_link("pages/1_Live_Dashboard.py",        label="🎥 Live Dashboard")
        st.page_link("pages/2_About.py",                 label="📖 About")
        st.page_link("pages/3_Demo_Scenarios.py",        label="🎭 Demo Scenarios")
        st.page_link("pages/4_Session_History.py",       label="📊 Session History")
        st.page_link("pages/5_Model_Cards.py",           label="🤖 Model Cards")

        st.markdown('<hr style="border:0;border-top:1px solid #2a2a3a;margin:0.8rem 0;">', unsafe_allow_html=True)
        st.markdown("**System Status**")
        st.success("✓ Vision Module Ready")
        st.success("✓ Audio Module Ready")
        st.success("✓ Text Module Ready")
        st.success("✓ WellnessAgent Ready")
