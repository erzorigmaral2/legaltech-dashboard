import streamlit as st

st.set_page_config(
    page_title="LegalTech Analytics",
    page_icon="⚖️",
    layout="wide"
)

# Load CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("⚖️ Mongolia LegalTech Analytics Platform")

st.markdown("""
### AI-Powered Legal Services Intelligence Dashboard

This dashboard analyzes:

• Legal demand trends  
• Customer behavior  
• Lawyer marketplace supply  
• Conversion into paid legal consultations  

Use the sidebar to explore analytics modules.
""")

st.success("Select an analytics module from the sidebar.")
