import streamlit as st

st.set_page_config(
    page_title="LegalTech Analytics Dashboard",
    page_icon="⚖️",
    layout="wide"
)

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("⚖️ Mongolia LegalTech Analytics Platform")

st.markdown("""
This dashboard analyzes **legal service demand in Mongolia**.

Analytics modules:

• Descriptive Analytics  
• Diagnostic Analytics  
• Predictive Analytics  
• Prescriptive Analytics

Use the sidebar to navigate.
""")

st.success("Select a page from the sidebar.")
