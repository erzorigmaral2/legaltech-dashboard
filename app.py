import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Legal Advisor App – Mongolia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar content
st.sidebar.title("LegalTech Analytics Dashboard")

st.sidebar.markdown("""
### Navigation

Use the menu below to explore analytics modules.
""")

st.sidebar.info("""
Modules Available

• Descriptive Analytics  
• Diagnostic Analytics  
• Predictive Analytics  
• Prescriptive Analytics
""")

# Main page content
st.title("Legal Advisor App – Mongolia")

st.header("Objective")

st.write("""
Monitor and optimize the conversion of users seeking legal help into paying legal consultations
while ensuring lawyer availability and service quality.
""")

st.write("Use the sidebar to navigate through analytics modules.")

st.markdown("---")

st.subheader("Dashboard Modules")

st.markdown("""
• **Descriptive Analytics** — Understand user demographics and legal needs  

• **Diagnostic Analytics** — Identify challenges users face when finding lawyers  

• **Predictive Analytics** — Predict which users are likely to pay for consultations  

• **Prescriptive Analytics** — Generate recommendations using association rule mining  
""")

st.markdown("---")

st.success("Use the sidebar on the left to explore the analytics dashboard.")
