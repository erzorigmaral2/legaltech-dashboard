import streamlit as st

st.set_page_config(
    page_title="Legal Advisor App – Mongolia",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("LegalTech Analytics Dashboard")

st.sidebar.success("Select a page above.")

st.title("Legal Advisor App – Mongolia")

st.header("Objective")

st.write("""
Monitor and optimize the conversion of users seeking legal help into paying legal consultations 
while ensuring lawyer availability and service quality.
""")

st.markdown("---")

st.subheader("Analytics Modules")

st.markdown("""
**Descriptive Analytics**
- Understand user demographics and legal demand

**Diagnostic Analytics**
- Identify problems users face finding lawyers

**Predictive Analytics**
- Predict which users will pay for legal consultations

**Prescriptive Analytics**
- Generate recommendations using association rule mining
""")

st.info("Use the sidebar to navigate between analytics views.")
