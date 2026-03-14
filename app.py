import streamlit as st

st.set_page_config(
    page_title="Legal Advisor App – Mongolia",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("LegalTech Analytics Dashboard")

st.sidebar.info("""
Navigation

• Descriptive Analytics  
• Diagnostic Analytics  
• Predictive Analytics  
• Prescriptive Analytics
""")

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
- Understand user demographics
- Analyze legal service demand

**Diagnostic Analytics**
- Identify barriers to finding lawyers

**Predictive Analytics**
- Predict likelihood of users paying for consultations

**Prescriptive Analytics**
- Generate strategic recommendations
""")

st.success("Use the sidebar to explore analytics pages.")
