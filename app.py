import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="LegalTech Analytics Platform",
    layout="wide"
)

st.title("⚖️ Mongolia LegalTech Intelligence Dashboard")

st.markdown("AI-powered legal demand analytics platform")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# KPI HEADER
col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Survey Users", len(df))
col2.metric("Cities Covered", df["City"].nunique())
col3.metric("Legal Issues Types", df["LegalIssue"].nunique())
col4.metric("Average Budget", f"${int(df['ConsultBudget'].mean())}")

st.markdown("---")

st.markdown("""
### Platform Capabilities

• Descriptive analytics  
• Diagnostic user segmentation  
• Predictive ML demand modeling  
• Prescriptive AI lawyer recommendations
""")
