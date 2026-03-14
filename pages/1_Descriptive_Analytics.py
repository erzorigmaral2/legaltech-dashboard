import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Descriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Overview")
st.dataframe(df.head())

# KPI Metrics
col1,col2,col3 = st.columns(3)

col1.metric("Total Respondents", len(df))
col2.metric("Unique Legal Issues", df["LegalIssue"].nunique())
col3.metric("Average Budget", round(df["ConsultBudget"].mean(),2))

st.markdown("---")

# Legal Issue Distribution
fig = px.pie(df, names="LegalIssue",
             title="Distribution of Legal Issues")

st.plotly_chart(fig)

st.info("""
Insight:

• Civil and family legal issues dominate demand.  
• These categories should be prioritized when onboarding lawyers to the platform.
""")

st.markdown("---")

# Budget Distribution
fig = px.histogram(df, x="ConsultBudget",
                   nbins=30,
                   title="Legal Consultation Budget Distribution")

st.plotly_chart(fig)

st.info("""
Insight:

• Most users have moderate legal consultation budgets.  
• Affordable online consultations may attract the majority of users.
""")

st.markdown("---")

# Urgency Analysis
fig = px.box(df, y="UrgencyScore",
             title="Urgency Level of Legal Problems")

st.plotly_chart(fig)

st.info("""
Insight:

• Many legal cases have high urgency scores.  
• Real-time lawyer matching could improve response speed.
""")

st.success("Key Takeaway: Mongolia shows strong demand for quick and affordable legal consultation services.")
