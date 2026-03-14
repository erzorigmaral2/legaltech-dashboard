import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Descriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

# KPI Cards
col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Respondents", len(df))
col2.metric("Avg Budget", round(df["ConsultBudget"].mean(),2))
col3.metric("Avg Urgency", round(df["UrgencyScore"].mean(),2))
col4.metric("Legal Issue Types", df["LegalIssue"].nunique())

st.markdown("---")

# Legal Issue Distribution
fig = px.pie(
    df,
    names="LegalIssue",
    title="Legal Issue Distribution"
)

st.plotly_chart(fig,use_container_width=True)

st.info("""
Insight

Family disputes and employment issues dominate legal demand.
A digital legal advisor should prioritize these services.
""")

# Budget Distribution
fig2 = px.histogram(
    df,
    x="ConsultBudget",
    nbins=30,
    title="Legal Consultation Budget Distribution"
)

st.plotly_chart(fig2,use_container_width=True)

st.info("""
Insight

Most users fall within a moderate consultation budget,
suggesting strong demand for affordable online legal consultations.
""")
