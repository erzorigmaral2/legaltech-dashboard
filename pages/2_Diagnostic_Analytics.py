import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Diagnostic Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

fig = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color="LegalIssue",
    title="Budget vs Urgency Analysis"
)

st.plotly_chart(fig,use_container_width=True)

st.info("""
Insight

Users with urgent legal issues tend to allocate higher budgets,
indicating willingness to pay for quick legal consultation.
""")
