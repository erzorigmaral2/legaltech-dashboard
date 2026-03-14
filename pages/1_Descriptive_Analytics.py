import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Descriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Respondents", len(df))
col2.metric("Avg Budget", round(df["ConsultBudget"].mean(),2))
col3.metric("Avg Urgency", round(df["UrgencyScore"].mean(),2))
col4.metric("Legal Issue Types", df["LegalIssue"].nunique())

st.markdown("---")

fig1 = px.pie(df,names="LegalIssue",title="Legal Issue Distribution")
st.plotly_chart(fig1,use_container_width=True)

st.info("Family disputes and employment issues dominate legal demand.")

fig2 = px.histogram(df,x="ConsultBudget",nbins=30,title="Consultation Budget Distribution")
st.plotly_chart(fig2,use_container_width=True)

st.info("Most respondents fall into a moderate legal consultation budget.")
