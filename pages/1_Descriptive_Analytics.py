import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.title("Descriptive Analytics")

st.subheader("Legal Issue Distribution")

issue_counts = df["LegalIssue"].value_counts().reset_index()
issue_counts.columns = ["LegalIssue","Count"]

fig = px.bar(issue_counts, x="LegalIssue", y="Count")

st.plotly_chart(fig)

st.subheader("Age Group Distribution")

fig2 = px.pie(df, names="AgeGroup")

st.plotly_chart(fig2)

st.subheader("Consultation Budget Distribution")

fig3 = px.histogram(df, x="ConsultBudget")

st.plotly_chart(fig3)
