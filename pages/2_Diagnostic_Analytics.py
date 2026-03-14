import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Diagnostic Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Main Difficulties Finding Lawyers")

fig1 = px.histogram(df, x="MainDifficulty")

st.plotly_chart(fig1)

st.subheader("Time to Find Lawyer by Legal Issue")

fig2 = px.box(df, x="LegalIssue", y="TimeToFindLawyer")

st.plotly_chart(fig2)
