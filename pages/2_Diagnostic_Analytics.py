import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.title("Diagnostic Analytics")

st.subheader("Main Difficulties in Finding Lawyers")

difficulty = df["MainDifficulty"].value_counts().reset_index()
difficulty.columns = ["Difficulty","Count"]

fig = px.bar(difficulty, x="Difficulty", y="Count")

st.plotly_chart(fig)

st.subheader("Time to Find Lawyer by Legal Issue")

fig2 = px.box(df, x="LegalIssue", y="TimeToFindLawyer")

st.plotly_chart(fig2)
