import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.title("Diagnostic Analytics")

st.subheader("Main Difficulty Finding Lawyers")

fig = px.bar(
    df["MainDifficulty"].value_counts().reset_index(),
    x="index",
    y="MainDifficulty"
)

st.plotly_chart(fig)

st.subheader("Difficulty vs Legal Issue")

fig2 = px.box(
    df,
    x="LegalIssue",
    y="TimeToFindLawyer"
)

st.plotly_chart(fig2)
