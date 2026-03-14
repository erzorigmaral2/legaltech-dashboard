import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Descriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Overview")
st.dataframe(df.head())

st.subheader("Legal Issue Distribution")

fig1 = px.histogram(df, x="LegalIssue")

st.plotly_chart(fig1)

st.subheader("Age Group Distribution")

fig2 = px.pie(df, names="AgeGroup")

st.plotly_chart(fig2)

st.subheader("Consultation Budget Distribution")

fig3 = px.histogram(df, x="ConsultBudget")

st.plotly_chart(fig3)
