import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Descriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

col1,col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="LegalIssue",
        title="Legal Issue Distribution"
    )
    st.plotly_chart(fig,use_container_width=True)

with col2:
    fig = px.histogram(
        df,
        x="City",
        title="Demand by City"
    )
    st.plotly_chart(fig,use_container_width=True)

fig = px.box(
    df,
    x="LegalIssue",
    y="ConsultBudget",
    title="Budget Distribution by Legal Issue"
)

st.plotly_chart(fig,use_container_width=True)

st.info("""
Family and business law generate the highest consultation demand.
Cities like Ulaanbaatar dominate legal service usage.
""")
