import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Descriptive Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

# Identify categorical columns automatically
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Identify numeric columns
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()

# -------------------------------
# CATEGORICAL DISTRIBUTION
# -------------------------------

if len(cat_cols) > 0:

    column = st.selectbox(
        "Select category to analyze",
        cat_cols
    )

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(f"""
    This chart shows how users are distributed across **{column}**.
    It helps identify which categories dominate legal demand.
    """)

# -------------------------------
# NUMERIC ANALYSIS
# -------------------------------

if len(num_cols) > 0:

    column = st.selectbox(
        "Select numeric column",
        num_cols
    )

    fig = px.box(
        df,
        y=column,
        title=f"{column} Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(f"""
    This visualization highlights spread and outliers in **{column}** values.
    Useful for understanding consultation budgets or other numeric indicators.
    """)
