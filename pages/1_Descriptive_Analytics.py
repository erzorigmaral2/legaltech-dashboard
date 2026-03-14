import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Descriptive Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# -----------------------------
# Data Cleaning
# -----------------------------

# Convert numeric columns safely
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

# Fill missing values
df["ConsultBudget"].fillna(df["ConsultBudget"].median(), inplace=True)
df["UrgencyScore"].fillna(df["UrgencyScore"].median(), inplace=True)

# -----------------------------
# Dataset Preview
# -----------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# KPI Metrics
# -----------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Respondents", len(df))
col2.metric("Unique Legal Issues", df["LegalIssue"].nunique())
col3.metric("Average Budget", round(df["ConsultBudget"].mean(), 2))

st.markdown("---")

# -----------------------------
# Legal Issue Distribution
# -----------------------------

fig = px.pie(
    df,
    names="LegalIssue",
    title="Distribution of Legal Issues"
)

st.plotly_chart(fig)

st.info("""
Insight:

• Family and civil disputes represent a major portion of legal demand.
• A digital legal advisor platform should prioritize lawyers in these areas.
""")

st.markdown("---")

# -----------------------------
# Consultation Budget Distribution
# -----------------------------

fig = px.histogram(
    df,
    x="ConsultBudget",
    nbins=30,
    title="Legal Consultation Budget Distribution"
)

st.plotly_chart(fig)

st.info("""
Insight:

• Most users fall within the mid-range consultation budget.
• Affordable online legal consultation packages could attract more users.
""")

st.markdown("---")

# -----------------------------
# Urgency Analysis
# -----------------------------

fig = px.box(
    df,
    y="UrgencyScore",
    title="Urgency Level of Legal Issues"
)

st.plotly_chart(fig)

st.info("""
Insight:

• Many respondents report urgent legal needs.
• Fast lawyer matching and instant consultation features would improve platform adoption.
""")

st.success("""
Key Takeaway

The Mongolian legal services market shows strong demand for affordable and fast legal consultation services.
""")
