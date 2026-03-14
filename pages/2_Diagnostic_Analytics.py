import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans

st.title("Diagnostic Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows with missing clustering features
df_clean = df.dropna(subset=["ConsultBudget", "UrgencyScore"])

# -----------------------------
# BUDGET VS URGENCY ANALYSIS
# -----------------------------

st.subheader("Budget vs Urgency Analysis")

fig = px.scatter(
    df_clean,
    x="ConsultBudget",
    y="UrgencyScore",
    color="LegalIssue",
    title="Legal Demand Drivers"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Higher urgency legal problems tend to involve larger consultation budgets.
Users are willing to pay more for urgent legal help.
""")

# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------

st.subheader("Customer Segmentation")

features = df_clean[["ConsultBudget", "UrgencyScore"]]

# Safety check
if len(features) < 5:

    st.warning("Not enough clean data available for clustering.")

else:

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    df_clean["Segment"] = kmeans.fit_predict(features)

    fig2 = px.scatter(
        df_clean,
        x="ConsultBudget",
        y="UrgencyScore",
        color=df_clean["Segment"].astype(str),
        title="Legal Client Segmentation"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.info("""
    Segment 0 – Budget sensitive users  
    Segment 1 – Urgent legal cases  
    Segment 2 – Premium legal clients
    """)
