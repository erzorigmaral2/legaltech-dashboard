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

# Convert numeric columns safely
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows where clustering features are missing
df = df.dropna(subset=["ConsultBudget", "UrgencyScore"])

# -----------------------------
# BUDGET VS URGENCY ANALYSIS
# -----------------------------

st.subheader("Budget vs Urgency Analysis")

fig = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color="LegalIssue",
    title="Legal Demand Drivers"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Higher urgency legal issues often involve larger consultation budgets.
This indicates users are willing to pay more for urgent legal help.
""")

# -----------------------------
# USER SEGMENTATION
# -----------------------------

st.subheader("Customer Segmentation")

features = df[["ConsultBudget", "UrgencyScore"]]

# Ensure no invalid values remain
features = features.replace([np.inf, -np.inf], np.nan)
features = features.dropna()

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

segments = kmeans.fit_predict(features)

df = df.loc[features.index]
df["Segment"] = segments

fig2 = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color=df["Segment"].astype(str),
    title="Legal Client Segmentation"
)

st.plotly_chart(fig2, use_container_width=True)

st.info("""
Segment 0 – Budget-sensitive users  
Segment 1 – Urgent legal cases  
Segment 2 – Premium legal clients
""")
