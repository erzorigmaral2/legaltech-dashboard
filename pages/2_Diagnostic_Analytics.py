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

# Convert required columns to numeric
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing numeric values
df["ConsultBudget"] = df["ConsultBudget"].fillna(df["ConsultBudget"].median())
df["UrgencyScore"] = df["UrgencyScore"].fillna(df["UrgencyScore"].median())

# Ensure numeric type
df["ConsultBudget"] = df["ConsultBudget"].astype(float)
df["UrgencyScore"] = df["UrgencyScore"].astype(float)

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
Users with urgent legal problems typically have higher consultation budgets.
This suggests demand for fast-response legal services.
""")

# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------

st.subheader("Customer Segmentation")

# Build feature matrix
features = df[["ConsultBudget", "UrgencyScore"]].copy()

# Final safety cleaning
features = features.replace([np.inf, -np.inf], np.nan)
features = features.fillna(features.median())

# Convert to numpy
X = features.values

# Run clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

segments = kmeans.fit_predict(X)

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
Segment 1 – Urgent legal clients  
Segment 2 – Premium consultation users
""")
