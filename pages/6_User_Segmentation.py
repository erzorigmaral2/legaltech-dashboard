import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

st.title("User Segmentation (K-Means Clustering)")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Ensure Required Columns Exist
# -----------------------------

if "ConsultBudget" not in df.columns or "UrgencyScore" not in df.columns:
    st.error("Dataset must contain 'ConsultBudget' and 'UrgencyScore' columns.")
    st.stop()

# -----------------------------
# Convert Columns Safely
# -----------------------------

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing values instead of dropping rows
df["ConsultBudget"] = df["ConsultBudget"].fillna(df["ConsultBudget"].median())
df["UrgencyScore"] = df["UrgencyScore"].fillna(df["UrgencyScore"].median())

# -----------------------------
# Prepare Features
# -----------------------------

features = df[["ConsultBudget", "UrgencyScore"]].astype(float)

# -----------------------------
# Run K-Means
# -----------------------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Segment"] = kmeans.fit_predict(features)

# -----------------------------
# Visualization
# -----------------------------

st.subheader("Customer Segmentation Map")

fig = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color=df["Segment"].astype(str),
    title="Legal Client Segmentation",
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Segment Summary
# -----------------------------

st.subheader("Segment Profile")

summary = df.groupby("Segment")[["ConsultBudget","UrgencyScore"]].mean().round(2)

st.dataframe(summary)

# -----------------------------
# Insights
# -----------------------------

st.info("""
Segmentation Insights

Segment 0 → Budget-sensitive users  
Segment 1 → High urgency clients needing quick legal help  
Segment 2 → Premium legal consultation clients
""")

st.success("""
Business Strategy

Use segmentation to personalize pricing, lawyer recommendations, and consultation offers.
""")
