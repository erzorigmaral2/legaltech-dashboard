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
# Data Cleaning
# -----------------------------

# Convert to numeric safely
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

# Replace invalid values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing values
df["ConsultBudget"].fillna(df["ConsultBudget"].median(), inplace=True)
df["UrgencyScore"].fillna(df["UrgencyScore"].median(), inplace=True)

# -----------------------------
# Prepare Features
# -----------------------------

features = df[["ConsultBudget", "UrgencyScore"]].astype(float)

# -----------------------------
# KMeans Model
# -----------------------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Segment"] = kmeans.fit_predict(features)

# -----------------------------
# Visualization
# -----------------------------

fig = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color=df["Segment"].astype(str),
    title="Customer Segmentation Based on Budget and Urgency"
)

st.plotly_chart(fig)

# -----------------------------
# Segment Summary
# -----------------------------

segment_summary = df.groupby("Segment")[["ConsultBudget","UrgencyScore"]].mean()

st.subheader("Segment Profile")

st.dataframe(segment_summary)

# -----------------------------
# Insights
# -----------------------------

st.info("""
Insight

Segment 0 → Low budget users with moderate urgency  
Segment 1 → High urgency clients needing fast legal help  
Segment 2 → High budget premium legal service users
""")

st.success("""
Key Takeaway

Customer segmentation helps the legal advisor platform target users with personalized services and pricing strategies.
""")
