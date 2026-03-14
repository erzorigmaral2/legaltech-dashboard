import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

st.title("User Segmentation (K-Means ML Clustering)")

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

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing values with median
df["ConsultBudget"] = df["ConsultBudget"].fillna(df["ConsultBudget"].median())
df["UrgencyScore"] = df["UrgencyScore"].fillna(df["UrgencyScore"].median())

# -----------------------------
# Prepare Features
# -----------------------------

features = df[["ConsultBudget", "UrgencyScore"]]

# Drop any remaining invalid rows
features = features.dropna()

# Ensure numeric
features = features.astype(float)

# Reset index to keep alignment
features.reset_index(drop=True, inplace=True)

# -----------------------------
# Safety Check
# -----------------------------

if features.shape[0] == 0:
    st.error("No valid rows available for clustering after cleaning.")
    st.stop()

# -----------------------------
# KMeans Clustering
# -----------------------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

segments = kmeans.fit_predict(features)

# Add segments back to dataframe
df = df.loc[features.index]
df["Segment"] = segments

# -----------------------------
# Visualization
# -----------------------------

st.subheader("Customer Segmentation Map")

fig = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color=df["Segment"].astype(str),
    title="Legal Client Segmentation (Budget vs Urgency)",
    labels={
        "ConsultBudget": "Consultation Budget",
        "UrgencyScore": "Legal Urgency Score"
    }
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Segment Profiles
# -----------------------------

st.subheader("Segment Profile Summary")

summary = df.groupby("Segment")[["ConsultBudget", "UrgencyScore"]].mean().round(2)

st.dataframe(summary)

# -----------------------------
# Business Insights
# -----------------------------

st.info("""
Segmentation Insights

Segment 0 → Budget-sensitive users needing basic legal guidance  
Segment 1 → High urgency users who require immediate legal help  
Segment 2 → Premium clients willing to pay for professional legal consultation
""")

st.success("""
Startup Strategy Insight

The platform should create:
• Low-cost automated legal advice for Segment 0  
• Instant lawyer matching for Segment 1  
• Premium consultation packages for Segment 2
""")
