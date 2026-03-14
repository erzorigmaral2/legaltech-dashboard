import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

st.title("Diagnostic Analytics")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------

# Convert urgency values
urgency_map = {
    "Low":1,"Medium":3,"High":5,
    "low":1,"medium":3,"high":5
}

df["UrgencyScore"] = df["UrgencyScore"].replace(urgency_map)
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

# Clean budget column
df["ConsultBudget"] = (
    df["ConsultBudget"]
    .astype(str)
    .str.replace("$","",regex=False)
    .str.replace(",","",regex=False)
)

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")

# -----------------------------
# FEATURE MATRIX
# -----------------------------

features = df[["ConsultBudget","UrgencyScore"]].copy()

# remove infinite values
features.replace([np.inf,-np.inf], np.nan, inplace=True)

# drop rows containing NaN
features = features.dropna()

# ensure numeric
features = features.astype(float)

# -----------------------------
# CHECK DATA SIZE
# -----------------------------

if len(features) < 20:
    st.warning("Dataset contains too few valid rows for clustering.")
    st.stop()

# keep matching rows
df_clean = df.loc[features.index].copy()

# -----------------------------
# SCATTER ANALYSIS
# -----------------------------

st.subheader("Legal Demand Drivers")

fig = px.scatter(
    df_clean,
    x="ConsultBudget",
    y="UrgencyScore",
    color="LegalIssue",
    title="Budget vs Urgency for Legal Services"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Customers with urgent legal issues generally show higher willingness to pay.
This suggests emergency legal services could be a profitable segment.
""")

# -----------------------------
# KMEANS SEGMENTATION
# -----------------------------

st.subheader("Customer Segmentation")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

segments = kmeans.fit_predict(features)

df_clean["Segment"] = segments

fig2 = px.scatter(
    df_clean,
    x="ConsultBudget",
    y="UrgencyScore",
    color=df_clean["Segment"].astype(str),
    title="Legal Client Segments"
)

st.plotly_chart(fig2, use_container_width=True)

st.info("""
Segment 0 – Low budget / low urgency users  
Segment 1 – Moderate demand clients  
Segment 2 – High urgency premium clients
""")
