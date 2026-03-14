import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

st.title("Diagnostic Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# -----------------------------
# SAFE FEATURE PREPARATION
# -----------------------------

# Extract clustering columns
features = df[["ConsultBudget", "UrgencyScore"]].copy()

# Convert to numeric safely
features["ConsultBudget"] = pd.to_numeric(features["ConsultBudget"], errors="coerce")
features["UrgencyScore"] = pd.to_numeric(features["UrgencyScore"], errors="coerce")

# Remove infinite values
features.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with invalid values
features = features.dropna()

# If too few rows remain, warn instead of crashing
if len(features) < 10:
    st.warning("Not enough valid numeric data for clustering.")
else:

    # -----------------------------
    # BUDGET VS URGENCY SCATTER
    # -----------------------------

    st.subheader("Budget vs Urgency Analysis")

    plot_df = df.loc[features.index]

    fig = px.scatter(
        plot_df,
        x="ConsultBudget",
        y="UrgencyScore",
        color="LegalIssue",
        title="Legal Demand Drivers"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    Users with urgent legal problems often allocate higher consultation budgets.
    This indicates willingness to pay for fast legal support.
    """)

    # -----------------------------
    # CUSTOMER SEGMENTATION
    # -----------------------------

    st.subheader("Customer Segmentation")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    segments = kmeans.fit_predict(features)

    plot_df["Segment"] = segments

    fig2 = px.scatter(
        plot_df,
        x="ConsultBudget",
        y="UrgencyScore",
        color=plot_df["Segment"].astype(str),
        title="Legal Client Segmentation"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.info("""
    Segment 0 – Budget-sensitive users  
    Segment 1 – Urgent legal clients  
    Segment 2 – Premium consultation users
    """)
