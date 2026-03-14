import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.title("Diagnostic Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# -----------------------------
# FIX URGENCY SCORE
# -----------------------------

urgency_map = {
    "Low":1,
    "Medium":3,
    "High":5,
    "low":1,
    "medium":3,
    "high":5
}

df["UrgencyScore"] = df["UrgencyScore"].replace(urgency_map)

# convert remaining values to numeric
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

# -----------------------------
# FIX CONSULT BUDGET
# -----------------------------

# remove symbols like $ or text
df["ConsultBudget"] = (
    df["ConsultBudget"]
    .astype(str)
    .str.replace("$","",regex=False)
    .str.replace(",","",regex=False)
)

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")

# fill missing values
df["ConsultBudget"].fillna(df["ConsultBudget"].median(), inplace=True)
df["UrgencyScore"].fillna(df["UrgencyScore"].median(), inplace=True)

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
Higher urgency legal problems often require larger consultation budgets.
This indicates strong demand for quick legal services.
""")

# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------

st.subheader("Customer Segmentation")

features = df[["ConsultBudget","UrgencyScore"]]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Segment"] = kmeans.fit_predict(features)

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
