import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

st.title("User Segmentation – Legal Client Clustering")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Convert Columns Safely
# -----------------------------

for col in df.columns:

    if df[col].dtype == "object":
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

# -----------------------------
# Replace Infinite Values
# -----------------------------

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# -----------------------------
# Fill Missing Values
# -----------------------------

df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# Select Features for Clustering
# -----------------------------

possible_features = ["ConsultBudget", "UrgencyScore"]

features = []

for col in possible_features:
    if col in df.columns:
        features.append(col)

if len(features) == 0:
    st.error("No valid clustering features found in dataset.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

X = df[features]

# Final safety cleaning
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()
X = X.astype(float)

if len(X) == 0:
    st.error("No valid rows available for clustering.")
    st.stop()

# -----------------------------
# K-Means Clustering
# -----------------------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

segments = kmeans.fit_predict(X)

# Align index
df = df.loc[X.index]
df["Segment"] = segments

# -----------------------------
# Visualization
# -----------------------------

st.subheader("Customer Segmentation Map")

fig = px.scatter(
    df,
    x=features[0],
    y=features[1] if len(features) > 1 else features[0],
    color=df["Segment"].astype(str),
    title="Legal Client Segmentation"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Segment Profile
# -----------------------------

st.subheader("Segment Summary")

summary = df.groupby("Segment")[features].mean().round(2)

st.dataframe(summary)

# -----------------------------
# Insights
# -----------------------------

st.info("""
Segmentation Insights

Segment 0 → Budget-sensitive users needing affordable legal advice  
Segment 1 → High urgency clients requiring immediate legal consultation  
Segment 2 → Premium users willing to pay for professional legal services
""")

st.success("""
Business Strategy

Use segmentation to:
• personalize pricing
• recommend suitable lawyers
• optimize legal service marketing campaigns
""")
