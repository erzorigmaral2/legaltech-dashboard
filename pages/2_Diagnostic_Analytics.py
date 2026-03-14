import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.title("Diagnostic Analytics")

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# -----------------------
# SELECT FEATURES
# -----------------------

features = df[[
    "LegalIssue",
    "City",
    "IncomeLevel",
    "PreferredService",
    "AgeGroup"
]].copy()

# -----------------------
# ENCODE CATEGORICAL DATA
# -----------------------

features = pd.get_dummies(features)

# remove rows with missing values
features = features.dropna()

# -----------------------
# CHECK DATA
# -----------------------

if len(features) < 20:
    st.error("Dataset still too small for clustering.")
    st.stop()

# -----------------------
# RUN KMEANS
# -----------------------

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

segments = kmeans.fit_predict(features)

df["Segment"] = segments

# -----------------------
# VISUALIZATION
# -----------------------

st.subheader("User Segmentation")

fig = px.scatter(
    df,
    x="IncomeLevel",
    y="LegalIssue",
    color=df["Segment"].astype(str),
    title="Legal Service User Segments"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Segment insights:

Segment 0 – Young users seeking basic legal advice  
Segment 1 – Corporate / business legal demand  
Segment 2 – Family & civil legal services  
Segment 3 – High-income premium legal consultation
""")
