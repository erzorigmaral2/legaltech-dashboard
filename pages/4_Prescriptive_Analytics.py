import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

st.title("User Segmentation")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# Convert numeric
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

df.fillna(df.median(numeric_only=True), inplace=True)

features = df[["ConsultBudget","UrgencyScore"]]

kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)

df["Segment"] = kmeans.fit_predict(features)

fig = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color=df["Segment"].astype(str),
    title="Legal Client Segmentation"
)

st.plotly_chart(fig,use_container_width=True)

st.info("""
Insight

Segment 1: budget-sensitive users  
Segment 2: urgent legal cases  
Segment 3: premium clients willing to pay higher consultation fees
""")
