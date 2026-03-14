import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.title("User Segmentation")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

features = df[["UrgencyScore","ConsultBudget"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["Segment"] = kmeans.fit_predict(features)

fig = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color="Segment",
    title="Customer Segments"
)

st.plotly_chart(fig)

st.info("""
Insight

Segment 0: Low budget casual users  
Segment 1: High urgency clients  
Segment 2: High value legal customers
""")
