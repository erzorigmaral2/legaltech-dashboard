import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans

st.title("Diagnostic Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

df.fillna(df.median(numeric_only=True), inplace=True)

st.subheader("Budget vs Urgency Analysis")

fig = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color="LegalIssue"
)

st.plotly_chart(fig,use_container_width=True)

st.info("Higher urgency legal issues typically involve larger consultation budgets.")

st.subheader("User Segmentation")

features = df[["ConsultBudget","UrgencyScore"]]

kmeans = KMeans(n_clusters=3,n_init=10,random_state=42)

df["Segment"] = kmeans.fit_predict(features)

fig2 = px.scatter(
    df,
    x="ConsultBudget",
    y="UrgencyScore",
    color=df["Segment"].astype(str),
    title="Client Segmentation"
)

st.plotly_chart(fig2,use_container_width=True)

st.info("""
Segment 0: Budget sensitive users  
Segment 1: Urgent legal cases  
Segment 2: Premium legal clients
""")
