import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.title("Diagnostic Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

cols = ["City","LegalIssue","IncomeLevel","PreferredService"]

enc = LabelEncoder()

for c in cols:
    df[c] = enc.fit_transform(df[c].astype(str))

features = df[cols]

kmeans = KMeans(n_clusters=4, random_state=42,n_init=10)

df["Segment"] = kmeans.fit_predict(features)

fig = px.scatter(
    df,
    x="IncomeLevel",
    y="LegalIssue",
    color=df["Segment"].astype(str),
    title="User Segmentation"
)

st.plotly_chart(fig,use_container_width=True)

st.info("""
Segment 0 – Low income legal advice seekers  
Segment 1 – Family law clients  
Segment 2 – Business legal users  
Segment 3 – Premium legal consultation clients
""")
