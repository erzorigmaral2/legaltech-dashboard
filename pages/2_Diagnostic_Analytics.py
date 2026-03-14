import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.title("Diagnostic Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.write("Dataset preview")
st.dataframe(df.head())

# ---------------------------------
# AUTOMATIC FEATURE SELECTION
# ---------------------------------

# find categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

if len(cat_cols) < 2:
    st.error("Not enough categorical columns for clustering.")
    st.stop()

# encode categorical columns
enc = LabelEncoder()

encoded_df = pd.DataFrame()

for c in cat_cols:
    encoded_df[c] = enc.fit_transform(df[c].astype(str))

# ---------------------------------
# KMEANS CLUSTERING
# ---------------------------------

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

df["Segment"] = kmeans.fit_predict(encoded_df)

# ---------------------------------
# VISUALIZATION
# ---------------------------------

x_axis = cat_cols[0]
y_axis = cat_cols[1]

fig = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color=df["Segment"].astype(str),
    title="User Segmentation"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
This segmentation groups users based on their legal service behavior.
Clusters reveal different customer profiles in the legal market.
""")
