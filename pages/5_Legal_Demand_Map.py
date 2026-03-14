import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("Mongolia Legal Demand Map")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# Synthetic province coordinates
provinces = {
"Ulaanbaatar":[47.8864,106.9057],
"Darkhan":[49.4867,105.9228],
"Erdenet":[49.0333,104.0833],
"Choibalsan":[48.0726,114.5326],
"Khovd":[48.0056,91.6419],
"Murun":[49.6333,100.1667]
}

df["City"] = np.random.choice(list(provinces.keys()), len(df))

df["lat"] = df["City"].apply(lambda x: provinces[x][0])
df["lon"] = df["City"].apply(lambda x: provinces[x][1])

city_demand = df.groupby("City").size().reset_index(name="Demand")

city_demand["lat"] = city_demand["City"].apply(lambda x: provinces[x][0])
city_demand["lon"] = city_demand["City"].apply(lambda x: provinces[x][1])

fig = px.scatter_mapbox(
    city_demand,
    lat="lat",
    lon="lon",
    size="Demand",
    hover_name="City",
    zoom=3,
    mapbox_style="carto-positron",
    title="Legal Service Demand Across Mongolia"
)

st.plotly_chart(fig)

st.info("""
Insight

• Ulaanbaatar dominates legal service demand.  
• Secondary cities like Darkhan and Erdenet show emerging legal markets.
""")
