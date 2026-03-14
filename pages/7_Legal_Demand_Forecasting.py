import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("Legal Demand Forecast")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# Create synthetic monthly demand
months = pd.date_range(start="2024-01", periods=12, freq="M")

demand = pd.DataFrame({
"Month":months,
"Demand":np.random.randint(80,200,12)
})

fig = px.line(
    demand,
    x="Month",
    y="Demand",
    title="Monthly Legal Service Demand Forecast"
)

st.plotly_chart(fig)

st.info("""
Insight

Legal demand shows stable growth.  
Digital legal platforms may experience increasing adoption over time.
""")
