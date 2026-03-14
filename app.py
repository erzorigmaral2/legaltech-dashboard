import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="LegalTech Analytics Platform",
    layout="wide"
)

st.title("⚖️ LegalTech Intelligence Dashboard")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

# Detect columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()

# KPI Header
col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Records", len(df))

if len(cat_cols) > 0:
    col2.metric("Categorical Variables", len(cat_cols))
else:
    col2.metric("Categorical Variables", 0)

if len(num_cols) > 0:
    col3.metric("Numeric Variables", len(num_cols))
else:
    col3.metric("Numeric Variables", 0)

if len(num_cols) > 0:
    col4.metric("Average Numeric Value", round(df[num_cols[0]].mean(),2))
else:
    col4.metric("Average Numeric Value", "N/A")

st.markdown("---")

st.markdown("""
### LegalTech Analytics Capabilities

This dashboard demonstrates the **four levels of analytics** used in modern platforms:

1️⃣ Descriptive Analytics – understanding legal demand  
2️⃣ Diagnostic Analytics – identifying user segments  
3️⃣ Predictive Analytics – machine learning models  
4️⃣ Prescriptive Analytics – AI legal service recommendations
""")
