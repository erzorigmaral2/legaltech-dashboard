import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

st.title("Prescriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

data = pd.get_dummies(df[
    ["MainDifficulty","PreferredConsultation","TrustFactor"]
])

freq = apriori(data, min_support=0.05, use_colnames=True)

rules = association_rules(freq, metric="confidence", min_threshold=0.5)

st.subheader("Top Association Rules")

st.dataframe(rules.head())

fig = px.scatter(
    rules,
    x="confidence",
    y="lift",
    hover_data=["antecedents","consequents"]
)

st.plotly_chart(fig)
