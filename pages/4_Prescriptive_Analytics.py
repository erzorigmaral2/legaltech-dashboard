import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Prescriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

data = pd.get_dummies(df[[
    "MainDifficulty",
    "PreferredConsultation"
]])

freq = apriori(data, min_support=0.05, use_colnames=True)

rules = association_rules(freq, metric="confidence", min_threshold=0.5)

st.subheader("Top Association Rules")

st.dataframe(rules.head())
