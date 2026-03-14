import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Prescriptive Analytics")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Select Columns for Rules
# -----------------------------

candidate_columns = [
    "LegalIssue",
    "LawyerDifficulty",
    "OnlineSearch",
    "UrgencyScore",
    "ConsultBudget"
]

available_columns = [c for c in candidate_columns if c in df.columns]

if len(available_columns) < 2:
    st.error("Not enough columns for association rule mining.")
    st.stop()

data = df[available_columns].copy()

# -----------------------------
# Convert to categorical
# -----------------------------

data = data.astype(str)

# One-hot encoding
basket = pd.get_dummies(data)

# -----------------------------
# Apriori Algorithm
# -----------------------------

frequent_itemsets = apriori(
    basket,
    min_support=0.05,
    use_colnames=True
)

if frequent_itemsets.empty:
    st.warning("No frequent itemsets found. Try lowering support.")
    st.stop()

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.4
)

if rules.empty:
    st.warning("No strong association rules discovered.")
    st.stop()

# -----------------------------
# Sort Best Rules
# -----------------------------

rules = rules.sort_values(by="lift", ascending=False)

st.subheader("Top Association Rules")

st.dataframe(
    rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(20)
)

# -----------------------------
# Visualization
# -----------------------------

st.subheader("Association Rule Strength")

rules["antecedents"] = rules["antecedents"].astype(str)
rules["consequents"] = rules["consequents"].astype(str)

fig = px.scatter(
    rules,
    x="support",
    y="confidence",
    size="lift",
    color="lift",
    hover_data=["antecedents","consequents"],
    title="Association Rule Strength"
)

st.plotly_chart(fig)

# -----------------------------
# Prescriptive Insights
# -----------------------------

st.subheader("Prescriptive Recommendations")

top_rules = rules.head(5)

for i,row in top_rules.iterrows():

    st.write(
        f"If users experience **{row['antecedents']}**, "
        f"they are likely to need **{row['consequents']}** "
        f"(confidence {round(row['confidence'],2)})."
    )
