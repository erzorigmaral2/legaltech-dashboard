import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Prescriptive Analytics")

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------
# Select usable columns
# -------------------------

candidate_columns = []

for col in df.columns:
    if df[col].dtype == "object":
        candidate_columns.append(col)

if len(candidate_columns) < 2:
    st.error("Dataset does not contain enough categorical columns.")
    st.stop()

data = df[candidate_columns]

# -------------------------
# One-hot encode
# -------------------------

basket = pd.get_dummies(data)

st.write("Encoded feature count:", basket.shape[1])

# -------------------------
# Apriori
# -------------------------

frequent_itemsets = apriori(
    basket,
    min_support=0.02,   # lowered threshold
    use_colnames=True
)

if frequent_itemsets.empty:
    st.warning("No frequent itemsets found.")
    st.stop()

# -------------------------
# Association Rules
# -------------------------

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3
)

if rules.empty:

    # fallback using lift
    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1
    )

if rules.empty:
    st.warning("No association rules discovered.")
    st.stop()

# -------------------------
# Clean rule display
# -------------------------

rules["antecedents"] = rules["antecedents"].astype(str)
rules["consequents"] = rules["consequents"].astype(str)

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

# -------------------------
# Visualization
# -------------------------

st.subheader("Rule Strength Visualization")

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

# -------------------------
# Prescriptive Insights
# -------------------------

st.subheader("Platform Recommendations")

top_rules = rules.head(5)

for i,row in top_rules.iterrows():

    st.write(
        f"If users experience **{row['antecedents']}**, "
        f"they are likely to need **{row['consequents']}** "
        f"(confidence {round(row['confidence'],2)}, lift {round(row['lift'],2)})."
    )
