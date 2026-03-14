import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Prescriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# Select categorical columns
cat_cols = df.select_dtypes(include="object").columns

data = pd.get_dummies(df[cat_cols])

frequent = apriori(data, min_support=0.02, use_colnames=True)

rules = association_rules(frequent,
                          metric="confidence",
                          min_threshold=0.3)

if rules.empty:

    st.warning("No strong rules found with confidence threshold, showing lift-based rules.")

    rules = association_rules(frequent,
                              metric="lift",
                              min_threshold=1)

rules = rules.sort_values("lift", ascending=False)

st.subheader("Top Association Rules")

st.dataframe(rules[[
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]].head(20))

st.markdown("---")

# Rule visualization
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

st.info("""
Insight:

• Certain legal problems frequently occur with difficulty finding lawyers.  
• Automated lawyer recommendation systems could improve user experience.
""")

st.markdown("---")

# Top rule visualization
fig = px.bar(
    rules.head(10),
    x="lift",
    y="antecedents",
    title="Top Legal Service Associations"
)

st.plotly_chart(fig)

st.info("""
Insight:

• Strong associations indicate predictable legal service patterns.  
• The platform can recommend lawyers based on the user's legal issue.
""")

st.success("Key Takeaway: Data-driven lawyer recommendations can significantly improve legal service accessibility.")
