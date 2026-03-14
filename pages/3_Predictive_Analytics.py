import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.title("Predictive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

# Clean data
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["PayLikelihood"] = pd.to_numeric(df["PayLikelihood"], errors="coerce")

df.fillna(df.median(numeric_only=True), inplace=True)

X = df[["UrgencyScore","ConsultBudget"]]
y = (df["PayLikelihood"] >= 4).astype(int)

model = RandomForestClassifier(random_state=42)
model.fit(X,y)

st.markdown("---")

# User inputs
st.subheader("Customer Conversion Prediction")

urgency = st.slider("Urgency Score",1,5,3)
budget = st.slider("Consultation Budget",10000,200000,50000)

input_data = pd.DataFrame({
    "UrgencyScore":[urgency],
    "ConsultBudget":[budget]
})

prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

# Gauge Chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob*100,
    title={'text':"Conversion Probability"},
    gauge={'axis':{'range':[0,100]}}
))

st.plotly_chart(fig)

st.info("""
Insight:

• Higher urgency and larger budgets significantly increase the probability of paid consultations.
""")

st.markdown("---")

# Feature Importance
importance_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})

fig = px.bar(importance_df,
             x="Feature",
             y="Importance",
             title="Feature Importance")

st.plotly_chart(fig)

st.info("""
Insight:

• Urgency score is the strongest predictor of conversion.  
• Marketing efforts should target users with urgent legal issues.
""")

st.markdown("---")

# Legal demand prediction
demand = df["LegalIssue"].value_counts().reset_index()
demand.columns=["LegalIssue","Demand"]

fig = px.bar(demand,
             x="LegalIssue",
             y="Demand",
             title="Predicted Legal Service Demand")

st.plotly_chart(fig)

st.info("""
Insight:

• Civil and family law services show the highest demand.  
• Platform lawyer supply should prioritize these specialties.
""")

st.success("Key Takeaway: Users with urgent legal needs are the most likely to convert into paying clients.")
