import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.title("Predictive Analytics")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Ensure Required Columns
# -----------------------------

required = ["UrgencyScore","ConsultBudget","PayLikelihood"]

for col in required:
    if col not in df.columns:
        st.error(f"Column missing: {col}")
        st.stop()

# -----------------------------
# Convert Data Safely
# -----------------------------

df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["PayLikelihood"] = pd.to_numeric(df["PayLikelihood"], errors="coerce")

# Replace invalid values with median instead of deleting rows
df["UrgencyScore"].fillna(df["UrgencyScore"].median(), inplace=True)
df["ConsultBudget"].fillna(df["ConsultBudget"].median(), inplace=True)
df["PayLikelihood"].fillna(df["PayLikelihood"].median(), inplace=True)

# Replace infinite values
df.replace([np.inf,-np.inf],np.nan,inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# Prepare ML Data
# -----------------------------

X = df[["UrgencyScore","ConsultBudget"]]
y = (df["PayLikelihood"] >= 4).astype(int)

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestClassifier(random_state=42)
model.fit(X,y)

st.markdown("---")

# -----------------------------
# Prediction Input
# -----------------------------

st.subheader("Customer Conversion Prediction")

urgency = st.slider("Urgency Score",1,5,3)
budget = st.slider("Consultation Budget (MNT)",10000,200000,50000)

input_data = pd.DataFrame({
    "UrgencyScore":[urgency],
    "ConsultBudget":[budget]
})

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# -----------------------------
# Probability Gauge
# -----------------------------

st.subheader("Conversion Probability Gauge")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability*100,
    title={'text':"Customer Conversion Probability (%)"},
    gauge={
        'axis':{'range':[0,100]},
        'bar':{'color':"green"},
        'steps':[
            {'range':[0,40],'color':"lightgray"},
            {'range':[40,70],'color':"yellow"},
            {'range':[70,100],'color':"lightgreen"}
        ]
    }
))

st.plotly_chart(fig)

# -----------------------------
# Prediction Result
# -----------------------------

if prediction == 1:
    st.success("User likely to pay for legal consultation.")
else:
    st.warning("User unlikely to pay.")

st.write(f"Conversion Probability: {round(probability*100,2)}%")

st.markdown("---")

# -----------------------------
# Feature Importance
# -----------------------------

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})

fig2 = px.bar(
    importance,
    x="Feature",
    y="Importance",
    title="Feature Importance for Conversion Prediction"
)

st.plotly_chart(fig2)

# -----------------------------
# Legal Demand Insights
# -----------------------------

if "LegalIssue" in df.columns:

    st.subheader("Legal Service Demand")

    demand = df["LegalIssue"].value_counts().reset_index()
    demand.columns=["LegalIssue","Demand"]

    fig3 = px.bar(
        demand,
        x="LegalIssue",
        y="Demand",
        title="Demand for Legal Services"
    )

    st.plotly_chart(fig3)
