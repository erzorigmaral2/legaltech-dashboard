import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.title("Predictive Analytics")

# -------------------------
# Load Dataset
# -------------------------

try:
    df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")
except:
    st.error("Dataset could not be loaded.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------
# Validate Columns
# -------------------------

required_columns = ["UrgencyScore", "ConsultBudget", "PayLikelihood", "LegalIssue"]

missing = [col for col in required_columns if col not in df.columns]

if len(missing) > 0:
    st.error(f"Missing columns in dataset: {missing}")
    st.stop()

# -------------------------
# Clean Data
# -------------------------

df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["PayLikelihood"] = pd.to_numeric(df["PayLikelihood"], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df.dropna(subset=["UrgencyScore", "ConsultBudget", "PayLikelihood"])

# Ensure dataset still has rows
if df.shape[0] == 0:
    st.error("No valid rows available after cleaning.")
    st.stop()

# -------------------------
# Train ML Model
# -------------------------

X = df[["UrgencyScore", "ConsultBudget"]].astype(float)
y = (df["PayLikelihood"] >= 4).astype(int)

model = RandomForestClassifier(random_state=42)

try:
    model.fit(X, y)
except:
    st.error("Model training failed due to invalid data.")
    st.stop()

st.markdown("---")

# -------------------------
# User Input
# -------------------------

st.subheader("Customer Conversion Prediction")

urgency = st.slider("Urgency Score", 1, 5, 3)
budget = st.slider("Consultation Budget (MNT)", 10000, 200000, 50000)

input_df = pd.DataFrame({
    "UrgencyScore": [float(urgency)],
    "ConsultBudget": [float(budget)]
})

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# -------------------------
# Probability Gauge
# -------------------------

st.subheader("Prediction Probability Gauge")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100,
    title={'text': "Customer Conversion Probability (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 40], 'color': "lightgray"},
            {'range': [40, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "lightgreen"}
        ]
    }
))

st.plotly_chart(fig)

# -------------------------
# Prediction Message
# -------------------------

if prediction == 1:
    st.success("User likely to pay for consultation.")
else:
    st.warning("User unlikely to pay.")

st.write(f"Predicted Probability: {round(probability * 100, 2)}%")

st.markdown("---")

# -------------------------
# Feature Importance
# -------------------------

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

fig2 = px.bar(
    importance_df,
    x="Feature",
    y="Importance",
    title="Model Feature Importance"
)

st.plotly_chart(fig2)

st.markdown("---")

# -------------------------
# Legal Service Demand
# -------------------------

st.subheader("Legal Service Demand Forecast")

demand_df = df["LegalIssue"].value_counts().reset_index()
demand_df.columns = ["LegalIssue", "Demand"]

fig3 = px.bar(
    demand_df,
    x="LegalIssue",
    y="Demand",
    title="Legal Service Demand Distribution"
)

st.plotly_chart(fig3)
