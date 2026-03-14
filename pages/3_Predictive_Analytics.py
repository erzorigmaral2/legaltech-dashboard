import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.title("Predictive Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Ensure columns exist
required_columns = ["UrgencyScore","ConsultBudget","PayLikelihood","LegalIssue"]

for col in required_columns:
    if col not in df.columns:
        st.error(f"Column '{col}' missing from dataset.")
        st.stop()

# Clean data
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["PayLikelihood"] = pd.to_numeric(df["PayLikelihood"], errors="coerce")

df.replace([np.inf,-np.inf],np.nan,inplace=True)
df.dropna(inplace=True)

# Features
X = df[["UrgencyScore","ConsultBudget"]]
y = df["PayLikelihood"].apply(lambda x: 1 if x >= 4 else 0)

# Train ML model
model = RandomForestClassifier(random_state=42)
model.fit(X,y)

st.markdown("---")

# ----------------------------
# USER INPUT
# ----------------------------

st.subheader("Customer Conversion Prediction")

urgency = st.slider("Urgency Score",1,5,3)
budget = st.slider("Consultation Budget (MNT)",10000,200000,50000)

input_df = pd.DataFrame({
    "UrgencyScore":[urgency],
    "ConsultBudget":[budget]
})

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# ----------------------------
# GAUGE CHART
# ----------------------------

st.subheader("Prediction Probability Gauge")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability*100,
    title={'text':"Probability Customer Will Pay (%)"},
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

st.plotly_chart(gauge)

# Conversion message
if prediction == 1:
    st.success("High probability of conversion.")
else:
    st.warning("Low probability of conversion.")

st.write(f"Predicted Conversion Probability: {round(probability*100,2)}%")

st.markdown("---")

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------

st.subheader("Feature Importance")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":importance
})

fig = px.bar(
    importance_df,
    x="Feature",
    y="Importance",
    title="Model Feature Importance"
)

st.plotly_chart(fig)

st.markdown("---")

# ----------------------------
# LEGAL SERVICE DEMAND PREDICTION
# ----------------------------

st.subheader("Legal Service Demand Forecast")

demand = df["LegalIssue"].value_counts().reset_index()
demand.columns=["LegalIssue","Demand"]

fig2 = px.bar(
    demand,
    x="LegalIssue",
    y="Demand",
    title="Predicted Legal Service Demand by Issue"
)

st.plotly_chart(fig2)

st.markdown("---")

# ----------------------------
# CUSTOMER SEGMENT SIMULATION
# ----------------------------

st.subheader("Customer Conversion Simulation")

sim_urgency = st.slider("Simulated Urgency",1,5,3,key="sim1")
sim_budget = st.slider("Simulated Budget",10000,200000,50000,key="sim2")

sim_input = pd.DataFrame({
    "UrgencyScore":[sim_urgency],
    "ConsultBudget":[sim_budget]
})

sim_prob = model.predict_proba(sim_input)[0][1]

st.write("Simulated Conversion Probability:",round(sim_prob*100,2),"%")
