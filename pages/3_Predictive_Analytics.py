import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Predictive Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Convert columns to numeric safely
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["PayLikelihood"] = pd.to_numeric(df["PayLikelihood"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["UrgencyScore", "ConsultBudget", "PayLikelihood"])

# Features and target
features = ["UrgencyScore", "ConsultBudget"]
X = df[features]
y = df["PayLikelihood"] > 3

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.subheader("Predict Paying Customer")

urgency = st.slider("Urgency Score", 1, 5)
budget = st.slider("Consultation Budget", 10000, 200000)

input_data = pd.DataFrame([[urgency, budget]], columns=features)

prediction = model.predict(input_data)

st.write("Will the user likely pay for consultation?")
st.success("Yes" if prediction[0] else "No")
