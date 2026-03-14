import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Predictive Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Ensure required columns exist
required_columns = ["UrgencyScore", "ConsultBudget", "PayLikelihood"]

for col in required_columns:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in dataset.")
        st.stop()

# Convert to numeric safely
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")
df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["PayLikelihood"] = pd.to_numeric(df["PayLikelihood"], errors="coerce")

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with missing values
df = df.dropna(subset=["UrgencyScore", "ConsultBudget", "PayLikelihood"])

# If dataset becomes empty after cleaning
if df.shape[0] == 0:
    st.error("Dataset has no valid rows after cleaning.")
    st.stop()

# Define features and target
X = df[["UrgencyScore", "ConsultBudget"]]
y = df["PayLikelihood"].apply(lambda x: 1 if x >= 4 else 0)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.subheader("Predict Paying Customer")

# User inputs
urgency = st.slider("Urgency Score", 1, 5, 3)
budget = st.slider("Consultation Budget (MNT)", 10000, 200000, 50000)

# Prediction input
input_df = pd.DataFrame({
    "UrgencyScore": [urgency],
    "ConsultBudget": [budget]
})

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Show results
st.write("Prediction Result:")

if prediction == 1:
    st.success("User is likely to pay for consultation.")
else:
    st.warning("User is unlikely to pay.")

st.write(f"Probability of Paying: {round(probability*100,2)}%")
