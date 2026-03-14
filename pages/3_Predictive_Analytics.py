import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Predictive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

features = ["UrgencyScore", "ConsultBudget"]
target = "PayLikelihood"

df = df.dropna()

X = df[features]
y = df[target] > 3

model = RandomForestClassifier()

model.fit(X, y)

st.subheader("Predict Paying Customer")

urgency = st.slider("Urgency Score", 1, 5)
budget = st.slider("Consultation Budget", 10000, 200000)

prediction = model.predict([[urgency, budget]])

st.write("Will the user likely pay for consultation?")
st.write(prediction[0])
