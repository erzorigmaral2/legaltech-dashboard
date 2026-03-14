import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.title("Predictive Analytics")

features = ["AgeGroup","Income","LegalIssue","UrgencyScore"]

data = df[features + ["PayLikelihood"]].copy()

encoder = LabelEncoder()

for col in features:
    data[col] = encoder.fit_transform(data[col])

X = data[features]
y = data["PayLikelihood"] >= 4

model = RandomForestClassifier()

model.fit(X,y)

st.subheader("Predict Paying Customer")

age = st.selectbox("AgeGroup", df["AgeGroup"].unique())
income = st.selectbox("Income", df["Income"].unique())
issue = st.selectbox("LegalIssue", df["LegalIssue"].unique())
urgency = st.slider("UrgencyScore",1,5)

input_df = pd.DataFrame([[age,income,issue,urgency]], columns=features)

for col in features[:-1]:
    input_df[col] = encoder.fit_transform(input_df[col])

prediction = model.predict(input_df)

st.write("Prediction (1 = Likely to Pay):")
st.write(prediction)
