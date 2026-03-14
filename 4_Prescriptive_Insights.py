import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/mongolia_legal_survey_synthetic_2000.csv")

st.title("Prescriptive Insights")

st.subheader("Trust Factors for Legal Platform")

fig = plt.figure()

df["TrustFactor"].value_counts().plot(kind="bar")

st.pyplot(fig)


st.subheader("Conversion Funnel")

total = len(df)

interested = len(df[df["AppInterest"]=="Yes"])

likely_to_pay = len(df[df["PayLikelihood"]>=4])

funnel = {
    "Legal Need": total,
    "App Interest": interested,
    "Likely to Pay": likely_to_pay
}

fig = plt.figure()

plt.bar(funnel.keys(), funnel.values())

st.pyplot(fig)
