import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/mongolia_legal_survey_synthetic_2000.csv")

st.title("Predictive Analytics")

st.subheader("Urgency vs Pay Likelihood")

fig = plt.figure()

plt.scatter(df["UrgencyScore"], df["PayLikelihood"])

plt.xlabel("Urgency Score")
plt.ylabel("Pay Likelihood")

st.pyplot(fig)


st.subheader("Income vs Consultation Budget")

table = pd.crosstab(df["Income"], df["ConsultBudget"])

fig = plt.figure()

table.plot(kind="bar")

st.pyplot(fig)
