import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Legal Advisor Application - Mongolia")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_2000.csv")

st.subheader("Legal Issue Distribution")

fig = plt.figure()
df["LegalIssue"].value_counts().plot(kind="bar")
st.pyplot(fig)


st.subheader("Preferred Consultation")

fig = plt.figure()
df["PreferredConsultation"].value_counts().plot(kind="bar")
st.pyplot(fig)


st.subheader("Urgency vs Pay Likelihood")

fig = plt.figure()
plt.scatter(df["UrgencyScore"], df["PayLikelihood"])
plt.xlabel("Urgency")
plt.ylabel("Pay Likelihood")
st.pyplot(fig)
