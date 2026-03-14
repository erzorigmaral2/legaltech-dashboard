import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/mongolia_legal_survey_synthetic_2000.csv")

st.title("Descriptive Analytics")

st.subheader("Legal Issue Distribution")

fig = plt.figure()

df["LegalIssue"].value_counts().plot(kind="bar")

plt.xlabel("Legal Issue")
plt.ylabel("Respondents")

st.pyplot(fig)


st.subheader("Age Distribution")

fig = plt.figure()

df["AgeGroup"].value_counts().plot(kind="bar")

plt.xlabel("Age Group")
plt.ylabel("Respondents")

st.pyplot(fig)


st.subheader("Consultation Preference")

fig = plt.figure()

df["PreferredConsultation"].value_counts().plot(kind="bar")

st.pyplot(fig)
