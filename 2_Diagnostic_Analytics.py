import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/mongolia_legal_survey_synthetic_2000.csv")

st.title("Diagnostic Analytics")

st.subheader("Main Challenges When Finding Lawyers")

fig = plt.figure()

df["MainDifficulty"].value_counts().plot(kind="bar")

st.pyplot(fig)


st.subheader("Satisfaction vs Pay Likelihood")

grouped = df.groupby("Satisfaction")["PayLikelihood"].mean()

fig = plt.figure()

grouped.plot(kind="bar")

plt.xlabel("Satisfaction Score")
plt.ylabel("Average Pay Likelihood")

st.pyplot(fig)
