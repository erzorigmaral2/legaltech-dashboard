import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Prescriptive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"],errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"],errors="coerce")

df.fillna(df.median(numeric_only=True),inplace=True)

def recommendation(row):

    if row["ConsultBudget"] > 200 and row["UrgencyScore"] > 4:
        return "Premium Lawyer"

    elif row["UrgencyScore"] > 4:
        return "Fast Online Consultation"

    else:
        return "Standard Legal Advice"

df["Recommendation"] = df.apply(recommendation,axis=1)

fig = px.histogram(
    df,
    x="Recommendation",
    title="Recommended Legal Services"
)

st.plotly_chart(fig,use_container_width=True)

st.info("""
Strategy Recommendation

• Premium legal services for high budget urgent users  
• Fast online consultations for urgent issues  
• Standard advice packages for low urgency cases
""")
