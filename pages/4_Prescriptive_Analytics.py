import streamlit as st
import pandas as pd

st.title("AI Lawyer Recommendation Engine")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

issue = st.selectbox(
    "Select Legal Issue",
    df["LegalIssue"].unique()
)

budget = st.slider(
    "Consultation Budget",
    50,1000,200
)

lawyers = {
"Family":["Bat-Erdene Law Firm","UB Family Legal"],
"Business":["Mongol Legal Partners","Steppe Corporate Law"],
"Criminal":["Justice Advocates","UB Criminal Defense"],
"Property":["Land Rights Law Group"]
}

rec = lawyers.get(issue,["General Legal Advisor"])

st.subheader("Recommended Lawyers")

for r in rec:
    st.write(f"⚖️ {r}")

st.info("""
Recommendation is based on legal issue type and consultation budget.
Premium budgets unlock access to specialized legal experts.
""")
