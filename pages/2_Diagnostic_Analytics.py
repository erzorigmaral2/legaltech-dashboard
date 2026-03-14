import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Diagnostic Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Overview")
st.dataframe(df.head())

st.markdown("---")

# Difficulty Finding Lawyers
if "LawyerDifficulty" in df.columns:

    fig = px.histogram(df,
                       x="LawyerDifficulty",
                       title="Difficulty in Finding Lawyers")

    st.plotly_chart(fig)

    st.info("""
Insight:

• Many respondents report difficulty locating reliable lawyers.  
• This indicates a gap in the legal services market.
""")

st.markdown("---")

# Budget vs Legal Issue
fig = px.box(df,
             x="LegalIssue",
             y="ConsultBudget",
             title="Consultation Budget by Legal Issue")

st.plotly_chart(fig)

st.info("""
Insight:

• Corporate and contract disputes tend to have higher consultation budgets.  
• These segments could generate higher platform revenue.
""")

st.markdown("---")

# Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

corr = numeric_df.corr()

fig = px.imshow(corr,
                text_auto=True,
                title="Correlation Between Legal Variables")

st.plotly_chart(fig)

st.info("""
Insight:

• Urgency and budget are positively correlated.  
• Users with urgent cases are more willing to pay for legal consultation.
""")

st.success("Key Takeaway: Difficulty finding lawyers and urgency are key drivers of legal service demand.")
