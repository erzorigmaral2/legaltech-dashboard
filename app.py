import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="LegalTech Growth Dashboard", layout="wide")

st.title("LegalTech Growth Intelligence Dashboard")

# Load data directly from repository root
users = pd.read_csv("users.csv")
questions = pd.read_csv("questions.csv")
consultations = pd.read_csv("consultations.csv")
lawyers = pd.read_csv("lawyers.csv")
sessions = pd.read_csv("sessions.csv")
reviews = pd.read_csv("reviews.csv")
marketing = pd.read_csv("marketing_spend.csv")

paid_consults = consultations[consultations["paid"] == 1]

# KPIs
total_users = len(users)
total_questions = len(questions)
paid_consultations = len(paid_consults)
revenue = paid_consults["consultation_price"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Users", total_users)
col2.metric("Questions Submitted", total_questions)
col3.metric("Paid Consultations", paid_consultations)
col4.metric("Revenue", f"${revenue:,.0f}")

st.markdown("---")

# Revenue by category
revenue_category = (
    paid_consults.groupby("category")["consultation_price"]
    .sum()
    .reset_index()
)

chart = alt.Chart(revenue_category).mark_bar().encode(
    x="category:N",
    y="consultation_price:Q",
    tooltip=["category", "consultation_price"]
).properties(
    title="Revenue by Legal Category"
)

st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# Traffic source analysis
traffic = users["traffic_source"].value_counts().reset_index()
traffic.columns = ["traffic_source", "users"]

traffic_chart = alt.Chart(traffic).mark_bar().encode(
    x="traffic_source:N",
    y="users:Q",
    tooltip=["traffic_source", "users"]
).properties(
    title="User Acquisition by Traffic Source"
)

st.altair_chart(traffic_chart, use_container_width=True)

st.markdown("---")

# Session duration distribution
session_chart = alt.Chart(sessions).mark_bar().encode(
    alt.X("session_duration_sec", bin=True),
    y="count()"
).properties(
    title="Session Duration Distribution"
)

st.altair_chart(session_chart, use_container_width=True)

st.write("Dashboard ready for GitHub + Streamlit deployment.")
