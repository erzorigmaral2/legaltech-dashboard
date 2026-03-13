import streamlit as st
import pandas as pd
import altair as alt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="LegalTech Growth Intelligence Dashboard", layout="wide")

st.title("LegalTech Growth Intelligence Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------

users = pd.read_csv("users.csv")
questions = pd.read_csv("questions.csv")
consultations = pd.read_csv("consultations.csv")
lawyers = pd.read_csv("lawyers.csv")
sessions = pd.read_csv("sessions.csv")
reviews = pd.read_csv("reviews.csv")
marketing = pd.read_csv("marketing_spend.csv")

paid_consults = consultations[consultations["paid"] == 1]

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------

view = st.sidebar.radio(
    "Analytics View",
    [
        "Descriptive Analytics",
        "Diagnostic Analytics",
        "Predictive Analytics",
        "Prescriptive Analytics"
    ]
)

# =========================================================
# DESCRIPTIVE ANALYTICS
# =========================================================

if view == "Descriptive Analytics":

    st.header("Descriptive Analytics — Platform Performance")

    total_visitors = len(users)
    new_users = users["signup_date"].count()
    legal_questions = len(questions)
    revenue = paid_consults["consultation_price"].sum()
    avg_price = paid_consults["consultation_price"].mean()
    active_lawyers = lawyers["lawyer_id"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Visitors", total_visitors)
    col2.metric("New Users", new_users)
    col3.metric("Legal Questions", legal_questions)
    col4.metric("Revenue", f"${revenue:,.0f}")
    col5.metric("Active Lawyers", active_lawyers)

    st.markdown("---")

    # Traffic source
    traffic = users["traffic_source"].value_counts().reset_index()
    traffic.columns = ["source", "users"]

    chart = alt.Chart(traffic).mark_bar().encode(
        x="source:N",
        y="users:Q",
        tooltip=["source", "users"]
    ).properties(title="User Acquisition by Traffic Source")

    st.altair_chart(chart, use_container_width=True)

    # Revenue by category
    revenue_category = (
        paid_consults.groupby("category")["consultation_price"]
        .sum()
        .reset_index()
    )

    chart2 = alt.Chart(revenue_category).mark_bar().encode(
        x="category:N",
        y="consultation_price:Q",
        tooltip=["category", "consultation_price"]
    ).properties(title="Revenue by Legal Category")

    st.altair_chart(chart2, use_container_width=True)

# =========================================================
# DIAGNOSTIC ANALYTICS
# =========================================================

elif view == "Diagnostic Analytics":

    st.header("Diagnostic Analytics — Conversion Funnel")

    visitors = len(users)
    questions_count = len(questions)
    consultations_count = len(consultations)
    paid_count = len(paid_consults)

    funnel = pd.DataFrame({
        "stage": ["Visitors", "Questions", "Consultations", "Paid Consultations"],
        "users": [visitors, questions_count, consultations_count, paid_count]
    })

    funnel_chart = alt.Chart(funnel).mark_bar().encode(
        x="stage",
        y="users",
        tooltip=["stage", "users"]
    ).properties(title="Conversion Funnel")

    st.altair_chart(funnel_chart, use_container_width=True)

    st.markdown("---")

    # Session duration impact
    chart = alt.Chart(sessions).mark_circle(size=60).encode(
        x="session_duration_sec",
        y="pages_viewed",
        tooltip=["session_duration_sec", "pages_viewed"]
    ).properties(title="User Engagement Behavior")

    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # Lawyer rating distribution
    rating_chart = alt.Chart(reviews).mark_bar().encode(
        alt.X("rating", bin=True),
        y="count()"
    ).properties(title="Lawyer Rating Distribution")

    st.altair_chart(rating_chart, use_container_width=True)

# =========================================================
# PREDICTIVE ANALYTICS
# =========================================================

elif view == "Predictive Analytics":

    st.header("Predictive Analytics")

    st.subheader("Revenue Forecast")

    consultations["date"] = pd.to_datetime(consultations["consultation_date"])

    revenue_time = consultations.groupby("date")["consultation_price"].sum().reset_index()

    revenue_time["day"] = np.arange(len(revenue_time))

    X = revenue_time[["day"]]
    y = revenue_time["consultation_price"]

    model = LinearRegression()
    model.fit(X, y)

    revenue_time["prediction"] = model.predict(X)

    chart = alt.Chart(revenue_time).mark_line().encode(
        x="date:T",
        y="consultation_price:Q"
    )

    pred = alt.Chart(revenue_time).mark_line(color="red").encode(
        x="date:T",
        y="prediction:Q"
    )

    st.altair_chart(chart + pred, use_container_width=True)

    st.markdown("---")

    st.subheader("Customer Segmentation")

    seg_data = consultations.groupby("user_id").agg({
        "consultation_price": "sum",
        "consultation_id": "count"
    }).reset_index()

    kmeans = KMeans(n_clusters=3)
    seg_data["segment"] = kmeans.fit_predict(seg_data[["consultation_price", "consultation_id"]])

    chart = alt.Chart(seg_data).mark_circle(size=60).encode(
        x="consultation_price",
        y="consultation_id",
        color="segment:N",
        tooltip=["user_id"]
    ).properties(title="Customer Segmentation (CLV vs Frequency)")

    st.altair_chart(chart, use_container_width=True)

# =========================================================
# PRESCRIPTIVE ANALYTICS
# =========================================================

elif view == "Prescriptive Analytics":

    st.header("Prescriptive Analytics")

    st.subheader("Marketing ROI")

    marketing["CPA"] = marketing["spend"] / marketing["users_acquired"]

    chart = alt.Chart(marketing).mark_bar().encode(
        x="channel",
        y="CPA",
        tooltip=["channel", "CPA"]
    ).properties(title="Cost per Acquisition by Channel")

    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    st.subheader("Lawyer Utilization")

    util = consultations.groupby("lawyer_id").size().reset_index(name="consultations")

    chart = alt.Chart(util).mark_bar().encode(
        x="lawyer_id:N",
        y="consultations:Q"
    ).properties(title="Lawyer Utilization Rate")

    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    st.success(
        "Recommendation: Increase SEO investment and improve response time to increase consultation conversion."
    )
