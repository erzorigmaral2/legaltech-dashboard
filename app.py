import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="LegalTech Growth Dashboard", layout="wide")

st.title("LegalTech Growth Intelligence Dashboard")

# -------------------------------------------------
# DATA LOADING (cached)
# -------------------------------------------------

@st.cache_data
def load_data():
    users = pd.read_csv("users.csv")
    questions = pd.read_csv("questions.csv")
    consultations = pd.read_csv("consultations.csv")
    lawyers = pd.read_csv("lawyers.csv")
    sessions = pd.read_csv("sessions.csv")
    reviews = pd.read_csv("reviews.csv")
    marketing = pd.read_csv("marketing_spend.csv")
    return users, questions, consultations, lawyers, sessions, reviews, marketing

users, questions, consultations, lawyers, sessions, reviews, marketing = load_data()

paid_consults = consultations[consultations["paid"] == 1]

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

view = st.sidebar.radio(
    "Analytics Layer",
    [
        "Descriptive Analytics",
        "Diagnostic Analytics",
        "Predictive Analytics",
        "Prescriptive Analytics"
    ]
)

# =================================================
# DESCRIPTIVE ANALYTICS
# =================================================

if view == "Descriptive Analytics":

    st.header("Platform KPI Overview")

    total_visitors = len(users)
    legal_questions = len(questions)
    revenue = paid_consults["consultation_price"].sum()
    avg_price = paid_consults["consultation_price"].mean()
    active_lawyers = lawyers["lawyer_id"].nunique()

    repeat_users = consultations["user_id"].value_counts()
    repeat_rate = (repeat_users > 1).sum() / len(repeat_users)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Total Visitors", total_visitors)
    col2.metric("Legal Questions", legal_questions)
    col3.metric("Revenue", f"${revenue:,.0f}")
    col4.metric("Avg Consultation Price", f"${avg_price:,.0f}")
    col5.metric("Active Lawyers", active_lawyers)
    col6.metric("Repeat Consultation Rate", f"{repeat_rate:.2%}")

    st.markdown("---")

    # Traffic source
    traffic = users["traffic_source"].value_counts().reset_index()
    traffic.columns = ["source", "users"]

    traffic_chart = alt.Chart(traffic).mark_bar().encode(
        x="source:N",
        y="users:Q",
        tooltip=["source", "users"]
    ).properties(title="Traffic Source")

    st.altair_chart(traffic_chart, use_container_width=True)

    # Revenue by category
    revenue_cat = paid_consults.groupby("category")["consultation_price"].sum().reset_index()

    rev_chart = alt.Chart(revenue_cat).mark_bar().encode(
        x="category:N",
        y="consultation_price:Q",
        tooltip=["category","consultation_price"]
    ).properties(title="Revenue by Legal Category")

    st.altair_chart(rev_chart, use_container_width=True)

# =================================================
# DIAGNOSTIC ANALYTICS
# =================================================

elif view == "Diagnostic Analytics":

    st.header("Conversion Funnel Analysis")

    visitors = len(users)
    questions_count = len(questions)
    consultations_count = len(consultations)
    paid_count = len(paid_consults)

    funnel = pd.DataFrame({
        "stage": ["Visitors","Questions","Consultations","Paid"],
        "users":[visitors,questions_count,consultations_count,paid_count]
    })

    funnel_chart = alt.Chart(funnel).mark_bar().encode(
        x="stage:N",
        y="users:Q",
        tooltip=["stage","users"]
    ).properties(title="User Conversion Funnel")

    st.altair_chart(funnel_chart,use_container_width=True)

    st.markdown("---")

    st.subheader("User Engagement")

    session_chart = alt.Chart(sessions).mark_circle(size=80).encode(
        x="session_duration_sec:Q",
        y="pages_viewed:Q",
        tooltip=["session_duration_sec","pages_viewed"]
    )

    st.altair_chart(session_chart,use_container_width=True)

    st.markdown("---")

    st.subheader("Lawyer Rating Distribution")

    rating_chart = alt.Chart(reviews).mark_bar().encode(
        alt.X("rating:Q", bin=True),
        y="count()"
    )

    st.altair_chart(rating_chart,use_container_width=True)

# =================================================
# PREDICTIVE ANALYTICS
# =================================================

elif view == "Predictive Analytics":

    st.header("Predictive Models")

    consultations["date"] = pd.to_datetime(consultations["consultation_date"])

    revenue_time = consultations.groupby("date")["consultation_price"].sum().reset_index()

    revenue_time["day_index"] = np.arange(len(revenue_time))

    X = revenue_time[["day_index"]]
    y = revenue_time["consultation_price"]

    model = LinearRegression()
    model.fit(X,y)

    revenue_time["forecast"] = model.predict(X)

    actual = alt.Chart(revenue_time).mark_line().encode(
        x="date:T",
        y="consultation_price:Q"
    )

    forecast = alt.Chart(revenue_time).mark_line(color="red").encode(
        x="date:T",
        y="forecast:Q"
    )

    st.subheader("Revenue Forecast")

    st.altair_chart(actual + forecast,use_container_width=True)

    st.markdown("---")

    st.subheader("Customer Segmentation (CLV vs Frequency)")

    seg = consultations.groupby("user_id").agg({
        "consultation_price":"sum",
        "consultation_id":"count"
    }).reset_index()

    kmeans = KMeans(n_clusters=3,n_init=10)
    seg["segment"] = kmeans.fit_predict(seg[["consultation_price","consultation_id"]])

    seg_chart = alt.Chart(seg).mark_circle(size=80).encode(
        x="consultation_price",
        y="consultation_id",
        color="segment:N",
        tooltip=["user_id"]
    )

    st.altair_chart(seg_chart,use_container_width=True)

# =================================================
# PRESCRIPTIVE ANALYTICS
# =================================================

elif view == "Prescriptive Analytics":

    st.header("Optimization & Recommendations")

    st.subheader("Cost Per Acquisition")

    spend_col = None
    users_col = None

    for col in marketing.columns:
        if "spend" in col.lower():
            spend_col = col
        if "user" in col.lower():
            users_col = col

    if spend_col and users_col:

        marketing["CPA"] = marketing[spend_col] / marketing[users_col]

        cpa_chart = alt.Chart(marketing).mark_bar().encode(
            x="channel:N",
            y="CPA:Q",
            tooltip=["channel","CPA"]
        )

        st.altair_chart(cpa_chart,use_container_width=True)

    else:
        st.warning("Marketing dataset columns not detected.")

    st.markdown("---")

    st.subheader("Lawyer Utilization")

    util = consultations.groupby("lawyer_id").size().reset_index(name="consultations")

    util_chart = alt.Chart(util).mark_bar().encode(
        x="lawyer_id:N",
        y="consultations:Q"
    )

    st.altair_chart(util_chart,use_container_width=True)

    st.markdown("---")

    st.success(
        "Recommended Strategy: Improve lawyer response time and increase SEO traffic to improve consultation conversion."
    )
