import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="LegalTech Growth Intelligence Dashboard", layout="wide")

st.title("LegalTech Growth Intelligence Dashboard")

# ======================================================
# utils/load_data.py
# ======================================================

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

# ======================================================
# utils/metrics.py
# ======================================================

def compute_metrics():

    m = {}

    m["total_visitors"] = len(users)
    m["new_users"] = users["signup_date"].count()
    m["questions"] = len(questions)
    m["consultations"] = len(consultations)
    m["paid"] = len(paid_consults)

    m["revenue"] = paid_consults["consultation_price"].sum()
    m["avg_price"] = paid_consults["consultation_price"].mean()

    m["active_lawyers"] = lawyers["lawyer_id"].nunique()

    return m


metrics = compute_metrics()

# ======================================================
# SIDEBAR — ANALYTICS LAYER
# ======================================================

analytics_view = st.sidebar.radio(
    "Analytics Layer",
    [
        "Descriptive",
        "Diagnostic",
        "Predictive",
        "Prescriptive"
    ]
)

# ======================================================
# DESCRIPTIVE ANALYTICS
# ======================================================

if analytics_view == "Descriptive":

    page = st.sidebar.radio(
        "Section",
        ["Acquisition", "Engagement"]
    )

    if page == "Acquisition":

        st.header("User Acquisition")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Visitors", metrics["total_visitors"])
        col2.metric("New Users", metrics["new_users"])
        col3.metric("Legal Questions", metrics["questions"])

        traffic = users["traffic_source"].value_counts().reset_index()
        traffic.columns = ["source","users"]

        chart = alt.Chart(traffic).mark_bar().encode(
            x="source:N",
            y="users:Q",
            tooltip=["source","users"]
        )

        st.altair_chart(chart, width="stretch")


    if page == "Engagement":

        st.header("User Engagement")

        chart = alt.Chart(sessions).mark_circle(size=60).encode(
            x="session_duration_sec:Q",
            y="pages_viewed:Q",
            tooltip=["session_duration_sec","pages_viewed"]
        )

        st.altair_chart(chart, width="stretch")

        rating_chart = alt.Chart(reviews).mark_bar().encode(
            alt.X("rating:Q", bin=True),
            y="count()"
        )

        st.altair_chart(rating_chart, width="stretch")


# ======================================================
# DIAGNOSTIC ANALYTICS
# ======================================================

elif analytics_view == "Diagnostic":

    page = st.sidebar.radio(
        "Section",
        ["Conversion", "Lawyer Marketplace"]
    )

    if page == "Conversion":

        st.header("Conversion Funnel")

        funnel = pd.DataFrame({

            "stage":[
                "Visitors",
                "Questions",
                "Consultations",
                "Paid"
            ],

            "users":[
                metrics["total_visitors"],
                metrics["questions"],
                metrics["consultations"],
                metrics["paid"]
            ]

        })

        chart = alt.Chart(funnel).mark_bar().encode(
            x="stage:N",
            y="users:Q",
            tooltip=["stage","users"]
        )

        st.altair_chart(chart, width="stretch")


    if page == "Lawyer Marketplace":

        st.header("Marketplace Health")

        util = consultations.groupby("lawyer_id").size().reset_index(name="consultations")

        chart = alt.Chart(util).mark_bar().encode(
            x="lawyer_id:N",
            y="consultations:Q"
        )

        st.altair_chart(chart, width="stretch")

        col1,col2 = st.columns(2)

        col1.metric("Active Lawyers", metrics["active_lawyers"])
        col2.metric("Average Consultation Price", f"${metrics['avg_price']:.0f}")


# ======================================================
# PREDICTIVE ANALYTICS
# ======================================================

elif analytics_view == "Predictive":

    page = st.sidebar.radio(
        "Section",
        ["Revenue Forecast", "Customer Segmentation"]
    )

    if page == "Revenue Forecast":

        st.header("Revenue Prediction")

        consultations["date"] = pd.to_datetime(consultations["consultation_date"])

        revenue_time = consultations.groupby("date")["consultation_price"].sum().reset_index()

        revenue_time["day"] = np.arange(len(revenue_time))

        X = revenue_time[["day"]]
        y = revenue_time["consultation_price"]

        model = LinearRegression()
        model.fit(X,y)

        revenue_time["prediction"] = model.predict(X)

        actual = alt.Chart(revenue_time).mark_line().encode(
            x="date:T",
            y="consultation_price:Q"
        )

        pred = alt.Chart(revenue_time).mark_line(color="red").encode(
            x="date:T",
            y="prediction:Q"
        )

        st.altair_chart(actual + pred, width="stretch")


    if page == "Customer Segmentation":

        st.header("Customer Segmentation")

        seg = consultations.groupby("user_id").agg(

            total_spent=("consultation_price","sum"),
            consult_count=("consultation_id","count")

        ).reset_index()

        kmeans = KMeans(n_clusters=3, random_state=0)

        seg["segment"] = kmeans.fit_predict(
            seg[["total_spent","consult_count"]]
        )

        chart = alt.Chart(seg).mark_circle(size=60).encode(
            x="total_spent:Q",
            y="consult_count:Q",
            color="segment:N",
            tooltip=["user_id"]
        )

        st.altair_chart(chart, width="stretch")


# ======================================================
# PRESCRIPTIVE ANALYTICS
# ======================================================

elif analytics_view == "Prescriptive":

    page = st.sidebar.radio(
        "Section",
        ["Marketing Optimization", "Marketplace Optimization"]
    )

    if page == "Marketing Optimization":

        st.header("Cost Per Acquisition")

        m = marketing.groupby("traffic_source").agg(

            spend=("ad_spend","sum"),
            users=("users_acquired","sum")

        ).reset_index()

        m["CPA"] = m["spend"]/m["users"]

        chart = alt.Chart(m).mark_bar().encode(
            x="traffic_source:N",
            y="CPA:Q",
            tooltip=["traffic_source","CPA"]
        )

        st.altair_chart(chart, width="stretch")

        st.success("Recommendation: Increase SEO traffic which has the lowest CPA.")


    if page == "Marketplace Optimization":

        st.header("Lawyer Utilization")

        util = consultations.groupby("lawyer_id").size().reset_index(name="consultations")

        chart = alt.Chart(util).mark_bar().encode(
            x="lawyer_id:N",
            y="consultations:Q"
        )

        st.altair_chart(chart, width="stretch")

        st.success("Recommendation: Improve lawyer availability during peak hours.")
