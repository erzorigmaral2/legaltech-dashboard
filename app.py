import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="LegalTech Growth Intelligence Dashboard", layout="wide")

st.title("LegalTech Growth Intelligence Dashboard")

# =========================================================
# utils/load_data.py
# =========================================================

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

# =========================================================
# utils/metrics.py
# =========================================================

def compute_metrics():

    metrics = {}

    metrics["total_visitors"] = len(users)
    metrics["new_users"] = users["signup_date"].count()
    metrics["questions"] = len(questions)

    metrics["consultations"] = len(consultations)

    metrics["paid_consults"] = len(paid_consults)

    metrics["revenue"] = paid_consults["consultation_price"].sum()

    metrics["avg_price"] = paid_consults["consultation_price"].mean()

    metrics["active_lawyers"] = lawyers["lawyer_id"].nunique()

    return metrics


metrics = compute_metrics()

# =========================================================
# Sidebar Navigation
# =========================================================

page = st.sidebar.radio(
    "Dashboard Sections",
    [
        "Acquisition",
        "Engagement",
        "Conversion",
        "Lawyer Marketplace",
        "Revenue"
    ]
)

# =========================================================
# pages/1_Acquisition.py
# =========================================================

def acquisition_page():

    st.header("User Acquisition")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Visitors", metrics["total_visitors"])
    col2.metric("New Users", metrics["new_users"])
    col3.metric("Legal Questions", metrics["questions"])

    st.markdown("---")

    traffic = users["traffic_source"].value_counts().reset_index()
    traffic.columns = ["source", "users"]

    chart = alt.Chart(traffic).mark_bar().encode(
        x=alt.X("source:N", title="Traffic Source"),
        y=alt.Y("users:Q", title="Users"),
        tooltip=["source", "users"]
    ).properties(title="User Acquisition Channels")

    st.altair_chart(chart, width="stretch")

    # CPA

    if {"traffic_source","ad_spend","users_acquired"}.issubset(marketing.columns):

        m = marketing.groupby("traffic_source").agg(
            spend=("ad_spend","sum"),
            users=("users_acquired","sum")
        ).reset_index()

        m["CPA"] = m["spend"]/m["users"]

        cpa_chart = alt.Chart(m).mark_bar().encode(
            x="traffic_source:N",
            y="CPA:Q",
            tooltip=["traffic_source","CPA"]
        ).properties(title="Cost per Acquisition")

        st.altair_chart(cpa_chart, width="stretch")


# =========================================================
# pages/2_Engagement.py
# =========================================================

def engagement_page():

    st.header("User Engagement")

    chart = alt.Chart(sessions).mark_circle(size=60).encode(
        x=alt.X("session_duration_sec:Q", title="Session Duration"),
        y=alt.Y("pages_viewed:Q", title="Pages Viewed"),
        tooltip=["session_duration_sec","pages_viewed"]
    )

    st.altair_chart(chart, width="stretch")

    st.markdown("---")

    rating_chart = alt.Chart(reviews).mark_bar().encode(
        x=alt.X("rating:Q", bin=True),
        y="count()"
    ).properties(title="Lawyer Rating Distribution")

    st.altair_chart(rating_chart, width="stretch")


# =========================================================
# pages/3_Conversion.py
# =========================================================

def conversion_page():

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
            metrics["paid_consults"]
        ]

    })

    funnel_chart = alt.Chart(funnel).mark_bar().encode(
        x="stage:N",
        y="users:Q",
        tooltip=["stage","users"]
    )

    st.altair_chart(funnel_chart, width="stretch")

    st.markdown("---")

    # Customer segmentation

    seg = consultations.groupby("user_id").agg(
        total_spent=("consultation_price","sum"),
        consult_count=("consultation_id","count")
    ).reset_index()

    kmeans = KMeans(n_clusters=3, random_state=0)
    seg["segment"] = kmeans.fit_predict(seg[["total_spent","consult_count"]])

    seg_chart = alt.Chart(seg).mark_circle(size=70).encode(
        x="total_spent:Q",
        y="consult_count:Q",
        color="segment:N",
        tooltip=["user_id","total_spent","consult_count"]
    )

    st.altair_chart(seg_chart, width="stretch")


# =========================================================
# pages/4_Lawyer_Marketplace.py
# =========================================================

def marketplace_page():

    st.header("Lawyer Marketplace Health")

    col1,col2 = st.columns(2)

    col1.metric("Active Lawyers", metrics["active_lawyers"])
    col2.metric("Average Consultation Price", f"${metrics['avg_price']:.0f}")

    st.markdown("---")

    util = consultations.groupby("lawyer_id").size().reset_index(name="consultations")

    util_chart = alt.Chart(util).mark_bar().encode(
        x="lawyer_id:N",
        y="consultations:Q"
    ).properties(title="Lawyer Utilization")

    st.altair_chart(util_chart, width="stretch")


# =========================================================
# pages/5_Revenue.py
# =========================================================

def revenue_page():

    st.header("Revenue Analytics")

    st.metric("Total Revenue", f"${metrics['revenue']:,.0f}")

    st.markdown("---")

    consultations["date"] = pd.to_datetime(consultations["consultation_date"])

    revenue_time = consultations.groupby("date")["consultation_price"].sum().reset_index()

    revenue_time["day_index"] = np.arange(len(revenue_time))

    X = revenue_time[["day_index"]]
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


# =========================================================
# ROUTER
# =========================================================

if page == "Acquisition":
    acquisition_page()

elif page == "Engagement":
    engagement_page()

elif page == "Conversion":
    conversion_page()

elif page == "Lawyer Marketplace":
    marketplace_page()

elif page == "Revenue":
    revenue_page()
