import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="LegalTech Growth Intelligence Dashboard", layout="wide")

st.title("LegalTech Growth Intelligence Dashboard")

# =====================================================
# LOAD DATA
# =====================================================

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

# =====================================================
# SIDEBAR
# =====================================================

view = st.sidebar.radio(
    "Analytics View",
    [
        "Descriptive Analytics",
        "Diagnostic Analytics",
        "Predictive Analytics",
        "Prescriptive Analytics"
    ]
)

# =====================================================
# DESCRIPTIVE ANALYTICS
# =====================================================

if view == "Descriptive Analytics":

    st.header("Platform Overview")

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

    # Traffic Source

    traffic = users["traffic_source"].value_counts().reset_index()
    traffic.columns = ["source", "users"]

    chart = alt.Chart(traffic).mark_bar().encode(
        x=alt.X("source:N", title="Traffic Source"),
        y=alt.Y("users:Q", title="Users"),
        tooltip=["source", "users"]
    ).properties(title="User Acquisition by Channel")

    st.altair_chart(chart, width="stretch")

    # Revenue by category

    revenue_category = (
        paid_consults.groupby("category")["consultation_price"]
        .sum()
        .reset_index()
    )

    chart2 = alt.Chart(revenue_category).mark_bar().encode(
        x=alt.X("category:N"),
        y=alt.Y("consultation_price:Q"),
        tooltip=["category", "consultation_price"]
    ).properties(title="Revenue by Legal Category")

    st.altair_chart(chart2, width="stretch")

# =====================================================
# DIAGNOSTIC ANALYTICS
# =====================================================

elif view == "Diagnostic Analytics":

    st.header("Conversion Funnel")

    visitors = len(users)
    questions_count = len(questions)
    consultations_count = len(consultations)
    paid_count = len(paid_consults)

    funnel = pd.DataFrame({
        "stage": ["Visitors", "Questions", "Consultations", "Paid"],
        "users": [visitors, questions_count, consultations_count, paid_count]
    })

    funnel_chart = alt.Chart(funnel).mark_bar().encode(
        x=alt.X("stage:N"),
        y=alt.Y("users:Q"),
        tooltip=["stage", "users"]
    ).properties(title="User Conversion Funnel")

    st.altair_chart(funnel_chart, width="stretch")

    st.markdown("---")

    # Engagement

    engagement = alt.Chart(sessions).mark_circle(size=60).encode(
        x=alt.X("session_duration_sec:Q", title="Session Duration"),
        y=alt.Y("pages_viewed:Q"),
        tooltip=["session_duration_sec", "pages_viewed"]
    ).properties(title="User Engagement Behavior")

    st.altair_chart(engagement, width="stretch")

    st.markdown("---")

    rating_chart = alt.Chart(reviews).mark_bar().encode(
        x=alt.X("rating:Q", bin=True),
        y="count()"
    ).properties(title="Lawyer Rating Distribution")

    st.altair_chart(rating_chart, width="stretch")

# =====================================================
# PREDICTIVE ANALYTICS
# =====================================================

elif view == "Predictive Analytics":

    st.header("Predictive Analytics")

    st.subheader("Revenue Forecast")

    consultations["date"] = pd.to_datetime(consultations["consultation_date"])

    revenue_time = consultations.groupby("date")["consultation_price"].sum().reset_index()

    revenue_time["day_index"] = np.arange(len(revenue_time))

    X = revenue_time[["day_index"]]
    y = revenue_time["consultation_price"]

    model = LinearRegression()
    model.fit(X, y)

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

    st.markdown("---")

    # CUSTOMER SEGMENTATION

    st.subheader("Customer Segmentation")

    seg = consultations.groupby("user_id").agg(
        total_spent=("consultation_price", "sum"),
        consult_count=("consultation_id", "count")
    ).reset_index()

    kmeans = KMeans(n_clusters=3, random_state=0)
    seg["segment"] = kmeans.fit_predict(seg[["total_spent", "consult_count"]])

    seg_chart = alt.Chart(seg).mark_circle(size=70).encode(
        x="total_spent:Q",
        y="consult_count:Q",
        color="segment:N",
        tooltip=["user_id", "total_spent", "consult_count"]
    ).properties(title="Customer Segments")

    st.altair_chart(seg_chart, width="stretch")

# =====================================================
# PRESCRIPTIVE ANALYTICS
# =====================================================

elif view == "Prescriptive Analytics":

    st.header("Optimization & Decision Insights")

    st.subheader("Cost Per Acquisition (CPA)")

    required_cols = ["traffic_source", "ad_spend", "users_acquired"]

    if all(col in marketing.columns for col in required_cols):

        marketing_grouped = (
            marketing.groupby("traffic_source")
            .agg(
                total_spend=("ad_spend", "sum"),
                total_users=("users_acquired", "sum")
            )
            .reset_index()
        )

        marketing_grouped["CPA"] = (
            marketing_grouped["total_spend"] /
            marketing_grouped["total_users"]
        )

        cpa_chart = alt.Chart(marketing_grouped).mark_bar().encode(
            x=alt.X("traffic_source:N", title="Traffic Source"),
            y=alt.Y("CPA:Q", title="Cost Per Acquisition"),
            tooltip=["traffic_source", "CPA"]
        ).properties(title="Cost Per Acquisition by Channel")

        st.altair_chart(cpa_chart, width="stretch")

    else:

        st.error("Marketing dataset missing required columns")
        st.write("Detected columns:", marketing.columns)

    st.markdown("---")

    # CHANNEL EFFICIENCY

    st.subheader("Channel Efficiency")

    efficiency_chart = alt.Chart(marketing).mark_circle(size=80).encode(
        x=alt.X("ad_spend:Q"),
        y=alt.Y("users_acquired:Q"),
        color="traffic_source:N",
        tooltip=["traffic_source", "ad_spend", "users_acquired"]
    )

    st.altair_chart(efficiency_chart, width="stretch")

    st.markdown("---")

    # LAWYER UTILIZATION

    st.subheader("Lawyer Utilization")

    util = consultations.groupby("lawyer_id").size().reset_index(name="consultations")

    util_chart = alt.Chart(util).mark_bar().encode(
        x="lawyer_id:N",
        y="consultations:Q"
    ).properties(title="Consultations per Lawyer")

    st.altair_chart(util_chart, width="stretch")

    st.success(
        "Recommendation: Increase SEO investment and reduce lawyer response time to improve conversion."
    )
