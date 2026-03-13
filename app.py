import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="LegalTech Growth Intelligence Dashboard", layout="wide")

st.title("LegalTech Growth Intelligence Dashboard")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

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

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

view = st.sidebar.radio(
    "Analytics View",
    [
        "Descriptive Analytics",
        "Diagnostic Analytics",
        "Predictive Analytics",
        "Prescriptive Analytics"
    ]
)

# ====================================================
# DESCRIPTIVE ANALYTICS
# ====================================================

if view == "Descriptive Analytics":

    st.header("Platform KPI Overview")

    total_visitors = len(users)
    legal_questions = len(questions)
    revenue = paid_consults["consultation_price"].sum()
    avg_price = paid_consults["consultation_price"].mean()
    active_lawyers = lawyers["lawyer_id"].nunique()

    repeat_users = consultations["user_id"].value_counts()
    repeat_rate = (repeat_users > 1).sum() / len(repeat_users)

    arpu = revenue / total_visitors if total_visitors > 0 else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Total Visitors", total_visitors)
    col2.metric("Legal Questions", legal_questions)
    col3.metric("Revenue", f"${revenue:,.0f}")
    col4.metric("ARPU", f"${arpu:,.2f}")
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
    ).properties(title="User Acquisition by Traffic Source")

    st.altair_chart(traffic_chart, width="stretch")

    # Revenue by legal category
    revenue_cat = paid_consults.groupby("category")["consultation_price"].sum().reset_index()

    revenue_chart = alt.Chart(revenue_cat).mark_bar().encode(
        x="category:N",
        y="consultation_price:Q",
        tooltip=["category", "consultation_price"]
    ).properties(title="Revenue by Legal Category")

    st.altair_chart(revenue_chart, width="stretch")

# ====================================================
# DIAGNOSTIC ANALYTICS
# ====================================================

elif view == "Diagnostic Analytics":

    st.header("Conversion Funnel Analysis")

    visitors = len(users)
    questions_count = len(questions)
    consultations_count = len(consultations)
    paid_count = len(paid_consults)

    funnel = pd.DataFrame({
        "stage": ["Visitors", "Questions", "Consultations", "Paid Consultations"],
        "users": [visitors, questions_count, consultations_count, paid_count]
    })

    funnel_chart = alt.Chart(funnel).mark_bar().encode(
        x="stage:N",
        y="users:Q",
        tooltip=["stage", "users"]
    ).properties(title="User Conversion Funnel")

    st.altair_chart(funnel_chart, width="stretch")

    st.markdown("---")

    st.subheader("User Engagement")

    engagement_chart = alt.Chart(sessions).mark_circle(size=80).encode(
        x="session_duration_sec:Q",
        y="pages_viewed:Q",
        tooltip=["session_duration_sec", "pages_viewed"]
    ).properties(title="Session Engagement vs Pages Viewed")

    st.altair_chart(engagement_chart, width="stretch")

    st.markdown("---")

    st.subheader("Lawyer Rating Distribution")

    rating_chart = alt.Chart(reviews).mark_bar().encode(
        alt.X("rating:Q", bin=True),
        y="count()"
    ).properties(title="Customer Satisfaction Distribution")

    st.altair_chart(rating_chart, width="stretch")

# ====================================================
# PREDICTIVE ANALYTICS
# ====================================================

elif view == "Predictive Analytics":

    st.header("Predictive Models")

    consultations["date"] = pd.to_datetime(consultations["consultation_date"])

    revenue_time = consultations.groupby("date")["consultation_price"].sum().reset_index()

    revenue_time["time_index"] = np.arange(len(revenue_time))

    X = revenue_time[["time_index"]]
    y = revenue_time["consultation_price"]

    model = LinearRegression()
    model.fit(X, y)

    revenue_time["forecast"] = model.predict(X)

    actual = alt.Chart(revenue_time).mark_line().encode(
        x="date:T",
        y="consultation_price:Q"
    ).properties(title="Actual Revenue")

    forecast = alt.Chart(revenue_time).mark_line(color="red").encode(
        x="date:T",
        y="forecast:Q"
    ).properties(title="Revenue Forecast")

    st.subheader("Revenue Forecast")

    st.altair_chart(actual + forecast, width="stretch")

    st.markdown("---")

    st.subheader("Customer Segmentation (CLV vs Frequency)")

    seg = consultations.groupby("user_id").agg({
        "consultation_price": "sum",
        "consultation_id": "count"
    }).reset_index()

    kmeans = KMeans(n_clusters=3, n_init=10)
    seg["segment"] = kmeans.fit_predict(seg[["consultation_price", "consultation_id"]])

    seg_chart = alt.Chart(seg).mark_circle(size=90).encode(
        x="consultation_price:Q",
        y="consultation_id:Q",
        color="segment:N",
        tooltip=["user_id", "consultation_price", "consultation_id"]
    ).properties(title="Customer Segmentation")

    st.altair_chart(seg_chart, width="stretch")

# ====================================================
# PRESCRIPTIVE ANALYTICS
# ====================================================

elif view == "Prescriptive Analytics":

    st.header("Optimization & Decision Insights")

    st.subheader("Cost Per Acquisition (CPA)")

    spend_col = None
    users_col = None

    # Detect columns automatically
    for col in marketing.columns:
        if "spend" in col.lower() or "cost" in col.lower():
            spend_col = col
        if "user" in col.lower() or "acquisition" in col.lower():
            users_col = col

    if spend_col and users_col:

        marketing["CPA"] = marketing[spend_col] / marketing[users_col]

        cpa_chart = alt.Chart(marketing).mark_bar().encode(
            x="channel:N",
            y="CPA:Q",
            tooltip=["channel", "CPA"]
        ).properties(title="Cost Per Acquisition by Channel")

        st.altair_chart(cpa_chart, width="stretch")

    else:

        st.warning("Marketing dataset does not contain recognizable spend/user columns.")
        st.write("Detected columns:", marketing.columns)

    st.markdown("---")

    st.subheader("Lawyer Utilization")

    util = consultations.groupby("lawyer_id").size().reset_index(name="consultations")

    util_chart = alt.Chart(util).mark_bar().encode(
        x="lawyer_id:N",
        y="consultations:Q"
    ).properties(title="Lawyer Utilization Rate")

    st.altair_chart(util_chart, width="stretch")

    st.markdown("---")

    st.success(
        "Recommendation: Improve lawyer response time and invest more in high-performing SEO channels to increase consultation conversion."
    )
