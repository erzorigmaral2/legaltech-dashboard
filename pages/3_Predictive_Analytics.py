import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

st.title("Predictive Analytics - Legal Consultation Conversion")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# Encode Categorical Variables
# -----------------------------

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# Define Target Variable
# -----------------------------

target_column = "WillUseApp"

if target_column not in df.columns:
    st.error("Target column 'WillUseApp' not found in dataset.")
    st.stop()

X = df.drop(columns=[target_column])
y = df[target_column]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Models
# -----------------------------

rf = RandomForestClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier()
lr = LogisticRegression(max_iter=1000)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# -----------------------------
# Model Predictions
# -----------------------------

rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
lr_pred = lr.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)
lr_acc = accuracy_score(y_test, lr_pred)

# -----------------------------
# Model Accuracy Comparison
# -----------------------------

st.subheader("Model Performance Comparison")

accuracy_df = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "Logistic Regression"],
    "Accuracy": [rf_acc, gb_acc, lr_acc]
})

fig = px.bar(
    accuracy_df,
    x="Model",
    y="Accuracy",
    title="ML Model Accuracy Comparison"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Insight:

Tree-based models such as Random Forest and Gradient Boosting often perform better
for behavioral prediction problems like legal service adoption.
""")

# -----------------------------
# Feature Importance
# -----------------------------

st.subheader("Feature Importance (Random Forest)")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values("Importance", ascending=False)

fig2 = px.bar(
    importance.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Features Influencing Legal Consultation Conversion"
)

st.plotly_chart(fig2, use_container_width=True)

st.info("""
Insight:

Features such as legal urgency, consultation budget, and legal issue type
are key drivers of whether users will convert into paying clients.
""")

# -----------------------------
# Conversion Probability Tool
# -----------------------------

st.subheader("Predict User Conversion Probability")

sample = X.sample(1)

prediction_prob = rf.predict_proba(sample)[0][1]

st.metric(
    "Predicted Conversion Probability",
    str(round(prediction_prob * 100, 2)) + "%"
)

st.success("""
Business Insight:

Users with high predicted probability should receive targeted offers,
lawyer recommendations, or discounts to increase conversion rates.
""")
