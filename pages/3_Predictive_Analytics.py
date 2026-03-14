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

st.title("Predictive Analytics - Legal App Adoption Prediction")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Detect Target Column Automatically
# -----------------------------

possible_targets = [
    "AppInterest",
    "UseLegalApp",
    "WillUseApp",
    "InterestedInApp",
]

target_column = None

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    st.error("No valid target column found in dataset.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

st.write("Using target column:", target_column)

# -----------------------------
# Convert Target to Binary
# -----------------------------

df[target_column] = df[target_column].map({
    "Yes": 1,
    "Maybe": 1,
    "No": 0
})

# -----------------------------
# Convert Numeric Columns
# -----------------------------

if "ConsultBudget" in df.columns:
    df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"], errors="coerce")

if "UrgencyScore" in df.columns:
    df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"], errors="coerce")

df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# Encode Categorical Variables
# -----------------------------

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# Split Features and Target
# -----------------------------

X = df.drop(columns=[target_column])
y = df[target_column]

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
# Model Accuracy
# -----------------------------

rf_acc = accuracy_score(y_test, rf.predict(X_test))
gb_acc = accuracy_score(y_test, gb.predict(X_test))
lr_acc = accuracy_score(y_test, lr.predict(X_test))

accuracy_df = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "Logistic Regression"],
    "Accuracy": [rf_acc, gb_acc, lr_acc]
})

st.subheader("Model Performance")

fig = px.bar(
    accuracy_df,
    x="Model",
    y="Accuracy",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

st.subheader("Top Features Influencing Legal App Adoption")

fig2 = px.bar(
    importance.head(10),
    x="Importance",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Conversion Probability
# -----------------------------

st.subheader("Predicted Conversion Probability")

sample = X.sample(1)

prob = rf.predict_proba(sample)[0][1]

st.metric("Conversion Probability", str(round(prob*100,2))+"%")

st.success("""
Users with high probability should receive targeted legal consultation offers.
""")
