import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Predictive Analytics")

# Load dataset
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.write("Dataset shape:", df.shape)

# -------------------------
# DATA CLEANING
# -------------------------

# Convert numeric columns safely
numeric_cols = ["ConsultBudget", "UrgencyScore"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill numeric NaN with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill categorical NaN
df.fillna("Unknown", inplace=True)

# -------------------------
# ENCODE CATEGORICAL
# -------------------------

label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# -------------------------
# TARGET VARIABLE
# -------------------------

# Automatically choose last column as target
target = df.columns[-1]

st.write("Target column used:", target)

X = df.drop(columns=[target])
y = df[target]

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# MODELS
# -------------------------

rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
lr = LogisticRegression(max_iter=1000)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# -------------------------
# ACCURACY
# -------------------------

rf_acc = accuracy_score(y_test, rf.predict(X_test))
gb_acc = accuracy_score(y_test, gb.predict(X_test))
lr_acc = accuracy_score(y_test, lr.predict(X_test))

results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "Logistic Regression"],
    "Accuracy": [rf_acc, gb_acc, lr_acc]
})

# -------------------------
# VISUALIZATION
# -------------------------

fig = px.bar(
    results,
    x="Model",
    y="Accuracy",
    title="Model Performance Comparison"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# INSIGHT
# -------------------------

st.info("""
Machine learning models are used to predict whether users are likely
to adopt a digital legal advisory application.

Tree-based models like Random Forest and Gradient Boosting usually
perform better because they capture nonlinear relationships
in legal service demand.
""")
