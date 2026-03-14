import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

st.title("Predictive Analytics – Legal App Conversion Prediction")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Detect Target Column
# -----------------------------

possible_targets = ["AppInterest","UseLegalApp","WillUseApp"]

target = None
for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    st.error("Target column not found in dataset.")
    st.write("Columns available:", df.columns.tolist())
    st.stop()

st.write("Target column:", target)

# -----------------------------
# Convert Target
# -----------------------------

df[target] = df[target].map({
    "Yes":1,
    "Maybe":1,
    "No":0
})

# -----------------------------
# Convert All Columns Safely
# -----------------------------

for col in df.columns:

    if df[col].dtype == "object":

        try:
            df[col] = pd.to_numeric(df[col])
        except:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

# -----------------------------
# Replace Infinite Values
# -----------------------------

df.replace([np.inf,-np.inf],np.nan,inplace=True)

# -----------------------------
# Fill Missing Values
# -----------------------------

df.fillna(df.median(numeric_only=True),inplace=True)

# -----------------------------
# Final Safety Cleaning
# -----------------------------

df = df.dropna()

# -----------------------------
# Split Features / Target
# -----------------------------

X = df.drop(columns=[target])
y = df[target]

# Ensure numeric matrix
X = X.astype(float)

# -----------------------------
# Train/Test Split
# -----------------------------

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# -----------------------------
# Train Models
# -----------------------------

rf = RandomForestClassifier(n_estimators=200,random_state=42)
gb = GradientBoostingClassifier()
lr = LogisticRegression(max_iter=1000)

rf.fit(X_train,y_train)
gb.fit(X_train,y_train)
lr.fit(X_train,y_train)

# -----------------------------
# Accuracy Comparison
# -----------------------------

rf_acc = accuracy_score(y_test,rf.predict(X_test))
gb_acc = accuracy_score(y_test,gb.predict(X_test))
lr_acc = accuracy_score(y_test,lr.predict(X_test))

acc_df = pd.DataFrame({
    "Model":["Random Forest","Gradient Boosting","Logistic Regression"],
    "Accuracy":[rf_acc,gb_acc,lr_acc]
})

st.subheader("Model Accuracy Comparison")

fig = px.bar(
    acc_df,
    x="Model",
    y="Accuracy",
    title="ML Model Performance"
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":rf.feature_importances_
}).sort_values("Importance",ascending=False)

st.subheader("Top Features Driving Conversion")

fig2 = px.bar(
    importance.head(10),
    x="Importance",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig2,use_container_width=True)

# -----------------------------
# Conversion Probability
# -----------------------------

st.subheader("Sample Conversion Probability")

sample = X.sample(1)

prob = rf.predict_proba(sample)[0][1]

st.metric("Predicted Conversion Probability",str(round(prob*100,2))+"%")

st.success("""
Users with high predicted probability should receive targeted legal consultation offers.
""")
