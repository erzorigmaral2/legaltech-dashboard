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
# Load dataset
# -----------------------------

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Detect target column
# -----------------------------

possible_targets = ["AppInterest","UseLegalApp","WillUseApp"]

target = None

for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    st.error("Target column not found in dataset.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

st.write("Target column used:", target)

# -----------------------------
# Convert target to binary
# -----------------------------

df[target] = df[target].map({
    "Yes":1,
    "Maybe":1,
    "No":0
})

# -----------------------------
# Convert numeric columns
# -----------------------------

numeric_cols = ["ConsultBudget","UrgencyScore"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Remove infinite values
# -----------------------------

df.replace([np.inf,-np.inf],np.nan,inplace=True)

# -----------------------------
# Fill missing values
# -----------------------------

for col in df.columns:
    if df[col].dtype in ["float64","int64"]:
        df[col].fillna(df[col].median(),inplace=True)
    else:
        df[col].fillna(df[col].mode()[0],inplace=True)

# -----------------------------
# Encode categorical features
# -----------------------------

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# Features and target
# -----------------------------

X = df.drop(columns=[target])
y = df[target]

# Final safety cleaning
X.replace([np.inf,-np.inf],np.nan,inplace=True)
X.fillna(X.median(),inplace=True)

# -----------------------------
# Train test split
# -----------------------------

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# -----------------------------
# Train ML models
# -----------------------------

rf = RandomForestClassifier(n_estimators=200,random_state=42)
gb = GradientBoostingClassifier()
lr = LogisticRegression(max_iter=1000)

rf.fit(X_train,y_train)
gb.fit(X_train,y_train)
lr.fit(X_train,y_train)

# -----------------------------
# Accuracy comparison
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

st.info("""
Tree-based models such as Random Forest and Gradient Boosting often perform
better for behavioral prediction tasks like legal service adoption.
""")

# -----------------------------
# Feature importance
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
# Conversion probability
# -----------------------------

st.subheader("Sample Conversion Probability")

sample = X.sample(1)

prob = rf.predict_proba(sample)[0][1]

st.metric("Predicted Conversion Probability",str(round(prob*100,2))+"%")

st.success("""
Users with high predicted probability should receive targeted legal consultation
offers and lawyer recommendations to maximize conversion.
""")
