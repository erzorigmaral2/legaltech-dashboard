import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Predictive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

df["ConsultBudget"] = pd.to_numeric(df["ConsultBudget"],errors="coerce")
df["UrgencyScore"] = pd.to_numeric(df["UrgencyScore"],errors="coerce")

df.replace([np.inf,-np.inf],np.nan,inplace=True)

df.fillna(df.median(numeric_only=True),inplace=True)
df.fillna("Unknown",inplace=True)

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

target = df.columns[-1]

X = df.drop(columns=[target])
y = df[target]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
lr = LogisticRegression(max_iter=1000)

rf.fit(X_train,y_train)
gb.fit(X_train,y_train)
lr.fit(X_train,y_train)

results = pd.DataFrame({
"Model":["Random Forest","Gradient Boosting","Logistic Regression"],
"Accuracy":[
accuracy_score(y_test,rf.predict(X_test)),
accuracy_score(y_test,gb.predict(X_test)),
accuracy_score(y_test,lr.predict(X_test))
]
})

fig = px.bar(results,x="Model",y="Accuracy",title="Model Performance")

st.plotly_chart(fig,use_container_width=True)

st.info("Tree based models capture complex legal service adoption patterns.")
