import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

st.title("Predictive Analytics")

df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

cols = ["City","LegalIssue","IncomeLevel","PreferredService"]

enc = LabelEncoder()

for c in cols:
    df[c] = enc.fit_transform(df[c].astype(str))

df["Target"] = (df["ConsultBudget"]>df["ConsultBudget"].median()).astype(int)

X = df[cols]
y = df["Target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
    "Logistic Regression":LogisticRegression(max_iter=500),
    "Random Forest":RandomForestClassifier(),
    "Gradient Boosting":GradientBoostingClassifier()
}

results = {}

for name,model in models.items():
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    results[name] = accuracy_score(y_test,pred)

acc_df = pd.DataFrame(list(results.items()),columns=["Model","Accuracy"])

fig = px.bar(
    acc_df,
    x="Model",
    y="Accuracy",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig,use_container_width=True)

best_model = acc_df.sort_values("Accuracy",ascending=False).iloc[0]

st.success(f"Best performing model: {best_model['Model']}")
