import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.title("Predictive Analytics")

# Load data
df = pd.read_csv("data/mongolia_legal_survey_synthetic_dataset_2000.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

# ------------------------------------
# AUTOMATIC FEATURE ENGINEERING
# ------------------------------------

# Detect categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Detect numeric columns
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()

# Encode categorical columns
enc = LabelEncoder()

for c in cat_cols:
    df[c] = enc.fit_transform(df[c].astype(str))

# ------------------------------------
# CREATE TARGET VARIABLE
# ------------------------------------

if len(num_cols) > 0:
    target_col = num_cols[0]
else:
    target_col = cat_cols[0]

df["Target"] = (df[target_col] > df[target_col].median()).astype(int)

# ------------------------------------
# FEATURES
# ------------------------------------

X = df.drop(columns=["Target"])
y = df["Target"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# ------------------------------------
# MODELS
# ------------------------------------

models = {
    "Logistic Regression":LogisticRegression(max_iter=500),
    "Random Forest":RandomForestClassifier(),
    "Gradient Boosting":GradientBoostingClassifier()
}

results = {}

for name,model in models.items():

    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test,preds)

    results[name] = acc

# ------------------------------------
# VISUALIZATION
# ------------------------------------

acc_df = pd.DataFrame(
    list(results.items()),
    columns=["Model","Accuracy"]
)

fig = px.bar(
    acc_df,
    x="Model",
    y="Accuracy",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig,use_container_width=True)

best = acc_df.sort_values("Accuracy",ascending=False).iloc[0]

st.success(f"Best performing model: {best['Model']}")
