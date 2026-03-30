import streamlit as st
import shap
import joblib
import pandas as pd
import plotly.express as px

st.title("🧠 Explainable AI (SHAP)")

# Load model
model = joblib.load("models/churn_model.pkl")

# Load data
df = pd.read_csv("data/churn_modelling.csv")

# Preprocess (same as training)
X = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)

FEATURE_COLUMNS = model.get_booster().feature_names
X = X[FEATURE_COLUMNS]

# SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Convert SHAP to DataFrame
shap_df = pd.DataFrame(shap_values.values, columns=X.columns)

# Mean importance
importance = shap_df.abs().mean().sort_values(ascending=False).reset_index()
importance.columns = ["Feature", "Impact"]

# Plotly interactive chart
fig = px.bar(
    importance,
    x="Impact",
    y="Feature",
    orientation="h",
    title="Feature Impact on Churn (Interactive)",
)

st.plotly_chart(fig, use_container_width=True)