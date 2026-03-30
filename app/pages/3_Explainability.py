import streamlit as st
import shap
import joblib
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from preprocess import load_data

st.title("🧠 Explainable AI (SHAP)")

model = joblib.load("models/churn_model.pkl")

df = load_data()
X = df.drop("Exited", axis=1)

FEATURE_COLUMNS = model.get_booster().feature_names
X = X[FEATURE_COLUMNS]

explainer = shap.Explainer(model)
shap_values = explainer(X)

shap_df = pd.DataFrame(shap_values.values, columns=X.columns)

importance = shap_df.abs().mean().sort_values(ascending=False).reset_index()
importance.columns = ["Feature", "Impact"]

fig = px.bar(
    importance,
    x="Impact",
    y="Feature",
    orientation="h",
    title="Feature Impact on Churn"
)

st.plotly_chart(fig, use_container_width=True)