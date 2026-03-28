import streamlit as st
import shap
import joblib
import pandas as pd

st.title("🧠 Explainable AI (SHAP)")

model = joblib.load("models/churn_model.pkl")
df = pd.read_csv("data/churn_modelling.csv")

X = df.drop(["RowNumber","CustomerId","Surname","Exited"], axis=1)

explainer = shap.Explainer(model)
shap_values = explainer(X)

st.write("Feature Impact on Churn:")
st.pyplot(shap.summary_plot(shap_values, X, show=False))