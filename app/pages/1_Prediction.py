import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Churn Prediction")

model = joblib.load("models/churn_model.pkl")

# Inputs
age = st.slider("Age", 18, 80)
balance = st.number_input("Balance", value=50000)
credit = st.slider("Credit Score", 300, 900)

input_df = pd.DataFrame({
    "CreditScore":[credit],
    "Age":[age],
    "Tenure":[5],
    "Balance":[balance],
    "NumOfProducts":[1],
    "HasCrCard":[1],
    "IsActiveMember":[1],
    "EstimatedSalary":[50000],
    "Geography_Germany":[0],
    "Geography_Spain":[0],
    "Gender_Male":[1]
})

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    st.metric("Churn Probability", f"{prob*100:.2f}%")

# Download sample predictions
if st.button("Download Predictions"):
    df = pd.read_csv("data/churn_modelling.csv")
    df["Prediction"] = model.predict(df.drop(["RowNumber","CustomerId","Surname","Exited"], axis=1))
    st.download_button("Download CSV", df.to_csv(index=False), "predictions.csv")