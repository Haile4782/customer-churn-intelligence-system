import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Churn Prediction")

# Load model
model = joblib.load("models/churn_model.pkl")

# User inputs
age = st.slider("Age", 18, 80)
balance = st.number_input("Balance", value=50000)
credit = st.slider("Credit Score", 300, 900)

input_df = pd.DataFrame({
    "CreditScore": [credit],
    "Age": [age],
    "Tenure": [5],
    "Balance": [balance],
    "NumOfProducts": [1],
    "HasCrCard": [1],
    "IsActiveMember": [1],
    "EstimatedSalary": [50000],
    "Geography_France": [0],
    "Geography_Germany": [0],
    "Geography_Spain": [1],  # Example default
    "Gender_Female": [0],
    "Gender_Male": [1]
})

# Make sure columns match model
FEATURE_COLUMNS = model.get_booster().feature_names
input_df = input_df[FEATURE_COLUMNS]

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    st.metric("Churn Probability", f"{prob*100:.2f}%")

# Download predictions
if st.button("Download Predictions"):
    df = pd.read_csv("data/churn_modelling.csv")
    X = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)
    X = X[FEATURE_COLUMNS]  # reorder columns
    df["Prediction"] = model.predict(X)
    st.download_button("Download CSV", df.to_csv(index=False), "predictions.csv")