import streamlit as st
import pandas as pd
import joblib
import shap

st.title("🔮 Churn Prediction")

model = joblib.load("models/churn_model.pkl")

# Inputs
age = st.slider("Age", 18, 80)
balance = st.number_input("Balance", value=50000)
credit = st.slider("Credit Score", 300, 900)

# Input dataframe (MATCH TRAINING)
input_df = pd.DataFrame({
    "CreditScore": [credit],
    "Age": [age],
    "Tenure": [5],
    "Balance": [balance],
    "NumOfProducts": [1],
    "HasCrCard": [1],
    "IsActiveMember": [1],
    "EstimatedSalary": [50000],
    "Geography_Germany": [0],
    "Geography_Spain": [1],
    "Gender_Male": [1],
    "Balance_per_Product": [balance / (1 + 1)],
    "Age_Group": [age // 10]
})

FEATURE_COLUMNS = model.get_booster().feature_names
input_df = input_df[FEATURE_COLUMNS]

# Prediction
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    st.metric("Churn Probability", f"{prob*100:.2f}%")

# SHAP explanation
if st.button("Explain Prediction"):
    explainer = shap.Explainer(model)
    shap_val = explainer(input_df)

    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Impact": shap_val.values[0]
    }).sort_values(by="Impact", key=abs, ascending=False)

    st.bar_chart(shap_df.set_index("Feature"))