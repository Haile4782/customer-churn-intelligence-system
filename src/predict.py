import pandas as pd 
import joblib

model = joblib.load("models/churn_model.pkl")

# List of columns used in training
FEATURE_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male"
]

def predict_customer(data_dict):
    df = pd.DataFrame([data_dict])
    # Ensure all columns are present
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLUMNS]  # order columns
    prediction = model.predict(df)[0]
    return prediction

# Example test
if __name__ == "__main__":
    sample = {
        "CreditScore":600,
        "Age":40,
        "Tenure":5,
        "Balance":60000,
        "NumOfProducts":2,
        "HasCrCard":1,
        "IsActiveMember":1,
        "EstimatedSalary":50000,
        "Geography_Germany":0,
        "Geography_Spain":1,
        "Gender_Male":1
    }

    print("Prediction:", predict_customer(sample))