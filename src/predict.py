import pandas as pd
import joblib

model = joblib.load("models/churn_model.pkl")

def predict_customer(data_dict):
    df = pd.DataFrame([data_dict])
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