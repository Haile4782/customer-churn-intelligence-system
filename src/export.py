import pandas as pd
import joblib
from preprocess import load_data

df = load_data()
model = joblib.load("models/churn_model.pkl")

df["Prediction"] = model.predict(df.drop("Exited", axis=1))

df.to_csv("outputs/churn_predictions.csv", index=False)

print("✅ File ready for Power BI")