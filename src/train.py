from preprocess import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import pandas as pd

# Load data
df = load_data()

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")
print("✅ Model saved successfully")