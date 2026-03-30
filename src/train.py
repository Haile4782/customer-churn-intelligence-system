from preprocess import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os

# Load processed data
df = load_data()

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model comparison
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier()
}

print("🔍 Model Comparison:")
for name, m in models.items():
    m.fit(X_train, y_train)
    acc = m.score(X_test, y_test)
    print(f"{name}: {acc:.4f}")

# Grid Search for XGBoost
params = {
    "n_estimators": [200, 300],
    "max_depth": [4, 5],
    "learning_rate": [0.01, 0.05]
}

grid = GridSearchCV(xgb.XGBClassifier(), params, cv=3, scoring="accuracy")
grid.fit(X_train, y_train)

model = grid.best_estimator_

print("✅ Best Params:", grid.best_params_)

# Final training
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("\n📊 Final Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")

print("💾 Model saved successfully!")