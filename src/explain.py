import shap
import joblib
from preprocess import load_data

# Load data & model
df = load_data()
model = joblib.load("models/churn_model.pkl")

X = df.drop("Exited", axis=1)

# SHAP Explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Summary plot
shap.summary_plot(shap_values, X)