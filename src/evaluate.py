from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from preprocess import load_data
import joblib

df = load_data()
model = joblib.load("models/churn_model.pkl")

X = df.drop("Exited", axis=1)
y = df["Exited"]

y_prob = model.predict_proba(X)[:,1]

fpr, tpr, _ = roc_curve(y, y_prob)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()