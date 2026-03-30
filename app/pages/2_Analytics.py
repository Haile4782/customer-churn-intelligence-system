import streamlit as st
import plotly.express as px
import joblib
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from preprocess import load_data

st.title("📊 Data Analytics")

model = joblib.load("models/churn_model.pkl")

df = load_data()

X = df.drop("Exited", axis=1)
FEATURE_COLUMNS = model.get_booster().feature_names
X = X[FEATURE_COLUMNS]

df["Prediction"] = model.predict(X)

# Charts
fig1 = px.histogram(df, x="Age", color="Exited", title="Churn by Age")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(df, x="Balance", y="Age", color="Exited")
st.plotly_chart(fig2, use_container_width=True)

# High risk customers
st.subheader("🚨 High Risk Customers")
st.dataframe(df[df["Prediction"] == 1].head(10))