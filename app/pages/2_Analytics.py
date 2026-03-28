import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Data Analytics")

df = pd.read_csv("data/churn_modelling.csv")

fig = px.histogram(df, x="Age", color="Exited", title="Churn by Age")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.scatter(df, x="Balance", y="Age", color="Exited")
st.plotly_chart(fig2, use_container_width=True)