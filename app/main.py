import streamlit as st

st.set_page_config(
    page_title="Churn Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode styling
st.markdown("""
<style>
body { background-color: #0E1117; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Customer Churn Intelligence System")

st.markdown("""
Welcome to a full **AI-powered churn analytics platform**:

- 🔮 Prediction
- 📊 Analytics
- 🧠 Explainable AI (SHAP)
- 📈 Power BI Integration
""")