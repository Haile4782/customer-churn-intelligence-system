import streamlit as st

st.set_page_config(
    page_title="Churn Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme
st.markdown("""
<style>
body { background-color: #0E1117; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Customer Churn Intelligence System")

st.info("⏳ If app is slow, it may be waking up from sleep (free hosting).")

st.markdown("""
### 🚀 Features:
- 🔮 Prediction
- 📊 Analytics
- 🧠 Explainable AI (SHAP)
""")