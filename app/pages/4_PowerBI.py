import streamlit as st

st.title("📈 Power BI Integration")

st.markdown("""
### How to Use in Power BI:

1. Run:

python src/export.py


2. Open Power BI  
3. Import:

outputs/churn_predictions.csv


4. Build:
- KPI Cards
- Churn %
- High Risk Customers
- Age vs Churn
""")

st.info("This system supports enterprise BI workflows.")