# 🏦 Customer Churn Intelligence System

## 🌐 Live Application
👉 https://customer-churn-intelligence-system.streamlit.app/

---

## 🚀 Overview
The **Customer Churn Intelligence System** is an end-to-end Machine Learning and Business Intelligence platform designed to predict, analyze, and explain customer churn behavior in banking environments.

This system integrates **advanced ML models, explainable AI (SHAP), and interactive dashboards** to support data-driven decision-making and customer retention strategies.

---

## 🔥 Key Features

### 🔮 Predictive Analytics
- Customer churn prediction using **XGBoost (optimized with GridSearchCV)**
- Real-time prediction via interactive UI

### 🧠 Explainable AI (SHAP)
- Global feature importance (interactive)
- Individual prediction explanation (per customer)
- Transparent and trustworthy AI decisions

### 📊 Interactive Dashboards
- Built with **Streamlit + Plotly**
- Multi-page architecture:
  - Prediction
  - Analytics
  - Explainability

### 📈 Business Intelligence
- High-risk customer identification
- Churn pattern analysis
- KPI monitoring (churn rate, customer behavior)

### 📥 Data Export
- Download model predictions for further analysis (CSV)
- Ready for integration with BI tools like Power BI

---

## 🧠 Technologies Used

| Category | Tools |
|--------|------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Explainable AI | SHAP |
| Visualization | Plotly |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🏗️ System Architecture

```

Raw Data → Preprocessing → Feature Engineering → ML Model → Prediction API → Dashboard → BI Reporting

```

---

## 🧪 Machine Learning Pipeline

- Data Cleaning & Preprocessing
- One-Hot Encoding (Geography, Gender)
- Feature Engineering:
  - Balance_per_Product
  - Age_Group
- Model Comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Hyperparameter Tuning (GridSearchCV)
- Model Evaluation & Deployment

---

## 📊 Business Value

This system enables organizations to:

- 🎯 Identify high-risk customers early
- 📉 Reduce churn rate through targeted interventions
- 📊 Gain insights into customer behavior patterns
- 🤖 Trust AI decisions via explainability
- 💼 Support strategic decision-making with data

---

## 📂 Project Structure

```text

churn-ml-system/
│
├── src/                # ML pipeline (preprocess, train, explain, predict, evaluate)
├── app/                # Streamlit multi-page app
│   └── pages/          # Prediction, Analytics, Explainability
├── models/             # Trained ML models
├── data/               # Dataset
├── outputs/            # Predictions / exports
├── requirements.txt
├── runtime.txt
└── README.md

````

---

## ⚙️ Installation & Run Locally

```bash
# Clone repo
git clone https://github.com/Haile4782/customer-churn-intelligence-system.git

# Navigate
cd customer-churn-intelligence-system

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Run app
streamlit run app/main.py
````

---

## 📌 Notes

* ⏳ App may take a few seconds to wake up (free hosting limitation)
* 🔄 Ensure consistent preprocessing across all modules
* 📊 Designed for scalability and enterprise BI integration

---

## 👨‍💻 Author

**Haiyleyesus Abayneh Belay**

```text
Engineering Data Analyst | Machine Learning Enthusiast
```