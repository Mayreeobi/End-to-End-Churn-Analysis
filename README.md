# End-to-End Customer Churn Analysis & Predictive Modeling Project  
### *(Data Cleaning → EDA → Feature Engineering → Logistic Regression → Tableau Dashboard)*  

This project showcases a complete **End-to-End Telco Churn Analysis & Prediction Pipeline** — from raw messy data all the way to an interactive dashboard in Tableau.

[**→ View Live Tableau Dashboard**](https://public.tableau.com/app/profile/chinyere.obi8867/viz/ChurnAnalysis_17302938038440/ChurnAnalysis)                  

---

## 🚀 Project Overview

SaaS businesses rely heavily on **customer retention**.  
The goal of this project is to:

- Understand **why customers churn**
- Identify key **churn drivers**
- Build a **predictive churn model**
- Quantify revenue at risk
- Provide **data-driven business recommendations**
- Build a **stakeholder-ready Tableau dashboard**

This project uses:
- **Python** for cleaning, EDA, and modeling  
- **Tableau** for dashboard visualizations  
- **Logistic Regression** for churn prediction  

---

## 🎯 Project Highlights

| Metric | Value | Impact |
|--------|-------|--------|
| **Model Accuracy** | 75% | Predicts churn effectively |
| **Recall Rate** | 81% | Captures 8 out of 10 churners |
| **High-Risk Customers** | 1,190 | Require proactive retention |
| **Monthly Revenue at Risk** | $86K | Annual impact: $1.04M |
| **Estimated ROI** | 131% | $236K net benefit (Year 1) |

---

## 🛠️ Tech Stack

- **Python:** pandas, NumPy, scikit-learn  
- **Visualization:** Tableau  
- **Plotting:** Matplotlib, Seaborn  
---


##  Key Findings

### ⚠️ The Churn Crisis
- **26.54% overall churn rate** → $1.67M lost revenue (2023–2024)
- **1,869 churned customers**
- **89%** of churners are *month-to-month customers*
- **Electronic Check** users churn the most (57%)

### 📅 When Customers Churn
1. **0–3 months:** 21% → onboarding issues  
2. **3–6 months:** 27% → poor value realization  
3. **12-month mark:** spike → contract renewal friction  

### 💡 Top 5 Churn Drivers
1. **Contract Type** → 2-year contracts reduce churn by **85%**
2. **Fiber Optic Internet** → 40% higher churn vs. DSL  
3. **Monthly Charges** → Price sensitivity above $80  
4. **Tenure < 24 months** → Early lifecycle risk  
5. **No Tech Support** → 2.3× higher churn likelihood  

### 🧠 Model Performance (2023)
- **Accuracy:** 75%  
- **Precision:** 52%  
- **Recall:** 81% (prioritized for business use case)  
- **AUC-ROC:** 0.91  

**Why high recall?**  
Missing churners is more expensive than false alarms.  
- Missed churner cost: **$770**  
- False alarm cost: **$50**  
- Trade-off = justified  

The model predicts churn **30 days in advance**, enabling proactive retention.

---

## 💰 Financial Impact & ROI

| Cost Category | Amount |
|---------------|--------|
| CS outreach (1,190 × $50) | $59,500 |
| Automation | $15,000 |
| Training & tools | $10,000 |
| Incentives (476 saves × $200) | $95,200 |
| **Total Investment** | **$179,700** |

---

##  Business Recommendations

### 🔴 High-Priority (Immediate Action: 1,190 Customers | $86K MRR)
- Customer success calls within 7 days  
- 20% discount + contract upgrades  
- 30-day premium support trial  
**Expected impact:** 40–50% save rate

### 🟡 Medium-Risk (686 Customers | $28K MRR)
- 3-touch automated email workflow  
- Webinars and product education  
**Expected saves:** 25–35%

### 🟢 Low-Risk (310 Customers | $8K MRR)
- Quarterly check-ins  
- Product newsletters  
- Community programs  

---

## 📊 Dashboard Features

### **Page 1 — Churn Analysis**
- Churn rate overview  
- Revenue loss  
- Contract type impact  
- Payment method trends  
- Tenure & monthly charges impact  

### **Page 2 — Predictive Analysis**
- Customer-level churn probability  
- High-/Medium-/Low-risk segmentation  
- Confusion matrix  
- Model key drivers  

Stakeholders can move from:  
**Descriptive → Diagnostic → Predictive insights**

---

## 📂 Project Structure
```
End-to-End-Churn-Analysis/
├── data/
│   ├── raw/                        # Original telecom customer data
│   └── processed/                  # Cleaned and engineered datasets
├── notebooks/
│   ├── predictive analysis.ipynb   # Containing EDA, Feature engineering and model
├── dashboards/
│   └── churn analysis              # Tableau dashboard
│   └── predictive analysis  
└── README.md
```
---

## 📝 Methodology
```
### 1. Data Cleaning  
- Removed duplicates  
- Standardized date formats (2023–2024)  
- Handled missing values  
- Validated pricing & subscription consistency  

### 2. Exploratory Data Analysis  
Focused on uncovering:  
- Churn distribution  
- Payment method patterns  
- Revenue at risk  
- Churn by contract type & tenure  
- Numerical correlations  

### 3. Feature Engineering  
- Tenure groups  
- Price sensitivity metrics  
- Contract splits  
- Average monthly spend  
- Binary encodings  
- Revenue-based features  

### 4. Modeling  
- **Algorithm:** Logistic Regression  
- **Scaling:** StandardScaler  
- **Validation:** 80/20 stratified split  
Model optimized for **RECALL** to maximize saved customers.
```
---

## 🏁 Final Thoughts  
This project represents a full analytics lifecycle — blending **data cleaning**, **feature engineering**, **predictive modeling**, and **business-focused insights** into one strategic solution.

It directly supports **retention**, **customer success**, and **revenue protection** initiatives.

[**→ View Python Code**](https://github.com/Mayreeobi/End-to-End-Churn-Analysis/blob/main/Churn%20Predictive%20Analysis.ipynb)


