# Customer Churn Analysis & Predictive Modeling
#### *(Data Cleaning → EDA → Feature Engineering → Logistic Regression → Tableau Dashboard)*  

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange) ![Tableau](https://img.shields.io/badge/Tableau-Dashboard-blue)

> **End-to-end ML pipeline predicting customer churn with 75% accuracy; delivering 30-day advance warning and up to $125K+ in recoverable annual revenue.**
> 
[**View Live Dashboard**](https://public.tableau.com/app/profile/chinyere.obi8867/viz/ChurnAnalysis_17302938038440/ChurnAnalysis)  |  [**View Python Code**](https://github.com/Mayreeobi/End-to-End-Churn-Analysis/blob/main/Churn%20Predictive%20Analysis.ipynb)

---

## Table of Contents

- [Situation](#situation)
- [Task](#task)
- [Action](#action)
- [Result](#result)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)

---

## Situation

Subscription businesses have a quiet revenue problem. Customers cancel without warning, and by the time the pattern shows up in a quarterly review, thousands of dollars in recurring revenue have already walked out the door. Acquiring a new customer costs 5–7x more than retaining an existing one, yet most retention teams only find out who churned after the fact; with no way to prioritize who actually needed attention.

**The specific problem:** Customer success teams were working off reactive quarterly reports. There was no way to know in advance which customers were drifting toward cancellation, which meant outreach was either too late or scattered across the wrong accounts. High-value customers at genuine risk of leaving were getting the same attention (or less) as stable ones.

| The Problem | Scale |
|---|---|
| Overall churn rate | 26.54% |
| Churned customers | 1,869 |
| Lost revenue (2023–2024) | $1.67M |
| Month-to-month churners | 89% of all churned customers |
| Retention approach | Reactive quarterly reviews |

Without a predictive layer, retention was essentially guesswork.

---

## Task

Build an end-to-end machine learning pipeline that could:

- Analyze historical customer behavior to identify patterns that predict churn before it happens
- Score every customer by churn probability so retention teams can prioritize the right people
- Surface the specific factors driving churn; not just who is leaving, but why
- Deliver findings through an interactive Tableau dashboard that non-technical teams can actually use
- Quantify the business case for acting on the model's predictions

Scope: 7,043 customer records, 21 features, Logistic Regression classification model.

---

## Action

### 1. Data Cleaning

```python
# Standardize column names
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

# Convert TotalCharges to numeric (raw data contains spaces)
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

# Parse date columns
df['signupdate'] = pd.to_datetime(df['signupdate'], errors='coerce')
df['churndate']  = pd.to_datetime(df['churndate'],  errors='coerce')
```

Handled 11 null values in `totalcharges` using median imputation during the modeling stage. Standardized column names across the full dataset for consistent downstream processing.

---

### 2. Feature Engineering

Four engineered features were created to capture behavioral signals that the raw columns couldn't express on their own:

```python
# Average monthly spend — normalizes total charges against tenure
df['avg_monthly_spend'] = df['totalcharges'] / (df['tenure'] + 1)

# Price sensitivity — are they paying more than their average suggests?
df['price_sensitivity'] = df['monthlycharges'] / df['avg_monthly_spend']

# Tenure group — segments lifecycle stage for pattern detection
df['tenuregroup'] = pd.cut(
    df['tenure'],
    bins=[0, 6, 12, 24, 999],
    labels=['0-6 months', '6-12 months', '12-24 months', '24+ months']
)

# High-risk payment flag — electronic check users churn at 2.1x the rate
df['high_risk_payment'] = (df['paymentmethod'] == 'Electronic check').astype(int)
```

---

### 3. Exploratory Data Analysis 

**When customers churn:**

| Tenure Window | Churn Rate | Root Cause |
|---|---|---|
| 0–3 months | 21% | Onboarding failure |
| 3–6 months | 27% | Poor value realization |
| 12-month mark | Spike | Contract renewal friction |

> **The 90-Day Cliff:** 38% of all churns happen within the first 90 days, with a spike around day 60. Customers who don't see value before their second payment leave before retention has any chance to intervene.

**Churn rates by key features:**

| Feature | Finding |
|---|---|
| Contract type | Month-to-month: 42% churn. Annual: 12%. Two-year: 6.3% |
| Fiber optic internet | 40% higher churn than DSL - counterintuitive for a premium tier |
| Monthly charges | Peak churn (45%) at $70-$90/month; sharp increase above $80 |
| Tenure | Customers in first 24 months are highest risk |
| Payment method | Electronic check: 2.1x higher churn than other methods |
| Tech support | No subscription: 2.3x higher churn. 3+ tickets/month: 2.8x higher churn |

---

### 4. Model Training & Evaluation

**Algorithm:** Logistic Regression with `class_weight='balanced'` to handle natural class imbalance (26.5% churn vs. 73.5% active). Optimized for **recall over precision** - missing a real churner costs more ($770) than a false alarm ($50).

```python
logreg = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
logreg.fit(X_train, y_train)
```

**Model performance:**

| Metric | Score | What it means |
|---|---|---|
| Accuracy | 75% | Overall correct predictions on held-out test set |
| Precision | 52% | Of customers flagged as likely to churn, 52% actually did |
| Recall | 81% | Of actual churners, 81% were correctly identified |
| F1-Score | 0.74 | Balanced measure of precision/recall trade-off |
| AUC-ROC | 0.91 | Strong discriminatory power across all thresholds |


---
### 5. Churn Risk Segmentation

```python
# Score all customers
df['churn_probability'] = logreg.predict_proba(X_scaled_all)[:, 1] * 100

# Assign risk tiers
df['risk_level'] = pd.cut(
    df['churn_probability'],
    bins=[0, 30, 70, 100],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)
```

| Risk Tier | Customers | MRR at Risk | Action |
|---|---|---|---|
| 🔴 High Risk | 1,190 | $86K | Customer success calls within 7 days |
| 🟡 Medium Risk | 686 | $28K | 3-touch automated email workflow |
| 🟢 Low Risk | 310 | $8K | Quarterly check-ins |

---

### 6. Tableau Dashboard

**Page 1: Churn Analysis** ![Churn Analysis](https://github.com/mayreeobi/End-to-End-Churn-Analysis/blob/main/Churn_Analysis.png)
- Segment patterns across contract type, tenure, payment method, internet service, and monthly charges.

**Page 2: Predictive Analysis** ![Predictive Analysis](https://github.com/mayreeobi/End-to-End-Churn-Analysi/blob/main/Predictive_Analysis.png)
- Feature importance, churn risk segmentation, revenue at risk, confusion matrix, and top 10 highest-risk customers ready for immediate outreach.

[View Live Dashboard](https://public.tableau.com/views/ChurnAnalysis_17302938038440/PredictiveAnalysis)

---
# Result

| Metric | Value |
|---|---|
| Prediction accuracy | **75%** on held-out test set |
| Recall (churners caught) | **81%** of actual churners identified |
| AUC-ROC | **0.91** |
| Early warning lead time | **30 days** before churn vs. reactive quarterly reviews |
| High-risk customers identified | **1,190** customers requiring proactive retention |
| Revenue recovery potential | **$125K+** annually from retaining 50 high-value customers |
| Churn reduction potential | **12–15%** from contract conversion strategy alone |

The model shifts retention from damage control to proactive revenue protection. Customer success teams work from a ranked priority list rather than guessing who needs attention. Beyond individual flags, feature importance analysis exposed systemic issues — payment friction, onboarding gaps, a fiber optic quality problem — that if addressed, would reduce churn across the entire customer base.

---

## Insights & Recommendations

**Business Health Assessment: High churn, highly fixable** — the drivers are structural and addressable.

<details>
<summary><strong>Insight 1: This Is an Onboarding Problem, Not a Product Problem</strong></summary>

38% of all churns happen within the first 90 days, spiking around day 60. That's not customers discovering the product is bad — it's customers never discovering the product at all. They're leaving before their second payment because no one guided them to value. A structured onboarding program through month 3 would address nearly 4 in 10 churns at the source.

</details>

<details>
<summary><strong>Insight 2: Contract Type Is the Single Biggest Lever</strong></summary>

Month-to-month contracts churn at 42%. Two-year contracts churn at 6.3%. That's an 85% reduction in churn from one variable. Customers on short-term contracts aren't necessarily less loyal — they just haven't been given a compelling reason to commit. Converting even 20% of the month-to-month base to annual contracts could reduce overall churn by 12–15%.

</details>

<details>
<summary><strong>Insight 3: Fiber Optic Has a Quality Problem, Not a Pricing Problem</strong></summary>

Fiber optic internet churns 40% more than DSL — counterintuitive for a premium tier. When the more expensive product loses customers faster than the cheaper one, the issue is almost never pricing. It's a quality or expectation gap: outages, speeds that don't match what was sold, or support that doesn't resolve issues. Retention messaging won't fix this. The product team needs a root cause investigation first.

</details>

<details>
<summary><strong>Insight 4: There's a Price Cliff at $80</strong></summary>

Churn peaks at 45% in the $70–$90 monthly charge range with a sharp acceleration above $80. This is a price sensitivity threshold, not random variation. Customers hitting that number are doing the math and deciding it doesn't add up. A $79 bundled tier with high-perceived-value features could reduce price-driven attrition without cannibalizing higher tiers.

</details>

<details>
<summary><strong>Insight 5: Electronic Check Is a Churn Signal, Not Just a Payment Method</strong></summary>

Electronic check users churn at 2.1x the rate of customers using other payment methods and represent 57% of all churned customers. This correlates with lower engagement and lower commitment — customers who haven't set up auto-pay are already mentally less invested. Migrating this group to auto-pay (with a small incentive) would reduce both churn and payment friction simultaneously.

</details>

---

### Recommendations

#### Immediate Actions (Next 30 Days)

**1. Work the High-Risk List**
- 1,190 customers · $86K MRR at risk
- Customer success calls within 7 days for top 200 by revenue
- Offer 20% discount + contract upgrade conversation
- 30-day premium support trial to demonstrate value
- 📈 *Expected save rate: 40–50% = $34K–$43K MRR retained*

**2. Launch a Contract Conversion Campaign**
- Target all month-to-month customers with tenure 6–24 months
- Offer meaningful incentive for annual commitment (first month free, locked pricing)
- Frame as protection against price increases, not just a discount
- 📈 *Expected impact: 12–15% overall churn reduction*

**3. Fix the Electronic Check Friction**
- Auto-pay migration campaign with small bill credit incentive
- Reduces churn signal and payment failure risk simultaneously
- 📈 *Expected impact: 15–20% churn reduction in this segment*

#### Short-Term (3–6 Months)

**4. Redesign Onboarding Through Month 3**
- Milestone check-ins at days 14, 30, and 60
- Dedicated onboarding specialist for high-value new accounts
- Usage-based triggers: if a customer hasn't activated key features by day 30, flag for outreach
- 📈 *Expected impact: address 38% of all churns at the source*

**5. Investigate Fiber Optic Quality**
- Pull support ticket data for fiber vs. DSL customers
- Map churn events against outage and complaint history
- Identify whether the issue is speed, reliability, or expectation mismatch
- 📈 *Expected impact: 40% churn reduction in fiber tier if root cause addressed*

**6. Introduce a $79 Price Tier**
- Bundle high-value features at a price point just below the $80 cliff
- Reduces price-driven attrition without cannibalizing $90+ tiers
- 📈 *Expected impact: reduce churn in the $70–$90 charge segment by 20–25%*

#### Long-Term (6–12 Months)

- **Extend success coverage through month 24** - risk doesn't end at month 3, it extends through the full first two years; milestone check-ins at months 6, 12, and 18 with dedicated CSM for high-value accounts
- **Build a churn early warning system** - automate the model scoring pipeline to run monthly and push high-risk flags directly into the CRM so retention teams work from live data, not static exports
- **Track cohort retention by acquisition channel** - understanding which channels bring customers who stay vs. churn early allows marketing to optimize spend toward higher-LTV acquisition

---

### Financial Impact Summary

| Initiative | Customers Affected | Expected Outcome |
|---|---|---|
| High-risk outreach | 1,190 | $34K–$43K MRR retained |
| Contract conversions | ~2,000 month-to-month | 12–15% churn reduction |
| Auto-pay migration | Electronic check segment | 15–20% segment churn reduction |
| Onboarding redesign | All new customers | Address 38% of churns at source |
| $79 price tier | $70–$90 charge segment | 20–25% segment churn reduction |
| **Total annual impact** | | **$125K+ recoverable revenue** |

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Logistic Regression, StandardScaler, SimpleImputer) |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Tableau |
| Environment | Jupyter Notebook |

---

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── telecom_churn.csv               # 7,043 records, 21 features
│   └── processed/
│       ├── churn_predictions_full.csv      # All customers scored with risk levels
│       ├── churn_master_data.csv           # Tableau master dataset
│       └── feature_importance.csv          # Top 15 churn drivers
│
├── notebooks/
│   └── Churn_Predictive_Analysis.ipynb     # Full pipeline: EDA → model → export
│
├── assets/
│   └── dashboard_preview.png              # Tableau dashboard screenshot
│
└── README.md
```

---

*Chinyere Obi · Data Analyst · [chinyereobi.netlify.app](https://chinyereobi.netlify.app) · [LinkedIn](https://www.linkedin.com/in/chinyere-obi/)*








