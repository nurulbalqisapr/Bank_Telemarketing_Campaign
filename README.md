# 🏦 Bank Telemarketing Campaign Optimization: Predicting Term Deposit Subscription

## Project Overview

This project analyzes a **Portugal Bank Marketing Campaign dataset** to help a bank improve the effectiveness of its telemarketing campaign for a **term deposit product**.

The campaign was conducted mostly through **direct phone calls**. Each row in the dataset represents one customer or campaign contact, including customer profile, financial condition, campaign information, previous campaign history, economic indicators, and the final outcome of whether the customer subscribed to a term deposit.

The main objective of this project is to identify which customers are more likely to subscribe, so the bank can prioritize high-potential customers and reduce inefficient calls.

---

## 👥 Created By

**Gamma Group**

1. **Nurul Balqis Apriany**  
2. **Muhamad Fahrizal Safalah**

---

##  Business Problem

Telemarketing campaigns require cost, time, and agent effort. If all customers are contacted without any selection or prioritization process, the bank may spend resources on customers who have a low probability of subscribing to a term deposit.

The dataset also shows that the number of customers who subscribe to a term deposit is much lower than those who do not subscribe. This indicates a relatively low campaign conversion rate and creates a need for a more data-driven campaign prioritization strategy.

---

## ❓ Problem Statement

**How can the bank identify the key factors that influence customers' tendency to subscribe to a term deposit and predict which customers should be prioritized in telemarketing campaigns to improve campaign efficiency?**

---

## ✅ Project Goals

1. Identify key factors that influence customers' tendency to subscribe to a term deposit based on:
   - customer profile
   - financial condition
   - campaign information
   - previous campaign history
   - economic indicators

2. Build a classification model to predict whether a customer is likely to subscribe to a term deposit.

3. Help the bank prioritize high-potential customers, reduce inefficient calls to low-potential customers, and improve telemarketing campaign efficiency.

---
## 📊 Tableau Dashboard

The Tableau dashboard summarizes the main EDA findings and turns them into business guidance for campaign prioritization.

🔗 **Public Tableau Dashboard:**  
https://public.tableau.com/app/profile/nrl.balqis/viz/finpro4/Dashboard1#2

### Dashboard Preview

The Tableau dashboard summarizes the main EDA findings and turns them into business guidance for campaign prioritization.

<img width="1599" height="899" alt="Dashboard 1" src="https://github.com/user-attachments/assets/6b13114f-f598-451c-bfe8-e45f10de2395" />


### Dashboard Highlights

The dashboard includes:

- **Target Distribution**  
  Shows the overall subscription imbalance between customers who subscribed and did not subscribe.

- **Previous Campaign Outcome**  
  Highlights that customers with previous campaign success show much stronger conversion behavior.

- **Contact Method**  
  Compares conversion rate between cellular and telephone contact methods.

- **Campaign Timing by Month**  
  Shows how conversion rate and customer volume differ across campaign months.

- **High-Potential Job Segments**  
  Identifies customer job groups with stronger subscription conversion.

- **Euribor Level Signal**  
  Shows how economic conditions are associated with customer subscription behavior.

- **Actionable Campaign Guidance**  
  Converts EDA insights into practical recommendations for telemarketing strategy.

---

## Analytical Approach

This project uses **supervised learning with binary classification** because the target variable has two classes:

- `yes`: customer subscribes to a term deposit
- `no`: customer does not subscribe to a term deposit

The workflow follows these main stages:

1. **Problem Statement**  
   Define the business problem and how machine learning can support the decision-making process.

2. **Data Understanding**  
   Understand the dataset structure, unit of analysis, feature groups, target variable, missing values, duplicates, and potential leakage.

3. **Exploratory Data Analysis**  
   Explore customer behavior and campaign patterns through visualizations and data storytelling.

4. **Data Preprocessing**  
   Handle duplicates, unknown values, feature engineering, encoding, scaling, and data leakage prevention.

5. **Modeling and Evaluation**  
   Compare several classification models using the same preprocessing pipeline and select the best model based on the main business metric.

6. **Conclusion and Recommendation**  
   Translate findings and model results into actionable telemarketing recommendations.

---

## Dataset Understanding

### Dataset Source

Bank Marketing Campaign dataset from Kaggle:  
https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset

### Unit of Analysis

Each row represents **one customer or one campaign contact record**.

### Target Variable

| Target | Meaning |
|---|---|
| `yes` | Customer subscribed to a term deposit |
| `no` | Customer did not subscribe to a term deposit |

### Feature Groups

| Feature Group | Description |
|---|---|
| Customer Profile | age, job, marital status, education |
| Financial Condition | default, housing loan, personal loan |
| Current Campaign | contact method, month, day of week, campaign frequency |
| Previous Campaign | previous contacts, days since last contact, previous outcome |
| Economic Indicators | employment variation rate, consumer price index, consumer confidence index, euribor3m, number of employees |
| Target | whether the customer subscribed to a term deposit |

---

## Key EDA Insights

### 1. Low Overall Conversion Rate

Only a small proportion of customers subscribed to the term deposit product. This confirms that the campaign has a low success rate and customer prioritization is needed.

### 2. Previous Campaign Success Is a Strong Signal

Customers with a successful previous campaign outcome show much higher conversion behavior. This group should be prioritized for follow-up campaigns.

### 3. Cellular Contact Performs Better

Cellular contact shows a stronger conversion rate than telephone contact, indicating that channel selection matters in telemarketing effectiveness.

### 4. Campaign Timing Matters

Some months show higher conversion rates than others. Campaign timing should consider both conversion rate and customer volume.

### 5. Job Segment Differences Exist

Certain job segments, such as students and retired customers, show higher conversion behavior. However, segment volume should also be considered before making business decisions.

### 6. Economic Context Influences Conversion

Euribor level and other economic indicators show different conversion patterns, meaning macroeconomic conditions should be considered when planning campaigns.

---

## Machine Learning Methodology

### Models Compared

Several classification models were compared using the same preprocessing pipeline:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- K-Nearest Neighbors

### Imbalance Handling

Because the target variable is imbalanced, imbalance handling was tested. The project uses:

- class weight adjustment
- stratified cross-validation
- F-beta Score as the main evaluation metric

### Best Model

The best model selected is:

**Tuned Logistic Regression with imbalance handling**

This model was selected because it provides the strongest performance based on the project’s main evaluation metric and remains interpretable for business decision support.

---

## Evaluation Metric

Accuracy is not used as the main metric because the dataset is imbalanced. A model could achieve high accuracy by mostly predicting the majority class (`no`), but this would not solve the business problem.

### Primary Metric

**F-beta Score with β > 1 for the positive class (`yes`)**

This metric is used because the project focuses on identifying potential subscribers and reducing the risk of missing customers who are likely to subscribe.

### Supporting Metrics

- ROC-AUC
- PR-AUC
- Confusion Matrix

---

## Deployment

This project also includes a Streamlit application for portfolio deployment.

### Streamlit App Features

- Executive overview of the business case
- Interactive EDA filters
- Customer-level subscription prediction
- Campaign prioritization recommendation
- Professional portfolio-ready UI

### Required Deployment Files

```text
app.py
requirements.txt
bank-additional-full.csv
best_lr_model.pkl
best_threshold.pkl
model_columns.pkl
assets/bank_hero.jpg
.streamlit/config.toml
```

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Repository Structure

```text
Bank_Telemarketing_Campaign_Gamma/
│
├── Final_Project_Gamma_Group.ipynb
├── README.md
├── app.py
├── bank-additional-full.csv
├── best_lr_model.pkl
├── best_threshold.pkl
├── model_columns.pkl
├── requirements.txt
├── tableu.png
│
└── assets/
    └── bank_hero.jpg
└── .streamlit/
    └── config.toml
```

---

## Tools and Libraries

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical processing |
| Matplotlib / Seaborn | Exploratory visualization |
| Scikit-learn | Machine learning pipeline and evaluation |
| XGBoost | Model comparison |
| Joblib | Saving model artifacts |
| Tableau | Interactive business dashboard |
| Streamlit | Model deployment and interactive app |
| GitHub | Project documentation and version control |

---

## Business Recommendations

Based on the analysis and modeling results, the bank should:

1. Prioritize customers with stronger subscription signals, especially those with successful previous campaign outcomes.
2. Use cellular-first outreach because it shows better conversion behavior than telephone contact.
3. Consider campaign timing, especially months with stronger conversion patterns.
4. Use the prediction model as a lead scoring tool before contacting customers.
5. Avoid contacting customers randomly and focus telemarketing effort on higher-potential customers.
6. Continue monitoring model performance when new campaign data becomes available.

---

## Project Limitations

- The model output should be used as a decision-support tool, not as the only business decision rule.
- The `duration` feature requires special attention because it may cause data leakage if used before the phone call is completed.
- Actual campaign performance may vary depending on customer behavior, campaign execution, economic condition, and business policy.
- Financial impact results are scenario-based and should be adjusted using the bank’s actual operational cost and revenue assumptions.

---

## Conclusion

This project demonstrates how data science can support telemarketing campaign optimization. By combining EDA, machine learning, Tableau dashboarding, and Streamlit deployment, the bank can move from broad customer outreach to a more focused and data-driven prioritization strategy.

The final output helps stakeholders understand campaign performance, identify high-potential customer groups, and support better telemarketing decisions.

---

## Streamlit Deployment
The application has been successfully deployed on Streamlit Community Cloud.

🔗 Live Demo
https://banktelemarketingcampaign-nrbcappbfxnww6vgbaaubuh.streamlit.app/
