# CreditWise — AI-Powered Loan Approval System

CreditWise is an end-to-end machine learning system that predicts whether a loan application should be approved or rejected based on an applicant's financial and personal profile. It uses a Gaussian Naive Bayes classifier and is deployed as an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Live Demo:** [creditwise-loan-approval-system-fk2mekg.streamlit.app](https://creditwise-loan-approval-system-fk2mekg.streamlit.app/)

---

## Overview

The project covers the full machine learning pipeline — from data preprocessing and exploratory data analysis (EDA) to feature engineering, multi-model comparison, and Streamlit deployment. Three classifiers were evaluated (Logistic Regression, KNN, and Gaussian Naive Bayes), with Naive Bayes selected as the final model based on overall performance metrics.

---

## Features

- Real-time loan approval prediction with a probability confidence score
- Model performance dashboard showing Accuracy, Precision, Recall, F1 Score, and Confusion Matrix
- Multi-model comparison across Logistic Regression, KNN, and Naive Bayes
- Feature engineering — DTI Ratio² and Credit Score² added to capture non-linear patterns
- CSV upload support to retrain the model on your own dataset
- Dark-themed, responsive UI built with custom CSS

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.8+ |
| ML / Data | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Notebook | Jupyter Notebook |

---

## Project Structure

```
creditwise-loan-system/
│
├── loan_system.ipynb          # Full ML pipeline notebook (EDA to model selection)
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── loan_approval_data.csv     # Dataset
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/creditwise-loan-system.git
cd creditwise-loan-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

If the above doesn't work, try:
```bash
python -m streamlit run app.py
```

### 4. Upload your dataset
Once the app opens at `http://localhost:8501`, upload `loan_approval_data.csv` from the sidebar. The model trains automatically.

---

## ML Pipeline

```
Raw CSV Data
    |
Missing Value Imputation (mean for numeric, mode for categorical)
    |
Label Encoding (Education Level, Target variable)
    |
One-Hot Encoding (Gender, Employment, Marital Status, Loan Purpose, etc.)
    |
Exploratory Data Analysis (EDA)
    |
Feature Engineering (DTI Ratio², Credit Score²)
    |
Train-Test Split (80/20)
    |
StandardScaler Normalization
    |
Model Comparison (Logistic Regression | KNN | Gaussian Naive Bayes)
    |
Best Model Selected: Gaussian Naive Bayes
    |
Streamlit Deployment
```

---

## Dataset Features

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Applicant's age |
| Gender | Categorical | Gender identity |
| Marital Status | Categorical | Marital situation |
| Education Level | Ordinal | Highest qualification |
| Employment Status | Categorical | Current employment type |
| Employer Category | Categorical | Government / Private / NGO / Self |
| Applicant Income | Numeric | Monthly income |
| Co-Applicant Income | Numeric | Co-applicant's monthly income |
| Loan Amount | Numeric | Requested loan amount |
| Loan Term | Numeric | Repayment period in months |
| Credit Score | Numeric | Credit bureau score (300–900) |
| DTI Ratio | Numeric | Debt-to-income ratio (%) |
| Savings | Numeric | Total savings |
| Existing Loan Balance | Numeric | Outstanding loan balance |
| Number of Dependents | Numeric | Number of financial dependents |
| Loan Purpose | Categorical | Home / Education / Business / Medical / Other |
| Property Area | Categorical | Urban / Semi-Urban / Rural |
| Loan Approved | Target | Yes / No |

---

## License

This project is licensed under the [MIT License](LICENSE).
