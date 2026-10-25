"""
app.py  -  Streamlit web app for the Loan Approval Prediction System

Run locally:   streamlit run app.py
"""

import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_common import (CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES,
                         TARGET, VALID_CATEGORIES, VALID_RANGES, add_features)

MODEL_PATH = os.path.join("models", "loan_model.joblib")
DATA_PATH = os.path.join("data", "loan_data_cleaned.csv")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")


@st.cache_resource
def get_model():
    """Load the saved model. If it cannot be loaded (for example the online server
    has a different scikit-learn version), train a fresh one from the cleaned data."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        df = pd.read_csv(DATA_PATH)
        prep = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             CATEGORICAL_FEATURES),
        ])
        model = Pipeline([("prep", prep),
                          ("model", RandomForestClassifier(n_estimators=300, max_depth=10,
                                                           min_samples_leaf=3, random_state=42,
                                                           n_jobs=-1))])
        return model.fit(df[MODEL_FEATURES], df[TARGET])


st.title("🏦 Loan Approval Prediction")
st.caption("Enter the applicant's details to predict whether the loan will be approved. "
           "Built on a synthetic dataset for a machine learning internship project.")

tab_predict, tab_insights = st.tabs(["Predict", "Data insights"])

# ----------------------------------------------------------------------------
# Tab 1: prediction form
# ----------------------------------------------------------------------------
with tab_predict:
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        age = col1.number_input("Age", *VALID_RANGES["Age"], value=32, step=1)
        income = col2.number_input("Monthly income (Rs)", *VALID_RANGES["MonthlyIncome"],
                                   value=55000, step=1000)
        amount = col1.number_input("Loan amount (Rs)", *VALID_RANGES["LoanAmount"],
                                   value=900000, step=10000)
        credit = col2.number_input("Credit score", *VALID_RANGES["CreditScore"], value=740, step=5)
        employment = col1.selectbox("Employment status", VALID_CATEGORIES["EmploymentStatus"])
        existing = col2.number_input("Existing active loans", *VALID_RANGES["ExistingLoans"],
                                     value=1, step=1)
        term = col1.number_input("Loan term (months)", *VALID_RANGES["LoanTerm"], value=60, step=6)
        area = col2.selectbox("Property area", VALID_CATEGORIES["PropertyArea"])
        submitted = st.form_submit_button("Predict loan decision", width="stretch")

    if submitted:
        row = add_features(pd.DataFrame([{
            "Age": age, "MonthlyIncome": income, "LoanAmount": amount, "CreditScore": credit,
            "ExistingLoans": existing, "LoanTerm": term,
            "EmploymentStatus": employment, "PropertyArea": area,
        }]))[MODEL_FEATURES]
        probability = float(get_model().predict_proba(row)[0][1])
        emi_ratio = float(row["EMIToIncomeRatio"].iloc[0])

        if probability >= 0.5:
            st.success(f"✅ Loan APPROVED  (approval chance {probability:.1%})")
        else:
            st.error(f"❌ Loan REJECTED  (approval chance {probability:.1%})")
        st.progress(probability)
        st.caption(f"Monthly instalment is about {emi_ratio:.0%} of monthly income "
                   "(interest ignored).")

# ----------------------------------------------------------------------------
# Tab 2: charts and model results
# ----------------------------------------------------------------------------
with tab_insights:
    st.subheader("What drives loan approval?")
    charts = [
        ("01_approval_by_employment.png", "Approvals by employment status"),
        ("05_credit_income_vs_approval.png", "Credit score and income vs approval"),
        ("02_correlation_heatmap.png", "Correlation heatmap"),
        ("03_income_vs_loan_amount.png", "Income vs loan amount"),
        ("08_feature_importance.png", "Feature importance (Random Forest)"),
        ("07_confusion_matrix.png", "Confusion matrix of the best model"),
    ]
    for filename, caption in charts:
        path = os.path.join("images", filename)
        if os.path.exists(path):
            st.image(path, caption=caption, width="stretch")
    st.info("Model: Random Forest  |  Accuracy 83.2%  |  Precision 83.4%  |  "
            "Recall 88.8%  |  F1-score 86.0%")
