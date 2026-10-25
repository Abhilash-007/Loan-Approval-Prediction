"""
loan_common.py
--------------
Small helpers shared by every file in the project (dataset generator,
main pipeline and the console prediction app), so the same rules are
used everywhere.
"""

import pandas as pd

# ----------------------------------------------------------------------------
# Allowed values (used for data validation AND for checking user input)
# ----------------------------------------------------------------------------
VALID_RANGES = {
    "Age": (18, 70),
    "MonthlyIncome": (1_000, 1_000_000),      # in rupees per month
    "LoanAmount": (10_000, 50_000_000),       # in rupees
    "CreditScore": (300, 900),                # CIBIL-style score
    "ExistingLoans": (0, 10),
    "LoanTerm": (6, 360),                     # in months
}

VALID_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "EmploymentStatus": ["Employed", "Self-Employed", "Unemployed"],
    "PropertyArea": ["Urban", "Semi-Urban", "Rural"],
}

# ----------------------------------------------------------------------------
# Columns the model is trained on.
# Gender is deliberately NOT used: a lending decision should not depend on
# gender, and it keeps the model fair. ApplicantID is just a label.
# ----------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Age", "MonthlyIncome", "LoanAmount", "CreditScore",
    "ExistingLoans", "LoanTerm", "EMIToIncomeRatio",
]
CATEGORICAL_FEATURES = ["EmploymentStatus", "PropertyArea"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "LoanApproved"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered feature EMIToIncomeRatio.

    EMIToIncomeRatio = (LoanAmount / LoanTerm) / MonthlyIncome
    It tells us what share of monthly income goes into the loan instalment
    (interest ignored to keep it simple). Banks look at this closely.
    """
    df = df.copy()
    df["EMIToIncomeRatio"] = (df["LoanAmount"] / df["LoanTerm"]) / df["MonthlyIncome"]
    return df
