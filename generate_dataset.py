"""
generate_dataset.py
-------------------
Creates a synthetic loan-applicant dataset (2000 records + a few duplicates)
and saves it as data/loan_data.csv.

The approval decision is NOT random: it is built from realistic rules
(good credit score, healthy income, being employed, a small instalment
compared with income, and few existing loans all raise the chance of
approval) plus some random noise, just like real life.

Run:  python generate_dataset.py
"""

import os
import numpy as np
import pandas as pd

N_RECORDS = 2000
SEED = 42                      # fixed seed -> same dataset every time
OUTPUT_PATH = os.path.join("data", "loan_data.csv")


def sigmoid(z):
    """Turns any number into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-z))


def generate(n: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Basic applicant details -------------------------------------------
    applicant_id = np.arange(1001, 1001 + n)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    age = np.clip(rng.normal(36, 10, n), 21, 65).astype(int)
    employment = rng.choice(["Employed", "Self-Employed", "Unemployed"],
                            size=n, p=[0.62, 0.28, 0.10])
    area = rng.choice(["Urban", "Semi-Urban", "Rural"], size=n, p=[0.38, 0.36, 0.26])

    # --- Monthly income depends on employment type and age -----------------
    income = np.where(
        employment == "Employed", rng.lognormal(np.log(35000), 0.55, n),
        np.where(employment == "Self-Employed", rng.lognormal(np.log(40000), 0.75, n),
                 rng.lognormal(np.log(8000), 0.60, n)))
    income = income * (1 + np.clip(age - 30, -8, 15) * 0.01)       # experience bonus
    income = (np.clip(income, 5_000, 400_000) / 100).round() * 100

    # --- Credit score is linked to income and employment -------------------
    credit = (rng.normal(690, 85, n)
              + 40 * (np.log(income) - np.log(35000))
              - np.where(employment == "Unemployed", 30, 0))
    credit = np.clip(credit, 300, 900).astype(int)

    # --- People with lower scores tend to have more running loans ----------
    existing = np.clip(rng.poisson(np.clip(1.0 + (680 - credit) / 250, 0.2, 3)), 0, 6)

    # --- Loan term and amount ----------------------------------------------
    term = rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360], size=n,
                      p=[0.08, 0.10, 0.14, 0.12, 0.18, 0.08, 0.10, 0.08, 0.08, 0.04])
    emi_share = rng.lognormal(np.log(0.28), 0.45, n)     # share of income going to EMI
    amount = emi_share * income * term
    amount = (np.clip(amount, 10_000, 50_000_000) / 5000).round() * 5000

    # --- Approval rule (the "hidden bank policy") --------------------------
    emi_ratio = (amount / term) / income
    score = (
        0.25
        + 0.018 * (credit - 650)                      # better credit -> approve
        - 5.0 * (emi_ratio - 0.25)                    # big instalment -> reject
        + 0.9 * np.log(income / 30000)                # higher income -> approve
        + np.where(employment == "Employed", 0.6,
                   np.where(employment == "Self-Employed", 0.1, -1.8))
        - 0.45 * existing                             # many loans -> reject
        + np.where(area == "Urban", 0.3, np.where(area == "Semi-Urban", 0.15, 0.0))
        - 0.015 * np.abs(age - 40)
        - np.where(credit < 550, 1.5, 0.0)            # very low score penalty
        + rng.normal(0, 0.6, n)                       # real-life randomness
    )
    approved = (rng.random(n) < sigmoid(score)).astype(int)

    df = pd.DataFrame({
        "ApplicantID": applicant_id,
        "Gender": gender,
        "Age": age,
        "MonthlyIncome": income.astype(int),
        "LoanAmount": amount.astype(int),
        "CreditScore": credit,
        "EmploymentStatus": employment,
        "ExistingLoans": existing,
        "LoanTerm": term,
        "PropertyArea": area,
        "LoanApproved": approved,
    })

    # --- Make the data "messy" on purpose so preprocessing has real work ---
    # 1) Missing values in a few columns
    for col, frac in [("Age", 0.02), ("MonthlyIncome", 0.03), ("CreditScore", 0.03),
                      ("EmploymentStatus", 0.015), ("PropertyArea", 0.01)]:
        idx = rng.choice(df.index, size=int(frac * n), replace=False)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)      # numbers need float to hold NaN
        df.loc[idx, col] = np.nan

    # 2) Some duplicate rows
    duplicates = df.sample(30, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    data = generate()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to {OUTPUT_PATH}  |  shape = {data.shape}")
    print(f"Approval rate (before cleaning): {data['LoanApproved'].mean():.1%}")
