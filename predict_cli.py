"""
predict_cli.py
--------------
Console app: asks for applicant details, checks every answer, and prints
whether the loan would be APPROVED or REJECTED.

Run:  python predict_cli.py
(First run loan_approval.py once so that models/loan_model.joblib exists.)
"""

import os
import sys

import joblib
import pandas as pd

from loan_common import MODEL_FEATURES, VALID_CATEGORIES, VALID_RANGES, add_features

MODEL_PATH = os.path.join("models", "loan_model.joblib")


# ----------------------------------------------------------------------------
# Input helpers - each one keeps asking until the answer is valid
# ----------------------------------------------------------------------------
class QuitProgram(Exception):
    """Raised when the user types 'q' to leave the program."""


def read_text(prompt):
    """input() wrapper: handles Ctrl+C / end of input and the 'q' (quit) command."""
    try:
        text = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise QuitProgram
    if text.lower() in ("q", "quit", "exit"):
        raise QuitProgram
    return text


def ask_number(label, field, as_int=True, hint=""):
    """Ask for a number and make sure it is inside the allowed range."""
    low, high = VALID_RANGES[field]
    while True:
        text = read_text(f"{label} ({low:,} - {high:,}){hint}: ").replace(",", "")
        if text == "":
            print("  ! This value is required. Please type a number.")
            continue
        try:
            value = float(text)
        except ValueError:
            print(f"  ! '{text}' is not a number. Please type digits only, e.g. 45000.")
            continue
        if value != value or value in (float("inf"), float("-inf")):
            print("  ! Please enter a normal number.")
            continue
        if as_int and value != int(value):
            print("  ! Please enter a whole number (no decimals).")
            continue
        if not low <= value <= high:
            print(f"  ! {label} must be between {low:,} and {high:,}.")
            continue
        return int(value) if as_int else value


def ask_choice(label, field):
    """Ask the user to pick one option from a numbered list."""
    options = VALID_CATEGORIES[field]
    menu = "  ".join(f"{i}) {opt}" for i, opt in enumerate(options, start=1))
    while True:
        text = read_text(f"{label} - {menu}: ")
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        for opt in options:                       # also accept the typed name
            if text.lower() == opt.lower():
                return opt
        print(f"  ! Please type a number from 1 to {len(options)} (or the option name).")


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found.")
        print("Please run  python loan_approval.py  first to train and save the model.")
        sys.exit(1)
    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:                    # damaged or incompatible file
        print(f"Could not load the model ({error}). Re-run loan_approval.py to rebuild it.")
        sys.exit(1)


def predict_loan(model, applicant: dict):
    """Return (decision text, probability of approval) for one applicant."""
    row = add_features(pd.DataFrame([applicant]))[MODEL_FEATURES]
    probability = float(model.predict_proba(row)[0][1])
    decision = "APPROVED" if probability >= 0.5 else "REJECTED"
    return decision, probability


def main():
    print("=" * 56)
    print("        LOAN APPROVAL PREDICTION SYSTEM")
    print("=" * 56)
    print("Type 'q' at any question to exit.\n")
    model = load_model()

    try:
        while True:
            applicant = {
                "Age": ask_number("Age", "Age"),
                "MonthlyIncome": ask_number("Monthly income (Rs)", "MonthlyIncome"),
                "LoanAmount": ask_number("Loan amount (Rs)", "LoanAmount"),
                "CreditScore": ask_number("Credit score", "CreditScore"),
                "EmploymentStatus": ask_choice("Employment status", "EmploymentStatus"),
                "ExistingLoans": ask_number("Existing active loans", "ExistingLoans"),
                "LoanTerm": ask_number("Loan term in months", "LoanTerm"),
                "PropertyArea": ask_choice("Property area", "PropertyArea"),
            }
            try:
                decision, probability = predict_loan(model, applicant)
            except Exception as error:
                print(f"\nSorry, the prediction failed ({error}). Please try again.\n")
                continue

            print("\n" + "-" * 56)
            print(f"  LOAN PREDICTION : {decision}")
            print(f"  Approval chance : {probability:.1%}")
            print("-" * 56)

            again = read_text("\nCheck another applicant? (y/n): ").lower()
            if again not in ("y", "yes"):
                break
            print()
    except QuitProgram:
        pass
    print("\nThank you for using the Loan Approval Prediction System.")


if __name__ == "__main__":
    main()
