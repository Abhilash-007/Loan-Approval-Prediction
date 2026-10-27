# Loan Approval Prediction System

A beginner-friendly machine learning project that predicts whether a loan application
will be **Approved** or **Rejected** from the applicant's details and financial information.
It covers the full workflow: data creation, cleaning, EDA, visualization, model training,
evaluation and a console-based prediction app.

## Project overview

| Item | Details |
|---|---|
| Problem type | Binary classification (0 = Rejected, 1 = Approved) |
| Dataset | Synthetic, 2,030 raw rows (2,000 after removing duplicates) |
| Models compared | Logistic Regression, Decision Tree, Random Forest |
| Best model | Random Forest (highest F1-score) |
| Interface | Console app with input validation (`predict_cli.py`) |

## Project structure

```
loan_approval_project/
├── Loan_Approval_Prediction.ipynb   # Notebook: full walkthrough with outputs and charts
├── loan_approval.py                 # Same pipeline as a script (clean -> EDA -> train -> save)
├── generate_dataset.py              # Creates the synthetic dataset
├── loan_common.py                   # Shared settings, validation ranges, feature engineering
├── predict_cli.py                   # Interactive console prediction system
├── app.py                           # Streamlit web app (prediction form + insights)
├── requirements.txt
├── data/
│   ├── loan_data.csv                # Raw dataset (has missing values + duplicates on purpose)
│   └── loan_data_cleaned.csv        # Dataset after preprocessing
├── models/loan_model.joblib         # Saved trained pipeline
└── images/                          # All saved charts (PNG)
```

## Dataset generation

`generate_dataset.py` uses NumPy and Pandas (fixed seed 42, so results are repeatable):

1. Creates 2,000 applicants with Gender, Age, EmploymentStatus, PropertyArea.
2. Income depends on employment type and age; credit score depends on income and employment;
   applicants with lower scores tend to hold more existing loans.
3. Loan amount is based on income and loan term.
4. `LoanApproved` is produced from a "bank policy" score: higher credit score, higher income,
   being employed, a small instalment compared with income and few existing loans raise the
   chance of approval; random noise is added so the data is not perfectly predictable.
5. Adds ~2-3% missing values and 30 duplicate rows so preprocessing has real work to do.

## Libraries and tools

Python 3, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, joblib, Jupyter Notebook.

## Steps to run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 2. install libraries
pip install -r requirements.txt

# 3. create the dataset (optional - loan_approval.py does it automatically if missing)
python generate_dataset.py

# 4. run the full pipeline: cleaning, EDA, charts, training, saving the model
python loan_approval.py

# 5. start the interactive prediction system
python predict_cli.py
```

To see the charts and explanations step by step, open `Loan_Approval_Prediction.ipynb`
in Jupyter Notebook or VS Code and choose *Run All*.

## Web app (Streamlit)

```bash
streamlit run app.py
```

## What the pipeline does

1. **Preprocessing:** removes duplicates, fills missing numbers with the median and missing
   categories with the most frequent value, validates every column against realistic ranges,
   and adds the feature `EMIToIncomeRatio` (monthly instalment / monthly income).
2. **Encoding and scaling:** one-hot encoding for categories and standard scaling for numbers,
   built into a scikit-learn `Pipeline`, so new applicants are processed the same way.
3. **Feature selection:** `ApplicantID` (just a label) and `Gender` (a lending decision should
   not depend on gender) are excluded from training.
4. **Training:** 80/20 stratified train-test split, three models, plus 5-fold cross-validation.
5. **Evaluation:** accuracy, precision, recall, F1-score and confusion matrix.

## Key insights from the analysis

- **Credit score is the strongest factor.** Approval rate rises from about 1% (score 550 or
  below) to 30% (551-650), 66% (651-750) and 92% (above 750). It is also the top feature in the
  Random Forest importance ranking (about 38%).
- **Income matters:** approval rate climbs from 33% in the lowest income quarter to 77% in the highest.
- **Employment:** employed (64%) and self-employed (60%) applicants are approved at similar rates,
  while unemployed applicants are rarely approved (11%).
- **Debt burden:** a high instalment compared with income, or several existing loans, lowers the chance of approval.
- **Gender and property area** have little effect (men 59%, women 56%; urban 60%, rural 56%).
- Income and loan amount are positively related (see the scatter plot), but a bigger loan is not
  automatically rejected: what counts is whether the instalment is affordable.

> The dataset is synthetic, so these patterns reflect the rules used to generate it. On real bank
> data the numbers would differ.

## Model performance (20% test set, 400 applicants)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.825 | 0.832 | 0.875 | 0.853 |
| Decision Tree | 0.808 | 0.835 | 0.832 | 0.834 |
| **Random Forest** | **0.832** | **0.834** | **0.888** | **0.860** |

Confusion matrix of the Random Forest: 127 correct rejections, 206 correct approvals,
41 wrongly approved, 26 wrongly rejected.

Accuracy is around 83% rather than 99% on purpose: real approval decisions contain randomness
that no model can predict, and the synthetic data includes that noise.

## Error handling in the console app

- Every number is checked: text, blank values, decimals where a whole number is needed and
  out-of-range values are rejected with a clear message, and the question is asked again.
- Categories are chosen from a numbered menu (typing the name also works).
- Typing `q`, pressing Ctrl+C or ending the input exits cleanly without a crash.
- A missing or damaged model file shows a helpful message telling the user what to do.

## Ideas to extend the project

Hyperparameter tuning with `GridSearchCV`, SHAP explanations, deploying the model online.


## Streamlit app link 

https://loan-approval-prediction-fmsqyagmhajqn9eykqwewh.streamlit.app/

