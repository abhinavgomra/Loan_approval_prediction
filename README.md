# Loan Approval Predictor

End-to-end supervised learning project to predict whether a loan application is **approved (Y)** or **rejected (N)** using borrower and application features.

**Dataset:** [Loan Approval Prediction: Case Study](https://www.kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study) (Kaggle, `bhanupratapbiswas`)

## Dataset overview

| Item | Detail |
|------|--------|
| File | `data/loan_prediction.csv` |
| Rows | 614 loan applications |
| Features | 12 predictors + `Loan_Status` target |
| Target | `Loan_Status`: `Y` (approved) / `N` (rejected) |

### Features

| Column | Description |
|--------|-------------|
| `Loan_ID` | Unique application identifier (dropped for modeling) |
| `Gender` | Applicant gender |
| `Married` | Marital status |
| `Dependents` | Number of dependents (`0`, `1`, `2`, `3+`) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employment flag |
| `ApplicantIncome` | Applicant monthly income |
| `CoapplicantIncome` | Co-applicant monthly income |
| `LoanAmount` | Loan amount (in thousands) |
| `Loan_Amount_Term` | Term in months |
| `Credit_History` | 1 = meets credit guidelines, 0 = does not |
| `Property_Area` | Urban / Semiurban / Rural |

### Initial observations (EDA)

1. **Class imbalance:** ~69% approved (`Y`) vs ~31% rejected (`N`). Models that optimize accuracy alone can look good while missing rejections.
2. **Missing values:** Notable gaps in `Credit_History` (50), `Self_Employed` (32), `LoanAmount` (22), `Dependents` (15), `Gender` (13), and smaller counts elsewhere. `Credit_History` is a strong business signal and needs careful imputation.
3. **Mixed types:** Categorical fields (`Gender`, `Property_Area`, …) and numeric income/loan fields; encoding and scaling are required before linear models.
4. **Credit history:** When present, `Credit_History = 1` aligns strongly with approval—useful for both modeling and business rules.
5. **Income & loan amount:** Right-skewed distributions; median imputation for missing numeric fields and scaling help tree and linear models alike.

## Project structure

```
loan/
├── data/loan_prediction.csv      # Raw dataset (from Kaggle)
├── loan_approval_prediction.ipynb # EDA + preprocessing + modeling
├── MODEL_REPORT.md               # Model trade-offs & deployment threshold
├── requirements.txt
└── README.md
```

## Setup

```bash
cd loan
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download data (requires [Kaggle API](https://www.kaggle.com/docs/api) credentials):

```bash
kaggle datasets download -d bhanupratapbiswas/loan-approval-prediction-case-study -p data --unzip
```

## Run in Jupyter (browser, cell by cell)

```bash
cd internship/Zomato/loan
python3 -m venv .venv          # skip if .venv already exists
source .venv/bin/activate
pip install -r requirements.txt

# Start Jupyter in your browser
jupyter notebook
# OR: jupyter lab
```

1. Your browser opens (usually `http://localhost:8888`).
2. Click **`loan_approval_prediction.ipynb`**.
3. Run cells yourself: select a cell → **Shift + Enter** (runs and moves down).
4. Run all from top: menu **Kernel → Restart & Run All** (only when you want the full pipeline at once).

**Tip:** Use **Kernel → Restart & Clear Output** if you want to re-run everything from scratch.

Dataset must be at `data/loan_prediction.csv` (see Kaggle download above).

## Deliverables

- **Notebook:** Full pipeline—EDA (4+ plots), preprocessing, imbalance strategies (SMOTE, undersampling, class weights), model comparison (logistic regression, decision tree, random forest), metrics (precision, recall, F1, ROC-AUC), and business interpretation.
- **Report:** `MODEL_REPORT.md` — trade-offs between models and a suggested probability threshold for deployment.
- **Submission:** See `SUBMISSION.md` for portal steps; run `python generate_submission_pdf.py` to create `Loan_Approval_Submission.pdf`.
# Loan_approval_predictor
