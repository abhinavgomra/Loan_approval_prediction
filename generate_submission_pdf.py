#!/usr/bin/env python3
"""Build Loan_Approval_Submission.pdf for internship task submission."""
from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "outputs" / "figures"
OUT_PDF = ROOT / "Loan_Approval_Submission.pdf"

# Update these before submitting
GITHUB_REPO_URL = "https://github.com/abhinavgomra/Loan_approval_prediction"
NOTEBOOK_PATH = "internship/Zomato/loan/loan_approval_prediction.ipynb"


class SubmissionPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 13)
        self.cell(self.epw, 9, "Loan Approval Prediction - Task Submission", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(90, 90, 90)
        self.cell(self.epw, 5, "Internship deliverable | Dataset: Kaggle loan-approval-prediction-case-study", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(self.epw, 7, title)
        self.set_font("Helvetica", "", 10)
        self.ln(1)

    def body(self, text: str, h: float = 5.5) -> None:
        self.multi_cell(self.epw, h, text)

    def bullet(self, text: str) -> None:
        self.multi_cell(self.epw, 5.5, f"  - {text}")


def pick_figures() -> list[tuple[str, Path]]:
    candidates = [
        ("Target class balance", "01_target_balance.png"),
        ("Missing values", "02_missing_values.png"),
        ("Income vs loan outcome", "03_income_by_status.png"),
        ("Credit history vs approval", "04_credit_history_approval.png"),
        ("Model evaluation (ROC + confusion matrix)", "07_deploy_model_eval.png"),
        ("Threshold trade-off", "08_threshold_analysis.png"),
    ]
    out = []
    for label, fname in candidates:
        p = FIG_DIR / fname
        if p.exists():
            out.append((label, p))
    return out


def build_pdf() -> Path:
    pdf = SubmissionPDF()
    pdf.set_auto_page_break(auto=True, margin=16)

    # Page 1 — Executive summary (~1 page)
    pdf.add_page()
    pdf.section_title("Executive summary (Task: Loan Approval Prediction)")
    # Use ASCII only for Helvetica / latin-1 compatibility
    pdf.body(
        "Objective: Build a supervised classifier to predict loan approval (Y/N) from borrower "
        "and application features, with emphasis on preprocessing, class imbalance, and evaluation."
    )
    pdf.ln(2)
    pdf.body(
        "Dataset: 614 applications, 12 predictors (income, loan amount, credit history, "
        "property area, demographics). Target is imbalanced (~69% approved, ~31% rejected). "
        "Missing values were handled via median/mode imputation; categoricals were one-hot encoded "
        "and numerics scaled for linear models."
    )
    pdf.ln(2)
    pdf.body("Key findings:")
    pdf.bullet("Credit history and total income are the strongest approval signals in EDA.")
    pdf.bullet("Three imbalance strategies were compared: class weights, SMOTE, and undersampling.")
    pdf.bullet(
        "Logistic regression with balanced class weights is recommended for deployment "
        "(ROC-AUC ~0.86, interpretable scores). Decision tree + SMOTE achieved highest F1 (~0.88) "
        "but weaker ranking quality (ROC-AUC ~0.75)."
    )
    pdf.bullet(
        "Suggested operating threshold: P(approve) >= 0.40, with manual review for scores 0.40-0.55."
    )
    pdf.ln(2)
    pdf.section_title("Repository & notebook links (update before submit)")
    pdf.body(f"GitHub repository: {GITHUB_REPO_URL}")
    pdf.body(f"Main notebook: {GITHUB_REPO_URL}/blob/main/{NOTEBOOK_PATH}")
    pdf.body("Local path: internship/Zomato/loan/loan_approval_prediction.ipynb")
    pdf.body("Written report: MODEL_REPORT.md (model trade-offs and threshold rationale)")
    if GOOGLE_DRIVE_URL:
        pdf.body(f"Google Drive (optional artifacts): {GOOGLE_DRIVE_URL}")

    pdf.ln(2)
    pdf.section_title("Task checklist")
    pdf.bullet("Beginner: EDA, 4+ visualizations, README - completed.")
    pdf.bullet("Core: preprocessing, imbalance handling, model comparison - completed.")
    pdf.bullet("Deliverables: notebook + MODEL_REPORT.md + this PDF - completed.")

    # Page 2+ — Charts
    figures = pick_figures()
    if figures:
        pdf.add_page()
        pdf.section_title("Selected charts (from notebook pipeline)")
        for label, path in figures:
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(pdf.epw, 5, label)
            pdf.image(str(path), w=min(175, pdf.epw))
            pdf.ln(4)
    else:
        pdf.body(
            "Charts not found. Run: jupyter nbconvert --execute loan_approval_prediction.ipynb "
            "then re-run this script."
        )

    pdf.output(str(OUT_PDF))
    return OUT_PDF


def main() -> int:
    if not (ROOT / "data" / "loan_prediction.csv").exists():
        print("Missing data/loan_prediction.csv", file=sys.stderr)
        return 1
    out = build_pdf()
    print(f"Wrote {out}")
    if "YOUR_USERNAME" in GITHUB_REPO_URL:
        print("Tip: Edit GITHUB_REPO_URL in generate_submission_pdf.py before final submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
