# How to Submit — Loan Approval Prediction

Follow these steps for **each task** on the internship portal.

---

## Step 1: Prepare your artifacts

| Artifact | Location | Notes |
|----------|----------|--------|
| Jupyter notebook | `loan_approval_prediction.ipynb` | Run end-to-end; outputs visible |
| README | `README.md` | Dataset + setup + reproduction |
| Model report | `MODEL_REPORT.md` | Trade-offs + threshold |
| Submission PDF | `Loan_Approval_Submission.pdf` | Executive summary + charts |
| Metrics CSV | `outputs/model_metrics.csv` | Optional proof of results |

### Generate the submission PDF

```bash
cd internship/Zomato/loan
source .venv/bin/activate
pip install fpdf2   # if not installed

# Ensure notebook has been executed (charts in outputs/figures/)
JUPYTER_CONFIG_DIR=.jupyter jupyter nbconvert --execute loan_approval_prediction.ipynb

# Build PDF
python generate_submission_pdf.py
```

Edit `GITHUB_REPO_URL` in `generate_submission_pdf.py` **before** generating the final PDF.

### Optional: export notebook with screenshots

In Jupyter: **File → Save and Export Notebook As → PDF** (or HTML), if your portal accepts a second document with cell outputs.

---

## Step 2: Push to GitHub

```bash
cd /path/to/your/repo
git init   # skip if repo already exists
git add internship/Zomato/loan/
git add internship/Zomato/loan/.gitignore
# Do NOT commit .venv/ or Kaggle credentials
git commit -m "Add loan approval prediction project (EDA, modeling, report)"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Do not push:** `.venv/`, `~/.kaggle/kaggle.json`, large model pickles (use Drive instead).

**Share these links:**

- Repo: `https://github.com/YOUR_USERNAME/YOUR_REPO`
- Notebook: `https://github.com/YOUR_USERNAME/YOUR_REPO/blob/main/internship/Zomato/loan/loan_approval_prediction.ipynb`
- README: `https://github.com/YOUR_USERNAME/YOUR_REPO/blob/main/internship/Zomato/loan/README.md`

---

## Step 3: Upload large files (if any)

If you add saved models (`.pkl`, `.joblib`) or datasets > 25 MB:

1. Upload folder to **Google Drive**
2. Set sharing to **Anyone with the link → Viewer**
3. Paste the Drive link in the submission form and in `TASK_SUBMISSION_LINKS.txt`

The raw CSV (~45 KB) can stay in GitHub.

---

## Step 4: Submit on the Task Submission page

Copy from `TASK_SUBMISSION_LINKS.txt` (fill in your URLs), e.g.:

```
Task: Loan Approval Prediction

GitHub repo: https://github.com/...
Notebook: https://github.com/.../loan_approval_prediction.ipynb
PDF report: https://github.com/.../Loan_Approval_Submission.pdf
  (or Google Drive link if PDF is uploaded there)
Model report (MD): https://github.com/.../MODEL_REPORT.md
Google Drive (optional): ...
```

Label each link with the **task number** exactly as the portal asks.

---

## Reviewer checklist (what you already have)

- [x] Sections in notebook (EDA → preprocess → imbalance → models → threshold → business notes)
- [x] 4+ visualizations
- [x] README with dataset description
- [x] Preprocessing, imbalance, logistic + tree models, metrics
- [x] MODEL_REPORT with deployment threshold
- [ ] Update `GITHUB_REPO_URL` in PDF generator
- [ ] Push to GitHub and paste links on portal
- [ ] Upload `Loan_Approval_Submission.pdf` (GitHub or Drive)

---

## Tips

1. **One PDF per task** — executive summary on page 1, charts on following pages.
2. **Reproducibility** — `requirements.txt` + README commands let reviewers rerun in &lt; 5 minutes.
3. **Tidy notebook** — clear markdown headers; avoid huge cell outputs (clear outputs if needed).
4. **Screenshots** — if the portal wants screenshots, export key cells (EDA plots, metrics table) from Jupyter or use the PNGs in `outputs/figures/`.
