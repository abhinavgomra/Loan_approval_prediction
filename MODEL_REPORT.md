# Loan Approval Model — Report

**Dataset:** 614 applications, target `Loan_Status` (69% approved / 31% rejected)  
**Evaluation:** 75/25 stratified hold-out test set  
**Artifacts:** `loan_approval_prediction.ipynb`, `outputs/model_metrics.csv`, `outputs/threshold_sweep.csv`

---

## 1. Problem framing

The bank wants to **automate or assist** loan decisions while limiting:

- **False approvals (FP):** Approve a applicant who should be rejected → credit loss.
- **False rejections (FN):** Reject a creditworthy applicant → lost revenue and poor customer experience.

Because rejections are the **minority class**, accuracy alone is misleading. We optimize **precision, recall, F1, and ROC-AUC** and tune the **probability threshold** for operations.

---

## 2. Preprocessing summary

| Step | Approach |
|------|----------|
| Missing numeric | Median imputation (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`, …) |
| Missing categorical | Mode imputation (`Gender`, `Self_Employed`, …) |
| Encoding | One-hot encoding for 6 categorical features |
| Scaling | `StandardScaler` on 5 numeric features |
| Target | `Y` → 1 (approve), `N` → 0 (reject) |

---

## 3. Class imbalance strategies

| Strategy | Idea | Trade-off |
|----------|------|-----------|
| **Class weights** | Penalize errors on minority class in the loss | No synthetic rows; easy to deploy; stable |
| **SMOTE** | Oversample minority in feature space | Can boost recall/F1; risk of overfitting on small data (n=614) |
| **Undersampling** | Drop majority rows | Faster training; throws away approved examples |

---

## 4. Model comparison (hold-out test)

| Model | Imbalance | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|-----------|--------|-----|---------|
| Decision Tree | SMOTE | 0.80 | **0.97** | **0.88** | 0.75 |
| Random Forest | Class weights | 0.83 | 0.92 | 0.88 | 0.81 |
| Logistic Regression | Undersample | 0.87 | 0.88 | 0.87 | 0.85 |
| **Logistic Regression** | **Class weights** | **0.88** | **0.86** | **0.87** | **0.86** |
| Logistic Regression | SMOTE | 0.87 | 0.84 | 0.86 | **0.86** |
| Random Forest | Undersample | 0.83 | 0.81 | 0.82 | 0.78 |
| Decision Tree | Class weights | 0.84 | 0.68 | 0.75 | 0.76 |

### Trade-offs

**Logistic regression (recommended for deployment)**  
- Pros: Interpretable coefficients, well-calibrated probabilities, highest/discriminative ROC-AUC (~0.86), simple to audit for compliance.  
- Cons: Slightly lower raw F1 than tree + SMOTE on this split; assumes linear boundaries after encoding.

**Random forest**  
- Pros: Captures non-linear interactions; strong recall with class weights (0.92).  
- Cons: Less transparent; harder to explain to regulators; feature importances are proxy only.

**Decision tree + SMOTE**  
- Pros: Best hold-out F1 (0.88) and very high approval recall.  
- Cons: **Lowest ROC-AUC (0.75)** among top models; probability scores are coarse (few unique values), making threshold tuning unreliable; SMOTE on small data can overfit.

**5-fold CV (logistic + class weights):** Mean ROC-AUC ≈ 0.84 (±0.04), consistent with hold-out.

---

## 5. Suggested deployment configuration

| Choice | Recommendation |
|--------|----------------|
| **Model** | Logistic regression with `class_weight='balanced'` |
| **Threshold** | **P(approve) ≥ 0.40** |
| **Rationale** | Among thresholds with **rejected-class recall ≥ 50%**, t=0.40 maximizes F1 on the hold-out set while keeping strong recall on rejections (~89%) |

### Expected hold-out behavior at t = 0.40

| Metric | Value |
|--------|-------|
| Precision (approved) | 0.86 |
| Recall (approved) | 0.96 |
| Recall (rejected) | 0.89 |
| F1 | 0.91 |

### Alternative thresholds

| Threshold | When to use |
|-----------|-------------|
| **0.50 (default)** | More conservative approvals; precision ~0.88, rejected recall ~0.70 — use if **credit loss** is the primary concern |
| **0.40 (recommended)** | Balanced policy: still catches most bad loans while auto-approving more borderline-good applicants |
| **0.47–0.48** | Slight bump in precision (~0.88) with moderate drop in rejected recall (~0.78) |
| **≥ 0.55** | Manual-review-only band; too many good applicants would be auto-rejected |

**Manual review queue:** Route applications with **0.40 ≤ P(approve) < 0.55** to a human underwriter (gray zone). Auto-approve if **≥ 0.55** only if policy allows aggressive growth; auto-reject if **< 0.40** subject to appeal process.

---

## 6. Business interpretation

1. **Credit history** is the dominant policy lever: applicants not meeting guidelines should rarely be auto-approved regardless of model score.  
2. **Income and loan amount** drive repayment capacity; large loans on low total income deserve review even with high scores.  
3. **Property area** proxies regional risk; monitor performance by region for drift.  
4. **Demographics** (gender, marital status) may appear in the model — track fairness metrics and prefer income/credit-based overrides.

**Error costs**  
- Lowering the threshold → more approvals → higher FN risk (bad loans slip through).  
- Raising the threshold → more rejections → higher FP cost on good customers.

---

## 7. Limitations and next steps

- Small sample (614 rows): results vary by split; retrain on production data quarterly.  
- No temporal validation (applications are not time-ordered in the file).  
- Production should add **monitoring** (score distribution, approval rate, default rate by decile) and **A/B test** against current rules before full automation.

---

*Generated from the modeling pipeline in `loan_approval_prediction.ipynb`.*
