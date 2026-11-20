# HireWise AI — MLOps Model Monitoring & Retraining Assessment Report

**Execution Timestamp:** `2026-11-20T05:00:57.506499`  
**Project:** HireWise AI (Resume Inflation Detector)  
**Quality Gate Status:** `PROMOTED_TO_CHAMPION`  

---

## 1. Task A — Data Ingestion & Validation Summary
- **Incoming Batch Size:** `300` candidate interview records.
- **Required Columns Validated:** `candidate_id, claimed_skill, measured_ability, difference, is_inflated`
- **Missing Values:** `0` (clean data stream).
- **Duplicates Found & Resolved:** `0`
- **Target Distribution (`is_inflated`):**
  - **Honest (Class 0):** `182` records (`60.7%`)
  - **Inflated (Class 1):** `118` records (`39.3%`)

---

## 2. Task B — Existing Model (Champion) Evaluation on New Batch
The currently deployed champion model was evaluated against the newly arrived labelled batch:

| Metric | Champion Value | Status |
| :--- | :--- | :--- |
| **Accuracy** | `28.67%` | Evaluated |
| **Precision** | `30.95%` | Evaluated |
| **Recall** | `66.10%` | Evaluated |
| **F1-Score** | `42.16%` | Evaluated |

- **Confusion Matrix:** TN=`8`, FP=`174`, FN=`40`, TP=`78`

---

## 3. Task C — Candidate Model Retraining
- **Combined Training Samples:** `1300` (1,000 baseline historical + 300 incoming batch).
- **Model Architecture:** `RandomForestClassifier(n_estimators=100, max_depth=6)`
- **Candidate Model Artifact:** Saved to `mlops/models/candidate_model.joblib`
- **MLflow Tracking Run ID:** `d335d0ee6b7546048067d8ac4a6078e8`

| Metric | Candidate Value (Holdout) | Candidate Value (New Batch) |
| :--- | :--- | :--- |
| **Accuracy** | `93.85%` | `94.00%` |
| **Precision** | `96.84%` | - |
| **Recall** | `87.62%` | - |
| **F1-Score** | `92.00%` | `92.11%` |

---

## 4. Task D — Quality Gate Decision & Metric Justification

### Selected Metric: **F1-Score**
> **Justification:**  
> F1-Score was selected as the primary quality gate metric because Resume Inflation Detection is an imbalanced classification problem where both False Positives and False Negatives have high business costs. A False Positive wrongly penalizes an honest applicant, damaging candidate experience; whereas a False Negative allows an unqualified applicant to slip into technical rounds, wasting engineering interview bandwidth. F1-Score takes the harmonic mean of Precision and Recall, preventing the accuracy paradox.

### Decision Evaluation:
1. **Minimum Performance Floor (F1 >= 0.82):** `PASSED` (`0.9200`)
2. **Outperformed / Matched Champion (Candidate >= Champion):** `PASSED` (`0.9200` vs `0.4216`)

### Final Decision:
**`PROMOTED_TO_CHAMPION`**  
*Candidate model achieved F1-Score of 0.9200 (>= 0.82) and outperformed/tied the existing champion (0.4216). Promoted to production.*

---

## 5. Artifacts Generated
- `mlops/data/baseline_data.csv` (1,000 samples)
- `mlops/data/incoming_batch.csv` (300 samples)
- `mlops/models/champion_model.joblib` (Production Champion Model)
- `mlops/models/candidate_model.joblib` (Newly Retrained Candidate Model)
- `mlops/reports/evaluation_report.json` (Structured JSON Metrics)
- `mlops/reports/evaluation_report.md` (Markdown Report for Submission)
- `mlops/reports/quality_gate_decision.json` (Quality Gate Audit Trail)
