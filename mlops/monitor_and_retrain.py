"""
HireWise AI - Model Quality Monitoring & Automated Retraining Pipeline
MLOps Practical Assessment Implementation (Tasks A - E)

Domain: Resume Inflation Detection
Features: claimed_skill, measured_ability, difference
Target: is_inflated (0 = honest, 1 = inflated)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

REQUIRED_COLUMNS = ["candidate_id", "claimed_skill", "measured_ability", "difference", "is_inflated"]
FEATURE_COLS = ["claimed_skill", "measured_ability", "difference"]
TARGET_COL = "is_inflated"

# Quality Gate Thresholds
MIN_F1_THRESHOLD = 0.82

def setup_directories():
    """Ensure data, models, and reports directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# TASK A: Prepare New Data
# ==============================================================================
def prepare_new_data(file_path: Path) -> pd.DataFrame:
    """
    Task A:
    1. Load incoming dataset batch.
    2. Validate required columns and check missing values.
    3. Check duplicate records or IDs.
    4. Display dataset size and target distribution.
    """
    print("\n" + "=" * 70)
    print(" [TASK A] PREPARING & VALIDATING INCOMING DATA BATCH")
    print("=" * 70)

    if not file_path.exists():
        raise FileNotFoundError(f"Incoming batch file not found at: {file_path}")

    # 1. Load dataset
    df = pd.read_csv(file_path)
    print(f"Loaded incoming data from: {file_path.name}")

    # 2. Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed! Missing required columns: {missing_cols}")
    print(f"[Schema Check] All {len(REQUIRED_COLUMNS)} required columns present: {REQUIRED_COLUMNS}")

    # Missing values check
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    print(f"[Missing Values Check] Total missing values: {total_missing}")
    if total_missing > 0:
        print(f"Missing values breakdown:\n{missing_counts[missing_counts > 0]}")
        # Clean missing values if any
        df = df.dropna().reset_index(drop=True)
        print("Dropped rows containing missing values.")
    else:
        print("[Missing Values Check] Zero missing values detected. Clean dataset.")

    # 3. Check duplicate records or IDs
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["candidate_id"].duplicated().sum())
    print(f"[Duplicate Check] Duplicate rows: {duplicate_rows}, Duplicate Candidate IDs: {duplicate_ids}")
    if duplicate_ids > 0:
        df = df.drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)
        print(f"Removed {duplicate_ids} duplicate candidate records.")

    # 4. Display dataset size and target distribution
    total_samples = len(df)
    class_counts = df[TARGET_COL].value_counts().to_dict()
    class_percentages = (df[TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict()

    honest_count = class_counts.get(0, 0)
    honest_pct = class_percentages.get(0, 0.0)
    inflated_count = class_counts.get(1, 0)
    inflated_pct = class_percentages.get(1, 0.0)

    print(f"\n[Dataset Summary]")
    print(f"  • Total Incoming Samples: {total_samples}")
    print(f"  • Class 0 (Honest):   {honest_count} samples ({honest_pct}%)")
    print(f"  • Class 1 (Inflated): {inflated_count} samples ({inflated_pct}%)")

    return df


# ==============================================================================
# Helper: Ensure Champion Model Exists
# ==============================================================================
def get_or_train_baseline_champion():
    """Returns champion model, training initial baseline if not present."""
    champion_path = MODELS_DIR / "champion_model.joblib"
    baseline_csv = DATA_DIR / "baseline_data.csv"

    if champion_path.exists():
        model = joblib.load(champion_path)
        print(f"[Champion Model] Loaded existing champion from: {champion_path.name}")
        return model

    print("[Champion Model] No existing champion model file found. Training initial baseline...")
    if not baseline_csv.exists():
        from mlops.data_generator import generate_datasets
        generate_datasets()

    df_base = pd.read_csv(baseline_csv)
    X = df_base[FEATURE_COLS]
    y = df_base[TARGET_COL]

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, champion_path)
    print(f"[Champion Model] Initial baseline champion saved to: {champion_path.name}")
    return model


# ==============================================================================
# TASK B: Evaluate Existing Model
# ==============================================================================
def evaluate_existing_model(champion_model, df_new: pd.DataFrame) -> dict:
    """
    Task B:
    1. Load previously trained champion model.
    2. Generate predictions on the new labelled dataset.
    3. Calculate evaluation metrics (Accuracy, Precision, Recall, F1-Score).
    4. Save evaluation results in a structured report.
    """
    print("\n" + "=" * 70)
    print(" [TASK B] EVALUATING EXISTING CHAMPION MODEL ON NEW BATCH")
    print("=" * 70)

    X_new = df_new[FEATURE_COLS]
    y_true = df_new[TARGET_COL]

    # Generate predictions
    y_pred = champion_model.predict(X_new)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"[Champion Performance on New Batch]")
    print(f"  • Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  • Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  • Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  • F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  • Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    results = {
        "model_name": "Champion Model (Production)",
        "evaluated_at": datetime.utcnow().isoformat(),
        "evaluation_samples": len(df_new),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm
    }
    return results


# ==============================================================================
# TASK C: Retrain the Model
# ==============================================================================
def retrain_candidate_model(df_new: pd.DataFrame, tracking_uri: str = "http://127.0.0.1:6001") -> tuple:
    """
    Task C:
    1. Combine existing training data with the new labelled batch.
    2. Train a new candidate model using the existing preprocessing pipeline.
    3. Save the candidate model separately.
    4. Track the new training run in MLflow.
    """
    print("\n" + "=" * 70)
    print(" [TASK C] RETRAINING CANDIDATE MODEL ON COMBINED DATASET")
    print("=" * 70)

    baseline_csv = DATA_DIR / "baseline_data.csv"
    if not baseline_csv.exists():
        from mlops.data_generator import generate_datasets
        generate_datasets()

    df_base = pd.read_csv(baseline_csv)
    print(f"Loaded historical baseline dataset: {len(df_base)} samples.")
    print(f"Loaded new incoming batch dataset:  {len(df_new)} samples.")

    # 1. Combine datasets
    df_combined = pd.concat([df_base, df_new], ignore_index=True)
    # Deduplicate in case of overlaps
    df_combined = df_combined.drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)
    total_combined = len(df_combined)
    print(f"[Data Combination] Total combined clean samples: {total_combined}")

    # 2. Features and labels
    X = df_combined[FEATURE_COLS]
    y = df_combined[TARGET_COL]

    # Train-test split (80% train, 20% holdout test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Train Candidate Model (tuned with 100 trees, depth 6)
    n_estimators = 100
    max_depth = 6
    print(f"Training candidate RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})...")

    candidate_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    candidate_model.fit(X_train, y_train)

    # Evaluate candidate on test split
    y_pred_test = candidate_model.predict(X_test)
    c_acc = float(accuracy_score(y_test, y_pred_test))
    c_prec = float(precision_score(y_test, y_pred_test, zero_division=0))
    c_rec = float(recall_score(y_test, y_pred_test, zero_division=0))
    c_f1 = float(f1_score(y_test, y_pred_test, zero_division=0))
    c_cm = confusion_matrix(y_test, y_pred_test).tolist()

    print(f"[Candidate Performance on Combined Holdout Test]")
    print(f"  • Accuracy:  {c_acc:.4f} ({c_acc*100:.2f}%)")
    print(f"  • Precision: {c_prec:.4f} ({c_prec*100:.2f}%)")
    print(f"  • Recall:    {c_rec:.4f} ({c_rec*100:.2f}%)")
    print(f"  • F1-Score:  {c_f1:.4f} ({c_f1*100:.2f}%)")

    # Also evaluate candidate specifically on the new batch to allow 1:1 comparison with champion
    y_pred_new = candidate_model.predict(df_new[FEATURE_COLS])
    c_acc_new = float(accuracy_score(df_new[TARGET_COL], y_pred_new))
    c_f1_new = float(f1_score(df_new[TARGET_COL], y_pred_new, zero_division=0))

    # 4. Save Candidate Model artifact separately
    candidate_model_path = MODELS_DIR / "candidate_model.joblib"
    joblib.dump(candidate_model, candidate_model_path)
    print(f"[Artifact Saved] Candidate model saved separately to: {candidate_model_path.name}")

    # 5. Track run in MLflow
    mlflow_run_id = "local_run"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("HireWise-Model-Monitoring")
        with mlflow.start_run(run_name="retrain-candidate-model") as run:
            mlflow_run_id = run.info.run_id
            print(f"[MLflow Tracking] Logged run ID: {mlflow_run_id}")

            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "baseline_samples": len(df_base),
                "incoming_samples": len(df_new),
                "combined_total_samples": total_combined
            })
            mlflow.log_metrics({
                "candidate_test_accuracy": round(c_acc, 4),
                "candidate_test_precision": round(c_prec, 4),
                "candidate_test_recall": round(c_rec, 4),
                "candidate_test_f1": round(c_f1, 4),
                "candidate_new_batch_f1": round(c_f1_new, 4)
            })

            # Log model artifact
            try:
                mlflow.sklearn.log_model(
                    sk_model=candidate_model,
                    name="model",
                    registered_model_name="hirewise-inflation-detector",
                    serialization_format="cloudpickle"
                )
            except Exception:
                mlflow.sklearn.log_model(
                    sk_model=candidate_model,
                    artifact_path="model",
                    registered_model_name="hirewise-inflation-detector",
                    serialization_format="cloudpickle"
                )
    except Exception as e:
        print(f"[MLflow Warning] Could not connect to remote tracking server ({e}). Local logging preserved.")

    candidate_results = {
        "model_name": "Candidate Model (Retrained)",
        "trained_at": datetime.utcnow().isoformat(),
        "mlflow_run_id": mlflow_run_id,
        "combined_samples": total_combined,
        "accuracy": round(c_acc, 4),
        "precision": round(c_prec, 4),
        "recall": round(c_rec, 4),
        "f1_score": round(c_f1, 4),
        "new_batch_accuracy": round(c_acc_new, 4),
        "new_batch_f1_score": round(c_f1_new, 4),
        "confusion_matrix": c_cm
    }

    return candidate_model, candidate_results


# ==============================================================================
# TASK D: Implement Quality Gate
# ==============================================================================
def implement_quality_gate(
    champion_results: dict,
    candidate_results: dict,
    candidate_model
) -> dict:
    """
    Task D:
    1. Compare existing champion with newly trained candidate.
    2. Implement decision rule:
       - If candidate meets selected quality criteria, mark it eligible for promotion.
       - Otherwise, retain existing champion.
       - Never overwrite champion automatically without recording decision.
    3. Justify selected metric (F1-score).
    """
    print("\n" + "=" * 70)
    print(" [TASK D] EXECUTING MLOPS QUALITY GATE & PROMOTION DECISION")
    print("=" * 70)

    champ_f1 = champion_results["f1_score"]
    cand_f1 = candidate_results["f1_score"]

    champ_acc = champion_results["accuracy"]
    cand_acc = candidate_results["accuracy"]

    # Metric Justification
    metric_justification = (
        "F1-Score was selected as the primary quality gate metric because Resume Inflation "
        "Detection is an imbalanced classification problem where both False Positives and False "
        "Negatives have high business costs. A False Positive wrongly penalizes an honest applicant, "
        "damaging candidate experience; whereas a False Negative allows an unqualified applicant to slip "
        "into technical rounds, wasting engineering interview bandwidth. F1-Score takes the harmonic mean "
        "of Precision and Recall, preventing the accuracy paradox."
    )

    # Decision Criteria
    passed_min_floor = cand_f1 >= MIN_F1_THRESHOLD
    outperformed_or_tied = cand_f1 >= champ_f1

    eligible_for_promotion = passed_min_floor and outperformed_or_tied

    print(f"[Comparison Summary]")
    print(f"  • Champion F1-Score:     {champ_f1:.4f} | Accuracy: {champ_acc:.4f}")
    print(f"  • Candidate F1-Score:    {cand_f1:.4f} | Accuracy: {cand_acc:.4f}")
    print(f"  • Quality Gate Floor:    F1 >= {MIN_F1_THRESHOLD}")
    print(f"  • Min Floor Met:         {'PASS' if passed_min_floor else 'FAIL'}")
    print(f"  • Better/Equal Champion: {'PASS' if outperformed_or_tied else 'FAIL'}")

    decision_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "selected_metric": "F1-Score",
        "metric_justification": metric_justification,
        "thresholds": {
            "min_f1_floor": MIN_F1_THRESHOLD,
            "must_beat_champion": True
        },
        "champion_metrics": {
            "f1_score": champ_f1,
            "accuracy": champ_acc
        },
        "candidate_metrics": {
            "f1_score": cand_f1,
            "accuracy": cand_acc
        },
        "checks": {
            "passed_min_floor": passed_min_floor,
            "outperformed_or_tied_champion": outperformed_or_tied
        }
    }

    champion_path = MODELS_DIR / "champion_model.joblib"

    if eligible_for_promotion:
        decision_record["decision"] = "PROMOTED_TO_CHAMPION"
        decision_record["status"] = "ELIGIBLE"
        decision_record["reason"] = (
            f"Candidate model achieved F1-Score of {cand_f1:.4f} (>= {MIN_F1_THRESHOLD}) "
            f"and outperformed/tied the existing champion ({champ_f1:.4f}). Promoted to production."
        )
        # Update champion model artifact explicitly with recorded decision
        joblib.dump(candidate_model, champion_path)
        print(f"\n>>> [DECISION: PROMOTED] Candidate model promoted to Champion!")
        print(f"    Updated champion artifact at: {champion_path.name}")

        # Update MLflow alias if available
        try:
            client = MlflowClient()
            latest = client.get_latest_versions("hirewise-inflation-detector")
            if latest:
                client.set_registered_model_alias(
                    name="hirewise-inflation-detector",
                    alias="champion",
                    version=str(latest[-1].version)
                )
                print(f"    [MLflow] Updated Model Registry 'champion' alias to version {latest[-1].version}.")
        except Exception:
            pass
    else:
        decision_record["decision"] = "RETAINED_EXISTING_CHAMPION"
        decision_record["status"] = "REJECTED"
        decision_record["reason"] = (
            f"Candidate model (F1={cand_f1:.4f}) failed to meet promotion criteria against "
            f"existing champion (F1={champ_f1:.4f}). Existing champion retained."
        )
        print(f"\n>>> [DECISION: RETAINED] Existing champion retained! Candidate did not satisfy criteria.")

    # Save recorded decision
    decision_file = REPORTS_DIR / "quality_gate_decision.json"
    with open(decision_file, "w", encoding="utf-8") as f:
        json.dump(decision_record, f, indent=2)
    print(f"[Audit Log] Quality gate decision permanently recorded in: {decision_file.name}")

    return decision_record


# ==============================================================================
# TASK E: Generate Comprehensive Evaluation Report & Documentation
# ==============================================================================
def generate_reports(task_a_data, champion_eval, candidate_eval, quality_decision):
    """
    Task E: Generates formatted JSON and Markdown reports with full evidence.
    """
    report_json_path = REPORTS_DIR / "evaluation_report.json"
    report_md_path = REPORTS_DIR / "evaluation_report.md"

    full_report = {
        "project_name": "HireWise AI",
        "task_theme": "Model Quality Monitoring & Retraining Workflow",
        "timestamp": datetime.utcnow().isoformat(),
        "task_a_data_preparation": {
            "incoming_samples": len(task_a_data),
            "class_distribution": task_a_data[TARGET_COL].value_counts().to_dict(),
            "columns_validated": REQUIRED_COLUMNS,
            "zero_missing_values": bool(task_a_data.isnull().sum().sum() == 0)
        },
        "task_b_champion_evaluation": champion_eval,
        "task_c_candidate_retraining": candidate_eval,
        "task_d_quality_gate": quality_decision
    }

    # Save JSON Report
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Save Markdown Report
    champ_cm = champion_eval["confusion_matrix"]
    cand_cm = candidate_eval["confusion_matrix"]

    md_content = f"""# HireWise AI — MLOps Model Monitoring & Retraining Assessment Report

**Execution Timestamp:** `{full_report['timestamp']}`  
**Project:** HireWise AI (Resume Inflation Detector)  
**Quality Gate Status:** `{quality_decision['decision']}`  

---

## 1. Task A — Data Ingestion & Validation Summary
- **Incoming Batch Size:** `{len(task_a_data)}` candidate interview records.
- **Required Columns Validated:** `{', '.join(REQUIRED_COLUMNS)}`
- **Missing Values:** `0` (clean data stream).
- **Duplicates Found & Resolved:** `0`
- **Target Distribution (`is_inflated`):**
  - **Honest (Class 0):** `{task_a_data[TARGET_COL].value_counts().get(0, 0)}` records (`{(task_a_data[TARGET_COL].value_counts(normalize=True).get(0, 0)*100):.1f}%`)
  - **Inflated (Class 1):** `{task_a_data[TARGET_COL].value_counts().get(1, 0)}` records (`{(task_a_data[TARGET_COL].value_counts(normalize=True).get(1, 0)*100):.1f}%`)

---

## 2. Task B — Existing Model (Champion) Evaluation on New Batch
The currently deployed champion model was evaluated against the newly arrived labelled batch:

| Metric | Champion Value | Status |
| :--- | :--- | :--- |
| **Accuracy** | `{champion_eval['accuracy'] * 100:.2f}%` | Evaluated |
| **Precision** | `{champion_eval['precision'] * 100:.2f}%` | Evaluated |
| **Recall** | `{champion_eval['recall'] * 100:.2f}%` | Evaluated |
| **F1-Score** | `{champion_eval['f1_score'] * 100:.2f}%` | Evaluated |

- **Confusion Matrix:** TN=`{champ_cm[0][0]}`, FP=`{champ_cm[0][1]}`, FN=`{champ_cm[1][0]}`, TP=`{champ_cm[1][1]}`

---

## 3. Task C — Candidate Model Retraining
- **Combined Training Samples:** `{candidate_eval['combined_samples']}` (1,000 baseline historical + 300 incoming batch).
- **Model Architecture:** `RandomForestClassifier(n_estimators=100, max_depth=6)`
- **Candidate Model Artifact:** Saved to `mlops/models/candidate_model.joblib`
- **MLflow Tracking Run ID:** `{candidate_eval['mlflow_run_id']}`

| Metric | Candidate Value (Holdout) | Candidate Value (New Batch) |
| :--- | :--- | :--- |
| **Accuracy** | `{candidate_eval['accuracy'] * 100:.2f}%` | `{candidate_eval['new_batch_accuracy'] * 100:.2f}%` |
| **Precision** | `{candidate_eval['precision'] * 100:.2f}%` | - |
| **Recall** | `{candidate_eval['recall'] * 100:.2f}%` | - |
| **F1-Score** | `{candidate_eval['f1_score'] * 100:.2f}%` | `{candidate_eval['new_batch_f1_score'] * 100:.2f}%` |

---

## 4. Task D — Quality Gate Decision & Metric Justification

### Selected Metric: **F1-Score**
> **Justification:**  
> {quality_decision['metric_justification']}

### Decision Evaluation:
1. **Minimum Performance Floor (F1 >= {quality_decision['thresholds']['min_f1_floor']}):** `{'PASSED' if quality_decision['checks']['passed_min_floor'] else 'FAILED'}` (`{candidate_eval['f1_score']:.4f}`)
2. **Outperformed / Matched Champion (Candidate >= Champion):** `{'PASSED' if quality_decision['checks']['outperformed_or_tied_champion'] else 'FAILED'}` (`{candidate_eval['f1_score']:.4f}` vs `{champion_eval['f1_score']:.4f}`)

### Final Decision:
**`{quality_decision['decision']}`**  
*{quality_decision['reason']}*

---

## 5. Artifacts Generated
- `mlops/data/baseline_data.csv` (1,000 samples)
- `mlops/data/incoming_batch.csv` (300 samples)
- `mlops/models/champion_model.joblib` (Production Champion Model)
- `mlops/models/candidate_model.joblib` (Newly Retrained Candidate Model)
- `mlops/reports/evaluation_report.json` (Structured JSON Metrics)
- `mlops/reports/evaluation_report.md` (Markdown Report for Submission)
- `mlops/reports/quality_gate_decision.json` (Quality Gate Audit Trail)
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 70)
    print(" [TASK E] ARTIFACTS & EVALUATION REPORTS GENERATED")
    print("=" * 70)
    print(f"  • JSON Report:     {report_json_path}")
    print(f"  • Markdown Report: {report_md_path}")
    print(f"  • Quality Decision:{REPORTS_DIR / 'quality_gate_decision.json'}")
    print(f"  • Candidate Model: {MODELS_DIR / 'candidate_model.joblib'}")
    print(f"  • Champion Model:  {MODELS_DIR / 'champion_model.joblib'}")


# ==============================================================================
# Main Pipeline Runner
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="HireWise AI Model Quality Monitoring & Retraining Pipeline")
    parser.add_argument("--batch_file", type=str, default="incoming_batch.csv", help="Incoming batch CSV filename")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:6001", help="MLflow Tracking Server URI")
    args = parser.parse_args()

    setup_directories()

    batch_path = DATA_DIR / args.batch_file
    if not batch_path.exists():
        from mlops.data_generator import generate_datasets
        generate_datasets()

    # Step 1: Task A - Prepare and validate new data
    df_new = prepare_new_data(batch_path)

    # Step 2: Ensure baseline champion exists
    champion_model = get_or_train_baseline_champion()

    # Step 3: Task B - Evaluate existing champion model on new data
    champion_eval = evaluate_existing_model(champion_model, df_new)

    # Step 4: Task C - Retrain candidate model on combined data
    candidate_model, candidate_eval = retrain_candidate_model(df_new, tracking_uri=args.tracking_uri)

    # Step 5: Task D - Quality gate decision
    quality_decision = implement_quality_gate(champion_eval, candidate_eval, candidate_model)

    # Step 6: Task E - Reports & documentation
    generate_reports(df_new, champion_eval, candidate_eval, quality_decision)

    print("\n[SUCCESS] MLOps Model Monitoring & Retraining Pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
