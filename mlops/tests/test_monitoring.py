"""
Unit Tests for MLOps Model Monitoring & Retraining Pipeline (Tasks A - E)
Runs isolated with Pytest.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from mlops.monitor_and_retrain import (
    prepare_new_data,
    evaluate_existing_model,
    retrain_candidate_model,
    implement_quality_gate,
    generate_reports,
    REQUIRED_COLUMNS,
    FEATURE_COLS,
    TARGET_COL,
    MIN_F1_THRESHOLD
)

@pytest.fixture
def sample_incoming_data(tmp_path):
    """Fixture providing clean incoming batch data."""
    csv_file = tmp_path / "test_incoming.csv"
    df = pd.DataFrame({
        "candidate_id": [f"TEST_{i}" for i in range(50)],
        "claimed_skill": np.round(np.random.uniform(4.0, 9.0, 50), 2),
        "measured_ability": np.round(np.random.uniform(2.0, 8.0, 50), 2),
        "difference": np.round(np.random.uniform(0.0, 4.0, 50), 2),
        "is_inflated": [1 if i % 2 == 0 else 0 for i in range(50)]
    })
    df.to_csv(csv_file, index=False)
    return csv_file, df

@pytest.fixture
def sample_champion_model():
    """Fixture providing a mock trained champion model."""
    X = np.random.uniform(1.0, 10.0, (100, 3))
    y = np.random.choice([0, 1], 100)
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


def test_task_a_data_validation(sample_incoming_data, tmp_path):
    """Task A Test: Validates required columns, missing values, duplicates, and distribution."""
    csv_file, raw_df = sample_incoming_data
    df = prepare_new_data(csv_file)

    # 1. Dataset size
    assert len(df) == 50

    # 2. Columns validated
    for col in REQUIRED_COLUMNS:
        assert col in df.columns

    # 3. Missing values check
    assert df.isnull().sum().sum() == 0

    # 4. Target distribution
    counts = df[TARGET_COL].value_counts().to_dict()
    assert 0 in counts and 1 in counts
    assert counts[0] + counts[1] == 50

    # Test failure on missing required column
    invalid_csv = tmp_path / "invalid.csv"
    raw_df.drop(columns=["is_inflated"]).to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_new_data(invalid_csv)


def test_task_b_evaluate_existing_model(sample_champion_model, sample_incoming_data):
    """Task B Test: Generates predictions and computes Accuracy, Precision, Recall, F1."""
    _, df = sample_incoming_data
    results = evaluate_existing_model(sample_champion_model, df)

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "confusion_matrix" in results

    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1_score"] <= 1.0


def test_task_c_retrain_candidate(sample_incoming_data):
    """Task C Test: Combines datasets and trains candidate model separately."""
    _, df = sample_incoming_data
    candidate_model, results = retrain_candidate_model(df, tracking_uri="sqlite:///test_mlflow.db")

    assert candidate_model is not None
    assert results["combined_samples"] >= len(df)
    assert 0.0 <= results["f1_score"] <= 1.0


def test_task_d_quality_gate_decision_logic(sample_champion_model):
    """Task D Test: Verifies both promotion and retention branches and justification."""
    # Scenario 1: Candidate beats champion -> PROMOTED
    champ_high = {"f1_score": 0.85, "accuracy": 0.86}
    cand_higher = {"f1_score": 0.90, "accuracy": 0.91}
    decision1 = implement_quality_gate(champ_high, cand_higher, sample_champion_model)

    assert decision1["decision"] == "PROMOTED_TO_CHAMPION"
    assert decision1["status"] == "ELIGIBLE"
    assert "F1-Score" in decision1["selected_metric"]
    assert len(decision1["metric_justification"]) > 20

    # Scenario 2: Candidate worse than champion -> RETAINED
    cand_lower = {"f1_score": 0.80, "accuracy": 0.82}
    decision2 = implement_quality_gate(champ_high, cand_lower, sample_champion_model)

    assert decision2["decision"] == "RETAINED_EXISTING_CHAMPION"
    assert decision2["status"] == "REJECTED"


def test_task_e_reports_creation(sample_incoming_data):
    """Task E Test: Verifies report generation and file persistence."""
    _, df = sample_incoming_data
    champ_res = {"accuracy": 0.90, "precision": 0.92, "recall": 0.88, "f1_score": 0.90, "confusion_matrix": [[20, 2], [3, 25]]}
    cand_res = {
        "combined_samples": 1050,
        "mlflow_run_id": "test_run_123",
        "accuracy": 0.93,
        "precision": 0.94,
        "recall": 0.92,
        "f1_score": 0.93,
        "new_batch_accuracy": 0.92,
        "new_batch_f1_score": 0.92,
        "confusion_matrix": [[21, 1], [2, 26]]
    }
    decision = {
        "decision": "PROMOTED_TO_CHAMPION",
        "selected_metric": "F1-Score",
        "metric_justification": "F1 balances precision and recall.",
        "thresholds": {"min_f1_floor": 0.82},
        "checks": {"passed_min_floor": True, "outperformed_or_tied_champion": True},
        "reason": "Outperformed champion."
    }

    generate_reports(df, champ_res, cand_res, decision)

    base_dir = Path(__file__).resolve().parent.parent
    assert (base_dir / "reports" / "evaluation_report.json").exists()
    assert (base_dir / "reports" / "evaluation_report.md").exists()
    assert (base_dir / "reports" / "quality_gate_decision.json").exists()
