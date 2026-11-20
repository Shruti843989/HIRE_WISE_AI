import os
import numpy as np
import pandas as pd
from pathlib import Path

def generate_datasets():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Baseline Dataset (1000 samples)
    np.random.seed(42)
    n_base = 1000
    base_ids = [f"CAND_{i+1:04d}" for i in range(n_base)]
    base_claimed = np.round(np.random.uniform(2.5, 10.0, n_base), 2)
    base_measured = np.round(np.random.uniform(1.0, 9.5, n_base), 2)
    base_diff = np.round(base_claimed - base_measured, 2)
    base_labels = (base_diff > 2.0).astype(int)
    
    # 6% realistic evaluator noise
    noise_idx = np.random.choice(n_base, size=int(0.06 * n_base), replace=False)
    base_labels[noise_idx] = 1 - base_labels[noise_idx]

    df_base = pd.DataFrame({
        "candidate_id": base_ids,
        "claimed_skill": base_claimed,
        "measured_ability": base_measured,
        "difference": base_diff,
        "is_inflated": base_labels
    })

    baseline_path = data_dir / "baseline_data.csv"
    df_base.to_csv(baseline_path, index=False)
    print(f"[OK] Baseline dataset generated: {baseline_path} ({len(df_base)} records)")

    # 2. Generate Incoming Data Batch (300 samples)
    np.random.seed(101)
    n_incoming = 300
    incoming_ids = [f"CAND_{1000 + i + 1:04d}" for i in range(n_incoming)]
    inc_claimed = np.round(np.random.uniform(3.0, 10.0, n_incoming), 2)
    inc_measured = np.round(np.random.uniform(1.5, 9.5, n_incoming), 2)
    inc_diff = np.round(inc_claimed - inc_measured, 2)
    inc_labels = (inc_diff > 2.0).astype(int)

    # 6% evaluator noise
    noise_inc = np.random.choice(n_incoming, size=int(0.06 * n_incoming), replace=False)
    inc_labels[noise_inc] = 1 - inc_labels[noise_inc]

    df_incoming = pd.DataFrame({
        "candidate_id": incoming_ids,
        "claimed_skill": inc_claimed,
        "measured_ability": inc_measured,
        "difference": inc_diff,
        "is_inflated": inc_labels
    })

    incoming_path = data_dir / "incoming_batch.csv"
    df_incoming.to_csv(incoming_path, index=False)
    print(f"[OK] Incoming batch generated: {incoming_path} ({len(df_incoming)} records)")

if __name__ == "__main__":
    generate_datasets()
