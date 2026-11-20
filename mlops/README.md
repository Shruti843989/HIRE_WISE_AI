# HireWise AI - MLOps Pipeline Documentation

This module contains a complete production-grade MLOps workflow for HireWise AI:
1. **Experiment Tracking & Model Registry** via MLflow
2. **Quality Gated Automated Training** (`test_accuracy >= 0.80`)
3. **Smarter Champion Model Promotion** (only updates alias if `new_accuracy >= champion_accuracy`)
4. **Realistic Synthetic Dataset** (with ~6% label noise to avoid artificial 100% accuracy)
5. **Model Serving** via FastAPI with Champion Model Dynamic Loading
6. **Isolated Automated Tests** with Pytest using `FakeModel`
7. **Containerization** via Docker (`python:3.12-slim`)
8. **Continuous Integration & Delivery (CI/CD)** via GitHub Actions (`.github/workflows/ci.yaml`)

---

## 📁 File Structure & Responsibilities

| File | Purpose & Description |
| :--- | :--- |
| `train.py` | Generates synthetic resume data with 6% label noise, trains a RandomForestClassifier, logs parameters & evaluation metrics to MLflow, enforces Quality Gate (test_accuracy >= 0.80), and conditionally promotes the champion model. |
| `serve.py` | FastAPI application serving predictions via application-factory pattern. Dynamically loads the `champion` model from MLflow Model Registry via lifespan context. |
| `tests/test_api.py` | Unit tests for `/health` and `/predict` endpoints using a `FakeModel` fixture. Runs in milliseconds without requiring an active MLflow server. |
| `requirements.txt` | Minimal dependencies needed for tracking, serving, testing, and training (Python 3.12). |
| `Dockerfile` | Multi-stage, cached container image definition based on `python:3.12-slim`. |
| `README.md` | Student-friendly guide and exam walkthrough. |

---

## 🚀 Step-by-Step Demo Guide (Windows Commands)

### 1. Virtual Environment Setup (Python 3.12)
```powershell
# Open terminal in project root
python -m venv mlops\.venv

# Activate venv in PowerShell
.\mlops\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r mlops\requirements.txt
```

### 2. Run Isolated Unit Tests
```powershell
python -m pytest mlops\tests
```
*Expected: All 5 tests pass instantly because FakeModel is used.*

### 3. Start MLflow Server
Open a dedicated PowerShell terminal:
```powershell
# Start local MLflow server with SQLite backend on port 6001
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 6001 --allowed-hosts "host.docker.internal,host.docker.internal:6001"
```
Open **[http://127.0.0.1:6001](http://127.0.0.1:6001)** in your browser to inspect the UI.

### 4. Train the Model (Run 3 Experiments)
In another PowerShell terminal:
```powershell
# Run 1: Baseline model (10 trees, depth 3) -> Initial Champion
python mlops\train.py --n_estimators 10 --max_depth 3

# Run 2: Intermediate model (50 trees, depth 5) -> Champion Unchanged if accuracy is lower
python mlops\train.py --n_estimators 50 --max_depth 5

# Run 3: Deep model (100 trees, depth 8) -> Promoted if accuracy >= current champion
python mlops\train.py --n_estimators 100 --max_depth 8
```

### 5. Start the FastAPI Serving Engine
```powershell
# Run FastAPI server locally
uvicorn mlops.serve:app --host 127.0.0.1 --port 8000 --reload
```

Test endpoints in PowerShell:
```powershell
# 1. Health Check
curl http://127.0.0.1:8000/health

# 2. Honest Candidate (Claimed: 7.0, Measured: 6.8) -> Not inflated
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"claimed_skill\": 7.0, \"measured_ability\": 6.8}"

# 3. Inflated Resume (Claimed: 9.5, Measured: 4.0) -> Inflated
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"claimed_skill\": 9.5, \"measured_ability\": 4.0}"
```

### 6. Model Monitoring & Retraining Pipeline (Assessment Tasks A - E)
Execute the complete automated monitoring, retraining, and quality-gated promotion pipeline:
```powershell
# Run the complete workflow
python mlops\monitor_and_retrain.py

# Run all 10 unit tests (Isolated FastAPI + Monitoring Pipeline tests)
python -m pytest mlops\tests
```

**Artifacts Generated & Verified:**
- `mlops/data/baseline_data.csv`: 1,000 historical training records.
- `mlops/data/incoming_batch.csv`: 300 incoming live interview records.
- `mlops/models/champion_model.joblib`: Deployed production champion model.
- `mlops/models/candidate_model.joblib`: Separately saved candidate model artifact.
- `mlops/reports/evaluation_report.md`: Complete formatted markdown evaluation report.
- `mlops/reports/evaluation_report.json`: Machine-readable metrics.
- `mlops/reports/quality_gate_decision.json`: Audit log of the promotion/retention decision.

---

### 7. Docker Containerization
```powershell
# 1. Build Docker image
docker build -t hirewise-mlops:latest ./mlops

# 2. Run Docker container connecting to local MLflow host
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:6001 hirewise-mlops:latest
```
