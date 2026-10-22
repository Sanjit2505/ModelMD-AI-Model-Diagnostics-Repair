# ModelMD — AI Model Diagnostic & Observability Platform

ModelMD is a local-first platform designed to evaluate trained machine learning models, calculate real baseline evaluation metrics, and monitor model health and data drift over time.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph UI[Modern Web Interface]
        Tab1[Module 01: Baseline Diagnostics]
        Tab2[Module 02: Health & Data Drift]
    end

    subgraph Backend[FastAPI Server: backend.app.main]
        API1[/api/diagnostics/run]
        API2[/api/drift/analyze]
        API3[/api/dataset/inspect]
    end

    subgraph Engines[Diagnostic Core]
        Worker[worker.py: Isolated Subprocess<br>Loads Model & Evaluates Inference]
        Drift[drift.py: Statistical Drift Engine<br>KS-Test, Chi-Square, PSI, Wasserstein]
    end

    UI --> Backend
    API1 --> Worker
    API2 --> Drift
    API2 -.->|Optional Degradation Tracking| Worker
```

---

## Module 01: Model Loader & Baseline Diagnostics
- **Upload Models**: Supports `.joblib` and `.pkl` scikit-learn classifiers and fitted `Pipeline`s.
- **Inspect CSV Datasets**: Displays row/column count, missing value analysis, duplicate row detection, and a 10-row preview.
- **Real Metrics**: Accuracy, Precision, Recall, F1-score (weighted, macro, micro).
- **Confusion Matrix**: Full multi-class breakdown with actual vs. predicted distribution.
- **Error Inspector**: Row-level inspector showing ground truth vs. model predictions, highlighting exact errors.
- **Worker Isolation**: Inferences run inside an isolated subprocess with a 30-second execution timeout.

---

## Module 02: Model Health & Data Drift Observability
- **Statistical Drift Detection**: Compares baseline training data against new incoming production batches.
  - **Numerical Features**: Two-sample Kolmogorov-Smirnov (KS) test, Wasserstein distance, and Population Stability Index (PSI).
  - **Categorical Features**: Chi-square contingency test and detection of new, unseen categories.
- **Feature-Level Pinpointing**: Details exactly which features drifted, the magnitude of shift (e.g. `+34.5%` mean change), and p-value significance.
- **Outlier & Range Warnings**: Flags values exceeding historical min/max boundaries.
- **Performance Degradation Tracking**: When labels are provided, evaluates the model on both datasets side-by-side to compute the accuracy/F1 drop.

---

## Quick Start Guide

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Generate Sample Artifacts
Generate the reference dataset, sample model, and drifted production dataset:
```powershell
# 1. Generate reference dataset and trained Pipeline
python scripts/generate_sample_data.py

# 2. Generate drifted current dataset
python scripts/generate_drifted_data.py
```
Generated files in `sample_data/`:
- `sample_model.joblib`: Trained scikit-learn `Pipeline` (StandardScaler + LogisticRegression).
- `sample_dataset.csv`: Reference dataset (180 clinical records).
- `sample_dataset_drifted.csv`: Current production dataset with simulated distribution drift.

### 3. Launch the Application
```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## Testing

Run the complete automated pytest suite:
```powershell
pytest -v
```
Verifies all 14 test scenarios covering:
- Baseline evaluation, missing values, column mismatches, invalid models.
- Data drift detection, KS-tests, p-value calculations, and performance degradation tracking.
