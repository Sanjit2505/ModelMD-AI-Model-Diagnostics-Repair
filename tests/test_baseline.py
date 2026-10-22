"""
Automated test suite for ModelMD Module 01 - Model Loader & Baseline Diagnostics.
"""

import os
import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

SAMPLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_model.joblib")
SAMPLE_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_dataset.csv")


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_data():
    """Ensure sample model and dataset exist before running tests."""
    if not os.path.exists(SAMPLE_MODEL_PATH) or not os.path.exists(SAMPLE_DATASET_PATH):
        from scripts.generate_sample_data import generate_sample_artifacts
        generate_sample_artifacts()


def test_status_endpoint():
    """Test health check endpoint."""
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["project"] == "ModelMD"


def test_dataset_inspect_success():
    """Test CSV inspection endpoint."""
    with open(SAMPLE_DATASET_PATH, "rb") as f:
        resp = client.post("/api/dataset/inspect", files={"file": ("sample.csv", f, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["row_count"] == 180
    assert "diagnosis" in data["columns"]
    assert len(data["preview"]) == 10
    assert data["total_missing_values"] >= 2  # Introduced in generator


def test_dataset_inspect_invalid_empty():
    """Test inspection fails cleanly on empty CSV."""
    empty_csv = io.BytesIO(b"")
    resp = client.post("/api/dataset/inspect", files={"file": ("empty.csv", empty_csv, "text/csv")})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_model_inspect_success():
    """Test model file validation endpoint."""
    with open(SAMPLE_MODEL_PATH, "rb") as f:
        resp = client.post("/api/model/inspect", files={"file": ("sample_model.joblib", f, "application/octet-stream")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "sample_model.joblib"
    assert data["extension"] == ".joblib"
    assert "security_warning" in data


def test_model_inspect_invalid_extension():
    """Test model upload rejects unsupported extensions."""
    fake_file = io.BytesIO(b"dummy code")
    resp = client.post("/api/model/inspect", files={"file": ("malicious.exe", fake_file, "application/octet-stream")})
    assert resp.status_code == 400
    assert "unsupported model file format" in resp.json()["detail"].lower()


def test_successful_baseline_evaluation():
    """Test complete baseline evaluation with real pipeline and dataset."""
    with open(SAMPLE_MODEL_PATH, "rb") as mf, open(SAMPLE_DATASET_PATH, "rb") as df:
        resp = client.post(
            "/api/diagnostics/run",
            files={
                "model_file": ("sample_model.joblib", mf, "application/octet-stream"),
                "dataset_file": ("sample_dataset.csv", df, "text/csv"),
            },
            data={
                "target_column": "diagnosis",
                "average_method": "weighted",
                "missing_strategy": "drop",
            },
        )

    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"

    summary = res["summary"]
    assert summary["total_evaluated_rows"] > 0
    assert summary["correct_predictions"] + summary["incorrect_predictions"] == summary["total_evaluated_rows"]
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["f1_score"] <= 1.0
    assert summary["dropped_rows_count"] == 2  # 2 rows with NaN

    # Confusion matrix checks
    cm = res["confusion_matrix"]
    assert "labels" in cm
    assert len(cm["labels"]) == len(cm["matrix"])

    # Per class checks
    per_class = res["per_class_metrics"]
    assert len(per_class) > 0
    for cls_data in per_class:
        assert "class_label" in cls_data
        assert "precision" in cls_data
        assert "recall" in cls_data
        assert "f1_score" in cls_data
        assert "support" in cls_data

    # Row-level checks
    row_table = res["row_level_predictions"]
    assert len(row_table) == summary["total_evaluated_rows"]
    assert "row_number" in row_table[0]
    assert "actual" in row_table[0]
    assert "predicted" in row_table[0]
    assert "is_correct" in row_table[0]


def test_missing_target_column():
    """Test evaluation fails when specified target column is not in CSV."""
    with open(SAMPLE_MODEL_PATH, "rb") as mf, open(SAMPLE_DATASET_PATH, "rb") as df:
        resp = client.post(
            "/api/diagnostics/run",
            files={
                "model_file": ("sample_model.joblib", mf, "application/octet-stream"),
                "dataset_file": ("sample_dataset.csv", df, "text/csv"),
            },
            data={
                "target_column": "non_existent_target",
                "average_method": "weighted",
                "missing_strategy": "drop",
            },
        )
    assert resp.status_code == 400
    assert "target column 'non_existent_target' does not exist" in resp.json()["detail"].lower()


def test_feature_mismatch():
    """Test evaluation fails cleanly when dataset features do not match model features."""
    mismatched_df = pd.DataFrame({
        "wrong_feature_1": [1, 2, 3],
        "wrong_feature_2": [4, 5, 6],
        "diagnosis": ["Low_Risk", "High_Risk", "Moderate_Risk"],
    })
    csv_bytes = io.BytesIO()
    mismatched_df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    with open(SAMPLE_MODEL_PATH, "rb") as mf:
        resp = client.post(
            "/api/diagnostics/run",
            files={
                "model_file": ("sample_model.joblib", mf, "application/octet-stream"),
                "dataset_file": ("mismatched.csv", csv_bytes, "text/csv"),
            },
            data={
                "target_column": "diagnosis",
                "average_method": "weighted",
                "missing_strategy": "drop",
            },
        )
    assert resp.status_code == 400
    assert "feature" in resp.json()["detail"].lower()


def test_missing_value_reject_strategy():
    """Test that missing_strategy='reject' stops execution when NaNs are present."""
    with open(SAMPLE_MODEL_PATH, "rb") as mf, open(SAMPLE_DATASET_PATH, "rb") as df:
        resp = client.post(
            "/api/diagnostics/run",
            files={
                "model_file": ("sample_model.joblib", mf, "application/octet-stream"),
                "dataset_file": ("sample_dataset.csv", df, "text/csv"),
            },
            data={
                "target_column": "diagnosis",
                "average_method": "weighted",
                "missing_strategy": "reject",
            },
        )
    assert resp.status_code == 400
    assert "missing values detected" in resp.json()["detail"].lower()


def test_invalid_model_prediction_failure():
    """Test worker failure when arbitrary invalid object without predict() is uploaded."""
    import joblib
    fake_obj = {"not_a_model": True}
    buf = io.BytesIO()
    joblib.dump(fake_obj, buf)
    buf.seek(0)

    with open(SAMPLE_DATASET_PATH, "rb") as df:
        resp = client.post(
            "/api/diagnostics/run",
            files={
                "model_file": ("invalid_model.joblib", buf, "application/octet-stream"),
                "dataset_file": ("sample_dataset.csv", df, "text/csv"),
            },
            data={
                "target_column": "diagnosis",
                "average_method": "weighted",
                "missing_strategy": "drop",
            },
        )
    assert resp.status_code == 400
    assert "predict" in resp.json()["detail"].lower()
