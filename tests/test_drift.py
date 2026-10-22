"""
Automated test suite for ModelMD Module 02 - Model Health & Data Drift.

Verifies:
1. Significant drift detection
2. No significant drift on identical/stable distributions
3. Strict exclusion of target/label column
4. Explicit handling of missing values and non-finite numbers (NaN, Inf)
5. Robust handling of constant columns (identical and shifted)
6. Consistent significance threshold and transparent severity classification
7. Out-of-range warnings without artificial severity elevation
8. Identification and exclusion of identifier columns with corrected denominator
9. Proper categorical feature distribution comparison (Chi-Square & TVD)
10. High-precision p-value formatting (no rounded zero)
11. Invalid CSV schema rejection
12. Dynamic HTTP API evaluation
"""

import os
import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.engine.drift import (
    compare_datasets_for_drift,
    analyze_numerical_feature_drift,
    analyze_categorical_feature_drift,
    format_p_value,
    DriftAnalysisError,
)

client = TestClient(app)

SAMPLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_model.joblib")
SAMPLE_REF_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_dataset.csv")
SAMPLE_DRIFT_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_dataset_drifted.csv")


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_drift_data():
    """Ensure sample reference, model, and drifted datasets exist."""
    if not os.path.exists(SAMPLE_REF_PATH) or not os.path.exists(SAMPLE_MODEL_PATH):
        from scripts.generate_sample_data import generate_sample_artifacts
        generate_sample_artifacts()
    if not os.path.exists(SAMPLE_DRIFT_PATH):
        from scripts.generate_drifted_data import generate_drifted_dataset
        generate_drifted_dataset()


def test_no_significant_drift():
    """Identical reference compared against itself should detect no drift."""
    ref_df = pd.read_csv(SAMPLE_REF_PATH)
    res = compare_datasets_for_drift(ref_df=ref_df, curr_df=ref_df, target_column="diagnosis")
    health = res["dataset_health"]

    assert health["health_status"] == "Healthy (Stable)"
    assert health["drifted_features_count"] == 0
    assert health["drift_share_pct"] == 0.0

    for feat_name, feat_res in res["features"].items():
        assert feat_res["drift_detected"] is False
        assert feat_res["severity"] == "None"
        assert feat_res["p_value"] >= 0.05


def test_significant_drift():
    """Shifted dataset should detect statistically significant drift with appropriate severity."""
    ref_df = pd.read_csv(SAMPLE_REF_PATH)
    curr_df = pd.read_csv(SAMPLE_DRIFT_PATH)
    res = compare_datasets_for_drift(ref_df=ref_df, curr_df=curr_df, target_column="diagnosis")
    health = res["dataset_health"]

    assert health["drifted_features_count"] > 0
    assert "glucose" in health["drifted_features_list"]

    glucose_stats = res["features"]["glucose"]
    assert glucose_stats["drift_detected"] is True
    assert glucose_stats["severity"] in ["Moderate", "High"]
    assert glucose_stats["p_value"] < 0.05
    assert glucose_stats["statistic"] > 0.15
    assert glucose_stats["curr_stats"]["mean"] > glucose_stats["ref_stats"]["mean"]


def test_target_exclusion():
    """Target column must be excluded from feature drift analysis and denominator."""
    ref_df = pd.read_csv(SAMPLE_REF_PATH)
    curr_df = pd.read_csv(SAMPLE_DRIFT_PATH)

    res = compare_datasets_for_drift(ref_df=ref_df, curr_df=curr_df, target_column="diagnosis")
    
    # Diagnosis must NOT be in features report
    assert "diagnosis" not in res["features"]
    assert "diagnosis" not in res["dataset_health"]["eligible_features"]
    assert res["dataset_health"]["excluded_columns"]["target_column"] == "diagnosis"
    
    # Target column must not inflate total_features count
    expected_eligible = [c for c in ref_df.columns if c != "diagnosis"]
    assert res["dataset_health"]["total_features"] == len(expected_eligible)


def test_identifier_exclusion_and_denominator():
    """Identifier columns (e.g., patient_id, uuid) must be excluded from analysis and denominator."""
    np.random.seed(42)
    n = 50
    ref_df = pd.DataFrame({
        "patient_id": [f"PID_{i:04d}" for i in range(n)],
        "feature_a": np.random.normal(50, 5, n),
        "feature_b": np.random.normal(100, 10, n),
        "target": np.random.choice([0, 1], n),
    })
    curr_df = pd.DataFrame({
        "patient_id": [f"PID_{i+100:04d}" for i in range(n)],
        "feature_a": np.random.normal(50, 5, n),
        "feature_b": np.random.normal(120, 10, n),  # drifted
        "target": np.random.choice([0, 1], n),
    })

    res = compare_datasets_for_drift(ref_df=ref_df, curr_df=curr_df, target_column="target")
    health = res["dataset_health"]

    # patient_id and target must be excluded
    assert "patient_id" not in res["features"]
    assert "target" not in res["features"]
    assert "patient_id" in health["excluded_columns"]["identifier_columns"]

    # Eligible features are strictly feature_a and feature_b
    assert health["total_features"] == 2
    assert set(health["eligible_features"]) == {"feature_a", "feature_b"}

    # Drift share denominator is exactly 2: 1 drifted out of 2 features = 50%
    assert health["drifted_features_count"] == 1
    assert health["drift_share_pct"] == 50.0


def test_missing_and_non_finite_values():
    """Explicitly handles NaNs, positive/negative infinities, and missing values."""
    ref_vals = pd.Series([10.0, 12.0, 14.0, np.nan, 16.0, 18.0, np.inf, 20.0, -np.inf, 22.0])
    curr_vals = pd.Series([np.nan, 11.0, 13.0, 15.0, 17.0, 19.0, np.nan, 21.0, 23.0, 25.0])

    res = analyze_numerical_feature_drift(ref_vals, curr_vals, p_value_threshold=0.05)

    assert res["feature_type"] == "numerical"
    assert res["ref_stats"]["missing_count"] == 4  # 1 NaN, 2 Infs
    assert res["curr_stats"]["missing_count"] == 2  # 2 NaNs
    assert res["ref_stats"]["valid_count"] == 6
    assert res["curr_stats"]["valid_count"] == 8
    assert any("Excluded non-finite" in w for w in res["warnings"])


def test_constant_columns_identical_and_shifted():
    """Constant columns must not cause ZeroDivisionError, and handle shifted constants properly."""
    # 1. Identical constant column
    ref_const = pd.Series([42.0] * 30)
    curr_const = pd.Series([42.0] * 30)

    res_same = analyze_numerical_feature_drift(ref_const, curr_const, p_value_threshold=0.05)
    assert res_same["drift_detected"] is False
    assert res_same["severity"] == "None"
    assert res_same["p_value"] == 1.0
    assert res_same["statistic"] == 0.0

    # 2. Shifted constant column
    curr_shifted = pd.Series([99.0] * 30)
    res_diff = analyze_numerical_feature_drift(ref_const, curr_shifted, p_value_threshold=0.05)
    assert res_diff["drift_detected"] is True
    assert res_diff["severity"] == "High"
    assert res_diff["p_value"] < 0.05
    assert res_diff["statistic"] == 1.0


def test_severity_consistency_and_threshold_enforcement():
    """
    Features with p-value >= threshold must strictly have severity 'None'.
    Out-of-range values must NOT elevate severity to High.
    """
    np.random.seed(123)
    # Distributions from same normal distribution (p >= 0.05)
    ref = pd.Series(np.random.normal(100, 15, 200))
    curr = pd.Series(np.random.normal(100, 15, 200))

    # Add a couple out-of-range values that don't shift the distribution significantly
    curr.iloc[0] = 250.0  # above ref_max
    curr.iloc[1] = 0.0    # below ref_min

    res = analyze_numerical_feature_drift(ref, curr, p_value_threshold=0.05)

    # Must have out of range warning
    assert res["out_of_range_count"] > 0
    assert any("outside historical reference range" in w for w in res["warnings"])

    # If p_value >= threshold, drift must NOT be detected and severity must be 'None'
    if res["p_value"] >= 0.05:
        assert res["drift_detected"] is False
        assert res["severity"] == "None"

    # Strict test with custom strict threshold p = 0.001
    res_strict = analyze_numerical_feature_drift(ref, curr, p_value_threshold=0.001)
    if res_strict["p_value"] >= 0.001:
        assert res_strict["drift_detected"] is False
        assert res_strict["severity"] == "None"


def test_categorical_feature_drift():
    """Categorical features use Chi-Square contingency test and TVD, detecting distribution shifts."""
    ref_cat = pd.Series(["A"] * 60 + ["B"] * 40)
    curr_cat = pd.Series(["A"] * 20 + ["B"] * 70 + ["C"] * 10)  # Significant shift + unseen 'C'

    res = analyze_categorical_feature_drift(ref_cat, curr_cat, p_value_threshold=0.05)

    assert res["feature_type"] == "categorical"
    assert res["drift_detected"] is True
    assert res["severity"] in ["Moderate", "High"]
    assert "C" in res["new_categories"]
    assert res["tvd"] > 0.2
    assert "Chi-Square" in res["test_used"]


def test_p_value_precision_formatting():
    """Verifies that p-values are never displayed as rounded zeros and formatted properly."""
    assert format_p_value(0.0) == "< 1e-15"
    assert format_p_value(1e-18) == "< 1e-15"
    assert format_p_value(1.23e-8) == "1.23e-08"
    assert format_p_value(0.00045) == "4.50e-04"
    assert format_p_value(0.03456) == "0.0346"
    assert format_p_value(0.85) == "0.8500"


def test_invalid_csv_schemas():
    """Validates error handling for empty datasets or non-overlapping columns."""
    empty_df = pd.DataFrame()
    valid_df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    disjoint_df = pd.DataFrame({"col_x": [1, 2, 3], "col_y": [4, 5, 6]})

    # Empty dataset error
    with pytest.raises(DriftAnalysisError, match="Reference or Current dataset is empty"):
        compare_datasets_for_drift(ref_df=empty_df, curr_df=valid_df)

    # Disjoint dataset error
    with pytest.raises(DriftAnalysisError, match="No eligible overlapping feature columns"):
        compare_datasets_for_drift(ref_df=valid_df, curr_df=disjoint_df)

    # All columns excluded error
    with pytest.raises(DriftAnalysisError, match="No eligible overlapping feature columns"):
        compare_datasets_for_drift(ref_df=valid_df, curr_df=valid_df, target_column="col_a", id_columns=["col_b"])


def test_drift_api_endpoint():
    """Testing dynamic multipart HTTP upload via /api/drift/analyze."""
    with open(SAMPLE_REF_PATH, "rb") as rf, open(SAMPLE_DRIFT_PATH, "rb") as cf:
        resp = client.post(
            "/api/drift/analyze",
            files={
                "reference_file": ("ref.csv", rf, "text/csv"),
                "current_file": ("curr.csv", cf, "text/csv"),
            },
            data={"target_column": "diagnosis", "p_value_threshold": "0.05"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "dataset_health" in data
    assert "features" in data
    assert data["dataset_health"]["drifted_features_count"] > 0
    assert "diagnosis" not in data["features"]


def test_drift_api_with_model_performance_degradation():
    """Testing /api/drift/analyze with model evaluation to check performance degradation."""
    with open(SAMPLE_REF_PATH, "rb") as rf, open(SAMPLE_DRIFT_PATH, "rb") as cf, open(SAMPLE_MODEL_PATH, "rb") as mf:
        resp = client.post(
            "/api/drift/analyze",
            files={
                "reference_file": ("ref.csv", rf, "text/csv"),
                "current_file": ("curr.csv", cf, "text/csv"),
                "model_file": ("model.joblib", mf, "application/octet-stream"),
            },
            data={"target_column": "diagnosis"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["performance_comparison"] is not None
    perf = data["performance_comparison"]
    assert "accuracy" in perf
    assert "baseline" in perf["accuracy"]
    assert "current" in perf["accuracy"]
    assert "delta" in perf["accuracy"]
