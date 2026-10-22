"""
Diagnostics coordinator for ModelMD.

Handles file validation, temporary workspace management, and subprocess isolation
for running scikit-learn models.
"""

import os
import sys
import json
import tempfile
import subprocess
from typing import Dict, Any, List
import pandas as pd
import io

from backend.engine.dataset import prepare_features_and_target, DatasetValidationError

# Security limits
MAX_MODEL_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_DATASET_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_MODEL_EXTENSIONS = {".joblib", ".pkl"}
WORKER_TIMEOUT_SECONDS = 30


class DiagnosticExecutionError(Exception):
    """Raised when model evaluation fails."""
    pass


def validate_model_upload(filename: str, size: int) -> str:
    """
    Validates file extension and size.
    Returns extension.
    """
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_MODEL_EXTENSIONS:
        raise DiagnosticExecutionError(
            f"Unsupported model file format '{ext}'. Allowed extensions are: {', '.join(ALLOWED_MODEL_EXTENSIONS)}. "
            "Please upload a fitted scikit-learn classifier or Pipeline serialized via joblib or pickle."
        )

    if size > MAX_MODEL_SIZE_BYTES:
        raise DiagnosticExecutionError(
            f"Model file size ({size / (1024*1024):.1f} MB) exceeds maximum allowed limit of {MAX_MODEL_SIZE_BYTES / (1024*1024):.0f} MB."
        )

    return ext


def validate_dataset_upload(filename: str, size: int):
    """Validates dataset file size and extension."""
    _, ext = os.path.splitext(filename.lower())
    if ext != ".csv":
        raise DiagnosticExecutionError(
            f"Unsupported dataset format '{ext}'. Only CSV (.csv) files are supported for baseline evaluation."
        )

    if size > MAX_DATASET_SIZE_BYTES:
        raise DiagnosticExecutionError(
            f"Dataset size ({size / (1024*1024):.1f} MB) exceeds maximum allowed limit of {MAX_DATASET_SIZE_BYTES / (1024*1024):.0f} MB."
        )


def run_baseline_evaluation(
    model_bytes: bytes,
    model_filename: str,
    csv_bytes: bytes,
    target_column: str,
    average_method: str = "weighted",
    missing_strategy: str = "drop",
) -> Dict[str, Any]:
    """
    Executes end-to-end baseline evaluation inside an isolated worker subprocess.
    """
    validate_model_upload(model_filename, len(model_bytes))
    validate_dataset_upload("dataset.csv", len(csv_bytes))

    if average_method not in ["weighted", "macro", "micro"]:
        raise DiagnosticExecutionError(f"Invalid average method: '{average_method}'.")

    # Parse and prepare dataset in-memory
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        raise DiagnosticExecutionError(f"Failed to read CSV dataset: {str(e)}")

    try:
        X, y, row_indices, dropped_rows_count = prepare_features_and_target(
            df=df,
            target_column=target_column,
            missing_strategy=missing_strategy,
        )
    except DatasetValidationError as e:
        raise DiagnosticExecutionError(str(e))

    # Convert records for IPC payload
    data_payload = {
        "X": X.to_dict(orient="records"),
        "y": y.tolist(),
        "row_indices": row_indices,
        "dropped_rows_count": dropped_rows_count,
    }

    # Execute in temporary isolated workspace
    with tempfile.TemporaryDirectory(prefix="modelmd_run_") as tmpdir:
        model_ext = os.path.splitext(model_filename)[1].lower()
        temp_model_path = os.path.join(tmpdir, f"model{model_ext}")
        temp_data_path = os.path.join(tmpdir, "input_data.json")
        temp_out_path = os.path.join(tmpdir, "output.json")

        with open(temp_model_path, "wb") as f:
            f.write(model_bytes)

        with open(temp_data_path, "w", encoding="utf-8") as f:
            json.dump(data_payload, f)

        cmd = [
            sys.executable,
            "-m",
            "backend.engine.worker",
            "--model",
            temp_model_path,
            "--data",
            temp_data_path,
            "--output",
            temp_out_path,
            "--average",
            average_method,
        ]

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                cwd=os.getcwd(),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise DiagnosticExecutionError(
                f"Worker evaluation timed out after {WORKER_TIMEOUT_SECONDS} seconds. "
                "The model inference took too long or hung during execution."
            )

        if not os.path.exists(temp_out_path):
            stderr_msg = completed.stderr.strip() if completed.stderr else "Unknown worker failure"
            raise DiagnosticExecutionError(
                f"Worker process terminated unexpectedly without producing output. Details:\n{stderr_msg}"
            )

        with open(temp_out_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        if result.get("status") == "error":
            raise DiagnosticExecutionError(result.get("message", "Model execution failed."))

        return result
