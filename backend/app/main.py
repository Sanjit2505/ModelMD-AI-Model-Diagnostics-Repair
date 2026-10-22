"""
ModelMD - AI Model Diagnostic Platform.
Module 01: Model Loader & Baseline Evaluation Service.
"""

import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.engine.dataset import inspect_csv, DatasetValidationError
from backend.engine.evaluator import (
    validate_model_upload,
    run_baseline_evaluation,
    DiagnosticExecutionError,
)

app = FastAPI(
    title="ModelMD",
    description="Local AI Model Diagnostic Platform - Module 01: Model Loader & Baseline",
    version="1.0.0",
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


@app.get("/api/status")
def get_status():
    """Health check endpoint."""
    return {
        "project": "ModelMD",
        "module": "01_model_loader_and_baseline",
        "status": "running",
        "message": "ModelMD local diagnostic engine is ready.",
    }


@app.post("/api/dataset/inspect")
async def inspect_dataset(file: UploadFile = File(...)):
    """
    Inspects an uploaded CSV dataset without saving it permanently.
    Returns preview rows, columns, duplicate count, and missing value statistics.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file '{file.filename}'. Only CSV files are supported for dataset inspection.",
        )

    try:
        content = await file.read()
        metadata = inspect_csv(content, filename=file.filename)
        return metadata
    except DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect CSV dataset: {str(e)}")


@app.post("/api/model/inspect")
async def inspect_model(file: UploadFile = File(...)):
    """
    Validates model upload artifact extension and file size.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No model file provided.")

    content = await file.read()
    try:
        ext = validate_model_upload(file.filename, len(content))
        return {
            "filename": file.filename,
            "size_bytes": len(content),
            "size_readable": f"{len(content) / (1024 * 1024):.2f} MB" if len(content) >= 1024 * 1024 else f"{len(content) / 1024:.1f} KB",
            "extension": ext,
            "security_warning": "Pickle/joblib deserialization can execute arbitrary code. Only load trusted local artifacts.",
        }
    except DiagnosticExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/diagnostics/run")
async def run_diagnostics(
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    target_column: str = Form(...),
    average_method: str = Form("weighted"),
    missing_strategy: str = Form("drop"),
):
    """
    Executes real baseline diagnostics for the uploaded model and CSV dataset.
    Runs in an isolated worker process.
    """
    if not model_file.filename:
        raise HTTPException(status_code=400, detail="Model file is required.")
    if not dataset_file.filename:
        raise HTTPException(status_code=400, detail="Dataset CSV file is required.")
    if not target_column or not target_column.strip():
        raise HTTPException(status_code=400, detail="Target column must be specified.")

    model_bytes = await model_file.read()
    dataset_bytes = await dataset_file.read()

    try:
        results = run_baseline_evaluation(
            model_bytes=model_bytes,
            model_filename=model_file.filename,
            csv_bytes=dataset_bytes,
            target_column=target_column.strip(),
            average_method=average_method.strip(),
            missing_strategy=missing_strategy.strip(),
        )
        return results
    except DiagnosticExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")


@app.post("/api/drift/analyze")
async def analyze_drift_endpoint(
    reference_file: UploadFile = File(...),
    current_file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    model_file: Optional[UploadFile] = File(None),
    p_value_threshold: float = Form(0.05),
):
    """
    Module 02 Endpoint: Analyzes statistical data drift between Reference and Current datasets.
    Optionally evaluates model performance degradation if model and labels are provided.
    """
    from backend.engine.drift import compare_datasets_for_drift, compare_model_performance, DriftAnalysisError
    import io
    import pandas as pd

    if not reference_file.filename or not reference_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Reference file must be a valid CSV.")
    if not current_file.filename or not current_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Current file must be a valid CSV.")

    ref_bytes = await reference_file.read()
    curr_bytes = await current_file.read()

    try:
        ref_df = pd.read_csv(io.BytesIO(ref_bytes))
        curr_df = pd.read_csv(io.BytesIO(curr_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV files: {str(e)}")

    target = target_column.strip() if target_column and target_column.strip() else None

    try:
        drift_results = compare_datasets_for_drift(
            ref_df=ref_df,
            curr_df=curr_df,
            target_column=target,
            p_value_threshold=p_value_threshold,
        )
    except DriftAnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift analysis error: {str(e)}")

    # If model is provided and target column exists in both datasets, compute performance degradation
    performance_comparison = None
    if model_file and model_file.filename and target and target in ref_df.columns and target in curr_df.columns:
        model_bytes = await model_file.read()
        try:
            ref_eval = run_baseline_evaluation(
                model_bytes=model_bytes,
                model_filename=model_file.filename,
                csv_bytes=ref_bytes,
                target_column=target,
            )
            curr_eval = run_baseline_evaluation(
                model_bytes=model_bytes,
                model_filename=model_file.filename,
                csv_bytes=curr_bytes,
                target_column=target,
            )
            performance_comparison = compare_model_performance(
                ref_metrics=ref_eval["summary"],
                curr_metrics=curr_eval["summary"],
            )
        except Exception as e:
            # Model eval failure doesn't invalidate drift results, but flag warning
            performance_comparison = {
                "evaluation_error": f"Could not compute performance degradation: {str(e)}"
            }

    drift_results["performance_comparison"] = performance_comparison
    return drift_results

def resolve_local_path(raw_path: str) -> str:
    """Resolves relative or absolute path on local disk."""
    if not raw_path or not raw_path.strip():
        raise HTTPException(status_code=400, detail="File path cannot be empty.")
    clean = raw_path.strip().strip('"').strip("'")
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Try relative to workspace root
    c1 = os.path.normpath(os.path.join(workspace_root, clean))
    if os.path.exists(c1) and os.path.isfile(c1):
        return c1

    # 2. Try as absolute or CWD relative
    c2 = os.path.normpath(os.path.abspath(clean))
    if os.path.exists(c2) and os.path.isfile(c2):
        return c2

    raise HTTPException(
        status_code=400,
        detail=f"File not found: '{clean}'. Please ensure the file exists on your computer.",
    )


@app.get("/api/files/browse")
def browse_local_workspace_files():
    """Scans workspace directory for .joblib, .pkl, and .csv files."""
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models = []
    datasets = []
    ignored = {".git", ".pytest_cache", "__pycache__", "venv", "env", ".vscode", ".agents"}

    for root, dirs, files in os.walk(workspace_root):
        dirs[:] = [d for d in dirs if d not in ignored]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, workspace_root).replace("\\", "/")
            try:
                sz = os.path.getsize(full_p)
                sz_str = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024 * 1024):.2f} MB"
            except Exception:
                sz_str = "Unknown"

            if ext in {".joblib", ".pkl"}:
                models.append({"filename": f, "path": rel_p, "size": sz_str})
            elif ext == ".csv":
                datasets.append({"filename": f, "path": rel_p, "size": sz_str})

    return {"models": models, "datasets": datasets}


class PathInspectRequest(BaseModel):
    file_path: str


class DiagnosticsPathRequest(BaseModel):
    model_path: str
    dataset_path: str
    target_column: str
    average_method: str = "weighted"
    missing_strategy: str = "drop"


class DriftPathRequest(BaseModel):
    ref_path: str
    curr_path: str
    target_column: Optional[str] = None
    model_path: Optional[str] = None
    p_value_threshold: float = 0.05


@app.post("/api/model/inspect-path")
def inspect_model_by_path(req: PathInspectRequest):
    """Inspects a model artifact directly from a local file path."""
    full_path = resolve_local_path(req.file_path)
    filename = os.path.basename(full_path)
    size = os.path.getsize(full_path)

    try:
        ext = validate_model_upload(filename, size)
        return {
            "filename": filename,
            "file_path": full_path,
            "size_bytes": size,
            "size_readable": f"{size / (1024 * 1024):.2f} MB" if size >= 1024 * 1024 else f"{size / 1024:.1f} KB",
            "extension": ext,
            "security_warning": "Pickle/joblib deserialization can execute arbitrary code. Only load trusted local artifacts.",
        }
    except DiagnosticExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/dataset/inspect-path")
def inspect_dataset_by_path(req: PathInspectRequest):
    """Inspects a CSV dataset directly from a local file path."""
    full_path = resolve_local_path(req.file_path)
    filename = os.path.basename(full_path)
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported.")

    try:
        with open(full_path, "rb") as f:
            content = f.read()
        metadata = inspect_csv(content, filename=filename)
        metadata["file_path"] = full_path
        return metadata
    except DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect CSV dataset: {str(e)}")


@app.post("/api/diagnostics/run-path")
def run_diagnostics_by_path(req: DiagnosticsPathRequest):
    """Executes baseline diagnostics given local file paths."""
    model_p = resolve_local_path(req.model_path)
    dataset_p = resolve_local_path(req.dataset_path)

    with open(model_p, "rb") as mf:
        model_bytes = mf.read()
    with open(dataset_p, "rb") as df:
        dataset_bytes = df.read()

    try:
        results = run_baseline_evaluation(
            model_bytes=model_bytes,
            model_filename=os.path.basename(model_p),
            csv_bytes=dataset_bytes,
            target_column=req.target_column.strip(),
            average_method=req.average_method.strip(),
            missing_strategy=req.missing_strategy.strip(),
        )
        return results
    except DiagnosticExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")


@app.post("/api/drift/analyze-path")
def analyze_drift_by_path(req: DriftPathRequest):
    """Executes drift analysis given local file paths."""
    from backend.engine.drift import compare_datasets_for_drift, compare_model_performance, DriftAnalysisError
    import pandas as pd

    ref_p = resolve_local_path(req.ref_path)
    curr_p = resolve_local_path(req.curr_path)

    try:
        ref_df = pd.read_csv(ref_p)
        curr_df = pd.read_csv(curr_p)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV files: {str(e)}")

    target = req.target_column.strip() if req.target_column and req.target_column.strip() else None

    try:
        drift_results = compare_datasets_for_drift(
            ref_df=ref_df,
            curr_df=curr_df,
            target_column=target,
            p_value_threshold=req.p_value_threshold,
        )
    except DriftAnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift analysis error: {str(e)}")

    performance_comparison = None
    if req.model_path and req.model_path.strip() and target and target in ref_df.columns and target in curr_df.columns:
        model_p = resolve_local_path(req.model_path)
        with open(model_p, "rb") as mf:
            model_bytes = mf.read()
        with open(ref_p, "rb") as rf:
            ref_bytes = rf.read()
        with open(curr_p, "rb") as cf:
            curr_bytes = cf.read()

        try:
            ref_eval = run_baseline_evaluation(
                model_bytes=model_bytes,
                model_filename=os.path.basename(model_p),
                csv_bytes=ref_bytes,
                target_column=target,
            )
            curr_eval = run_baseline_evaluation(
                model_bytes=model_bytes,
                model_filename=os.path.basename(model_p),
                csv_bytes=curr_bytes,
                target_column=target,
            )
            performance_comparison = compare_model_performance(
                ref_metrics=ref_eval["summary"],
                curr_metrics=curr_eval["summary"],
            )
        except Exception as e:
            performance_comparison = {
                "evaluation_error": f"Could not compute performance degradation: {str(e)}"
            }

    drift_results["performance_comparison"] = performance_comparison
    return drift_results


# Mount static assets and main frontend page
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    """Serves the ModelMD web application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ModelMD backend running. Static frontend not yet compiled."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)

