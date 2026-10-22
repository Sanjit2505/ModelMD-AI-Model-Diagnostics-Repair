"""
Isolated Worker Process for ModelMD.

Loads the scikit-learn model artifact, checks features, executes predictions,
and computes evaluation metrics in an isolated subprocess.
"""

import sys
import json
import argparse
import traceback
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def load_model_safely(model_path: str):
    """Loads a joblib or pickle artifact."""
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to deserialize model artifact: {str(e)}")

    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(
            f"The uploaded artifact of type '{type(model).__name__}' does not have a callable 'predict()' method. "
            "Please upload a fitted scikit-learn classifier or Pipeline."
        )

    # Check if classifier or fitted pipeline
    is_classifier = hasattr(model, "classes_")
    if not is_classifier and hasattr(model, "named_steps"):
        # Check final estimator in pipeline
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, "classes_"):
            is_classifier = True

    if not is_classifier:
        # Check if predict returns discrete predictions or if user uploaded regressor
        pass

    return model


def inspect_model_metadata(model) -> Dict[str, Any]:
    """Inspects model class name, expected feature names or feature count."""
    model_type = type(model).__name__
    is_pipeline = hasattr(model, "named_steps")
    steps = [name for name, _ in model.steps] if is_pipeline and hasattr(model, "steps") else []

    expected_features = None
    if hasattr(model, "feature_names_in_"):
        expected_features = [str(f) for f in model.feature_names_in_]
    elif is_pipeline and hasattr(model.steps[0][1], "feature_names_in_"):
        expected_features = [str(f) for f in model.steps[0][1].feature_names_in_]

    n_features_in = None
    if hasattr(model, "n_features_in_"):
        n_features_in = int(model.n_features_in_)

    classes = None
    if hasattr(model, "classes_"):
        classes = [str(c) for c in model.classes_]
    elif is_pipeline and hasattr(list(model.named_steps.values())[-1], "classes_"):
        classes = [str(c) for c in list(model.named_steps.values())[-1].classes_]

    return {
        "model_type": model_type,
        "is_pipeline": is_pipeline,
        "pipeline_steps": steps,
        "expected_features": expected_features,
        "n_features_in": n_features_in,
        "classes": classes,
    }


def validate_feature_columns(model, X: pd.DataFrame):
    """
    Validates that X matches the model's expected inputs strictly.
    Never silently reorders or guesses columns when this could change model behavior.
    """
    expected_features = None
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "steps") and hasattr(model.steps[0][1], "feature_names_in_"):
        expected_features = list(model.steps[0][1].feature_names_in_)

    current_columns = list(X.columns)

    if expected_features is not None:
        expected_str = [str(f) for f in expected_features]
        current_str = [str(c) for c in current_columns]

        if current_str != expected_str:
            # Check if missing or extra or out-of-order
            missing = [c for c in expected_str if c not in current_str]
            extra = [c for c in current_str if c not in expected_str]

            error_details = []
            if missing:
                error_details.append(f"Missing expected features: {missing}")
            if extra:
                error_details.append(f"Unexpected extra features: {extra}")
            if not missing and not extra and current_str != expected_str:
                error_details.append(
                    f"Feature column order does not match model expectations.\n"
                    f"Model expects: {expected_str}\n"
                    f"Provided: {current_str}\n"
                    f"ModelMD will not silently reorder columns to avoid unexpected model behavior."
                )

            raise ValueError(
                f"Dataset feature columns do not strictly match model requirements. "
                + " | ".join(error_details)
            )

    if hasattr(model, "n_features_in_"):
        expected_n = int(model.n_features_in_)
        if X.shape[1] != expected_n:
            raise ValueError(
                f"Feature count mismatch: Model expects {expected_n} features, but dataset provided {X.shape[1]} features."
            )


def execute_evaluation(
    model_path: str,
    data_path: str,
    average_method: str = "weighted",
) -> Dict[str, Any]:
    """
    Executes model inference and calculates comprehensive classification metrics.
    """
    # 1. Load model
    model = load_model_safely(model_path)
    meta = inspect_model_metadata(model)

    # 2. Load prepared dataset chunk
    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    X_records = payload["X"]
    y_raw = payload["y"]
    row_indices = payload.get("row_indices", list(range(1, len(y_raw) + 1)))
    dropped_rows = payload.get("dropped_rows_count", 0)

    X = pd.DataFrame.from_records(X_records)
    y_true = np.array(y_raw)

    # 3. Validate features
    validate_feature_columns(model, X)

    # 4. Execute actual predictions
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed during predict(X): {str(e)}")

    if len(y_pred) != len(y_true):
        raise RuntimeError(
            f"Prediction length mismatch: received {len(y_pred)} predictions for {len(y_true)} rows."
        )

    # Convert to string representations for uniform classification comparison
    y_true_str = np.array([str(item) for item in y_true])
    y_pred_str = np.array([str(item) for item in y_pred])

    # Establish explicit class labels
    if meta.get("classes") is not None and len(meta["classes"]) > 0:
        unique_labels = list(meta["classes"])
        # In case actual data or prediction has additional classes not in model classes
        for label in np.unique(np.concatenate([y_true_str, y_pred_str])):
            if label not in unique_labels:
                unique_labels.append(label)
    else:
        unique_labels = sorted(list(np.unique(np.concatenate([y_true_str, y_pred_str]))))

    # 5. Calculate Metrics
    accuracy = float(accuracy_score(y_true_str, y_pred_str))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_str,
        y_pred_str,
        average=average_method,
        zero_division=0,
    )

    # Classification report for per-class breakdown
    report_dict = classification_report(
        y_true_str,
        y_pred_str,
        labels=unique_labels,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = []
    for cls in unique_labels:
        cls_stats = report_dict.get(cls, {})
        per_class_metrics.append({
            "class_label": cls,
            "precision": round(float(cls_stats.get("precision", 0.0)), 4),
            "recall": round(float(cls_stats.get("recall", 0.0)), 4),
            "f1_score": round(float(cls_stats.get("f1-score", 0.0)), 4),
            "support": int(cls_stats.get("support", 0)),
        })

    # Confusion matrix
    cm = confusion_matrix(y_true_str, y_pred_str, labels=unique_labels)
    confusion_matrix_data = {
        "labels": unique_labels,
        "matrix": cm.tolist(),
    }

    # Row-level results
    is_correct_arr = (y_true_str == y_pred_str)
    correct_count = int(np.sum(is_correct_arr))
    incorrect_count = int(len(y_true_str) - correct_count)

    row_level_table = []
    for idx, (orig_row, actual_val, pred_val, is_corr) in enumerate(
        zip(row_indices, y_true_str, y_pred_str, is_correct_arr)
    ):
        row_level_table.append({
            "row_number": int(orig_row),
            "actual": str(actual_val),
            "predicted": str(pred_val),
            "is_correct": bool(is_corr),
        })

    return {
        "status": "success",
        "model_metadata": meta,
        "summary": {
            "total_evaluated_rows": len(y_true_str),
            "correct_predictions": correct_count,
            "incorrect_predictions": incorrect_count,
            "dropped_rows_count": dropped_rows,
            "accuracy": round(accuracy, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "average_method": average_method,
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix_data,
        "row_level_predictions": row_level_table,
    }


def main():
    parser = argparse.ArgumentParser(description="ModelMD Worker Process")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--data", required=True, help="Path to input json data")
    parser.add_argument("--output", required=True, help="Path to output json file")
    parser.add_argument("--average", default="weighted", choices=["macro", "micro", "weighted"])
    args = parser.parse_args()

    try:
        result = execute_evaluation(
            model_path=args.model,
            data_path=args.data,
            average_method=args.average,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        sys.exit(0)
    except Exception as e:
        err_payload = {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(err_payload, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
