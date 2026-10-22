"""
Dataset loading, schema validation, and summary utilities for ModelMD.
"""

from typing import Dict, Any, List, Tuple, Optional
import io
import pandas as pd


class DatasetValidationError(Exception):
    """Raised when CSV validation fails."""
    pass


def inspect_csv(content: bytes, filename: str = "dataset.csv") -> Dict[str, Any]:
    """
    Parses CSV bytes, extracts column metadata, summary stats, and a 10-row preview.
    """
    if not content or len(content.strip()) == 0:
        raise DatasetValidationError(f"File '{filename}' is completely empty.")

    try:
        # Read dataset with pandas
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise DatasetValidationError(f"Failed to parse CSV '{filename}': {str(e)}")

    if df.empty:
        raise DatasetValidationError(f"CSV file '{filename}' contains no rows or data.")

    if len(df.columns) < 2:
        raise DatasetValidationError(
            f"CSV file '{filename}' must contain at least 2 columns (1 feature + 1 target). Found {len(df.columns)}."
        )

    # Calculate statistics
    row_count = int(len(df))
    col_count = int(len(df.columns))
    missing_by_col = {str(col): int(df[col].isna().sum()) for col in df.columns}
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Preview first 10 rows
    preview_df = df.head(10).fillna("")
    preview_records = preview_df.to_dict(orient="records")
    # Clean records to native python types for JSON serialization
    clean_preview = []
    for row in preview_records:
        clean_preview.append({str(k): (v if not pd.isna(v) else "") for k, v in row.items()})

    columns = [str(c) for c in df.columns]
    column_types = {str(col): str(df[col].dtype) for col in df.columns}

    return {
        "filename": filename,
        "row_count": row_count,
        "column_count": col_count,
        "columns": columns,
        "column_types": column_types,
        "total_missing_values": total_missing,
        "missing_per_column": missing_by_col,
        "duplicate_rows": duplicate_rows,
        "preview": clean_preview,
    }


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    missing_strategy: str = "drop",
) -> Tuple[pd.DataFrame, pd.Series, List[int], int]:
    """
    Validates and separates features X and target y.
    Preserves original row indices.
    
    Returns:
        (X, y, original_row_indices, dropped_rows_count)
    """
    if target_column not in df.columns:
        raise DatasetValidationError(
            f"Selected target column '{target_column}' does not exist in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Retain original row indices (1-indexed for user readability)
    original_indices = [int(i + 1) for i in df.index]
    initial_count = len(df)

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if len(X.columns) == 0:
        raise DatasetValidationError("Dataset has no feature columns after excluding the target column.")

    # Check missing values
    has_nans = X.isna().any().any() or y.isna().any()

    if has_nans:
        if missing_strategy == "reject":
            nan_cols = [c for c in df.columns if df[c].isna().any()]
            raise DatasetValidationError(
                f"Missing values detected in columns {nan_cols}. "
                f"Evaluation rejected because missing strategy is set to 'reject'. "
                f"Switch to 'drop' or clean your dataset."
            )
        elif missing_strategy == "drop":
            combined = pd.concat([X, y], axis=1)
            valid_mask = combined.notna().all(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            original_indices = [original_indices[i] for i, valid in enumerate(valid_mask) if valid]
        else:
            raise DatasetValidationError(f"Unsupported missing value strategy: '{missing_strategy}'. Must be 'drop' or 'reject'.")

    dropped_count = initial_count - len(X)

    if len(X) == 0:
        raise DatasetValidationError(
            "All rows were dropped due to missing values. Cannot evaluate model on an empty dataset."
        )

    return X, y, original_indices, dropped_count
