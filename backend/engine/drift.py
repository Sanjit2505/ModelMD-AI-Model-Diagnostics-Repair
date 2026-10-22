"""
ModelMD Module 02: Model Health & Data Drift Engine.

Provides rigorous statistical distribution comparison, drift detection
(Two-Sample Kolmogorov-Smirnov test for numerical features, Chi-Square contingency
test and Total Variation Distance for categorical features), outlier & range shift
tracking, and performance degradation analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from scipy import stats


class DriftAnalysisError(Exception):
    """Raised when data drift analysis encounters invalid inputs or computation errors."""
    pass


def format_p_value(p_val: float) -> str:
    """
    Formats a p-value with sufficient precision to distinguish very small values.
    Never displays a rounded zero as if it were an exact value.
    """
    if p_val <= 0.0 or p_val < 1e-15:
        return "< 1e-15"
    if p_val < 0.0001:
        return f"{p_val:.2e}"
    if p_val < 0.01:
        return f"{p_val:.4f}"
    return f"{p_val:.4f}"


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between reference and current distributions.
    Handles constant distributions, small samples, and empty inputs gracefully.
    """
    ref_clean = reference[np.isfinite(reference)]
    curr_clean = current[np.isfinite(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # If reference is constant or has insufficient unique values, PSI is 0.0 if current matches
    if len(np.unique(ref_clean)) < 2:
        if len(np.unique(curr_clean)) < 2 and np.allclose(ref_clean[0], curr_clean[0]):
            return 0.0
        # If reference and current are constant but different, return high PSI
        return 1.0 if not np.allclose(np.mean(ref_clean), np.mean(curr_clean)) else 0.0

    try:
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref_clean, quantiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        bins[0] = -np.inf
        bins[-1] = np.inf
    except Exception:
        return 0.0

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    curr_counts, _ = np.histogram(curr_clean, bins=bins)

    ref_pct = (ref_counts / len(ref_clean))
    curr_pct = (curr_counts / len(curr_clean))

    # Apply Laplace smoothing to avoid division by zero or log(0)
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(round(max(0.0, psi), 4))


def analyze_numerical_feature_drift(
    ref_series: pd.Series,
    curr_series: pd.Series,
    p_value_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Runs two-sample Kolmogorov-Smirnov (KS) test, Wasserstein distance,
    PSI, and summary statistics comparison for numerical features.
    
    Handles missing values and non-finite values explicitly.
    Applies the significance threshold strictly: p_value >= threshold is never significant.
    Does not label a feature High merely because of out-of-range values.
    """
    warnings: List[str] = []

    # Explicitly coerce to numeric and filter non-finite values (NaN, Inf, -Inf)
    ref_num = pd.to_numeric(ref_series, errors="coerce")
    curr_num = pd.to_numeric(curr_series, errors="coerce")

    ref_valid = ref_num[np.isfinite(ref_num)].to_numpy(dtype=float)
    curr_valid = curr_num[np.isfinite(curr_num)].to_numpy(dtype=float)

    ref_missing = int(len(ref_series) - len(ref_valid))
    curr_missing = int(len(curr_series) - len(curr_valid))

    if ref_missing > 0 or curr_missing > 0:
        warnings.append(
            f"Explicitly excluded non-finite/missing values: {ref_missing} in reference, {curr_missing} in current."
        )

    if len(ref_valid) == 0 or len(curr_valid) == 0:
        return {
            "feature_type": "numerical",
            "drift_detected": False,
            "p_value": 1.0,
            "p_value_formatted": "1.0000",
            "test_used": "Insufficient data",
            "statistic": 0.0,
            "severity": "None",
            "ref_stats": {},
            "curr_stats": {},
            "warnings": warnings + ["No valid numerical data available for statistical comparison."],
            "out_of_range_count": 0,
            "mean_shift_pct": 0.0,
        }

    # Summary statistics
    ref_mean = float(np.mean(ref_valid))
    curr_mean = float(np.mean(curr_valid))
    ref_std = float(np.std(ref_valid))
    curr_std = float(np.std(curr_valid))
    ref_min = float(np.min(ref_valid))
    curr_min = float(np.min(curr_valid))
    ref_max = float(np.max(ref_valid))
    curr_max = float(np.max(curr_valid))

    # Out-of-range check against historical reference range
    out_of_range_count = int(np.sum((curr_valid < ref_min) | (curr_valid > ref_max)))
    if out_of_range_count > 0:
        warnings.append(
            f"{out_of_range_count} current value(s) outside historical reference range [{ref_min:.2f}, {ref_max:.2f}]."
        )

    # Constant column check
    ref_is_const = bool(len(ref_valid) > 0 and np.all(ref_valid == ref_valid[0]))
    curr_is_const = bool(len(curr_valid) > 0 and np.all(curr_valid == curr_valid[0]))

    if ref_is_const and curr_is_const:
        if np.isclose(ref_valid[0], curr_valid[0]):
            # Identical constants -> no drift
            ks_stat = 0.0
            p_value = 1.0
            wasserstein = 0.0
            psi_val = 0.0
            warnings.append("Feature is constant with identical values in both reference and current.")
        else:
            # Different constant values -> definite distribution shift
            ks_stat = 1.0
            p_value = 0.0
            wasserstein = float(abs(curr_valid[0] - ref_valid[0]))
            psi_val = 1.0
            warnings.append(f"Constant shifted: reference was {ref_valid[0]}, current is {curr_valid[0]}.")
    else:
        # Two-sample KS test
        ks_res = stats.ks_2samp(ref_valid, curr_valid)
        ks_stat = float(ks_res.statistic)
        p_value = float(ks_res.pvalue)
        wasserstein = float(stats.wasserstein_distance(ref_valid, curr_valid))
        psi_val = calculate_psi(ref_valid, curr_valid)

    mean_diff_pct = ((curr_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else (0.0 if curr_mean == 0 else 100.0)

    # STRICT SIGNIFICANCE RULE:
    # A feature with p-value >= threshold is NOT statistically significant, severity MUST be "None".
    if p_value >= p_value_threshold:
        drift_detected = False
        severity = "None"
    else:
        drift_detected = True
        # Transparent, documented severity rules considering statistical significance AND effect size:
        # High: Significant AND (KS statistic >= 0.30 OR PSI >= 0.25 OR |mean shift| >= 30%)
        # Moderate: Significant AND (KS statistic >= 0.15 OR PSI >= 0.10 OR |mean shift| >= 10%)
        # Low: Significant AND effect size is small (KS < 0.15 AND PSI < 0.10 AND |mean shift| < 10%)
        if ks_stat >= 0.30 or psi_val >= 0.25 or abs(mean_diff_pct) >= 30.0:
            severity = "High"
        elif ks_stat >= 0.15 or psi_val >= 0.10 or abs(mean_diff_pct) >= 10.0:
            severity = "Moderate"
        else:
            severity = "Low"

    return {
        "feature_type": "numerical",
        "drift_detected": bool(drift_detected),
        "severity": severity,
        "test_used": "Two-Sample Kolmogorov-Smirnov (KS)",
        "statistic": round(ks_stat, 4),
        "p_value": p_value,
        "p_value_formatted": format_p_value(p_value),
        "wasserstein_distance": round(wasserstein, 4),
        "psi": psi_val,
        "ref_stats": {
            "mean": round(ref_mean, 2),
            "std": round(ref_std, 2),
            "min": round(ref_min, 2),
            "max": round(ref_max, 2),
            "valid_count": len(ref_valid),
            "missing_count": ref_missing,
        },
        "curr_stats": {
            "mean": round(curr_mean, 2),
            "std": round(curr_std, 2),
            "min": round(curr_min, 2),
            "max": round(curr_max, 2),
            "valid_count": len(curr_valid),
            "missing_count": curr_missing,
        },
        "mean_shift_pct": round(mean_diff_pct, 1),
        "out_of_range_count": out_of_range_count,
        "warnings": warnings,
    }


def analyze_categorical_feature_drift(
    ref_series: pd.Series,
    curr_series: pd.Series,
    p_value_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Analyzes categorical feature drift using Chi-Square Contingency Test,
    Total Variation Distance (TVD), and Cramér's V effect size.
    
    Categorical distribution changes are evaluated appropriately rather than
    applying numerical KS tests.
    """
    warnings: List[str] = []

    ref_clean = ref_series.dropna().astype(str)
    curr_clean = curr_series.dropna().astype(str)

    ref_missing = int(len(ref_series) - len(ref_clean))
    curr_missing = int(len(curr_series) - len(curr_clean))

    if ref_missing > 0 or curr_missing > 0:
        warnings.append(
            f"Excluded missing values: {ref_missing} in reference, {curr_missing} in current."
        )

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return {
            "feature_type": "categorical",
            "drift_detected": False,
            "p_value": 1.0,
            "p_value_formatted": "1.0000",
            "test_used": "Insufficient data",
            "statistic": 0.0,
            "severity": "None",
            "new_categories": [],
            "missing_categories": [],
            "category_distribution": {},
            "warnings": warnings + ["No valid categorical data available for comparison."],
            "mean_shift_pct": None,
        }

    ref_counts = ref_clean.value_counts()
    curr_counts = curr_clean.value_counts()

    all_categories = sorted(list(set(ref_counts.index).union(set(curr_counts.index))))
    ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories], dtype=float)
    curr_freq = np.array([curr_counts.get(cat, 0) for cat in all_categories], dtype=float)

    # Detect unseen new categories and vanished categories
    new_categories = [cat for cat in curr_counts.index if cat not in ref_counts.index]
    missing_categories = [cat for cat in ref_counts.index if cat not in curr_counts.index]

    if new_categories:
        warnings.append(f"Encountered {len(new_categories)} new category/categories: {new_categories[:5]}")
    if missing_categories:
        warnings.append(f"Missing {len(missing_categories)} baseline category/categories in current data: {missing_categories[:5]}")

    # Total Variation Distance (TVD): 0.5 * sum(|P(x) - Q(x)|)
    ref_prob = ref_freq / np.sum(ref_freq)
    curr_prob = curr_freq / np.sum(curr_freq)
    tvd = float(0.5 * np.sum(np.abs(ref_prob - curr_prob)))

    # Chi-Square Contingency Test
    chi2_stat = 0.0
    p_value = 1.0
    cramers_v = 0.0
    test_used = "Chi-Square Test of Independence"

    if len(all_categories) <= 1:
        # Constant single category in both
        chi2_stat = 0.0
        p_value = 1.0
        cramers_v = 0.0
        test_used = "Chi-Square Test (single category)"
    else:
        try:
            contingency = np.array([ref_freq, curr_freq])
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            chi2_stat = float(chi2)
            p_value = float(p)
            total_n = np.sum(contingency)
            # Cramér's V for 2 x K contingency table: sqrt(chi2 / (total_n * (min(2-1, K-1)))) = sqrt(chi2 / total_n)
            cramers_v = float(np.sqrt(chi2_stat / total_n)) if total_n > 0 else 0.0
        except Exception as e:
            test_used = "Frequency Shift Comparison"
            warnings.append(f"Chi-square test calculation fallback: {str(e)}")
            p_value = 1.0 if tvd < 0.10 else 0.01

    # STRICT SIGNIFICANCE RULE:
    # A feature with p-value >= threshold is NOT statistically significant.
    if p_value >= p_value_threshold:
        drift_detected = False
        severity = "None"
    else:
        drift_detected = True
        # Transparent severity rules for categorical distributions:
        # High: Significant AND (Cramér's V >= 0.30 OR TVD >= 0.25)
        # Moderate: Significant AND (Cramér's V >= 0.15 OR TVD >= 0.10)
        # Low: Significant AND minor shift (Cramér's V < 0.15 AND TVD < 0.10)
        if cramers_v >= 0.30 or tvd >= 0.25:
            severity = "High"
        elif cramers_v >= 0.15 or tvd >= 0.10:
            severity = "Moderate"
        else:
            severity = "Low"

    return {
        "feature_type": "categorical",
        "drift_detected": bool(drift_detected),
        "severity": severity,
        "test_used": test_used,
        "statistic": round(chi2_stat, 4),
        "cramers_v": round(cramers_v, 4),
        "tvd": round(tvd, 4),
        "p_value": p_value,
        "p_value_formatted": format_p_value(p_value),
        "new_categories": new_categories,
        "missing_categories": missing_categories,
        "category_distribution": {
            cat: {
                "ref_count": int(ref_counts.get(cat, 0)),
                "curr_count": int(curr_counts.get(cat, 0)),
            }
            for cat in all_categories
        },
        "ref_stats": {
            "valid_count": len(ref_clean),
            "missing_count": ref_missing,
            "unique_categories": len(ref_counts),
        },
        "curr_stats": {
            "valid_count": len(curr_clean),
            "missing_count": curr_missing,
            "unique_categories": len(curr_counts),
        },
        "mean_shift_pct": None,
        "warnings": warnings,
    }


def is_identifier_column(col_name: str, series: pd.Series) -> bool:
    """
    Determines whether a column represents an identifier/key rather than an analytical feature.
    Checks column naming conventions and high cardinality unique strings.
    """
    clean_name = str(col_name).strip().lower()

    # Exact matches for common identifier names
    common_id_names = {
        "id", "uuid", "guid", "index", "idx", "row_id", "record_id",
        "sample_id", "patient_id", "subject_id", "case_id", "user_id",
        "client_id", "customer_id", "employee_id", "transaction_id",
    }
    if clean_name in common_id_names:
        return True

    # Suffixes or prefixes indicating identifiers
    if clean_name.endswith(("_id", "_uuid", "_guid", "_key", "id")):
        # Avoid false positives like 'valid', 'acid', 'fluid', 'grid'
        valid_words = {"valid", "acid", "fluid", "grid", "solid", "hybrid", "lipid", "pyramid", "rapid"}
        if clean_name not in valid_words:
            return True

    if clean_name.startswith(("id_", "uuid_")):
        return True

    # High cardinality string identifier heuristic (100% unique strings with at least 15 rows)
    if (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)) and len(series) >= 15:
        if series.dropna().nunique() == len(series.dropna()):
            return True

    return False


def compare_datasets_for_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    target_column: Optional[str] = None,
    p_value_threshold: float = 0.05,
    id_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Performs comprehensive dataset-level and feature-level drift analysis.
    
    1. Excludes configured target column from feature drift analysis.
    2. Excludes identifier columns from feature drift analysis.
    3. Corrects drift percentage denominator to count ONLY eligible feature columns.
    4. Computes KS-test for numerical features and Chi-Square for categorical features.
    5. Enforces consistent significance threshold and effect-size severity grading.
    """
    if ref_df is None or curr_df is None or ref_df.empty or curr_df.empty:
        raise DriftAnalysisError("Reference or Current dataset is empty or unreadable.")

    # 1. Resolve target column (case-insensitive strip matching)
    matched_target: Optional[str] = None
    if target_column and target_column.strip():
        target_clean = target_column.strip().lower()
        for col in ref_df.columns:
            if str(col).strip().lower() == target_clean:
                matched_target = col
                break

    # 2. Identify identifier columns to exclude
    explicit_ids = set(id_columns or [])
    detected_ids: List[str] = []

    for col in ref_df.columns:
        if matched_target and col == matched_target:
            continue
        if col in explicit_ids or is_identifier_column(col, ref_df[col]):
            detected_ids.append(col)

    excluded_columns: Set[str] = set()
    if matched_target:
        excluded_columns.add(matched_target)
    excluded_columns.update(detected_ids)

    # 3. Determine eligible feature columns (MUST be in both datasets and not excluded)
    eligible_features = [
        col for col in ref_df.columns
        if col in curr_df.columns and col not in excluded_columns
    ]

    if not eligible_features:
        raise DriftAnalysisError(
            "No eligible overlapping feature columns found between Reference and Current datasets to compare "
            "(columns are either missing, target, or identifiers)."
        )

    # Missing and extra columns tracking
    missing_in_current = [c for c in ref_df.columns if c not in curr_df.columns and c not in excluded_columns]
    extra_in_current = [c for c in curr_df.columns if c not in ref_df.columns and c not in excluded_columns]

    features_report: Dict[str, Any] = {}
    drifted_features: List[str] = []
    high_drift_count = 0

    # 4. Analyze each eligible feature
    for col in eligible_features:
        ref_col = ref_df[col]
        curr_col = curr_df[col]

        is_num = pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(curr_col)

        if is_num:
            res = analyze_numerical_feature_drift(ref_col, curr_col, p_value_threshold=p_value_threshold)
        else:
            res = analyze_categorical_feature_drift(ref_col, curr_col, p_value_threshold=p_value_threshold)

        features_report[col] = res
        if res["drift_detected"]:
            drifted_features.append(col)
            if res.get("severity") == "High":
                high_drift_count += 1

    # 5. Correct drift percentage denominator: counts ONLY eligible feature columns
    total_features = len(eligible_features)
    drift_share = (len(drifted_features) / total_features) if total_features > 0 else 0.0

    # Overall Dataset Health Status
    if high_drift_count > 0 or drift_share >= 0.5:
        health_status = "Critical Drift"
        status_color = "danger"
    elif len(drifted_features) > 0 or drift_share > 0.2:
        health_status = "Moderate Drift"
        status_color = "warning"
    else:
        health_status = "Healthy (Stable)"
        status_color = "success"

    return {
        "status": "success",
        "dataset_health": {
            "health_status": health_status,
            "status_color": status_color,
            "total_features": total_features,
            "eligible_features": eligible_features,
            "excluded_columns": {
                "target_column": matched_target,
                "identifier_columns": detected_ids,
            },
            "drifted_features_count": len(drifted_features),
            "drifted_features_list": drifted_features,
            "drift_share_pct": round(drift_share * 100, 1),
            "p_value_threshold_used": p_value_threshold,
            "ref_row_count": len(ref_df),
            "curr_row_count": len(curr_df),
            "missing_columns_in_current": missing_in_current,
            "extra_columns_in_current": extra_in_current,
        },
        "features": features_report,
    }


def compare_model_performance(
    ref_metrics: Dict[str, Any],
    curr_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes performance degradation delta when ground-truth labels are available in both datasets.
    """
    acc_ref = ref_metrics.get("accuracy", 0.0)
    acc_curr = curr_metrics.get("accuracy", 0.0)
    acc_delta = round(acc_curr - acc_ref, 4)

    f1_ref = ref_metrics.get("f1_score", 0.0)
    f1_curr = curr_metrics.get("f1_score", 0.0)
    f1_delta = round(f1_curr - f1_ref, 4)

    prec_ref = ref_metrics.get("precision", 0.0)
    prec_curr = curr_metrics.get("precision", 0.0)
    prec_delta = round(prec_curr - prec_ref, 4)

    rec_ref = ref_metrics.get("recall", 0.0)
    rec_curr = curr_metrics.get("recall", 0.0)
    rec_delta = round(rec_curr - rec_ref, 4)

    degraded = (acc_delta < -0.05) or (f1_delta < -0.05)

    return {
        "performance_degraded": degraded,
        "accuracy": {
            "baseline": acc_ref,
            "current": acc_curr,
            "delta": acc_delta,
            "delta_pct": round(acc_delta * 100, 1),
        },
        "f1_score": {
            "baseline": f1_ref,
            "current": f1_curr,
            "delta": f1_delta,
            "delta_pct": round(f1_delta * 100, 1),
        },
        "precision": {
            "baseline": prec_ref,
            "current": prec_curr,
            "delta": prec_delta,
        },
        "recall": {
            "baseline": rec_ref,
            "current": rec_curr,
            "delta": rec_delta,
        },
    }
