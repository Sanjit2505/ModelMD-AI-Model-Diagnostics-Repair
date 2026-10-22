"""
Synthetic data and sample model generation script for ModelMD.

Generates a realistic clinical classification dataset and trains a scikit-learn
Pipeline (StandardScaler + LogisticRegression) saved as a .joblib artifact.
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


def generate_sample_artifacts(output_dir: str = "sample_data"):
    """Creates a sample model and CSV dataset for testing ModelMD."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Generate 180 synthetic clinical patient profiles
    n_samples = 180
    age = np.random.randint(22, 75, size=n_samples)
    glucose = np.random.normal(110, 25, size=n_samples).clip(70, 220)
    blood_pressure = np.random.normal(125, 15, size=n_samples).clip(85, 190)
    bmi = np.random.normal(27.5, 5.0, size=n_samples).clip(18.0, 45.0)
    cholesterol = np.random.normal(205, 35, size=n_samples).clip(120, 320)

    # Risk score calculation
    risk_score = (
        (age - 45) * 0.04
        + (glucose - 100) * 0.05
        + (blood_pressure - 120) * 0.03
        + (bmi - 25) * 0.08
        + (cholesterol - 200) * 0.02
        + np.random.normal(0, 0.7, size=n_samples)
    )

    # Categorize into 3 classes: Low Risk, Moderate Risk, High Risk
    classes = []
    for score in risk_score:
        if score < -0.3:
            classes.append("Low_Risk")
        elif score < 1.2:
            classes.append("Moderate_Risk")
        else:
            classes.append("High_Risk")

    df = pd.DataFrame({
        "age": age,
        "glucose": np.round(glucose, 1),
        "blood_pressure": np.round(blood_pressure, 1),
        "bmi": np.round(bmi, 1),
        "cholesterol": np.round(cholesterol, 1),
        "diagnosis": classes,
    })

    # Introduce a couple of NaN values in bmi/cholesterol to allow testing missing value policies
    df.loc[12, "bmi"] = np.nan
    df.loc[35, "cholesterol"] = np.nan

    csv_path = os.path.join(output_dir, "sample_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated sample dataset: {csv_path} ({len(df)} rows, target: 'diagnosis')")

    # 2. Train a scikit-learn Pipeline on clean rows
    clean_df = df.dropna()
    X_train = clean_df.drop(columns=["diagnosis"])
    y_train = clean_df["diagnosis"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=300, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    model_path = os.path.join(output_dir, "sample_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved trained scikit-learn Pipeline: {model_path}")

    # Check model features
    if hasattr(pipeline, "feature_names_in_"):
        print(f"Model feature names: {list(pipeline.feature_names_in_)}")
    elif hasattr(pipeline.steps[0][1], "feature_names_in_"):
        print(f"Model feature names: {list(pipeline.steps[0][1].feature_names_in_)}")

    print("\nSample generation complete!")
    print(f"To test ModelMD:")
    print(f"1. Model file: {os.path.abspath(model_path)}")
    print(f"2. Dataset file: {os.path.abspath(csv_path)}")
    print(f"3. Target column: 'diagnosis'")


if __name__ == "__main__":
    generate_sample_artifacts()
