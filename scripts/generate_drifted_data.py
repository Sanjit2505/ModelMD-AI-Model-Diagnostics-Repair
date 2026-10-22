"""
Generates synthetic current (drifted) production dataset for testing ModelMD Module 02.

Simulates a real-world demographic and health shift:
- Glucose levels shifted upward by 35 mg/dL (severe drift)
- BMI shifted upward by 4.5 units (moderate drift)
- Age shifted slightly (mild shift)
- Evaluates actual labels to simulate a drop in classification accuracy
"""

import os
import numpy as np
import pandas as pd


def generate_drifted_dataset(output_path: str = "sample_data/sample_dataset_drifted.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(99)

    n_samples = 150
    # Simulate temporal shift in patients
    age = np.random.randint(35, 82, size=n_samples)  # Older patient group
    glucose = np.random.normal(148, 30, size=n_samples).clip(80, 240)  # Significant upward shift (Drift!)
    blood_pressure = np.random.normal(135, 18, size=n_samples).clip(90, 200)  # Moderate shift
    bmi = np.random.normal(32.0, 5.5, size=n_samples).clip(20.0, 48.0)  # Higher obesity rate
    cholesterol = np.random.normal(220, 40, size=n_samples).clip(130, 340)

    # Calculate real-world diagnosis under new clinical ground truth
    risk_score = (
        (age - 45) * 0.04
        + (glucose - 100) * 0.05
        + (blood_pressure - 120) * 0.03
        + (bmi - 25) * 0.08
        + (cholesterol - 200) * 0.02
        + np.random.normal(0, 0.7, size=n_samples)
    )

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

    df.to_csv(output_path, index=False)
    print(f"Generated drifted dataset: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    generate_drifted_dataset()
