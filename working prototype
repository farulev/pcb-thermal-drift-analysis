"""
pcb_thermal_drift_prototype.py

Prototype ML pipeline for predicting PCB calibration drift after thermal cycling.

What this does:
1. Simulates a dataset for PCB degradation under thermal cycling
2. Trains a regression model to predict drift_percent
3. Trains a classification model to predict threshold exceedance
4. Prints feature importance and evaluation metrics

Author: Rodion Farulev / prototype scaffold
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class SimulationConfig:
    n_samples: int = 1200
    drift_threshold_percent: float = 3.0


def simulate_pcb_dataset(config: SimulationConfig) -> pd.DataFrame:
    """Create a synthetic dataset that mimics PCB degradation after thermal cycling."""

    n = config.n_samples

    pcb_material = np.random.choice(
        ["FR4", "Polyimide", "Ceramic", "Rogers"],
        size=n,
        p=[0.55, 0.20, 0.10, 0.15],
    )
    solder_type = np.random.choice(
        ["SnAgCu", "SnPb", "LowTemp"],
        size=n,
        p=[0.65, 0.20, 0.15],
    )
    coating_type = np.random.choice(
        ["None", "Acrylic", "Silicone", "Parylene"],
        size=n,
        p=[0.30, 0.25, 0.25, 0.20],
    )
    component_density = np.random.choice(
        ["Low", "Medium", "High"],
        size=n,
        p=[0.25, 0.45, 0.30],
    )

    thickness_mm = np.round(np.random.uniform(0.8, 2.4, size=n), 2)
    copper_layers = np.random.choice([2, 4, 6, 8], size=n, p=[0.20, 0.45, 0.25, 0.10])
    thermal_cycles = np.random.randint(20, 1500, size=n)
    temp_min_c = np.random.choice([-60, -40, -20], size=n, p=[0.35, 0.45, 0.20])
    temp_max_c = np.random.choice([60, 85, 105, 125], size=n, p=[0.20, 0.35, 0.30, 0.15])
    ramp_rate_c_per_min = np.round(np.random.uniform(1.0, 12.0, size=n), 2)
    dwell_hot_min = np.random.randint(5, 90, size=n)
    dwell_cold_min = np.random.randint(5, 90, size=n)
    humidity_percent = np.round(np.random.uniform(5, 95, size=n), 1)
    vacuum_kpa = np.round(np.random.uniform(0.01, 100.0, size=n), 2)
    vibration_rms_g = np.round(np.random.uniform(0.0, 12.0, size=n), 2)
    initial_scale_coeff = np.round(np.random.normal(loc=1.0, scale=0.01, size=n), 5)

    temp_span = temp_max_c - temp_min_c

    # Material risk factors
    material_factor = np.array(
        [0.7 if m == "FR4" else 0.9 if m == "Polyimide" else 0.45 if m == "Ceramic" else 0.6 for m in pcb_material]
    )
    solder_factor = np.array(
        [0.65 if s == "SnAgCu" else 0.85 if s == "SnPb" else 1.05 for s in solder_type]
    )
    coating_factor = np.array(
        [1.0 if c == "None" else 0.9 if c == "Acrylic" else 0.8 if c == "Silicone" else 0.7 for c in coating_type]
    )
    density_factor = np.array(
        [0.75 if d == "Low" else 1.0 if d == "Medium" else 1.25 for d in component_density]
    )

    # Synthetic deformation proxy
    deformation_um = (
        0.0025 * thermal_cycles
        + 0.09 * temp_span
        + 0.45 * ramp_rate_c_per_min
        + 0.02 * (dwell_hot_min + dwell_cold_min)
        + 0.65 * vibration_rms_g
        + 0.012 * humidity_percent
        + 2.2 * density_factor
        + 1.5 * solder_factor
        - 1.8 * thickness_mm
        - 0.8 * copper_layers
        - 1.4 * coating_factor
        + np.random.normal(0, 4.0, size=n)
    )
    deformation_um = np.clip(deformation_um, 0.0, None)

    # Drift model: nonlinear-ish synthetic physics-inspired relation
    drift_percent = (
        0.0009 * thermal_cycles
        + 0.012 * temp_span
        + 0.055 * ramp_rate_c_per_min
        + 0.003 * (dwell_hot_min + dwell_cold_min)
        + 0.035 * vibration_rms_g
        + 0.0025 * humidity_percent
        + 0.010 * deformation_um
        + 0.35 * material_factor
        + 0.30 * solder_factor
        + 0.22 * density_factor
        - 0.25 * thickness_mm
        - 0.05 * copper_layers
        - 0.18 * coating_factor
        + np.random.normal(0, 0.45, size=n)
    )
    drift_percent = np.clip(drift_percent, 0.0, None)

    exceeds_threshold = (drift_percent >= config.drift_threshold_percent).astype(int)
    post_scale_coeff = initial_scale_coeff * (1.0 + drift_percent / 100.0)

    df = pd.DataFrame(
        {
            "pcb_material": pcb_material,
            "solder_type": solder_type,
            "coating_type": coating_type,
            "component_density": component_density,
            "thickness_mm": thickness_mm,
            "copper_layers": copper_layers,
            "thermal_cycles": thermal_cycles,
            "temp_min_c": temp_min_c,
            "temp_max_c": temp_max_c,
            "temp_span_c": temp_span,
            "ramp_rate_c_per_min": ramp_rate_c_per_min,
            "dwell_hot_min": dwell_hot_min,
            "dwell_cold_min": dwell_cold_min,
            "humidity_percent": humidity_percent,
            "vacuum_kpa": vacuum_kpa,
            "vibration_rms_g": vibration_rms_g,
            "initial_scale_coeff": initial_scale_coeff,
            "deformation_um": np.round(deformation_um, 3),
            "drift_percent": np.round(drift_percent, 3),
            "post_scale_coeff": np.round(post_scale_coeff, 5),
            "exceeds_threshold": exceeds_threshold,
        }
    )

    return df


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )
    return preprocessor, categorical_cols, numeric_cols


def train_regression_model(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=["drift_percent", "post_scale_coeff", "exceeds_threshold"])
    y = df["drift_percent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preprocessor, _, _ = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=12,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Regression: Predict drift_percent ===")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R^2  : {r2:.3f}")

    return model


def train_classification_model(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=["drift_percent", "post_scale_coeff", "exceeds_threshold"])
    y = df["exceeds_threshold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    preprocessor, _, _ = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Classification: Predict threshold exceedance ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    return model


def print_feature_importance(model: Pipeline, X: pd.DataFrame, top_n: int = 15) -> None:
    """Extract feature names after preprocessing and print top importances."""
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]

    # Feature names after preprocessing
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names = cat_names + num_cols

    if "regressor" in model.named_steps:
        estimator = model.named_steps["regressor"]
    else:
        estimator = model.named_steps["classifier"]

    importances = estimator.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print(f"\n=== Top {top_n} Feature Importances ===")
    print(importance_df.head(top_n).to_string(index=False))


def predict_single_example(model: Pipeline) -> None:
    """Demo inference on one synthetic board."""
    sample = pd.DataFrame(
        [
            {
                "pcb_material": "FR4",
                "solder_type": "LowTemp",
                "coating_type": "None",
                "component_density": "High",
                "thickness_mm": 1.0,
                "copper_layers": 4,
                "thermal_cycles": 900,
                "temp_min_c": -40,
                "temp_max_c": 105,
                "temp_span_c": 145,
                "ramp_rate_c_per_min": 7.5,
                "dwell_hot_min": 45,
                "dwell_cold_min": 40,
                "humidity_percent": 65.0,
                "vacuum_kpa": 1.0,
                "vibration_rms_g": 6.5,
                "initial_scale_coeff": 1.002,
                "deformation_um": 24.0,
            }
        ]
    )

    pred = model.predict(sample)[0]
    print("\n=== Example prediction ===")
    print(f"Predicted drift_percent: {pred:.3f}")


def main() -> None:
    config = SimulationConfig(n_samples=1200, drift_threshold_percent=3.0)
    df = simulate_pcb_dataset(config)

    print("Synthetic dataset created.")
    print(df.head().to_string(index=False))

    # Save demo dataset so you can inspect it or replace it later with real data
    df.to_csv("simulated_pcb_thermal_drift_data.csv", index=False)
    print("\nSaved: simulated_pcb_thermal_drift_data.csv")

    regression_model = train_regression_model(df)
    classification_model = train_classification_model(df)

    X_base = df.drop(columns=["drift_percent", "post_scale_coeff", "exceeds_threshold"])
    print_feature_importance(regression_model, X_base, top_n=12)
    predict_single_example(regression_model)

    # Optional: save models with joblib if needed later
    try:
        import joblib

        joblib.dump(regression_model, "pcb_drift_regression_model.joblib")
        joblib.dump(classification_model, "pcb_drift_classifier_model.joblib")
        print("\nSaved models:")
        print("- pcb_drift_regression_model.joblib")
        print("- pcb_drift_classifier_model.joblib")
    except ImportError:
        print("\njoblib not installed, skipping model save.")


if __name__ == "__main__":
    main()
