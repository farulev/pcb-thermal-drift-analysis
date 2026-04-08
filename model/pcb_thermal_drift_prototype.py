"""
pcb_thermal_drift_prototype.py

Hard-mode ML simulation pipeline for predicting PCB calibration drift after
thermal cycling. This version expands the original scaffold with:

1. Richer synthetic physics-inspired degradation signals
2. Batch effects, outliers, and injected missingness
3. Stacked regression and classification ensembles
4. Permutation importance and multi-scenario inference demos
"""

from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
TRAINING_FILE_NAME = "pcb_training_data.csv"
SCORE_FILE_NAME = "pcb_input.csv"
TEMPLATE_FILE_NAME = "pcb_input_template.csv"
REPORT_FILE_NAME = "pcb_drift_report.html"
SCENARIO_FILE_NAME = "pcb_demo_scenario_predictions.csv"
SCORED_FILE_NAME = "pcb_input_predictions.csv"

MATERIAL_BASE_RISK = {
    "FR4": 0.82,
    "HighTg_FR4": 0.62,
    "Polyimide": 0.70,
    "Ceramic": 0.28,
    "Rogers": 0.52,
}
MATERIAL_TG = {
    "FR4": 136.0,
    "HighTg_FR4": 172.0,
    "Polyimide": 255.0,
    "Ceramic": 400.0,
    "Rogers": 287.0,
}
MATERIAL_CTE_Z = {
    "FR4": 69.0,
    "HighTg_FR4": 55.0,
    "Polyimide": 47.0,
    "Ceramic": 8.0,
    "Rogers": 31.0,
}
MATERIAL_MOISTURE = {
    "FR4": 0.18,
    "HighTg_FR4": 0.13,
    "Polyimide": 0.27,
    "Ceramic": 0.01,
    "Rogers": 0.05,
}
SOLDER_CREEP_FACTOR = {
    "SnAgCu": 0.68,
    "SnPb": 0.94,
    "LowTemp": 1.18,
    "SACXPlus": 0.56,
}
SOLDER_FATIGUE_FACTOR = {
    "SnAgCu": 0.76,
    "SnPb": 0.91,
    "LowTemp": 1.09,
    "SACXPlus": 0.66,
}
COATING_BARRIER = {
    "None": 0.00,
    "Acrylic": 0.34,
    "Silicone": 0.52,
    "Parylene": 0.82,
    "NanoCeramic": 0.65,
}
DENSITY_FACTOR = {"Low": 0.74, "Medium": 1.00, "High": 1.24, "UltraHigh": 1.48}
VIA_FACTOR = {"Through": 0.78, "BlindBuried": 0.96, "Microvia": 1.15, "StackedMicrovia": 1.36}
FINISH_FACTOR = {"HASL": 0.97, "ENIG": 0.76, "OSP": 1.03, "ImmAg": 0.86}
PROCESS_FACTOR = {"LineA": 0.88, "LineB": 1.00, "LineC": 1.10, "LineD": 1.24}
CHAMBER_FACTOR = {"Chamber_1": 0.94, "Chamber_2": 1.00, "Chamber_3": 1.13}
VENDOR_FACTOR = {"VendorA": 0.92, "VendorB": 1.03, "VendorC": 1.00, "VendorD": 1.11, "VendorE": 0.96}

CATEGORICAL_INPUT_COLUMNS = [
    "pcb_material",
    "solder_type",
    "coating_type",
    "component_density",
    "via_structure",
    "surface_finish",
    "process_line",
    "chamber_id",
    "laminate_vendor",
]
NUMERIC_INPUT_COLUMNS = [
    "thickness_mm",
    "copper_layers",
    "copper_weight_oz",
    "thermal_cycles",
    "temp_min_c",
    "temp_max_c",
    "ramp_rate_c_per_min",
    "dwell_hot_min",
    "dwell_cold_min",
    "humidity_percent",
    "vacuum_kpa",
    "vibration_rms_g",
    "initial_scale_coeff",
    "rework_count",
    "pcb_area_cm2",
    "hole_density_per_dm2",
    "copper_imbalance_percent",
    "void_fraction_percent",
    "ionic_contamination_ug_cm2",
    "board_age_days",
    "cycles_per_day",
    "peak_assembly_temp_c",
    "inspection_delay_hours",
]
DERIVED_FEATURE_COLUMNS = [
    "temp_span_c",
    "tg_c",
    "cte_z_ppm",
    "moisture_absorption_percent",
    "warp_energy_index",
    "via_fatigue_index",
    "solder_creep_index",
    "delamination_risk_index",
    "residual_stress_index",
    "deformation_um",
]
BASE_FEATURE_DEFAULTS = {
    "pcb_material": "HighTg_FR4",
    "solder_type": "SnAgCu",
    "coating_type": "Silicone",
    "component_density": "Medium",
    "via_structure": "Through",
    "surface_finish": "ENIG",
    "process_line": "LineB",
    "chamber_id": "Chamber_2",
    "laminate_vendor": "VendorC",
    "thickness_mm": 1.6,
    "copper_layers": 4,
    "copper_weight_oz": 1.0,
    "thermal_cycles": 400,
    "temp_min_c": -40,
    "temp_max_c": 85,
    "ramp_rate_c_per_min": 5.0,
    "dwell_hot_min": 20,
    "dwell_cold_min": 20,
    "humidity_percent": 40.0,
    "vacuum_kpa": 5.0,
    "vibration_rms_g": 2.0,
    "initial_scale_coeff": 1.0,
    "rework_count": 0,
    "pcb_area_cm2": 80.0,
    "hole_density_per_dm2": 150.0,
    "copper_imbalance_percent": 8.0,
    "void_fraction_percent": 2.0,
    "ionic_contamination_ug_cm2": 0.4,
    "board_age_days": 60,
    "cycles_per_day": 1.0,
    "peak_assembly_temp_c": 245,
    "inspection_delay_hours": 8.0,
}


@dataclass(frozen=True)
class SimulationConfig:
    n_samples: int = 2600
    drift_threshold_percent: float = 3.0
    missing_rate: float = 0.04
    outlier_rate: float = 0.015
    random_seed: int = RANDOM_SEED


def make_onehot_encoder() -> OneHotEncoder:
    """Create a version-compatible encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def map_by_key(values: np.ndarray, mapping: dict[str, float]) -> np.ndarray:
    return np.array([mapping[item] for item in values], dtype=float)


def assign_if_missing_or_null(df: pd.DataFrame, column: str, values: pd.Series | np.ndarray) -> None:
    computed = pd.Series(values, index=df.index)
    if column not in df.columns:
        df[column] = computed
    else:
        current = pd.to_numeric(df[column], errors="coerce")
        df[column] = current.fillna(computed)


def build_input_template() -> pd.DataFrame:
    template = build_demo_scenarios().reset_index().rename(columns={"index": "board_id"})
    template = template[["board_id"] + CATEGORICAL_INPUT_COLUMNS + NUMERIC_INPUT_COLUMNS]
    return template


def prepare_prediction_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize user input and compute derived model features automatically."""
    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "board_id" not in df.columns:
        df.insert(0, "board_id", [f"board_{idx + 1:04d}" for idx in range(len(df))])

    for column in CATEGORICAL_INPUT_COLUMNS + NUMERIC_INPUT_COLUMNS:
        if column not in df.columns:
            df[column] = BASE_FEATURE_DEFAULTS[column]

    for column in CATEGORICAL_INPUT_COLUMNS:
        default_value = BASE_FEATURE_DEFAULTS[column]
        df[column] = (
            df[column]
            .astype("string")
            .fillna(default_value)
            .str.strip()
            .replace({"": default_value, "<NA>": default_value})
        )

    for column in NUMERIC_INPUT_COLUMNS + DERIVED_FEATURE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in NUMERIC_INPUT_COLUMNS:
        df[column] = df[column].fillna(BASE_FEATURE_DEFAULTS[column])

    assign_if_missing_or_null(df, "temp_span_c", df["temp_max_c"] - df["temp_min_c"])

    material_risk = df["pcb_material"].map(MATERIAL_BASE_RISK).fillna(MATERIAL_BASE_RISK[BASE_FEATURE_DEFAULTS["pcb_material"]])
    tg_c = df["pcb_material"].map(MATERIAL_TG).fillna(MATERIAL_TG[BASE_FEATURE_DEFAULTS["pcb_material"]])
    cte_z_ppm = df["pcb_material"].map(MATERIAL_CTE_Z).fillna(MATERIAL_CTE_Z[BASE_FEATURE_DEFAULTS["pcb_material"]])
    moisture_absorption_percent = df["pcb_material"].map(MATERIAL_MOISTURE).fillna(
        MATERIAL_MOISTURE[BASE_FEATURE_DEFAULTS["pcb_material"]]
    )
    solder_creep = df["solder_type"].map(SOLDER_CREEP_FACTOR).fillna(
        SOLDER_CREEP_FACTOR[BASE_FEATURE_DEFAULTS["solder_type"]]
    )
    solder_fatigue = df["solder_type"].map(SOLDER_FATIGUE_FACTOR).fillna(
        SOLDER_FATIGUE_FACTOR[BASE_FEATURE_DEFAULTS["solder_type"]]
    )
    coating_protection = df["coating_type"].map(COATING_BARRIER).fillna(
        COATING_BARRIER[BASE_FEATURE_DEFAULTS["coating_type"]]
    )
    density_load = df["component_density"].map(DENSITY_FACTOR).fillna(
        DENSITY_FACTOR[BASE_FEATURE_DEFAULTS["component_density"]]
    )
    via_complexity = df["via_structure"].map(VIA_FACTOR).fillna(VIA_FACTOR[BASE_FEATURE_DEFAULTS["via_structure"]])
    finish_risk = df["surface_finish"].map(FINISH_FACTOR).fillna(FINISH_FACTOR[BASE_FEATURE_DEFAULTS["surface_finish"]])
    process_instability = df["process_line"].map(PROCESS_FACTOR).fillna(PROCESS_FACTOR[BASE_FEATURE_DEFAULTS["process_line"]])
    chamber_stress = df["chamber_id"].map(CHAMBER_FACTOR).fillna(CHAMBER_FACTOR[BASE_FEATURE_DEFAULTS["chamber_id"]])
    vendor_variation = df["laminate_vendor"].map(VENDOR_FACTOR).fillna(
        VENDOR_FACTOR[BASE_FEATURE_DEFAULTS["laminate_vendor"]]
    )

    assign_if_missing_or_null(df, "tg_c", tg_c)
    assign_if_missing_or_null(df, "cte_z_ppm", cte_z_ppm)
    assign_if_missing_or_null(df, "moisture_absorption_percent", moisture_absorption_percent)

    tg_margin = np.maximum(df["tg_c"] - df["temp_max_c"], 8.0)
    thermal_aggression = (df["temp_span_c"] / tg_margin) * (0.9 + 0.12 * df["ramp_rate_c_per_min"])
    moisture_ingress_index = (
        df["humidity_percent"]
        * (0.55 + 3.2 * df["moisture_absorption_percent"])
        * (1.0 - 0.72 * coating_protection)
        * (1.0 + 0.0009 * df["board_age_days"])
        + 8.5 * df["ionic_contamination_ug_cm2"]
    )
    solder_creep_index = (
        0.0012 * df["thermal_cycles"] * df["temp_max_c"] * solder_creep
        + 0.085 * df["dwell_hot_min"] * solder_creep
        + 0.18 * df["rework_count"] * solder_creep * df["peak_assembly_temp_c"] / 100.0
    )
    via_fatigue_index = (
        0.042
        * np.sqrt(df["thermal_cycles"])
        * df["temp_span_c"]
        * via_complexity
        * solder_fatigue
        / np.sqrt(df["thickness_mm"])
        + 0.006 * df["hole_density_per_dm2"]
        + 0.35 * df["copper_layers"]
    )
    warp_energy_index = (
        0.012
        * df["pcb_area_cm2"]
        * (1.0 + df["copper_imbalance_percent"] / 12.0)
        * density_load
        * (df["temp_span_c"] / 100.0)
        / np.maximum(df["thickness_mm"], 0.65)
        + 0.35 * df["copper_weight_oz"]
    )
    delamination_risk_index = (
        0.08 * moisture_ingress_index
        + 0.95 * thermal_aggression
        + 0.018 * df["cte_z_ppm"]
        + 0.42 * df["rework_count"]
        + 0.15 * process_instability
    )
    residual_stress_index = (
        0.72 * process_instability
        + 0.48 * chamber_stress
        + 0.30 * vendor_variation
        + 0.008 * df["inspection_delay_hours"]
        + 0.028 * df["board_age_days"] / 30.0
        + 0.018 * df["cycles_per_day"] * df["thermal_cycles"] / 100.0
    )
    deformation_um = (
        0.028 * warp_energy_index
        + 0.065 * via_fatigue_index
        + 0.055 * solder_creep_index
        + 0.52 * df["vibration_rms_g"]
        + 0.018 * moisture_ingress_index
        + 0.020 * df["hole_density_per_dm2"] / 10.0
        + 1.4 * density_load
        - 1.9 * df["thickness_mm"]
        - 0.75 * df["copper_layers"]
        - 1.45 * coating_protection
    )

    assign_if_missing_or_null(df, "warp_energy_index", warp_energy_index)
    assign_if_missing_or_null(df, "via_fatigue_index", via_fatigue_index)
    assign_if_missing_or_null(df, "solder_creep_index", solder_creep_index)
    assign_if_missing_or_null(df, "delamination_risk_index", delamination_risk_index)
    assign_if_missing_or_null(df, "residual_stress_index", residual_stress_index)
    assign_if_missing_or_null(df, "deformation_um", np.clip(deformation_um, 0.0, None))

    # Keep computed values numeric after any fill operations.
    for column in DERIVED_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "predicted_physics_risk" not in df.columns:
        df["predicted_physics_risk"] = (
            0.16 * material_risk
            + 0.12 * finish_risk
            + 0.008 * df["via_fatigue_index"]
            + 0.006 * df["solder_creep_index"]
            + 0.005 * df["delamination_risk_index"]
        )

    return df


def inject_missing_values(
    df: pd.DataFrame,
    rng: np.random.Generator,
    base_rate: float,
) -> pd.DataFrame:
    """Introduce realistic test-bench gaps and metadata omissions."""
    numeric_cols = [
        "humidity_percent",
        "vacuum_kpa",
        "vibration_rms_g",
        "void_fraction_percent",
        "ionic_contamination_ug_cm2",
        "initial_scale_coeff",
        "deformation_um",
        "inspection_delay_hours",
    ]
    categorical_cols = ["coating_type", "surface_finish", "laminate_vendor"]

    line_d_bias = (df["process_line"] == "LineD").to_numpy(dtype=float)
    chamber_3_bias = (df["chamber_id"] == "Chamber_3").to_numpy(dtype=float)

    for col in numeric_cols:
        missing_prob = base_rate * (1.0 + 0.45 * line_d_bias + 0.15 * chamber_3_bias)
        mask = rng.random(len(df)) < missing_prob
        df.loc[mask, col] = np.nan

    for col in categorical_cols:
        mask = rng.random(len(df)) < (base_rate * 0.55)
        df.loc[mask, col] = np.nan

    return df


def simulate_pcb_dataset(config: SimulationConfig) -> pd.DataFrame:
    """Create a synthetic PCB drift dataset with nonlinear degradation signals."""
    rng = np.random.default_rng(config.random_seed)
    n = config.n_samples

    pcb_material = rng.choice(
        ["FR4", "HighTg_FR4", "Polyimide", "Ceramic", "Rogers"],
        size=n,
        p=[0.36, 0.22, 0.16, 0.08, 0.18],
    )
    solder_type = rng.choice(
        ["SnAgCu", "SnPb", "LowTemp", "SACXPlus"],
        size=n,
        p=[0.46, 0.18, 0.16, 0.20],
    )
    coating_type = rng.choice(
        ["None", "Acrylic", "Silicone", "Parylene", "NanoCeramic"],
        size=n,
        p=[0.24, 0.24, 0.22, 0.18, 0.12],
    )
    component_density = rng.choice(
        ["Low", "Medium", "High", "UltraHigh"],
        size=n,
        p=[0.18, 0.42, 0.28, 0.12],
    )
    via_structure = rng.choice(
        ["Through", "BlindBuried", "Microvia", "StackedMicrovia"],
        size=n,
        p=[0.38, 0.24, 0.24, 0.14],
    )
    surface_finish = rng.choice(
        ["HASL", "ENIG", "OSP", "ImmAg"],
        size=n,
        p=[0.24, 0.38, 0.18, 0.20],
    )
    process_line = rng.choice(
        ["LineA", "LineB", "LineC", "LineD"],
        size=n,
        p=[0.28, 0.34, 0.23, 0.15],
    )
    chamber_id = rng.choice(
        ["Chamber_1", "Chamber_2", "Chamber_3"],
        size=n,
        p=[0.42, 0.36, 0.22],
    )
    laminate_vendor = rng.choice(
        ["VendorA", "VendorB", "VendorC", "VendorD", "VendorE"],
        size=n,
        p=[0.20, 0.23, 0.24, 0.18, 0.15],
    )

    thickness_mm = np.round(rng.uniform(0.7, 2.8, size=n), 3)
    copper_layers = rng.choice([2, 4, 6, 8, 10, 12], size=n, p=[0.08, 0.32, 0.24, 0.18, 0.10, 0.08])
    copper_weight_oz = rng.choice([0.5, 1.0, 2.0, 3.0], size=n, p=[0.14, 0.50, 0.26, 0.10])
    thermal_cycles = rng.integers(40, 2201, size=n)
    temp_min_c = rng.choice([-65, -55, -40, -20], size=n, p=[0.16, 0.24, 0.42, 0.18])
    temp_max_c = rng.choice([70, 85, 105, 125, 150], size=n, p=[0.14, 0.28, 0.30, 0.18, 0.10])
    ramp_rate_c_per_min = np.round(rng.uniform(0.8, 18.0, size=n), 3)
    dwell_hot_min = rng.integers(6, 121, size=n)
    dwell_cold_min = rng.integers(6, 121, size=n)
    humidity_percent = np.round(rng.uniform(6.0, 98.0, size=n), 2)
    vacuum_kpa = np.round(np.exp(rng.uniform(np.log(0.05), np.log(100.0), size=n)), 3)
    vibration_rms_g = np.round(rng.uniform(0.0, 18.0, size=n), 3)
    initial_scale_coeff = np.round(rng.normal(loc=1.0, scale=0.012, size=n), 6)
    rework_count = rng.poisson(lam=0.9, size=n)
    pcb_area_cm2 = np.round(rng.uniform(18.0, 240.0, size=n), 2)
    hole_density_per_dm2 = np.round(rng.uniform(20.0, 700.0, size=n), 2)
    copper_imbalance_percent = np.round(rng.uniform(0.0, 34.0, size=n), 2)
    void_fraction_percent = np.round(rng.beta(1.8, 7.4, size=n) * 20.0, 3)
    ionic_contamination_ug_cm2 = np.round(rng.gamma(shape=2.2, scale=0.34, size=n), 3)
    board_age_days = rng.integers(1, 720, size=n)
    cycles_per_day = np.round(rng.uniform(0.4, 8.5, size=n), 3)
    peak_assembly_temp_c = rng.choice([235, 245, 255, 265], size=n, p=[0.12, 0.38, 0.36, 0.14])
    inspection_delay_hours = np.round(rng.uniform(0.5, 72.0, size=n), 2)

    temp_span_c = temp_max_c - temp_min_c

    material_base_risk = {
        "FR4": 0.82,
        "HighTg_FR4": 0.62,
        "Polyimide": 0.70,
        "Ceramic": 0.28,
        "Rogers": 0.52,
    }
    material_tg = {
        "FR4": 136.0,
        "HighTg_FR4": 172.0,
        "Polyimide": 255.0,
        "Ceramic": 400.0,
        "Rogers": 287.0,
    }
    material_cte_z = {
        "FR4": 69.0,
        "HighTg_FR4": 55.0,
        "Polyimide": 47.0,
        "Ceramic": 8.0,
        "Rogers": 31.0,
    }
    material_moisture = {
        "FR4": 0.18,
        "HighTg_FR4": 0.13,
        "Polyimide": 0.27,
        "Ceramic": 0.01,
        "Rogers": 0.05,
    }
    solder_creep_factor = {
        "SnAgCu": 0.68,
        "SnPb": 0.94,
        "LowTemp": 1.18,
        "SACXPlus": 0.56,
    }
    solder_fatigue_factor = {
        "SnAgCu": 0.76,
        "SnPb": 0.91,
        "LowTemp": 1.09,
        "SACXPlus": 0.66,
    }
    coating_barrier = {
        "None": 0.00,
        "Acrylic": 0.34,
        "Silicone": 0.52,
        "Parylene": 0.82,
        "NanoCeramic": 0.65,
    }
    density_factor = {"Low": 0.74, "Medium": 1.00, "High": 1.24, "UltraHigh": 1.48}
    via_factor = {"Through": 0.78, "BlindBuried": 0.96, "Microvia": 1.15, "StackedMicrovia": 1.36}
    finish_factor = {"HASL": 0.97, "ENIG": 0.76, "OSP": 1.03, "ImmAg": 0.86}
    process_factor = {"LineA": 0.88, "LineB": 1.00, "LineC": 1.10, "LineD": 1.24}
    chamber_factor = {"Chamber_1": 0.94, "Chamber_2": 1.00, "Chamber_3": 1.13}
    vendor_factor = {"VendorA": 0.92, "VendorB": 1.03, "VendorC": 1.00, "VendorD": 1.11, "VendorE": 0.96}

    material_risk = map_by_key(pcb_material, material_base_risk)
    tg_c = map_by_key(pcb_material, material_tg)
    cte_z_ppm = map_by_key(pcb_material, material_cte_z)
    moisture_absorption_percent = map_by_key(pcb_material, material_moisture)
    solder_creep = map_by_key(solder_type, solder_creep_factor)
    solder_fatigue = map_by_key(solder_type, solder_fatigue_factor)
    coating_protection = map_by_key(coating_type, coating_barrier)
    density_load = map_by_key(component_density, density_factor)
    via_complexity = map_by_key(via_structure, via_factor)
    finish_risk = map_by_key(surface_finish, finish_factor)
    process_instability = map_by_key(process_line, process_factor)
    chamber_stress = map_by_key(chamber_id, chamber_factor)
    vendor_variation = map_by_key(laminate_vendor, vendor_factor)

    tg_margin = np.maximum(tg_c - temp_max_c, 8.0)
    thermal_aggression = (temp_span_c / tg_margin) * (0.9 + 0.12 * ramp_rate_c_per_min)
    moisture_ingress_index = (
        humidity_percent
        * (0.55 + 3.2 * moisture_absorption_percent)
        * (1.0 - 0.72 * coating_protection)
        * (1.0 + 0.0009 * board_age_days)
        + 8.5 * ionic_contamination_ug_cm2
    )
    solder_creep_index = (
        0.0012 * thermal_cycles * temp_max_c * solder_creep
        + 0.085 * dwell_hot_min * solder_creep
        + 0.18 * rework_count * solder_creep * peak_assembly_temp_c / 100.0
    )
    via_fatigue_index = (
        0.042 * np.sqrt(thermal_cycles) * temp_span_c * via_complexity * solder_fatigue / np.sqrt(thickness_mm)
        + 0.006 * hole_density_per_dm2
        + 0.35 * copper_layers
    )
    warp_energy_index = (
        0.012 * pcb_area_cm2
        * (1.0 + copper_imbalance_percent / 12.0)
        * density_load
        * (temp_span_c / 100.0)
        / np.maximum(thickness_mm, 0.65)
        + 0.35 * copper_weight_oz
    )
    delamination_risk_index = (
        0.08 * moisture_ingress_index
        + 0.95 * thermal_aggression
        + 0.018 * cte_z_ppm
        + 0.42 * rework_count
        + 0.15 * process_instability
    )
    residual_stress_index = (
        0.72 * process_instability
        + 0.48 * chamber_stress
        + 0.30 * vendor_variation
        + 0.008 * inspection_delay_hours
        + 0.028 * board_age_days / 30.0
        + 0.018 * cycles_per_day * thermal_cycles / 100.0
    )

    deformation_um = (
        0.028 * warp_energy_index
        + 0.065 * via_fatigue_index
        + 0.055 * solder_creep_index
        + 0.52 * vibration_rms_g
        + 0.018 * moisture_ingress_index
        + 0.020 * hole_density_per_dm2 / 10.0
        + 1.4 * density_load
        - 1.9 * thickness_mm
        - 0.75 * copper_layers
        - 1.45 * coating_protection
        + rng.normal(0.0, 2.8 + 0.012 * temp_span_c, size=n)
    )
    deformation_um = np.clip(deformation_um, 0.0, None)

    base_drift = (
        0.032 * np.sqrt(thermal_cycles)
        + 0.0038 * temp_span_c
        + 0.019 * ramp_rate_c_per_min
        + 0.0016 * (dwell_hot_min + dwell_cold_min)
        + 0.018 * vibration_rms_g
        + 0.0011 * humidity_percent
        + 0.0095 * deformation_um
        + 0.0038 * moisture_ingress_index
        + 0.0060 * via_fatigue_index
        + 0.0048 * solder_creep_index
        + 0.0062 * delamination_risk_index
        + 0.110 * material_risk
        + 0.080 * finish_risk
        + 0.055 * residual_stress_index
        - 0.24 * thickness_mm
        - 0.030 * copper_layers
        - 0.140 * coating_protection
        - 0.050 * np.log1p(vacuum_kpa)
    )

    nonlinear_uplift = (
        0.12 * np.maximum(temp_max_c - 110.0, 0.0) / 10.0
        + 0.09 * np.maximum(humidity_percent - 75.0, 0.0) / 10.0
        + 0.22 * np.maximum(rework_count - 1, 0)
        + 0.16 * np.maximum(copper_imbalance_percent - 18.0, 0.0) / 10.0
        + 0.12 * np.maximum(void_fraction_percent - 7.0, 0.0) / 4.0
    )

    interaction_uplift = (
        0.000055 * temp_span_c * humidity_percent * via_complexity
        + 0.00014 * thermal_cycles * copper_imbalance_percent / 100.0
        + 0.00095 * temp_span_c * rework_count * solder_creep
        + 0.00130 * pcb_area_cm2 * density_load / np.maximum(thickness_mm, 0.65)
    )

    heteroscedastic_noise = rng.normal(
        loc=0.0,
        scale=0.18 + 0.028 * process_instability + 0.0008 * temp_span_c + 0.0015 * humidity_percent,
        size=n,
    )

    early_failure = (
        (temp_span_c > 175)
        & (humidity_percent > 82)
        & np.isin(via_structure, ["Microvia", "StackedMicrovia"])
        & (coating_type == "None")
    )
    rare_shock = rng.random(n) < config.outlier_rate
    robust_stack = (
        (pcb_material == "Ceramic")
        & np.isin(coating_type, ["Parylene", "NanoCeramic"])
        & (thickness_mm > 1.7)
        & (copper_imbalance_percent < 10.0)
    )

    drift_percent = (
        base_drift
        + nonlinear_uplift
        + interaction_uplift
        + heteroscedastic_noise
        + early_failure.astype(float) * rng.normal(1.8, 0.35, size=n)
        + rare_shock.astype(float) * rng.uniform(0.7, 2.4, size=n)
        - robust_stack.astype(float) * rng.uniform(0.3, 0.9, size=n)
        - 0.85
    )
    drift_percent = np.clip(drift_percent, 0.0, None)

    exceeds_threshold = (drift_percent >= config.drift_threshold_percent).astype(int)
    severe_threshold = (drift_percent >= (config.drift_threshold_percent * 1.75)).astype(int)
    post_scale_coeff = initial_scale_coeff * (1.0 + drift_percent / 100.0)

    df = pd.DataFrame(
        {
            "pcb_material": pcb_material,
            "solder_type": solder_type,
            "coating_type": coating_type,
            "component_density": component_density,
            "via_structure": via_structure,
            "surface_finish": surface_finish,
            "process_line": process_line,
            "chamber_id": chamber_id,
            "laminate_vendor": laminate_vendor,
            "thickness_mm": thickness_mm,
            "copper_layers": copper_layers,
            "copper_weight_oz": copper_weight_oz,
            "thermal_cycles": thermal_cycles,
            "temp_min_c": temp_min_c,
            "temp_max_c": temp_max_c,
            "temp_span_c": temp_span_c,
            "ramp_rate_c_per_min": ramp_rate_c_per_min,
            "dwell_hot_min": dwell_hot_min,
            "dwell_cold_min": dwell_cold_min,
            "humidity_percent": humidity_percent,
            "vacuum_kpa": vacuum_kpa,
            "vibration_rms_g": vibration_rms_g,
            "initial_scale_coeff": initial_scale_coeff,
            "rework_count": rework_count,
            "pcb_area_cm2": pcb_area_cm2,
            "hole_density_per_dm2": hole_density_per_dm2,
            "copper_imbalance_percent": copper_imbalance_percent,
            "void_fraction_percent": void_fraction_percent,
            "ionic_contamination_ug_cm2": ionic_contamination_ug_cm2,
            "board_age_days": board_age_days,
            "cycles_per_day": cycles_per_day,
            "peak_assembly_temp_c": peak_assembly_temp_c,
            "inspection_delay_hours": inspection_delay_hours,
            "tg_c": np.round(tg_c, 2),
            "cte_z_ppm": np.round(cte_z_ppm, 2),
            "moisture_absorption_percent": np.round(moisture_absorption_percent, 4),
            "warp_energy_index": np.round(warp_energy_index, 3),
            "via_fatigue_index": np.round(via_fatigue_index, 3),
            "solder_creep_index": np.round(solder_creep_index, 3),
            "delamination_risk_index": np.round(delamination_risk_index, 3),
            "residual_stress_index": np.round(residual_stress_index, 3),
            "deformation_um": np.round(deformation_um, 3),
            "drift_percent": np.round(drift_percent, 4),
            "post_scale_coeff": np.round(post_scale_coeff, 6),
            "exceeds_threshold": exceeds_threshold,
            "severe_threshold": severe_threshold,
        }
    )

    return inject_missing_values(df, rng, config.missing_rate)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor, categorical_cols, numeric_cols


def build_regression_pipeline(preprocessor: ColumnTransformer, random_seed: int) -> Pipeline:
    regressor = StackingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=140,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=random_seed,
                    n_jobs=1,
                ),
            ),
            (
                "et",
                ExtraTreesRegressor(
                    n_estimators=160,
                    max_depth=18,
                    min_samples_leaf=2,
                    random_state=random_seed,
                    n_jobs=1,
                ),
            ),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.045,
                    max_depth=3,
                    subsample=0.88,
                    random_state=random_seed,
                ),
            ),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
        cv=3,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def build_classification_pipeline(preprocessor: ColumnTransformer, random_seed: int) -> Pipeline:
    classifier = StackingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=140,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_seed,
                    n_jobs=1,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=160,
                    max_depth=16,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_seed,
                    n_jobs=1,
                ),
            ),
            (
                "gbc",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.88,
                    random_state=random_seed,
                ),
            ),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=3,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def split_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=["drift_percent", "post_scale_coeff", "exceeds_threshold", "severe_threshold"])
    y_reg = df["drift_percent"]
    y_cls = df["exceeds_threshold"]
    return X, y_reg, y_cls


def train_regression_model(
    df: pd.DataFrame,
    config: SimulationConfig,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, dict[str, float]]:
    X, y, _ = split_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.random_seed,
    )

    preprocessor, _, _ = build_preprocessor(X_train)
    model = build_regression_pipeline(preprocessor, config.random_seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Regression: Predict drift_percent ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")
    print(f"Target mean drift : {y_test.mean():.4f}%")
    print(f"Target max drift  : {y_test.max():.4f}%")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "target_mean": float(y_test.mean()),
        "target_max": float(y_test.max()),
    }
    return model, X_test, y_test, metrics


def train_classification_model(
    df: pd.DataFrame,
    config: SimulationConfig,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, dict[str, float]]:
    X, _, y = split_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.random_seed,
        stratify=y,
    )

    preprocessor, _, _ = build_preprocessor(X_train)
    model = build_classification_pipeline(preprocessor, config.random_seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification: Predict threshold exceedance ===")
    print(f"Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision         : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall            : {recall_score(y_test, y_pred):.4f}")
    print(f"F1                : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC           : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision : {average_precision_score(y_test, y_proba):.4f}")
    print("Confusion matrix  :")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
    }
    return model, X_test, y_test, metrics


def get_transformed_feature_names(model: Pipeline, X: pd.DataFrame) -> list[str]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    return cat_names + num_cols


def print_permutation_importance(
    model: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    scoring: str,
    label: str,
    top_n: int = 15,
) -> pd.DataFrame:
    sample_size = min(420, len(X_eval))
    X_sample = X_eval.iloc[:sample_size].copy()
    y_sample = y_eval.iloc[:sample_size].copy()
    transformed = model.named_steps["preprocessor"].transform(X_sample)
    feature_names = get_transformed_feature_names(model, X_eval)

    estimator = model.named_steps["regressor"] if "regressor" in model.named_steps else model.named_steps["classifier"]
    result = permutation_importance(
        estimator,
        transformed,
        y_sample,
        scoring=scoring,
        n_repeats=5,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .head(top_n)
    )

    print(f"\n=== Top {top_n} Permutation Importances ({label}) ===")
    print(importance_df.to_string(index=False))
    return importance_df.reset_index(drop=True)


def build_demo_scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pcb_material": "Ceramic",
                "solder_type": "SACXPlus",
                "coating_type": "Parylene",
                "component_density": "Medium",
                "via_structure": "Through",
                "surface_finish": "ENIG",
                "process_line": "LineA",
                "chamber_id": "Chamber_1",
                "laminate_vendor": "VendorA",
                "thickness_mm": 2.1,
                "copper_layers": 6,
                "copper_weight_oz": 1.0,
                "thermal_cycles": 320,
                "temp_min_c": -40,
                "temp_max_c": 85,
                "temp_span_c": 125,
                "ramp_rate_c_per_min": 3.4,
                "dwell_hot_min": 18,
                "dwell_cold_min": 20,
                "humidity_percent": 22.0,
                "vacuum_kpa": 8.0,
                "vibration_rms_g": 1.8,
                "initial_scale_coeff": 0.9985,
                "rework_count": 0,
                "pcb_area_cm2": 55.0,
                "hole_density_per_dm2": 90.0,
                "copper_imbalance_percent": 6.0,
                "void_fraction_percent": 1.2,
                "ionic_contamination_ug_cm2": 0.22,
                "board_age_days": 60,
                "cycles_per_day": 1.1,
                "peak_assembly_temp_c": 245,
                "inspection_delay_hours": 6.0,
                "tg_c": 400.0,
                "cte_z_ppm": 8.0,
                "moisture_absorption_percent": 0.0100,
                "warp_energy_index": 4.0,
                "via_fatigue_index": 8.0,
                "solder_creep_index": 5.5,
                "delamination_risk_index": 2.8,
                "residual_stress_index": 1.3,
                "deformation_um": 5.0,
            },
            {
                "pcb_material": "HighTg_FR4",
                "solder_type": "SnAgCu",
                "coating_type": "Silicone",
                "component_density": "High",
                "via_structure": "Microvia",
                "surface_finish": "ENIG",
                "process_line": "LineB",
                "chamber_id": "Chamber_2",
                "laminate_vendor": "VendorC",
                "thickness_mm": 1.3,
                "copper_layers": 8,
                "copper_weight_oz": 1.0,
                "thermal_cycles": 980,
                "temp_min_c": -55,
                "temp_max_c": 105,
                "temp_span_c": 160,
                "ramp_rate_c_per_min": 7.2,
                "dwell_hot_min": 38,
                "dwell_cold_min": 34,
                "humidity_percent": 58.0,
                "vacuum_kpa": 1.2,
                "vibration_rms_g": 5.1,
                "initial_scale_coeff": 1.0012,
                "rework_count": 1,
                "pcb_area_cm2": 112.0,
                "hole_density_per_dm2": 280.0,
                "copper_imbalance_percent": 15.0,
                "void_fraction_percent": 4.8,
                "ionic_contamination_ug_cm2": 0.64,
                "board_age_days": 210,
                "cycles_per_day": 3.1,
                "peak_assembly_temp_c": 255,
                "inspection_delay_hours": 18.0,
                "tg_c": 172.0,
                "cte_z_ppm": 55.0,
                "moisture_absorption_percent": 0.1300,
                "warp_energy_index": 17.0,
                "via_fatigue_index": 24.0,
                "solder_creep_index": 16.5,
                "delamination_risk_index": 11.0,
                "residual_stress_index": 3.4,
                "deformation_um": 19.0,
            },
            {
                "pcb_material": "FR4",
                "solder_type": "LowTemp",
                "coating_type": "None",
                "component_density": "UltraHigh",
                "via_structure": "StackedMicrovia",
                "surface_finish": "OSP",
                "process_line": "LineD",
                "chamber_id": "Chamber_3",
                "laminate_vendor": "VendorD",
                "thickness_mm": 0.9,
                "copper_layers": 10,
                "copper_weight_oz": 2.0,
                "thermal_cycles": 1650,
                "temp_min_c": -65,
                "temp_max_c": 150,
                "temp_span_c": 215,
                "ramp_rate_c_per_min": 14.2,
                "dwell_hot_min": 74,
                "dwell_cold_min": 62,
                "humidity_percent": 91.0,
                "vacuum_kpa": 0.2,
                "vibration_rms_g": 11.8,
                "initial_scale_coeff": 1.0048,
                "rework_count": 3,
                "pcb_area_cm2": 168.0,
                "hole_density_per_dm2": 540.0,
                "copper_imbalance_percent": 28.0,
                "void_fraction_percent": 10.5,
                "ionic_contamination_ug_cm2": 1.88,
                "board_age_days": 520,
                "cycles_per_day": 6.5,
                "peak_assembly_temp_c": 265,
                "inspection_delay_hours": 48.0,
                "tg_c": 136.0,
                "cte_z_ppm": 69.0,
                "moisture_absorption_percent": 0.1800,
                "warp_energy_index": 34.0,
                "via_fatigue_index": 44.0,
                "solder_creep_index": 33.0,
                "delamination_risk_index": 21.0,
                "residual_stress_index": 6.8,
                "deformation_um": 38.0,
            },
        ],
        index=["rugged_board", "production_typical", "extreme_failure_candidate"],
    )


def predict_demo_scenarios(regression_model: Pipeline, classification_model: Pipeline) -> pd.DataFrame:
    scenarios = build_demo_scenarios()
    drift_pred = regression_model.predict(scenarios)
    exceed_prob = classification_model.predict_proba(scenarios)[:, 1]

    output = pd.DataFrame(
        {
            "predicted_drift_percent": np.round(drift_pred, 4),
            "threshold_exceed_prob": np.round(exceed_prob, 4),
        },
        index=scenarios.index,
    )

    print("\n=== Scenario Inference ===")
    print(output.to_string())
    return output.reset_index(names="scenario_name")


def write_input_template(template_dir: Path) -> Path:
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / TEMPLATE_FILE_NAME
    build_input_template().to_csv(template_path, index=False)
    return template_path


def find_score_input_path(explicit_path: Path | None, search_dir: Path) -> Path | None:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path

    auto_candidate = search_dir / SCORE_FILE_NAME
    if auto_candidate.exists():
        return auto_candidate

    return None


def score_input_file(
    score_path: Path,
    regression_model: Pipeline,
    classification_model: Pipeline,
    threshold: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    raw_df = pd.read_csv(score_path)
    prepared_df = prepare_prediction_features(raw_df)
    features = prepared_df.drop(columns=["board_id"], errors="ignore")

    scored_df = raw_df.copy()
    if "board_id" not in scored_df.columns:
        scored_df.insert(0, "board_id", prepared_df["board_id"])

    scored_df["predicted_drift_percent"] = np.round(regression_model.predict(features), 4)
    scored_df["threshold_exceed_prob"] = np.round(classification_model.predict_proba(features)[:, 1], 4)
    scored_df["predicted_exceeds_threshold"] = (
        (scored_df["predicted_drift_percent"] >= threshold) | (scored_df["threshold_exceed_prob"] >= 0.5)
    ).astype(int)
    scored_df["predicted_risk_band"] = pd.cut(
        scored_df["threshold_exceed_prob"],
        bins=[-0.001, 0.20, 0.50, 0.80, 1.00],
        labels=["low", "moderate", "high", "critical"],
    ).astype("string")

    if "drift_percent" in raw_df.columns:
        actual = pd.to_numeric(raw_df["drift_percent"], errors="coerce")
        scored_df["prediction_error_percent"] = np.round(scored_df["predicted_drift_percent"] - actual, 4)

    predictions_path = output_dir / SCORED_FILE_NAME
    scored_df.to_csv(predictions_path, index=False)

    print("\n=== Batch Scoring ===")
    print(f"Loaded input boards: {score_path}")
    print(f"Saved predictions : {predictions_path}")
    preview_cols = ["board_id", "predicted_drift_percent", "threshold_exceed_prob", "predicted_risk_band"]
    print(scored_df[preview_cols].head(10).to_string(index=False))

    return scored_df, predictions_path


def render_bar_chart_svg(title: str, labels: list[str], values: list[float], color: str) -> str:
    if not labels or not values:
        return "<p>No chart data available.</p>"

    width = 920
    left_pad = 260
    right_pad = 30
    top_pad = 40
    row_height = 28
    usable_width = width - left_pad - right_pad
    max_value = max(values) if max(values) > 0 else 1.0
    height = top_pad + row_height * len(values) + 16

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="18" y="24" font-size="18" font-family="Segoe UI, Arial, sans-serif" fill="#17324d">{html.escape(title)}</text>',
    ]

    for idx, (label, value) in enumerate(zip(labels, values, strict=False)):
        y = top_pad + idx * row_height
        bar_width = 0 if max_value == 0 else (value / max_value) * usable_width
        svg_parts.append(
            f'<text x="18" y="{y + 16}" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="#364152">'
            f"{html.escape(str(label))}</text>"
        )
        svg_parts.append(
            f'<rect x="{left_pad}" y="{y + 2}" width="{bar_width:.1f}" height="16" rx="4" fill="{color}" opacity="0.88" />'
        )
        svg_parts.append(
            f'<text x="{left_pad + bar_width + 8:.1f}" y="{y + 16}" font-size="12" '
            f'font-family="Consolas, monospace" fill="#364152">{value:.4f}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def render_histogram_svg(title: str, values: pd.Series, color: str) -> str:
    clean_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if clean_values.size == 0:
        return "<p>No histogram data available.</p>"

    counts, bins = np.histogram(clean_values, bins=min(16, max(6, int(np.sqrt(clean_values.size)))))
    width = 920
    height = 280
    left_pad = 50
    right_pad = 20
    top_pad = 40
    bottom_pad = 40
    plot_width = width - left_pad - right_pad
    plot_height = height - top_pad - bottom_pad
    max_count = max(counts.max(), 1)
    bar_width = plot_width / len(counts)

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="18" y="24" font-size="18" font-family="Segoe UI, Arial, sans-serif" fill="#17324d">{html.escape(title)}</text>',
        f'<line x1="{left_pad}" y1="{height - bottom_pad}" x2="{width - right_pad}" y2="{height - bottom_pad}" stroke="#c6d1dc" />',
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{height - bottom_pad}" stroke="#c6d1dc" />',
    ]

    for idx, count in enumerate(counts):
        bar_height = (count / max_count) * plot_height
        x = left_pad + idx * bar_width + 2
        y = height - bottom_pad - bar_height
        svg_parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 4, 2):.1f}" height="{bar_height:.1f}" fill="{color}" opacity="0.82" />'
        )
        svg_parts.append(
            f'<text x="{x + (bar_width / 2):.1f}" y="{height - bottom_pad + 16}" text-anchor="middle" '
            f'font-size="10" font-family="Consolas, monospace" fill="#4b5563">{bins[idx]:.1f}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def save_html_report(
    output_dir: Path,
    training_df: pd.DataFrame,
    regression_metrics: dict[str, float],
    classification_metrics: dict[str, float],
    regression_importance: pd.DataFrame,
    classification_importance: pd.DataFrame,
    scenario_predictions: pd.DataFrame,
    scored_predictions: pd.DataFrame | None,
    score_input_path: Path | None,
    predictions_path: Path | None,
    template_path: Path,
) -> Path:
    report_path = output_dir / REPORT_FILE_NAME
    top_scored = None
    if scored_predictions is not None:
        top_scored = scored_predictions.sort_values("threshold_exceed_prob", ascending=False).head(15)

    regression_chart = render_bar_chart_svg(
        "Regression Feature Importance",
        regression_importance["feature"].tolist(),
        regression_importance["importance_mean"].tolist(),
        "#0f766e",
    )
    classification_chart = render_bar_chart_svg(
        "Classification Feature Importance",
        classification_importance["feature"].tolist(),
        classification_importance["importance_mean"].tolist(),
        "#b45309",
    )
    drift_histogram = render_histogram_svg("Training Drift Distribution", training_df["drift_percent"], "#2563eb")

    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>PCB Drift Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #1f2937; background: #f8fafc; }}
    h1, h2 {{ color: #0f172a; }}
    .card {{ background: white; border: 1px solid #dbe4ee; border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05); }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .metric {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 12px; }}
    .metric strong {{ display: block; font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
    th, td {{ border: 1px solid #dbe4ee; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef4fb; }}
    code {{ background: #e8eef6; padding: 2px 6px; border-radius: 6px; }}
    .note {{ color: #475569; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>PCB Thermal Drift Automatic Report</h1>

  <div class="card">
    <h2>Run Summary</h2>
    <p class="note">Drop a <code>{SCORE_FILE_NAME}</code> file next to the script and rerun to score your own boards automatically.</p>
    <p class="note">Input template: <code>{template_path}</code></p>
    <p class="note">Scored input source: <code>{score_input_path if score_input_path else "not provided"}</code></p>
    <p class="note">Scored predictions file: <code>{predictions_path if predictions_path else "not generated in this run"}</code></p>
  </div>

  <div class="card">
    <h2>Regression Metrics</h2>
    <div class="metrics">
      <div class="metric"><strong>MAE</strong>{regression_metrics["mae"]:.4f}</div>
      <div class="metric"><strong>RMSE</strong>{regression_metrics["rmse"]:.4f}</div>
      <div class="metric"><strong>R^2</strong>{regression_metrics["r2"]:.4f}</div>
      <div class="metric"><strong>Mean Drift</strong>{regression_metrics["target_mean"]:.4f}%</div>
      <div class="metric"><strong>Max Drift</strong>{regression_metrics["target_max"]:.4f}%</div>
    </div>
  </div>

  <div class="card">
    <h2>Classification Metrics</h2>
    <div class="metrics">
      <div class="metric"><strong>Accuracy</strong>{classification_metrics["accuracy"]:.4f}</div>
      <div class="metric"><strong>Precision</strong>{classification_metrics["precision"]:.4f}</div>
      <div class="metric"><strong>Recall</strong>{classification_metrics["recall"]:.4f}</div>
      <div class="metric"><strong>F1</strong>{classification_metrics["f1"]:.4f}</div>
      <div class="metric"><strong>ROC AUC</strong>{classification_metrics["roc_auc"]:.4f}</div>
      <div class="metric"><strong>Avg Precision</strong>{classification_metrics["avg_precision"]:.4f}</div>
    </div>
  </div>

  <div class="card">{drift_histogram}</div>
  <div class="card">{regression_chart}</div>
  <div class="card">{classification_chart}</div>

  <div class="card">
    <h2>Scenario Predictions</h2>
    {scenario_predictions.to_html(index=False, border=0)}
  </div>
"""

    if top_scored is not None:
        html_body += f"""
  <div class="card">
    <h2>Top Scored Boards</h2>
    {top_scored.to_html(index=False, border=0)}
  </div>
"""

    html_body += """
</body>
</html>
"""

    report_path.write_text(html_body, encoding="utf-8")
    return report_path


def save_artifacts(
    df: pd.DataFrame,
    regression_model: Pipeline,
    classification_model: Pipeline,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "simulated_pcb_thermal_drift_data.csv"
    df.to_csv(dataset_path, index=False)
    print(f"\nSaved dataset: {dataset_path}")

    try:
        import joblib

        regression_path = output_dir / "pcb_drift_regression_model.joblib"
        classification_path = output_dir / "pcb_drift_classifier_model.joblib"
        joblib.dump(regression_model, regression_path)
        joblib.dump(classification_model, classification_path)
        print("Saved models:")
        print(f"- {regression_path}")
        print(f"- {classification_path}")
    except ImportError:
        print("joblib not installed, skipping model export.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-mode PCB thermal drift simulation and ML pipeline.")
    parser.add_argument("--samples", type=int, default=2600, help="Number of synthetic boards to simulate.")
    parser.add_argument("--threshold", type=float, default=3.0, help="Drift threshold percentage for classification.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=f"Optional board CSV to score. If omitted, the script auto-detects {SCORE_FILE_NAME} in the working folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "pcb_drift_outputs",
        help="Directory for exported CSV and model artifacts.",
    )
    return parser.parse_args()


def print_dataset_snapshot(df: pd.DataFrame, threshold: float) -> None:
    positive_rate = df["exceeds_threshold"].mean()
    severe_rate = df["severe_threshold"].mean()

    print("Synthetic hard-mode PCB dataset created.")
    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    print(f"Drift threshold: {threshold:.2f}%")
    print(f"Threshold exceedance rate: {positive_rate:.3f}")
    print(f"Severe drift rate       : {severe_rate:.3f}")
    print("\nPreview:")
    print(df.head(5).to_string(index=False))


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        n_samples=args.samples,
        drift_threshold_percent=args.threshold,
        random_seed=args.seed,
    )

    df = simulate_pcb_dataset(config)
    print_dataset_snapshot(df, config.drift_threshold_percent)

    regression_model, X_reg_test, y_reg_test, regression_metrics = train_regression_model(df, config)
    classification_model, X_cls_test, y_cls_test, classification_metrics = train_classification_model(df, config)

    regression_importance = print_permutation_importance(
        regression_model,
        X_reg_test,
        y_reg_test,
        scoring="neg_root_mean_squared_error",
        label="regression",
        top_n=12,
    )
    classification_importance = print_permutation_importance(
        classification_model,
        X_cls_test,
        y_cls_test,
        scoring="roc_auc",
        label="classification",
        top_n=12,
    )

    scenario_predictions = predict_demo_scenarios(regression_model, classification_model)
    save_artifacts(df, regression_model, classification_model, args.output_dir)

    scenario_path = args.output_dir / SCENARIO_FILE_NAME
    scenario_predictions.to_csv(scenario_path, index=False)
    print(f"Saved scenario predictions: {scenario_path}")

    template_path = write_input_template(Path.cwd())
    print(f"Saved input template: {template_path}")

    score_input_path = find_score_input_path(args.input_csv, Path.cwd())
    scored_predictions = None
    predictions_path = None
    if score_input_path is not None:
        scored_predictions, predictions_path = score_input_file(
            score_input_path,
            regression_model,
            classification_model,
            config.drift_threshold_percent,
            args.output_dir,
        )
    else:
        print(f"\nNo {SCORE_FILE_NAME} detected. Fill in {template_path.name}, rename it to {SCORE_FILE_NAME}, and run again.")

    report_path = save_html_report(
        args.output_dir,
        df,
        regression_metrics,
        classification_metrics,
        regression_importance,
        classification_importance,
        scenario_predictions,
        scored_predictions,
        score_input_path,
        predictions_path,
        template_path,
    )
    print(f"Saved HTML report: {report_path}")


if __name__ == "__main__":
    main()
