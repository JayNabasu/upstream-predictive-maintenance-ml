"""
Feature Engineering Pipeline for Time-Series Downhole Telemetry.
Generates statistical rolling windows, degradation rates, and operational ratios.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

FEATURE_COLUMNS = [
    "intake_pressure_psi",
    "discharge_pressure_psi",
    "pressure_differential_psi",
    "motor_temperature_c",
    "vibration_rms_mms",
    "current_draw_amps",
    "drive_frequency_hz",
    "water_cut_pct",
    "vib_rolling_mean_6h",
    "vib_rolling_std_6h",
    "vib_delta_6h",
    "temp_rolling_mean_6h",
    "temp_delta_6h",
    "press_ratio"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes time-series rolling metrics and degradation indicators grouped by well."""
    df_out = df.sort_values(by=["well_id", "timestamp"]).copy()

    # Rolling window statistics grouped per well
    df_out["vib_rolling_mean_6h"] = df_out.groupby("well_id")["vibration_rms_mms"].transform(
        lambda s: s.rolling(window=6, min_periods=1).mean()
    )
    df_out["vib_rolling_std_6h"] = df_out.groupby("well_id")["vibration_rms_mms"].transform(
        lambda s: s.rolling(window=6, min_periods=1).std().fillna(0.0)
    )
    df_out["vib_delta_6h"] = df_out.groupby("well_id")["vibration_rms_mms"].diff(periods=6).fillna(0.0)

    df_out["temp_rolling_mean_6h"] = df_out.groupby("well_id")["motor_temperature_c"].transform(
        lambda s: s.rolling(window=6, min_periods=1).mean()
    )
    df_out["temp_delta_6h"] = df_out.groupby("well_id")["motor_temperature_c"].diff(periods=6).fillna(0.0)

    # Operational physics ratios
    df_out["press_ratio"] = df_out["pressure_differential_psi"] / (df_out["intake_pressure_psi"] + 1e-5)

    return df_out

def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits into train and test sets preserving chronological well batches."""
    df_feats = engineer_features(df)
    
    wells = df_feats["well_id"].unique()
    np.random.seed(42)
    test_wells = np.random.choice(wells, size=max(1, int(len(wells) * test_size)), replace=False)

    train_mask = ~df_feats["well_id"].isin(test_wells)
    test_mask = df_feats["well_id"].isin(test_wells)

    X_train = df_feats.loc[train_mask, FEATURE_COLUMNS]
    y_train = df_feats.loc[train_mask, "failure_next_72h"]
    X_test = df_feats.loc[test_mask, FEATURE_COLUMNS]
    y_test = df_feats.loc[test_mask, "failure_next_72h"]

    return X_train, X_test, y_train, y_test
