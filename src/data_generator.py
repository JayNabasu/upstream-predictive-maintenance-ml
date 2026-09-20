"""
Upstream Well Telemetry & Degradation Generator for Predictive Maintenance.
Simulates high-frequency ESP (Electrical Submersible Pump) operational data across OML assets.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

ASSET_IDS = ["OML-20", "OML-28", "OML-38", "OML-49", "OML-116"]

def generate_telemetry_stream(num_wells: int = 15, days: int = 45, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    records = []
    start_time = datetime(2024, 8, 1, 0, 0, 0)
    total_hours = days * 24

    for well_idx in range(1, num_wells + 1):
        asset = np.random.choice(ASSET_IDS)
        well_name = f"{asset}-ESP-{well_idx:02d}"

        # Well baseline characteristics
        base_intake_press = np.random.uniform(750.0, 1100.0)
        base_discharge_press = base_intake_press + np.random.uniform(1800.0, 2400.0)
        base_temp = np.random.uniform(92.0, 108.0)
        base_vib = np.random.uniform(1.8, 2.6)
        base_amps = np.random.uniform(55.0, 75.0)
        base_freq = np.random.uniform(52.0, 58.0)
        water_cut = np.random.uniform(15.0, 42.0)

        # Decide whether this well experiences a degradation event
        will_fail = np.random.choice([True, False], p=[0.35, 0.65])
        fail_hour = np.random.randint(int(total_hours * 0.5), total_hours - 24) if will_fail else -1

        current_vib = base_vib
        current_temp = base_temp
        current_intake = base_intake_press
        current_amps = base_amps

        for h in range(total_hours):
            timestamp = start_time + timedelta(hours=h)
            
            # Normal operational noise
            noise_vib = np.random.normal(0, 0.1)
            noise_temp = np.random.normal(0, 0.4)
            noise_press = np.random.normal(0, 5.0)
            noise_amps = np.random.normal(0, 0.8)

            # Check if entering degradation phase (72 hours prior to failure)
            is_in_failure_window = 0
            if will_fail and h >= (fail_hour - 72) and h <= fail_hour:
                is_in_failure_window = 1
                progression = (h - (fail_hour - 72)) / 72.0
                
                # Accelerate vibration and thermal signature
                current_vib += 0.08 * (1.0 + progression * 2.0)
                current_temp += 0.35 * (1.0 + progression * 1.5)
                current_intake -= 3.5 * progression
                current_amps += 0.4 * progression

            # Bounded values
            vib_val = max(0.5, round(current_vib + noise_vib, 2))
            temp_val = max(50.0, round(current_temp + noise_temp, 2))
            intake_val = max(200.0, round(current_intake + noise_press, 1))
            discharge_val = round(intake_val + (base_discharge_press - base_intake_press), 1)
            amps_val = max(20.0, round(current_amps + noise_amps, 1))

            records.append({
                "timestamp": timestamp,
                "asset_code": asset,
                "well_id": well_name,
                "intake_pressure_psi": intake_val,
                "discharge_pressure_psi": discharge_val,
                "pressure_differential_psi": round(discharge_val - intake_val, 1),
                "motor_temperature_c": temp_val,
                "vibration_rms_mms": vib_val,
                "current_draw_amps": amps_val,
                "drive_frequency_hz": round(base_freq + np.random.normal(0, 0.1), 1),
                "water_cut_pct": round(water_cut, 1),
                "failure_next_72h": is_in_failure_window
            })

            # If well reaches failure point, reset / simulate workover pump replacement
            if will_fail and h == fail_hour:
                current_vib = base_vib
                current_temp = base_temp
                current_intake = base_intake_press
                current_amps = base_amps
                will_fail = False

    return pd.DataFrame(records)
