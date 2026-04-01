# ============================================================
# Quality per Joule — Energy Tracking Utility
# utils/energy.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================
# Provides GPU power monitoring via CodeCarbon.
# Falls back gracefully when NVML is unavailable.
# ============================================================

import os
import glob
import csv
from codecarbon import EmissionsTracker


# NVML-based GPU power monitoring setup
NVML_AVAILABLE = False
gpu_handle = None

try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
    print("NVML initialized — GPU power monitoring enabled.")
except Exception:
    print("NVML not available — using CodeCarbon for energy measurement.")

# Returns instantaneous GPU power draw in Watts, or 0 if unavailable.
def get_gpu_power_watts():
    if NVML_AVAILABLE:
        return pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
    return 0.0

# Creates and returns a CodeCarbon EmissionsTracker with silent logging.
def create_tracker(project_name: str, output_dir: str) -> EmissionsTracker:
    import logging
    os.makedirs(output_dir, exist_ok=True)

    # Suppress all CodeCarbon logging
    for name in ["codecarbon", "codecarbon.emissions_tracker",
                 "codecarbon.external.geography", "codecarbon.core"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    return EmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        log_level="critical",
        save_to_file=True,
    )

# Reads total energy consumed (kWh) from CodeCarbon's CSV output for the given project.
def get_codecarbon_energy_kwh(output_dir: str, project_name: str) -> float:
    csv_path = os.path.join(output_dir, "emissions.csv")
    if not os.path.exists(csv_path):
        return 0.0

    energy_kwh = 0.0
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("project_name", "") == project_name:
                    energy_kwh = float(row.get("energy_consumed", 0.0))
    except Exception as e:
        print(f"  Warning: Could not read CodeCarbon CSV: {e}")

    return energy_kwh

# Converts kWh to Joules: 1 kWh = 3,600,000 J
def get_total_energy_joules(output_dir: str, project_name: str) -> float:
    kwh = get_codecarbon_energy_kwh(output_dir, project_name)
    return kwh * 3_600_000.0
