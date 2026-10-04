"""
NeuroPulse Parallel Dataset Ingestion Runner
=============================================
Entry point for running the full feature extraction pipeline across
multiple CHB-MIT patients either sequentially or in parallel.

Usage:
    # Process a single patient
    python -m ml.ingest_dataset --patients chb01

    # Process multiple patients
    python -m ml.ingest_dataset --patients chb01 chb02 chb03

    # Process all available patients
    python -m ml.ingest_dataset --all

    # Customize workers per patient
    python -m ml.ingest_dataset --all --workers 12
"""

import os
import time
import argparse
from typing import List, Optional

from ml.preprocessing.feature_extractor import ingest_patient, RAW_DIR

# All 24 CHB-MIT patient IDs
ALL_PATIENTS = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]


def get_available_patients(raw_dir: str = RAW_DIR) -> List[str]:
    """
    Scans the raw data directory and returns patient IDs
    that actually have downloaded EDF data.
    """
    available = []
    for patient_id in ALL_PATIENTS:
        patient_dir = os.path.join(raw_dir, patient_id)
        if os.path.isdir(patient_dir):
            edf_files = [f for f in os.listdir(patient_dir) if f.endswith(".edf")]
            if edf_files:
                available.append(patient_id)
    return available


def run_ingestion_pipeline(
    patients: List[str],
    workers_per_patient: int = 8,
) -> None:
    """
    Sequentially runs the full ingestion pipeline for each patient.
    Each patient's EDF files are processed in parallel within the patient.

    Args:
        patients: List of patient IDs to process.
        workers_per_patient: CPU workers for parallel EDF processing within each patient.
    """
    print(f"\n{'#'*65}")
    print(f"  NeuroPulse Dataset Ingestion Pipeline")
    print(f"  Patients to process : {len(patients)}")
    print(f"  Workers per patient : {workers_per_patient}")
    print(f"{'#'*65}\n")

    start_time = time.time()
    results = {}

    for i, patient_id in enumerate(patients, 1):
        print(f"[{i}/{len(patients)}] Starting {patient_id}...")
        try:
            saved_path = ingest_patient(patient_id, num_workers=workers_per_patient)
            results[patient_id] = saved_path if saved_path else "FAILED"
        except Exception as exc:
            print(f"  [CRITICAL] {patient_id} failed with exception: {exc}")
            results[patient_id] = "ERROR"

    # --- Summary Report ---
    elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"  INGESTION COMPLETE | Total time: {elapsed:.1f}s")
    print(f"{'='*65}")
    for pid, path in results.items():
        status = "[OK]" if path and path not in ("FAILED", "ERROR") else "[FAIL]"
        print(f"  {status} {pid}: {path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroPulse Parallel Dataset Ingestion Runner"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--patients", nargs="+",
        help="Specific patient IDs to process (e.g., --patients chb01 chb02)"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Process all available patients found in the raw data directory"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel CPU workers per patient. Default: 8"
    )
    args = parser.parse_args()

    if args.all:
        target_patients = get_available_patients()
        if not target_patients:
            print(f"[ERROR] No patient data found in '{RAW_DIR}'. Run download_data.py first.")
            exit(1)
        print(f"Auto-detected {len(target_patients)} available patients: {target_patients}")
    else:
        target_patients = args.patients

    run_ingestion_pipeline(target_patients, workers_per_patient=args.workers)
