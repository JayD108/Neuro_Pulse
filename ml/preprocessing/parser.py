"""
CHB-MIT Summary File Parser
============================
Parses the chbXX-summary.txt files from the CHB-MIT dataset to extract
exact seizure windows (start_sec, end_sec) for each EDF recording file.

Usage:
    from ml.preprocessing.parser import parse_patient_summary, assign_window_label
"""

import os
import re
from typing import Dict, List, Tuple


def parse_patient_summary(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parses a CHB-MIT summary .txt file to extract seizure time windows per EDF file.

    The CHB-MIT format varies slightly across patients. This parser handles:
    - Single seizure per file: "Seizure Start Time: 2996 seconds"
    - Multiple seizures per file: "Seizure 1 Start Time: 2996 seconds"
    - Files with zero seizures: "Number of Seizures in File: 0"

    Args:
        summary_path: Absolute path to the patient summary .txt file.
                      e.g., "./ml/data/raw/chb01/chb01-summary.txt"

    Returns:
        A dict mapping EDF filenames to a list of (start_sec, end_sec) tuples.
        e.g., {'chb01_03.edf': [(2996, 3036)], 'chb01_01.edf': []}
    """
    seizure_map: Dict[str, list] = {}
    current_file: str | None = None

    if not os.path.exists(summary_path):
        print(f"[WARN] Summary file not found: {summary_path}")
        return seizure_map

    with open(summary_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # --- Detect new file entry ---
        if line.startswith("File Name:"):
            current_file = line.split(":", 1)[-1].strip()
            seizure_map[current_file] = []

        if current_file is None:
            continue

        # --- Detect seizure start time ---
        # Matches both "Seizure Start Time:" and "Seizure N Start Time:"
        if re.search(r"Seizure(\s+\d+)?\s+Start\s+Time", line, re.IGNORECASE):
            # Avoid matching "File Start Time:"
            if "File" not in line:
                match = re.search(r"(\d+)\s+second", line)
                if match:
                    start_sec = int(match.group(1))
                    seizure_map[current_file].append({"start": start_sec})

        # --- Detect seizure end time ---
        elif re.search(r"Seizure(\s+\d+)?\s+End\s+Time", line, re.IGNORECASE):
            if "File" not in line:
                match = re.search(r"(\d+)\s+second", line)
                if match and seizure_map[current_file]:
                    end_sec = int(match.group(1))
                    seizure_map[current_file][-1]["end"] = end_sec

    # --- Convert to clean list of (start, end) tuples ---
    formatted_map: Dict[str, List[Tuple[int, int]]] = {}
    for filename, seizures in seizure_map.items():
        formatted_map[filename] = [
            (s["start"], s["end"])
            for s in seizures
            if "start" in s and "end" in s
        ]

    return formatted_map


def assign_window_label(
    window_start_sec: float,
    window_end_sec: float,
    seizure_times: List[Tuple[int, int]],
    preictal_window_min: int = 30,
    postictal_buffer_min: int = 5,
) -> int:
    """
    Assigns a label to a given time window based on its proximity to seizures.

    Label definitions:
        0: Interictal — Baseline brain state, well away from any seizure.
        1: Preictal   — Within `preictal_window_min` minutes BEFORE seizure onset.
                        This is the PRIMARY PREDICTION TARGET.
        2: Ictal      — Window overlaps with an active seizure. Typically excluded
                        from the Interictal vs. Preictal binary classification task.

    Args:
        window_start_sec:   Start time (in seconds) of the epoch window.
        window_end_sec:     End time (in seconds) of the epoch window.
        seizure_times:      List of (start_sec, end_sec) tuples for all seizures
                            in the recording file.
        preictal_window_min: How many minutes before seizure onset to label as Preictal.
                             Default: 30 minutes (1800 seconds).
        postictal_buffer_min: A buffer after seizure end to exclude from interictal.
                              Default: 5 minutes.

    Returns:
        Integer label: 0 (Interictal), 1 (Preictal), or 2 (Ictal).
    """
    preictal_sec = preictal_window_min * 60
    postictal_sec = postictal_buffer_min * 60

    for sz_start, sz_end in seizure_times:
        # --- Check Ictal: window overlaps with the seizure ---
        if not (window_end_sec <= sz_start or window_start_sec >= sz_end):
            return 2  # Ictal

        # --- Check Postictal buffer (exclude from interictal) ---
        postictal_end = sz_end + postictal_sec
        if sz_end <= window_start_sec < postictal_end:
            return 2  # Treat postictal as excluded (map to Ictal label for skipping)

        # --- Check Preictal: within the prediction horizon before seizure ---
        preictal_start = max(0, sz_start - preictal_sec)
        if preictal_start <= window_start_sec < sz_start:
            return 1  # Preictal — OUR PRIMARY PREDICTION TARGET

    return 0  # Interictal / Normal


if __name__ == "__main__":
    # Quick test on chb01
    test_path = "./ml/data/raw/chb01/chb01-summary.txt"
    result = parse_patient_summary(test_path)
    print(f"\nParsed {len(result)} files from summary:")
    for fname, seizures in result.items():
        if seizures:
            print(f"  {fname}: {seizures}")
