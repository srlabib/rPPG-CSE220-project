"""
Ground Truth parsing and evaluation utilities for rPPG benchmarking.
Supports:
1. Standard CSV format (timestamp, hr)
2. UBFC-style format (ground_truth.txt with PPG, HR, Timestep lines)
"""

import os
from typing import Dict, Any, Optional, Tuple, List, Sequence
import numpy as np

Sequence_or_array = Any


def parse_ground_truth(file_path: str) -> Dict[str, Any]:
    """
    Parses a ground truth file (.csv or .txt) and returns time-series arrays and summary stats.
    
    Returns:
        Dict with keys:
            - 'timestamps': List[float] (relative to 0.0s)
            - 'hr': List[float]
            - 'duration': float
            - 'mean_hr': float
            - 'min_hr': float
            - 'max_hr': float
            - 'sample_count': int
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    # Detect file type
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        raise ValueError("Ground truth file is empty.")

    # Check if UBFC format (first line contains space-separated floats without headers)
    # UBFC files typically have 3 long lines: Line 1 = PPG, Line 2 = HR, Line 3 = Timesteps
    if len(lines) >= 3 and len(lines) <= 5 and (" " in lines[0] or "\t" in lines[0]):
        try:
            # UBFC format
            gt_hr = np.fromstring(lines[1], sep=" ", dtype=np.float32)
            gt_times = np.fromstring(lines[2], sep=" ", dtype=np.float32)

            if len(gt_hr) > 0 and len(gt_times) > 0:
                # Ensure equal length
                min_len = min(len(gt_hr), len(gt_times))
                gt_hr = gt_hr[:min_len]
                gt_times = gt_times[:min_len]
                # Normalize time to start at 0.0s
                gt_times = gt_times - gt_times[0]
                return _build_response_dict(gt_times, gt_hr)
        except Exception:
            pass  # Fall through to CSV parser

    # Standard CSV / TSV parser
    times: List[float] = []
    hrs: List[float] = []

    delimiter = "," if "," in lines[0] else ("\t" if "\t" in lines[0] else None)
    start_idx = 0

    # Check for header
    first_row = [c.strip().lower() for c in (lines[0].split(delimiter) if delimiter else lines[0].split())]
    time_col = 0
    hr_col = 1

    if any(h in first_row for h in ["time", "timestamp", "sec", "seconds", "t"]):
        start_idx = 1
        for idx, col_name in enumerate(first_row):
            if col_name in ["time", "timestamp", "sec", "seconds", "t"]:
                time_col = idx
            elif col_name in ["hr", "bpm", "heart_rate", "rate", "pulse"]:
                hr_col = idx

    for line_idx in range(start_idx, len(lines)):
        row = lines[line_idx].split(delimiter) if delimiter else lines[line_idx].split()
        if len(row) >= 2:
            try:
                t_val = float(row[time_col])
                hr_val = float(row[hr_col])
                times.append(t_val)
                hrs.append(hr_val)
            except ValueError:
                continue

    if not times:
        raise ValueError("Could not parse timestamp and HR columns from ground truth file.")

    t_arr = np.array(times, dtype=np.float32)
    hr_arr = np.array(hrs, dtype=np.float32)

    # Normalize time to start at 0.0s
    if len(t_arr) > 0:
        t_arr = t_arr - t_arr[0]

    return _build_response_dict(t_arr, hr_arr)


def _build_response_dict(gt_times: np.ndarray, gt_hr: np.ndarray) -> Dict[str, Any]:
    """Helper to compute statistics and format downsampled response for charts."""
    # Remove NaN or non-positive values
    valid_mask = np.isfinite(gt_hr) & (gt_hr > 30.0) & (gt_hr < 220.0)
    gt_times = gt_times[valid_mask]
    gt_hr = gt_hr[valid_mask]

    if len(gt_times) == 0:
        raise ValueError("No valid physiological HR values found in ground truth (expected 30-220 BPM).")

    # Normalize time to start at 0.0s
    gt_times = gt_times - gt_times[0]

    # Automatically detect millisecond timestamps (e.g. gtdump.txt, gtdump.xmp, or ms pulse-oximeter logs)
    if len(gt_times) > 1:
        span = float(gt_times[-1] - gt_times[0])
        diffs = np.diff(gt_times)
        median_diff = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
        if span > 1000.0 or (span > 150.0 and median_diff >= 4.0):
            gt_times = gt_times / 1000.0

    # Downsample for frontend plotting if too dense (> 1000 points)
    if len(gt_times) > 1200:
        step = int(np.ceil(len(gt_times) / 1000))
        plot_times = gt_times[::step].round(2).tolist()
        plot_hr = gt_hr[::step].round(1).tolist()
    else:
        plot_times = gt_times.round(2).tolist()
        plot_hr = gt_hr.round(1).tolist()

    return {
        "timestamps": plot_times,
        "hr": plot_hr,
        "duration": float(gt_times[-1] - gt_times[0]) if len(gt_times) > 1 else 0.0,
        "mean_hr": float(np.mean(gt_hr)),
        "min_hr": float(np.min(gt_hr)),
        "max_hr": float(np.max(gt_hr)),
        "sample_count": len(gt_times),
        "raw_times": gt_times.tolist(),
        "raw_hr": gt_hr.tolist(),
    }


def interpolate_gt_hr(gt_times: Sequence_or_array, gt_hr: Sequence_or_array, current_time: float) -> Optional[float]:
    """Interpolates ground truth HR at the given timestamp."""
    times = np.asarray(gt_times)
    hrs = np.asarray(gt_hr)
    if len(times) == 0 or current_time < times[0] or current_time > times[-1]:
        return None
    return float(np.interp(current_time, times, hrs))


def compute_live_metrics(
    pred_history: List[Tuple[float, float]],
    gt_times: List[float],
    gt_hr: List[float],
    skip_seconds: float = 15.0
) -> Dict[str, Optional[float]]:
    """
    Computes real-time academic metrics: MAE, RMSE, Pearson r, current error.
    
    Args:
        pred_history: List of (timestamp, estimated_bpm)
        gt_times: Full ground truth timestamps
        gt_hr: Full ground truth HR values
        skip_seconds: Initial stabilization window to exclude from cumulative error
        
    Returns:
        Dict with mae, rmse, pearson_r, current_error, count
    """
    if not pred_history or not gt_times:
        return {
            "current_error": None,
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "count": 0
        }

    gt_t_arr = np.array(gt_times, dtype=np.float32)
    gt_h_arr = np.array(gt_hr, dtype=np.float32)

    pred_t_all = np.array([p[0] for p in pred_history], dtype=np.float32)
    pred_h_all = np.array([p[1] for p in pred_history], dtype=np.float32)

    latest_time = pred_t_all[-1]
    latest_pred = pred_h_all[-1]

    # Current instantaneous error
    latest_gt = float(np.interp(latest_time, gt_t_arr, gt_h_arr))
    current_error = abs(latest_pred - latest_gt)

    # Filter out initialization/warm-up period
    eval_mask = (pred_t_all >= skip_seconds) & (pred_t_all >= gt_t_arr[0]) & (pred_t_all <= gt_t_arr[-1])
    eval_pred_t = pred_t_all[eval_mask]
    eval_pred_h = pred_h_all[eval_mask]

    if len(eval_pred_t) < 3:
        return {
            "current_error": round(current_error, 2),
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "count": len(eval_pred_t)
        }

    interp_gt_h = np.interp(eval_pred_t, gt_t_arr, gt_h_arr)
    diff = eval_pred_h - interp_gt_h

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Pearson r
    if np.std(eval_pred_h) > 1e-4 and np.std(interp_gt_h) > 1e-4:
        r = float(np.corrcoef(eval_pred_h, interp_gt_h)[0, 1])
    else:
        r = 0.0

    return {
        "current_error": round(current_error, 2),
        "latest_gt": round(latest_gt, 1),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "pearson_r": round(r, 3),
        "count": len(eval_pred_t)
    }
