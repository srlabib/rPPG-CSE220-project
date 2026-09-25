"""
Offline evaluation script for UBFC-style rPPG datasets (e.g., subject10).
Compares estimated Heart Rate (BPM) against Ground Truth from a multi-line ground_truth.txt file:
    Line 1: PPG signal (Blood Volume Pulse)
    Line 2: Heart rate (HR in BPM)
    Line 3: Timestep (seconds, scientific notation)
"""

import argparse
import os
import sys
from typing import Tuple, Optional
import numpy as np
import cv2 as cv

try:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend to avoid GUI threading issues
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib is required for plotting. Please install it using:")
    print("        pip install matplotlib")
    sys.exit(1)

from config import DEFAULT_FPS
from face_processor import FaceROIProcessor
from rppg_pipeline import RPPGPipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rPPG algorithm on subject datasets with ground_truth.txt."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/subject10",
        help="Path to the dataset directory containing vid.avi and ground_truth.txt (default: data/subject10)."
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="If set, displays video frames and ROI overlays during evaluation."
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=15.0,
        help="Seconds to skip at the beginning before calculating metrics (stabilization period, default: 15.0)."
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="pos",
        choices=["pos", "chrom", "green"],
        help="rPPG extraction algorithm: pos, chrom, or green (default: pos)."
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="If set, exports timestamp, estimated HR, and ground truth HR to evaluation_metrics.csv."
    )
    return parser.parse_args()


def load_ground_truth(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses UBFC-style ground_truth.txt.
    
    Structure:
        Line 1: PPG signal (blood volume pulse values)
        Line 2: Heart rate (HR in BPM)
        Line 3: Timestep (seconds, in scientific notation)

    Returns:
        gt_times: 1D array of timestamps in seconds.
        gt_hr: 1D array of Ground Truth Heart Rate (BPM).
        gt_ppg: 1D array of Ground Truth PPG signal.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 3:
        raise ValueError(
            f"Invalid ground_truth.txt format in '{file_path}'. "
            f"Expected 3 lines (PPG, HR, Time), but found {len(lines)} line(s)."
        )

    # Parse whitespace-separated scientific notation floats
    gt_ppg = np.array([float(x) for x in lines[0].split()], dtype=np.float32)
    gt_hr = np.array([float(x) for x in lines[1].split()], dtype=np.float32)
    gt_times = np.array([float(x) for x in lines[2].split()], dtype=np.float32)

    return gt_times, gt_hr, gt_ppg


def calculate_metrics(
    est_times: np.ndarray,
    est_hr: np.ndarray,
    gt_times: np.ndarray,
    gt_hr: np.ndarray,
    skip_seconds: float = 15.0
) -> Optional[dict]:
    """
    Computes standard biomedical rPPG evaluation metrics:
    MAE, RMSE, Pearson Correlation (r), Mean Bias, and Error Standard Deviation.
    """
    valid_mask = est_times >= skip_seconds
    if not np.any(valid_mask):
        return None

    est_times_stable = est_times[valid_mask]
    est_hr_stable = est_hr[valid_mask]

    # Interpolate ground truth HR to align precisely with estimated timestamps
    interpolated_gt_hr = np.interp(est_times_stable, gt_times, gt_hr)

    errors = est_hr_stable - interpolated_gt_hr
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(squared_errors)))
    bias = float(np.mean(errors))
    std = float(np.std(errors))
    max_err = float(np.max(abs_errors))

    # Pearson correlation coefficient
    if len(est_hr_stable) > 1 and np.std(est_hr_stable) > 1e-6 and np.std(interpolated_gt_hr) > 1e-6:
        pearson_r = float(np.corrcoef(est_hr_stable, interpolated_gt_hr)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "bias": bias,
        "std": std,
        "max_err": max_err,
        "est_times_stable": est_times_stable,
        "est_hr_stable": est_hr_stable,
        "gt_hr_stable": interpolated_gt_hr,
        "errors": errors,
    }


def plot_results(
    gt_times: np.ndarray,
    gt_hr: np.ndarray,
    est_times: np.ndarray,
    est_hr: np.ndarray,
    metrics: Optional[dict],
    skip_seconds: float,
    output_path: str
):
    """Generates and saves a two-panel publication-quality evaluation figure."""
    fig, (ax_hr, ax_err) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- Top Panel: Heart Rate Trajectory ---
    if skip_seconds > 0:
        ax_hr.axvspan(0, skip_seconds, color='gray', alpha=0.18, label='Stabilization Period (Skipped)')

    ax_hr.plot(gt_times, gt_hr, label="Ground Truth HR", color="#1f77b4", linewidth=2.0, alpha=0.85)
    ax_hr.plot(est_times, est_hr, label="Estimated HR (rPPG)", color="#d62728", linewidth=2.2)

    ax_hr.set_ylabel("Heart Rate (BPM)", fontsize=11, fontweight="bold")
    title_str = "rPPG Performance Evaluation vs Ground Truth"
    if metrics:
        title_str += f" | MAE: {metrics['mae']:.2f} BPM | RMSE: {metrics['rmse']:.2f} BPM | r: {metrics['pearson_r']:.3f}"
    ax_hr.set_title(title_str, fontsize=13, fontweight="bold", pad=10)
    ax_hr.legend(loc="upper right", framealpha=0.9)
    ax_hr.grid(True, linestyle="--", alpha=0.5)

    # --- Bottom Panel: Error / Residuals ---
    if metrics:
        ax_err.plot(
            metrics["est_times_stable"],
            metrics["errors"],
            color="#2ca02c",
            linewidth=1.5,
            label="Error (Est - GT)"
        )
        ax_err.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_err.axhspan(-5, 5, color="green", alpha=0.10, label="±5 BPM Clinical Band")
        if skip_seconds > 0:
            ax_err.axvspan(0, skip_seconds, color='gray', alpha=0.18)
        ax_err.set_ylabel("Error (BPM)", fontsize=11, fontweight="bold")
        ax_err.legend(loc="upper right", framealpha=0.9)
    else:
        ax_err.text(0.5, 0.5, "Insufficient data for error plot", ha='center', va='center')

    ax_err.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax_err.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Evaluation plot saved successfully to: {output_path}")


def run_evaluation(
    dataset_dir: str,
    headless: bool = True,
    skip_seconds: float = 15.0,
    save_csv: bool = False
):
    video_path = os.path.join(dataset_dir, "vid.avi")
    gt_path = os.path.join(dataset_dir, "ground_truth.txt")

    print(f"\n{'=' * 60}")
    print(f"      rPPG EVALUATION ON UBFC DATASET: {dataset_dir}")
    print(f"{'=' * 60}")
    print(f"[INFO] Loading Ground Truth: {gt_path}")
    gt_times, gt_hr, gt_ppg = load_ground_truth(gt_path)
    print(f"[INFO] Loaded {len(gt_times)} GT samples (Duration: {gt_times[-1]:.2f}s, HR range: {np.min(gt_hr):.1f}-{np.max(gt_hr):.1f} BPM)")

    print(f"[INFO] Opening video: {video_path}")
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    detected_fps = cap.get(cv.CAP_PROP_FPS)
    fps = detected_fps if detected_fps > 0 else DEFAULT_FPS
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video resolution: {int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}")
    print(f"[INFO] Using rPPG Algorithm: {args.method.upper()}")
    pipeline = RPPGPipeline(fps=fps, method=args.method)

    est_times = []
    est_hr = []
    frame_idx = 0

    try:
        with FaceROIProcessor() as face_processor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Consistent frame downscaling if necessary
                frame = FaceROIProcessor._maybe_resize(frame)

                # Determine accurate frame timestamp
                pos_msec = cap.get(cv.CAP_PROP_POS_MSEC)
                current_time_sec = (pos_msec / 1000.0) if pos_msec > 0 else (frame_idx / fps)

                detection = face_processor.process_frame(frame)
                if detection.detected and detection.mean_rgbs is not None:
                    _, bpm, is_warming_up = pipeline.update(detection.mean_rgbs)

                    if not is_warming_up and bpm is not None and bpm > 0:
                        est_times.append(current_time_sec)
                        est_hr.append(bpm)

                if not headless:
                    display_frame = frame.copy()
                    if detection.detected and detection.polys is not None:
                        display_frame = face_processor.render_roi_overlay(
                            frame,
                            active_indices=pipeline.active_patch_indices,
                            polys=detection.polys
                        )
                    # HUD overlay
                    disp_bpm = pipeline.current_bpm if len(est_hr) > 0 else None
                    cv.putText(
                        display_frame,
                        f"Frame: {frame_idx}/{total_frames} | Est: {disp_bpm:.1f} BPM" if disp_bpm else "Warming up...",
                        (20, 40),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0) if disp_bpm else (0, 255, 255),
                        2
                    )
                    cv.imshow("Subject Evaluation", display_frame)
                    if cv.waitKey(1) & 0xFF in (ord('q'), 27):
                        print("\n[INFO] Evaluation interrupted by user.")
                        break

                frame_idx += 1
                if frame_idx % 200 == 0 or frame_idx == total_frames:
                    pct = (frame_idx / total_frames) * 100
                    print(f"  -> Processed {frame_idx}/{total_frames} frames ({pct:.1f}%)...")

    finally:
        cap.release()
        cv.destroyAllWindows()

    print("[INFO] Video frame processing complete.")

    if len(est_hr) == 0:
        print("[WARNING] No valid heart rate estimations produced. Was the video too short?")
        return

    est_times_arr = np.array(est_times, dtype=np.float32)
    est_hr_arr = np.array(est_hr, dtype=np.float32)

    # Compute metrics
    metrics = calculate_metrics(est_times_arr, est_hr_arr, gt_times, gt_hr, skip_seconds=skip_seconds)

    print("\n" + "=" * 50)
    print("           EVALUATION RESULTS (UBFC)")
    print("=" * 50)
    print(f"Stabilization period skipped:  {skip_seconds:.1f} seconds")
    print(f"Total valid estimated frames:  {len(est_hr_arr)}")

    if metrics:
        print(f"Mean Absolute Error (MAE):     {metrics['mae']:.2f} BPM")
        print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.2f} BPM")
        print(f"Pearson Correlation (r):       {metrics['pearson_r']:.3f}")
        print(f"Mean Error (Bias):             {metrics['bias']:+.2f} BPM")
        print(f"Error Std Deviation (SD):      {metrics['std']:.2f} BPM")
        print(f"Max Absolute Error:            {metrics['max_err']:.2f} BPM")
    else:
        print("[WARNING] Video is shorter than skip-seconds. Metrics could not be calculated.")
    print("=" * 50 + "\n")

    # Generate and save plot
    plot_output_path = os.path.join(dataset_dir, "evaluation_result.png")
    plot_results(gt_times, gt_hr, est_times_arr, est_hr_arr, metrics, skip_seconds, plot_output_path)

    # Optional CSV export
    if save_csv and metrics:
        csv_path = os.path.join(dataset_dir, "evaluation_metrics.csv")
        data_table = np.column_stack((
            metrics["est_times_stable"],
            metrics["est_hr_stable"],
            metrics["gt_hr_stable"],
            metrics["errors"]
        ))
        np.savetxt(
            csv_path,
            data_table,
            delimiter=",",
            header="Timestamp_sec,Estimated_HR_BPM,GroundTruth_HR_BPM,Error_BPM",
            comments="",
            fmt="%.4f"
        )
        print(f"[INFO] Exported metrics timeseries to: {csv_path}")


def main():
    args = parse_arguments()
    try:
        run_evaluation(
            dataset_dir=args.dataset,
            headless=not args.no_headless,
            skip_seconds=args.skip_seconds,
            save_csv=args.save_csv
        )
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
