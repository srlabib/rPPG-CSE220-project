"""
Offline evaluation script for the rPPG algorithm.
Compares estimated Heart Rate (BPM) from video against Ground Truth data from an .xmp file.
"""

import argparse
import os
import sys
import numpy as np
import cv2 as cv

try:
    import matplotlib
    matplotlib.use('Agg')  # Use headless backend to avoid GUI crashes
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib is required for plotting. Please install it using:")
    print("        pip install matplotlib")
    sys.exit(1)

from config import DEFAULT_FPS
from face_processor import FaceROIProcessor
from rppg_pipeline import RPPGPipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rPPG algorithm on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/10-gt",
        help="Path to the dataset directory containing vid.avi and gtdump.xmp."
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="If set, shows the video processing windows during evaluation."
    )
    return parser.parse_args()


def load_ground_truth(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses the ground truth .xmp file.
    Format: Timestamp(ms), HR, SpO2, PPG
    Returns:
        gt_times: 1D array of timestamps in seconds.
        gt_hr: 1D array of Ground Truth Heart Rate (BPM).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    # Load comma-separated data. We only care about cols 0 (Time) and 1 (HR)
    data = np.genfromtxt(file_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
        
    gt_times_ms = data[:, 0]
    gt_hr = data[:, 1]

    # Convert timestamp to seconds
    gt_times_sec = gt_times_ms / 1000.0
    return gt_times_sec, gt_hr


def run_evaluation(dataset_dir: str, headless: bool = True):
    video_path = os.path.join(dataset_dir, "vid.avi")
    gt_path = os.path.join(dataset_dir, "gtdump.xmp")

    print(f"[INFO] Starting evaluation on dataset: {dataset_dir}")
    print(f"[INFO] Loading Ground Truth from: {gt_path}")
    gt_times, gt_hr = load_ground_truth(gt_path)

    print(f"[INFO] Opening video: {video_path}")
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    detected_fps = cap.get(cv.CAP_PROP_FPS)
    fps = detected_fps if detected_fps > 0 else DEFAULT_FPS

    pipeline = RPPGPipeline(fps=fps)
    
    est_times = []
    est_hr = []

    frame_idx = 0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Processing {total_frames} frames at {fps:.1f} FPS...")

    try:
        with FaceROIProcessor() as face_processor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get current timestamp from OpenCV in seconds
                current_time_sec = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0

                detection = face_processor.process_frame(frame)
                if detection.detected and detection.mean_rgbs is not None:
                    _, bpm, is_warming_up = pipeline.update(detection.mean_rgbs)
                    
                    if not is_warming_up and bpm is not None and bpm > 0:
                        est_times.append(current_time_sec)
                        est_hr.append(bpm)

                if not headless:
                    display_frame = frame.copy()
                    if detection.detected and detection.masks is not None:
                        display_frame = face_processor.render_roi_overlay(
                            frame, 
                            detection.masks,
                            active_indices=pipeline.active_patch_indices
                        )
                    cv.imshow("Evaluation - Face Processing", display_frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Evaluation interrupted by user.")
                        break

                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"  -> Processed {frame_idx}/{total_frames} frames...")

    finally:
        cap.release()
        cv.destroyAllWindows()

    print("[INFO] Video processing complete.")
    
    if len(est_hr) == 0:
        print("[WARNING] No valid heart rate estimations were produced. Was the video too short?")
        return

    est_times = np.array(est_times)
    est_hr = np.array(est_hr)

    print("[INFO] Calculating performance metrics...")
    
    # We must compare estimated HR against GT HR at the same timestamps.
    # Since they are sampled differently, we interpolate the Ground Truth HR
    # to match the timestamps where we have valid estimations.
    interpolated_gt_hr = np.interp(est_times, gt_times, gt_hr)

    # Calculate Metrics
    absolute_errors = np.abs(est_hr - interpolated_gt_hr)
    squared_errors = (est_hr - interpolated_gt_hr) ** 2

    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))

    print("-" * 40)
    print("        EVALUATION RESULTS        ")
    print("-" * 40)
    print(f"Mean Absolute Error (MAE):  {mae:.2f} BPM")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} BPM")
    print("-" * 40)

    # Plotting
    plot_path = os.path.join(dataset_dir, "evaluation_result.png")
    plt.figure(figsize=(10, 5))
    plt.plot(gt_times, gt_hr, label="Ground Truth HR", color="blue", linewidth=1.5, alpha=0.7)
    plt.plot(est_times, est_hr, label="Estimated HR (rPPG)", color="red", linewidth=2.0)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title(f"rPPG Performance Evaluation (MAE: {mae:.2f}, RMSE: {rmse:.2f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] Evaluation plot saved to: {plot_path}")
    
    # Try to show the plot if running interactively
    try:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    except Exception:
        pass


def main():
    args = parse_arguments()
    try:
        run_evaluation(dataset_dir=args.dataset, headless=not args.no_headless)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
