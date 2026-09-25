"""
rPPG Video Processing Application.

Extracts real-time photoplethysmography (pulse) signals and heart rate (BPM)
from facial video feeds using MediaPipe FaceMesh and the POS algorithm.
"""

import argparse
import os
import sys
import time
from typing import Optional
import cv2 as cv
import numpy as np

# Re-exports for backward compatibility and clean modular imports
from config import (
    FOREHEAD_INDICES,
    LEFT_CHEEK_INDICES,
    RIGHT_CHEEK_INDICES,
    FORHEAD_INDICES,  # Typo alias preserved for backward compatibility
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_CAM_WIDTH,
    DEFAULT_CAM_HEIGHT,
    DEFAULT_FPS,
)
from signal_processing import (
    execute_pos_algorithm,
    calculate_bpm,
    butter_bandpass_filter,
)
from face_processor import FaceROIProcessor, FaceROIDetection
from visualizer import create_pulse_graph, draw_hud
from rppg_pipeline import RPPGPipeline

# Fast forward playback control boolean
FAST_FORWARD: bool = False


def is_fast_forward() -> bool:
    """Returns True if fast-forward playback is enabled."""
    return FAST_FORWARD


def set_fast_forward(enabled: bool) -> bool:
    """Sets the fast-forward boolean state."""
    global FAST_FORWARD
    FAST_FORWARD = bool(enabled)
    return FAST_FORWARD


def toggle_fast_forward() -> bool:
    """Toggles the fast-forward boolean state and returns the new value."""
    global FAST_FORWARD
    FAST_FORWARD = not FAST_FORWARD
    return FAST_FORWARD


def open_video_source(
    source: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    target_fps: float = DEFAULT_FPS,
    show_camera_settings: bool = False,
) -> cv.VideoCapture:
    """
    Opens and configures a video source (webcam index or video file path).

    Args:
        source: Camera index as string/int (e.g., '0', '1') or path to video file.
        width: Desired capture width (for webcams). Defaults to 640 for webcams.
        height: Desired capture height (for webcams). Defaults to 480 for webcams.
        target_fps: Desired capture frame rate.
        show_camera_settings: If True, opens Windows hardware camera settings dialog.

    Returns:
        Configured cv.VideoCapture instance.
    """
    is_camera = source.isdigit()
    device_id = int(source) if is_camera else source

    if is_camera and sys.platform == "win32":
        cap = cv.VideoCapture(device_id, cv.CAP_DSHOW)
        if not cap.isOpened() or not cap.read()[0]:
            cap.release()
            cap = cv.VideoCapture(device_id)
    else:
        cap = cv.VideoCapture(device_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: '{source}'. "
            f"Please verify device connection or video file path."
        )

    if is_camera:
        actual_w = width if width is not None else DEFAULT_CAM_WIDTH
        actual_h = height if height is not None else DEFAULT_CAM_HEIGHT

        # Request MJPG stream for maximum USB camera throughput and high FPS
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, actual_w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, actual_h)
        cap.set(cv.CAP_PROP_FPS, target_fps)

        if show_camera_settings:
            # Force the Windows hardware settings menu to pop up for manual exposure control
            cap.set(cv.CAP_PROP_SETTINGS, 1)

    return cap


def parse_arguments() -> argparse.Namespace:
    """Parses command-line options for running the rPPG processor."""
    parser = argparse.ArgumentParser(
        description="Real-time rPPG Heart Rate Estimation from Face Video."
    )
    
    # Default to data/10-gt/vid.avi if it exists, otherwise camera 0
    default_source = "data/10-gt/vid.avi" if os.path.exists("data/10-gt/vid.avi") else "0"

    parser.add_argument(
        "-s", "--source",
        type=str,
        default=default_source,
        help="Path to video file or camera index (e.g., 0, 1, 'data/10-gt/vid.avi').",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=f"Camera frame width (default: {DEFAULT_CAM_WIDTH} for webcams, {DEFAULT_FRAME_WIDTH} for files).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=f"Camera frame height (default: {DEFAULT_CAM_HEIGHT} for webcams, {DEFAULT_FRAME_HEIGHT} for files).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override frame rate (default: auto-detected or 30.0).",
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="pos",
        choices=["pos", "chrom", "green"],
        help="rPPG extraction algorithm: pos, chrom, or green (default: pos).",
    )
    parser.add_argument(
        "--camera-settings",
        action="store_true",
        help="Open camera driver hardware settings dialog (Windows only).",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable the pulse graph window.",
    )

    return parser.parse_args()


def run_rppg(
    source: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps_override: Optional[float] = None,
    show_camera_settings: bool = False,
    show_graph: bool = True,
    method: str = "pos",
):
    """
    Main processing loop for rPPG tracking and visualization.
    """
    is_camera = source.isdigit()

    cap = open_video_source(
        source=source,
        width=width,
        height=height,
        target_fps=DEFAULT_FPS,
        show_camera_settings=show_camera_settings,
    )

    # Determine frame rate and playback delay
    detected_fps = cap.get(cv.CAP_PROP_FPS)
    fps = fps_override if fps_override and fps_override > 0 else (
        detected_fps if detected_fps > 0 else DEFAULT_FPS
    )
    # For live camera, waitKey(1) avoids adding artificial 33ms sleep latency
    delay = 1 if is_camera else max(int(1000.0 / fps), 1)

    print(f"[INFO] Video source: {source} ({'Webcam' if is_camera else 'File'})")
    print(f"[INFO] Resolution: {cap.get(cv.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT):.0f}")
    print(f"[INFO] Initial frame rate estimate: {fps:.2f} FPS (Key wait: {delay} ms)")
    if is_camera and not fps_override:
        print("[INFO] Dynamic real-time FPS estimation enabled for webcam.")
    print(f"[INFO] rPPG Algorithm: {method.upper()}")
    print("[INFO] Press 'q' or ESC in any display window to exit.")

    pipeline = RPPGPipeline(fps=fps, method=method)
    measured_fps = fps
    last_frame_time = time.perf_counter()
    face_lost_count = 0

    try:
        with FaceROIProcessor() as face_processor:
            while True:
                frame_start_time = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream or cannot read frame. Exiting...")
                    break

                # Downscale large frames so display and processing are fast and fit screen
                frame = FaceROIProcessor._maybe_resize(frame)

                # Dynamic FPS measurement based on actual wall-clock inter-frame arrival time
                now = time.perf_counter()
                dt = now - last_frame_time
                last_frame_time = now
                if dt > 0:
                    instant_fps = 1.0 / dt
                    measured_fps = 0.9 * measured_fps + 0.1 * instant_fps
                    if is_camera and not fps_override:
                        pipeline.set_fps(measured_fps)

                current_fps = measured_fps if is_camera else fps
                reset_threshold = int(2.5 * current_fps)

                # 1. Detect face landmarks and extract skin ROI
                detection = face_processor.process_frame(frame)

                display_frame = frame.copy()
                filtered_signal = pipeline.filtered_signal
                bpm = pipeline.current_bpm
                is_warming_up = True

                if detection.detected and detection.mean_rgbs is not None:
                    # If returning after prolonged absence (> 2.5s), reset pipeline to prevent step discontinuity
                    if face_lost_count >= reset_threshold:
                        pipeline.reset()
                    face_lost_count = 0

                    # 2. Ingest RGB into rPPG pipeline
                    filtered_signal, bpm, is_warming_up = pipeline.update(detection.mean_rgbs)

                    # 3. Render semi-transparent green overlay on active ROIs
                    display_frame = face_processor.render_roi_overlay(
                        frame,
                        active_indices=pipeline.active_patch_indices,
                        polys=detection.polys
                    )
                else:
                    face_lost_count += 1
                    # Brief loss (< 2.5s): hold previous stable BPM estimate
                    if face_lost_count < reset_threshold and len(pipeline.pulse_buffer) >= pipeline.min_samples_for_bpm:
                        is_warming_up = False

                # 4. Render HUD (BPM readout / status and live FPS)
                draw_hud(display_frame, bpm=bpm, is_warming_up=is_warming_up, fps=current_fps)

                # 5. Display windows
                cv.imshow("Face mask", display_frame)

                if show_graph:
                    graph_image = create_pulse_graph(filtered_signal)
                    cv.imshow("Master pulse signal", graph_image)

                # Real-time frame synchronization: wait only the remaining time for target FPS
                elapsed = time.perf_counter() - frame_start_time
                target_frame_sec = 1.0 / fps
                remaining_ms = int((target_frame_sec - elapsed) * 1000)
                frame_delay = 1 if (is_camera or FAST_FORWARD) else max(1, remaining_ms)

                # Keyboard interaction
                key = cv.waitKey(frame_delay) & 0xFF
                if key in (ord('q'), 27):  # 'q' or ESC
                    break
                elif key in (ord('f'), ord('F')):  # 'f' toggles fast forward
                    toggle_fast_forward()
                    print(f"[INFO] Fast Forward: {'ON' if FAST_FORWARD else 'OFF'}")

    finally:
        cap.release()
        cv.destroyAllWindows()
        print("[INFO] Cleanup complete. Resources released.")



def main():
    """Entry point for command line invocation."""
    args = parse_arguments()
    try:
        run_rppg(
            source=args.source,
            width=args.width,
            height=args.height,
            fps_override=args.fps,
            show_camera_settings=args.camera_settings,
            show_graph=not args.no_graph,
            method=args.method,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting cleanly.")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()