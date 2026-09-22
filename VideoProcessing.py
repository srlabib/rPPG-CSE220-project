"""
rPPG Video Processing Application.

Extracts real-time photoplethysmography (pulse) signals and heart rate (BPM)
from facial video feeds using MediaPipe FaceMesh and the POS algorithm.
"""

import argparse
import os
import sys
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


def open_video_source(
    source: str,
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
    target_fps: float = DEFAULT_FPS,
    show_camera_settings: bool = False,
) -> cv.VideoCapture:
    """
    Opens and configures a video source (webcam index or video file path).

    Args:
        source: Camera index as string/int (e.g., '0', '1') or path to video file.
        width: Desired capture width (for webcams).
        height: Desired capture height (for webcams).
        target_fps: Desired capture frame rate.
        show_camera_settings: If True, opens Windows hardware camera settings dialog.

    Returns:
        Configured cv.VideoCapture instance.
    """
    is_camera = source.isdigit()
    device_id = int(source) if is_camera else source

    cap = cv.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: '{source}'. "
            f"Please verify device connection or video file path."
        )

    if is_camera:
        # Request uncompressed raw frames (YUYV) instead of compressed MJPEG for DroidCam/webcams
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
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
        default=DEFAULT_FRAME_WIDTH,
        help=f"Camera frame width (default: {DEFAULT_FRAME_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help=f"Camera frame height (default: {DEFAULT_FRAME_HEIGHT}).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override frame rate (default: auto-detected or 30.0).",
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
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
    fps_override: float = None,
    show_camera_settings: bool = False,
    show_graph: bool = True,
):
    """
    Main processing loop for rPPG tracking and visualization.
    """
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
    delay = max(int(1000.0 / fps), 1)

    print(f"[INFO] Video source: {source}")
    print(f"[INFO] Resolution: {cap.get(cv.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT):.0f}")
    print(f"[INFO] Frame rate: {fps:.2f} FPS (Frame delay: {delay} ms)")
    print("[INFO] Press 'q' or ESC in any display window to exit.")

    pipeline = RPPGPipeline(fps=fps)

    try:
        with FaceROIProcessor() as face_processor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream or cannot read frame. Exiting...")
                    break

                # 1. Detect face landmarks and extract skin ROI
                detection = face_processor.process_frame(frame)

                display_frame = frame.copy()
                filtered_signal = pipeline.filtered_signal
                bpm = pipeline.current_bpm
                is_warming_up = True

                if detection.detected and detection.mean_rgbs is not None:
                    # 2. Ingest RGB into rPPG pipeline
                    filtered_signal, bpm, is_warming_up = pipeline.update(detection.mean_rgbs)

                    # 3. Render semi-transparent green overlay on ROIs
                    if detection.masks is not None:
                        display_frame = face_processor.render_roi_overlay(
                            frame,
                            detection.masks,
                            active_indices=pipeline.active_patch_indices
                        )

                # 4. Render HUD (BPM readout / status)
                draw_hud(display_frame, bpm=bpm, is_warming_up=is_warming_up, fps=fps)

                # 5. Display windows
                cv.imshow("Face mask", display_frame)

                if show_graph:
                    graph_image = create_pulse_graph(filtered_signal)
                    cv.imshow("Master pulse signal", graph_image)

                # Keyboard interaction
                key = cv.waitKey(delay) & 0xFF
                if key in (ord('q'), 27):  # 'q' or ESC
                    break

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
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting cleanly.")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()