"""
Thread-safe rPPG Web Streaming and Processing Engine.

Manages video acquisition (webcam/file), FaceROIProcessor, and RPPGPipeline
in a background worker thread, producing real-time MJPEG frames and telemetry.
"""

import csv
import io
import os
import sys
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
import cv2 as cv
import numpy as np

from config import (
    DEFAULT_FPS,
    DEFAULT_CAM_WIDTH,
    DEFAULT_CAM_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    ROI_OVERLAY_ALPHA,
    NUM_PATCHES,
    TOP_K_PATCHES,
    MIN_FREQ_HZ,
    MAX_FREQ_HZ,
)
from face_processor import FaceROIProcessor, FaceROIDetection
from rppg_pipeline import RPPGPipeline
from signal_processing import calculate_snr
from visualizer import draw_hud


class WebRPPGPipeline:
    """
    Orchestrates video frame capture, face landmark extraction,
    and POS-based rPPG pulse estimation in a non-blocking background thread.
    """

    def __init__(self, initial_source: Optional[str] = None):
        self.source = initial_source
        self.is_running = False
        self.is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Processing options
        self.show_overlay = True
        self.show_hud = False  # Frontend has rich UI HUD; keep video HUD optional
        self.roi_alpha = ROI_OVERLAY_ALPHA
        self.jpeg_quality = 80
        self.loop_video = True

        # Pipeline state
        self.fps = DEFAULT_FPS
        self.measured_fps = DEFAULT_FPS
        self.current_bpm: float = 0.0
        self.is_warming_up: bool = True
        self.face_detected: bool = False
        self.active_patches: List[int] = []
        self.current_snr: float = 0.0
        self.latest_pulse_val: float = 0.0

        # Frame buffers
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_id: int = 0
        self._filtered_signal_cache: List[float] = []

        # Session & Recording stats
        self.session_start_time: float = 0.0
        self.is_recording = False
        self.recording_buffer: List[Dict[str, Any]] = []
        self.bpm_readings: List[float] = []

    def start(self):
        """Starts the background worker thread if a valid source is configured."""
        with self._lock:
            if not self.source:
                return  # Do not start without an explicitly chosen source
            if self.is_running:
                return
            self.is_running = True
            self.is_paused = False
            self.session_start_time = time.time()
            self.bpm_readings.clear()

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the worker thread."""
        with self._lock:
            self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def pause(self, paused: Optional[bool] = None):
        """Toggles or sets the paused state."""
        with self._lock:
            if paused is None:
                self.is_paused = not self.is_paused
            else:
                self.is_paused = paused

    def reset(self):
        """Resets the rPPG signal buffers and statistics."""
        with self._lock:
            self.bpm_readings.clear()
            self.session_start_time = time.time()
            self.current_bpm = 0.0
            self.is_warming_up = True
            self.latest_pulse_val = 0.0
            self._filtered_signal_cache.clear()
            if self.is_recording:
                self.recording_buffer.clear()

    def set_source(self, new_source: Optional[str]):
        """Changes the video source (webcam device id or video file path) and starts."""
        was_running = self.is_running
        if was_running:
            self.stop()
        with self._lock:
            self.source = new_source
            self.bpm_readings.clear()
            self.session_start_time = time.time()
            self.current_bpm = 0.0
            self.is_warming_up = True
            self._latest_jpeg = None
        if new_source:
            self.start()

    def update_settings(self, settings: Dict[str, Any]):
        """Dynamically updates runtime settings."""
        with self._lock:
            if "show_overlay" in settings:
                self.show_overlay = bool(settings["show_overlay"])
            if "show_hud" in settings:
                self.show_hud = bool(settings["show_hud"])
            if "roi_alpha" in settings:
                self.roi_alpha = float(np.clip(settings["roi_alpha"], 0.05, 0.95))
            if "jpeg_quality" in settings:
                self.jpeg_quality = int(np.clip(settings["jpeg_quality"], 30, 95))
            if "loop_video" in settings:
                self.loop_video = bool(settings["loop_video"])

    def start_recording(self):
        """Begins recording timestamped rPPG data for CSV export."""
        with self._lock:
            self.is_recording = True
            self.recording_buffer.clear()

    def stop_recording(self) -> int:
        """Stops recording and returns the number of captured samples."""
        with self._lock:
            self.is_recording = False
            return len(self.recording_buffer)

    def export_csv(self) -> str:
        """Serializes recorded session data into CSV format."""
        with self._lock:
            data = list(self.recording_buffer)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp_sec", "BPM", "Pulse_Value", "SNR", "Face_Detected"])
        for row in data:
            writer.writerow([
                f"{row.get('timestamp', 0.0):.3f}",
                f"{row.get('bpm', 0.0):.1f}" if row.get("bpm") is not None else "",
                f"{row.get('pulse', 0.0):.5f}",
                f"{row.get('snr', 0.0):.2f}",
                1 if row.get("face_detected") else 0,
            ])
        return output.getvalue()

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns the most recent JPEG-encoded video frame."""
        with self._lock:
            return self._latest_jpeg

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns a snapshot of the current vital signs and telemetry."""
        with self._lock:
            elapsed = time.time() - self.session_start_time if self.session_start_time > 0 else 0.0
            valid_bpms = [b for b in self.bpm_readings if b > 0]
            min_bpm = min(valid_bpms) if valid_bpms else 0.0
            max_bpm = max(valid_bpms) if valid_bpms else 0.0
            avg_bpm = sum(valid_bpms) / len(valid_bpms) if valid_bpms else 0.0

            return {
                "source": self.source,
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "is_recording": self.is_recording,
                "recorded_samples": len(self.recording_buffer),
                "bpm": self.current_bpm,
                "is_warming_up": self.is_warming_up,
                "face_detected": self.face_detected,
                "fps": round(self.measured_fps, 1),
                "snr": round(self.current_snr, 2),
                "active_patches": list(self.active_patches),
                "num_patches": NUM_PATCHES,
                "latest_pulse": round(float(self.latest_pulse_val), 5),
                "pulse_samples": list(self._filtered_signal_cache[-120:]),  # Last 120 samples
                "session_duration_sec": round(elapsed, 1),
                "min_bpm": round(min_bpm, 1),
                "max_bpm": round(max_bpm, 1),
                "avg_bpm": round(avg_bpm, 1),
                "timestamp": round(time.time(), 3),
            }

    def _open_capture(self) -> Optional[cv.VideoCapture]:
        """Safely opens a video source."""
        source_str = str(self.source).strip()
        is_camera = source_str.isdigit()
        device_id = int(source_str) if is_camera else source_str

        if is_camera and sys.platform == "win32":
            cap = cv.VideoCapture(device_id, cv.CAP_DSHOW)
            if not cap.isOpened() or not cap.read()[0]:
                cap.release()
                cap = cv.VideoCapture(device_id)
        else:
            cap = cv.VideoCapture(device_id)

        if not cap.isOpened():
            return None

        if is_camera:
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, DEFAULT_CAM_WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAM_HEIGHT)
            cap.set(cv.CAP_PROP_FPS, DEFAULT_FPS)

        return cap

    def _worker_loop(self):
        """Continuous background capture and processing loop."""
        cap = self._open_capture()
        if cap is None:
            print(f"[WARN] WebRPPGPipeline could not open source: {self.source}")
            with self._lock:
                self.is_running = False
            return

        detected_fps = cap.get(cv.CAP_PROP_FPS)
        source_is_camera = str(self.source).isdigit()
        base_fps = detected_fps if detected_fps > 0 else DEFAULT_FPS
        self.fps = base_fps
        self.measured_fps = base_fps

        pipeline = RPPGPipeline(fps=base_fps)
        last_frame_time = time.perf_counter()
        face_lost_count = 0

        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        try:
            with FaceROIProcessor() as face_processor:
                while self.is_running:
                    if self.is_paused:
                        time.sleep(0.05)
                        continue

                    frame_start_time = time.perf_counter()
                    ret, frame = cap.read()

                    if not ret:
                        # Handle end of video file: loop if enabled, else pause
                        if not source_is_camera and self.loop_video:
                            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                break
                        else:
                            time.sleep(0.05)
                            continue

                    # Downscale for performance if needed
                    frame = FaceROIProcessor._maybe_resize(frame)

                    # Dynamic FPS estimation
                    now = time.perf_counter()
                    dt = now - last_frame_time
                    last_frame_time = now
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        self.measured_fps = 0.9 * self.measured_fps + 0.1 * instant_fps
                        if source_is_camera:
                            pipeline.set_fps(self.measured_fps)

                    current_fps = self.measured_fps if source_is_camera else base_fps
                    reset_threshold = int(2.5 * current_fps)

                    # 1. Detect face landmarks and skin ROIs
                    detection = face_processor.process_frame(frame)
                    display_frame = frame.copy()

                    bpm = self.current_bpm
                    is_warming_up = True
                    pulse_val = 0.0

                    if detection.detected and detection.mean_rgbs is not None:
                        self.face_detected = True
                        if face_lost_count >= reset_threshold:
                            pipeline.reset()
                        face_lost_count = 0

                        # 2. Ingest RGB into rPPG pipeline
                        filtered_signal, calculated_bpm, is_warming_up = pipeline.update(detection.mean_rgbs)

                        if calculated_bpm is not None and calculated_bpm > 0:
                            bpm = calculated_bpm
                            with self._lock:
                                self.bpm_readings.append(bpm)

                        self.active_patches = list(pipeline.active_patch_indices)

                        if len(filtered_signal) > 0:
                            pulse_val = float(filtered_signal[-1])
                            self._filtered_signal_cache = [float(x) for x in filtered_signal]
                            if len(filtered_signal) >= pipeline.min_samples_for_bpm:
                                self.current_snr = calculate_snr(filtered_signal, current_fps)

                        # 3. Render ROI overlay if enabled
                        if self.show_overlay and detection.polys:
                            display_frame = face_processor.render_roi_overlay(
                                display_frame,
                                active_indices=pipeline.active_patch_indices,
                                polys=detection.polys,
                                alpha=self.roi_alpha,
                            )
                    else:
                        self.face_detected = False
                        face_lost_count += 1
                        if face_lost_count < reset_threshold and len(pipeline.pulse_buffer) >= pipeline.min_samples_for_bpm:
                            is_warming_up = False

                    # 4. Optional OpenCV HUD
                    if self.show_hud:
                        draw_hud(display_frame, bpm=bpm, is_warming_up=is_warming_up, fps=current_fps)

                    # Update internal states
                    with self._lock:
                        self.current_bpm = bpm
                        self.is_warming_up = is_warming_up
                        self.latest_pulse_val = pulse_val

                        # Recording
                        if self.is_recording:
                            self.recording_buffer.append({
                                "timestamp": time.time() - self.session_start_time,
                                "bpm": bpm if not is_warming_up else None,
                                "pulse": pulse_val,
                                "snr": self.current_snr,
                                "face_detected": self.face_detected,
                            })

                    # Encode JPEG for MJPEG stream
                    encode_param[1] = self.jpeg_quality
                    success, jpeg_buffer = cv.imencode(".jpg", display_frame, encode_param)
                    if success:
                        with self._lock:
                            self._latest_jpeg = jpeg_buffer.tobytes()
                            self._latest_frame_id += 1

                    # Synchronization delay for video files
                    if not source_is_camera:
                        elapsed = time.perf_counter() - frame_start_time
                        target_sec = 1.0 / base_fps
                        sleep_time = target_sec - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

        finally:
            cap.release()
            print("[INFO] WebRPPGPipeline worker exited and released capture source.")
