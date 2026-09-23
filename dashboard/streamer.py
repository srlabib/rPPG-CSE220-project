"""
Thread-safe video streaming and real-time rPPG analysis engine.
Supports both live webcam capture and uploaded video file playback with ground truth synchronization.
"""

import time
import sys
import threading
from typing import Optional, Dict, Any, List, Tuple
import cv2 as cv
import numpy as np

from face_processor import FaceROIProcessor
from rppg_pipeline import RPPGPipeline
from config import DEFAULT_FPS, DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT
from dashboard.ground_truth import compute_live_metrics


class RPPGStreamManager:
    """Singleton manager controlling the active video processing worker."""
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.is_paused = False
        self.mode = "webcam"  # "webcam" or "upload"
        self.source_path: Any = 0
        self.gt_times: List[float] = []
        self.gt_hr: List[float] = []

        # Thread handle
        self.worker_thread: Optional[threading.Thread] = None

        # Latest frame JPEG bytes
        self.latest_jpeg: Optional[bytes] = None

        # Telemetry & Metrics state
        self.current_bpm: Optional[float] = None
        self.latest_pulse: float = 0.0
        self.pulse_history: List[float] = []
        self.pred_history: List[Tuple[float, float]] = []  # (timestamp, bpm)
        self.current_timestamp: float = 0.0
        self.fps: float = 30.0
        self.sqi: int = 0
        self.face_detected: bool = False
        self.status_text: str = "Ready"
        self.video_progress: float = 0.0  # 0 to 100%

        # Live performance metrics against ground truth
        self.live_metrics: Dict[str, Any] = {
            "current_error": None,
            "latest_gt": None,
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "count": 0
        }

    def start_webcam(self, camera_index: int = 0):
        """Starts real-time analysis using local webcam."""
        self.stop()
        with self.lock:
            self.mode = "webcam"
            self.source_path = camera_index
            self.gt_times = []
            self.gt_hr = []
            self._reset_state()
            self.is_running = True
            self.is_paused = False

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def start_video(self, video_path: str, gt_times: Optional[List[float]] = None, gt_hr: Optional[List[float]] = None):
        """Starts playback and evaluation of an uploaded video."""
        self.stop()
        with self.lock:
            self.mode = "upload"
            self.source_path = video_path
            self.gt_times = gt_times or []
            self.gt_hr = gt_hr or []
            self._reset_state()
            self.is_running = True
            self.is_paused = False

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def pause(self):
        """Toggles pause/resume."""
        with self.lock:
            self.is_paused = not self.is_paused

    def stop(self):
        """Stops the current streaming worker."""
        with self.lock:
            self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)
        self.worker_thread = None

    def _reset_state(self):
        self.current_bpm = None
        self.latest_pulse = 0.0
        self.pulse_history = []
        self.pred_history = []
        self.current_timestamp = 0.0
        self.sqi = 0
        self.face_detected = False
        self.status_text = "Initializing pipeline..."
        self.video_progress = 0.0
        self.live_metrics = {
            "current_error": None,
            "latest_gt": None,
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "count": 0
        }

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_telemetry(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "running": self.is_running,
                "paused": self.is_paused,
                "mode": self.mode,
                "timestamp": round(self.current_timestamp, 2),
                "fps": round(self.fps, 1),
                "current_bpm": round(self.current_bpm, 1) if self.current_bpm else None,
                "latest_pulse": round(self.latest_pulse, 4),
                "pulse_history": list(self.pulse_history),
                "sqi": self.sqi,
                "face_detected": self.face_detected,
                "status_text": self.status_text,
                "progress": round(self.video_progress, 1),
                "metrics": self.live_metrics,
            }

    def _worker_loop(self):
        """Worker thread processing video frames and running rPPG algorithms."""
        cap = None
        face_processor = None
        pipeline = None

        try:
            # 1. Initialize VideoCapture
            is_camera = (self.mode == "webcam")
            if is_camera:
                src = int(self.source_path) if str(self.source_path).isdigit() else 0
                if sys.platform == "win32":
                    cap = cv.VideoCapture(src, cv.CAP_DSHOW)
                    if not cap.isOpened() or not cap.read()[0]:
                        cap.release()
                        cap = cv.VideoCapture(src)
                else:
                    cap = cv.VideoCapture(src)

                cap.set(cv.CAP_PROP_FRAME_WIDTH, DEFAULT_CAM_WIDTH)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAM_HEIGHT)
                detected_fps = cap.get(cv.CAP_PROP_FPS)
                target_fps = detected_fps if (detected_fps and 15.0 <= detected_fps <= 60.0) else DEFAULT_FPS
                total_frames = 0
            else:
                cap = cv.VideoCapture(self.source_path)
                detected_fps = cap.get(cv.CAP_PROP_FPS)
                target_fps = detected_fps if (detected_fps and 10.0 <= detected_fps <= 120.0) else DEFAULT_FPS
                total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            if not cap.isOpened():
                with self.lock:
                    self.status_text = f"Error: Cannot open video source ({self.source_path})"
                    self.is_running = False
                return

            self.fps = target_fps
            face_processor = FaceROIProcessor()
            pipeline = RPPGPipeline(fps=target_fps)

            frame_interval = 1.0 / target_fps
            frame_count = 0
            failed_reads = 0

            while True:
                with self.lock:
                    if not self.is_running:
                        break
                    paused = self.is_paused

                if paused:
                    time.sleep(0.05)
                    continue

                loop_start = time.perf_counter()
                ret, frame = cap.read()

                if not ret:
                    if is_camera:
                        failed_reads += 1
                        if failed_reads > 30:  # ~1 second of continuous drop
                            with self.lock:
                                self.status_text = "Camera connection lost"
                                self.is_running = False
                            break
                        time.sleep(0.02)
                        continue
                    else:
                        # Video completed
                        with self.lock:
                            self.status_text = "Video Completed"
                            self.video_progress = 100.0
                            self.is_running = False
                        break

                failed_reads = 0
                frame_count += 1
                current_time_sec = frame_count / target_fps

                # Downscale large frames if needed
                frame = FaceROIProcessor._maybe_resize(frame)
                display_frame = frame.copy()

                # Process face and skin patches
                detection = face_processor.process_frame(frame)
                has_face = detection.detected and (detection.mean_rgbs is not None)

                est_bpm = None
                pulse_val = 0.0

                if has_face:
                    filtered_signal, raw_bpm, is_warming_up = pipeline.update(detection.mean_rgbs)
                    if not is_warming_up and raw_bpm and raw_bpm > 0:
                        est_bpm = raw_bpm

                    # Render semi-transparent patch overlays
                    display_frame = face_processor.render_roi_overlay(
                        frame,
                        active_indices=pipeline.active_patch_indices,
                        polys=detection.polys
                    )

                    # Extract the true Butterworth-filtered BVP waveform for the oscilloscope
                    norm_bvp = []
                    if len(filtered_signal) >= 8:
                        recent_len = min(len(filtered_signal), 140)
                        recent_signal = filtered_signal[-recent_len:]
                        peak_amp = float(np.max(np.abs(recent_signal)))
                        if peak_amp > 1e-6:
                            norm_bvp = (recent_signal / peak_amp).round(4).tolist()
                        else:
                            norm_bvp = [0.0] * recent_len
                        pulse_val = float(filtered_signal[-1])

                # Signal Quality Index (SQI)
                sqi_val = 0
                if has_face and len(pipeline.pulse_buffer) >= pipeline.min_samples_for_bpm:
                    sqi_val = 85 if est_bpm is not None else 50
                elif has_face:
                    sqi_val = 40

                # Update shared state
                with self.lock:
                    self.face_detected = has_face
                    self.current_timestamp = current_time_sec
                    self.current_bpm = est_bpm
                    self.latest_pulse = pulse_val
                    if norm_bvp:
                        self.pulse_history = norm_bvp
                    self.sqi = sqi_val
                    if total_frames > 0:
                        self.video_progress = min(100.0, (frame_count / total_frames) * 100.0)

                    if has_face:
                        if est_bpm is not None:
                            self.status_text = f"Tracking: {est_bpm:.1f} BPM"
                            self.pred_history.append((current_time_sec, est_bpm))
                        else:
                            self.status_text = "Buffering skin pulse..."
                    else:
                        self.status_text = "Searching for face..."

                    # Live metrics against ground truth
                    if self.gt_times and self.gt_hr and len(self.pred_history) > 0:
                        self.live_metrics = compute_live_metrics(
                            self.pred_history,
                            self.gt_times,
                            self.gt_hr,
                            skip_seconds=15.0
                        )

                # Draw minimal HUD on video frame
                self._draw_overlay_badge(display_frame, est_bpm, sqi_val, has_face)

                # Encode frame to JPEG
                _, buffer = cv.imencode(".jpg", display_frame, [cv.IMWRITE_JPEG_QUALITY, 80])
                jpeg_bytes = buffer.tobytes()

                with self.lock:
                    self.latest_jpeg = jpeg_bytes

                # Throttle playback to match native video framerate
                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

        except Exception as e:
            with self.lock:
                self.status_text = f"Error: {str(e)}"
                self.is_running = False
        finally:
            if cap:
                cap.release()
            if face_processor:
                face_processor.close()

    def _draw_overlay_badge(self, frame: np.ndarray, bpm: Optional[float], sqi: int, has_face: bool):
        """Draws subtle HUD information in the corner of the video frame."""
        # Top banner background
        cv.rectangle(frame, (10, 10), (220, 52), (15, 23, 42), -1)
        cv.rectangle(frame, (10, 10), (220, 52), (51, 65, 85), 1)

        if has_face:
            status_color = (0, 230, 115)  # neon green
            status_str = f"BPM: {bpm:.1f}" if bpm else "BUFFERING..."
        else:
            status_color = (80, 80, 240)  # red/amber
            status_str = "SEARCHING FACE"

        cv.circle(frame, (25, 31), 6, status_color, -1)
        cv.putText(frame, status_str, (40, 37), cv.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv.LINE_AA)
