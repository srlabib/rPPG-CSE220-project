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
import VideoProcessing


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
        self.method: str = "pos"  # "pos", "chrom", "green", or "all"
        self.source_path: Any = 0
        self.gt_times: List[float] = []
        self.gt_hr: List[float] = []

        # Thread handle
        self.worker_thread: Optional[threading.Thread] = None

        # Latest frame JPEG bytes
        self.latest_jpeg: Optional[bytes] = None

        # Telemetry & Metrics state
        self.current_bpm: Optional[float] = None
        self.method_bpms: Dict[str, Optional[float]] = {"pos": None, "chrom": None, "green": None}
        self.latest_pulse: float = 0.0
        self.pulse_history: List[float] = []
        self.raw_pulse_history: List[float] = []
        self.method_pulses: Dict[str, List[float]] = {"pos": [], "chrom": [], "green": []}
        self.method_raw_pulses: Dict[str, List[float]] = {"pos": [], "chrom": [], "green": []}
        self.pred_history: List[Tuple[float, float]] = []  # active method (timestamp, bpm)
        self.pred_histories: Dict[str, List[Tuple[float, float]]] = {"pos": [], "chrom": [], "green": []}
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
        self.all_metrics: Dict[str, Dict[str, Any]] = {
            "pos": {"current_error": None, "latest_gt": None, "mae": None, "rmse": None, "pearson_r": None, "count": 0},
            "chrom": {"current_error": None, "latest_gt": None, "mae": None, "rmse": None, "pearson_r": None, "count": 0},
            "green": {"current_error": None, "latest_gt": None, "mae": None, "rmse": None, "pearson_r": None, "count": 0},
        }

    def set_method(self, method: str):
        """Switches the active evaluation / display method ('pos', 'chrom', 'green', or 'all')."""
        with self.lock:
            m = (method or "pos").lower().strip()
            if m in ["pos", "chrom", "green", "all"]:
                self.method = m
                # Update current_bpm and live_metrics to reflect newly selected method
                if m != "all" and m in self.method_bpms:
                    self.current_bpm = self.method_bpms[m]
                    self.live_metrics = self.all_metrics.get(m, self.live_metrics)

    def start_webcam(self, camera_index: int = 0, method: str = "pos"):
        """Starts real-time analysis using local webcam."""
        self.stop()
        with self.lock:
            self.mode = "webcam"
            self.method = (method or "pos").lower().strip()
            self.source_path = camera_index
            self.gt_times = []
            self.gt_hr = []
            self._reset_state()
            self.is_running = True
            self.is_paused = False

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def start_video(self, video_path: str, gt_times: Optional[List[float]] = None, gt_hr: Optional[List[float]] = None, method: str = "pos"):
        """Starts playback and evaluation of an uploaded video."""
        self.stop()
        with self.lock:
            self.mode = "upload"
            self.method = (method or "pos").lower().strip()
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
        VideoProcessing.set_fast_forward(False)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)
        self.worker_thread = None

    def _reset_state(self):
        self.current_bpm = None
        self.method_bpms = {"pos": None, "chrom": None, "green": None}
        self.latest_pulse = 0.0
        self.pulse_history = []
        self.raw_pulse_history = []
        self.method_pulses = {"pos": [], "chrom": [], "green": []}
        self.method_raw_pulses = {"pos": [], "chrom": [], "green": []}
        self.pred_history = []
        self.pred_histories = {"pos": [], "chrom": [], "green": []}
        self.current_timestamp = 0.0
        self.sqi = 0
        self.face_detected = False
        self.status_text = "Initializing rPPG pipelines..."
        self.video_progress = 0.0
        self.live_metrics = {
            "current_error": None,
            "latest_gt": None,
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "count": 0
        }
        self.all_metrics = {
            "pos": {"current_error": None, "latest_gt": None, "mae": None, "rmse": None, "pearson_r": None, "count": 0},
            "chrom": {"current_error": None, "latest_gt": None, "mae": None, "rmse": None, "pearson_r": None, "count": 0},
            "green": {"current_error": None, "latest_gt": None, "mae": None, "rmse": None, "pearson_r": None, "count": 0},
        }

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_telemetry(self) -> Dict[str, Any]:
        with self.lock:
            active_m = self.method
            active_bpm = self.current_bpm
            if active_m in self.method_bpms and self.method_bpms[active_m] is not None:
                active_bpm = self.method_bpms[active_m]
            elif active_m == "all" and self.method_bpms["pos"] is not None:
                active_bpm = self.method_bpms["pos"]

            return {
                "running": self.is_running,
                "paused": self.is_paused,
                "mode": self.mode,
                "method": self.method,
                "timestamp": round(self.current_timestamp, 2),
                "fps": round(self.fps, 1),
                "current_bpm": round(active_bpm, 1) if active_bpm else None,
                "method_bpms": {
                    k: (round(v, 1) if v is not None else None)
                    for k, v in self.method_bpms.items()
                },
                "latest_pulse": round(self.latest_pulse, 4),
                "pulse_history": list(self.pulse_history),
                "raw_pulse_history": list(self.raw_pulse_history),
                "method_pulses": self.method_pulses,
                "sqi": self.sqi,
                "face_detected": self.face_detected,
                "status_text": self.status_text,
                "progress": round(self.video_progress, 1),
                "metrics": self.live_metrics,
                "all_metrics": self.all_metrics,
                "fast_forward": VideoProcessing.is_fast_forward(),
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
            # Initialize concurrent rPPG pipelines for all three algorithms
            pipelines = {
                "pos": RPPGPipeline(fps=target_fps, method="pos"),
                "chrom": RPPGPipeline(fps=target_fps, method="chrom"),
                "green": RPPGPipeline(fps=target_fps, method="green"),
            }

            frame_interval = 1.0 / target_fps
            frame_count = 0
            failed_reads = 0

            while True:
                with self.lock:
                    if not self.is_running:
                        break
                    paused = self.is_paused
                    current_method = self.method

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

                results = {}
                active_key = current_method if current_method in pipelines else "pos"
                method_bvp_dict = {"pos": [], "chrom": [], "green": []}
                method_raw_dict = {"pos": [], "chrom": [], "green": []}
                method_bpms_now = {"pos": None, "chrom": None, "green": None}
                est_bpm = None
                pulse_val = 0.0
                norm_bvp = []
                norm_raw = []

                if has_face:
                    # Run all pipelines concurrently on the extracted skin mean RGBs
                    for m_name, pipe in pipelines.items():
                        filt, raw_bpm, is_warming = pipe.update(detection.mean_rgbs)
                        valid_bpm = raw_bpm if (not is_warming and raw_bpm and raw_bpm > 0) else None
                        results[m_name] = {
                            "filtered": filt,
                            "bpm": valid_bpm,
                            "warming": is_warming,
                            "pipe": pipe
                        }
                        method_bpms_now[m_name] = valid_bpm

                        # Extract filtered BVP
                        if len(filt) >= 8:
                            rec_len = min(len(filt), 140)
                            rec = filt[-rec_len:]
                            p_max = float(np.max(np.abs(rec)))
                            method_bvp_dict[m_name] = (rec / p_max).round(4).tolist() if p_max > 1e-6 else [0.0] * rec_len

                        # Extract raw pulse
                        raw_arr = np.asarray(pipe.pulse_buffer, dtype=np.float32)
                        if len(raw_arr) >= 8:
                            raw_rec_len = min(len(raw_arr), 140)
                            raw_rec = raw_arr[-raw_rec_len:]
                            r_max = float(np.max(np.abs(raw_rec)))
                            method_raw_dict[m_name] = (raw_rec / r_max).round(4).tolist() if r_max > 1e-6 else [0.0] * raw_rec_len

                    # Active overlay pipeline
                    active_pipe = pipelines[active_key]
                    display_frame = face_processor.render_roi_overlay(
                        frame,
                        active_indices=active_pipe.active_patch_indices,
                        polys=detection.polys
                    )

                    norm_bvp = method_bvp_dict.get(active_key, [])
                    norm_raw = method_raw_dict.get(active_key, [])
                    est_bpm = method_bpms_now.get(active_key)
                    if len(results[active_key]["filtered"]) > 0:
                        pulse_val = float(results[active_key]["filtered"][-1])
                else:
                    active_pipe = pipelines[active_key]

                # Signal Quality Index (SQI)
                sqi_val = 0
                if has_face and len(active_pipe.pulse_buffer) >= active_pipe.min_samples_for_bpm:
                    sqi_val = 85 if est_bpm is not None else 50
                elif has_face:
                    sqi_val = 40

                # Update shared state
                with self.lock:
                    self.face_detected = has_face
                    self.current_timestamp = current_time_sec
                    self.method_bpms = method_bpms_now
                    self.current_bpm = est_bpm
                    self.latest_pulse = pulse_val
                    self.method_pulses = method_bvp_dict
                    self.method_raw_pulses = method_raw_dict
                    if norm_bvp:
                        self.pulse_history = norm_bvp
                    if norm_raw:
                        self.raw_pulse_history = norm_raw
                    self.sqi = sqi_val
                    if total_frames > 0:
                        self.video_progress = min(100.0, (frame_count / total_frames) * 100.0)

                    if has_face:
                        for m in ["pos", "chrom", "green"]:
                            if method_bpms_now[m] is not None:
                                self.pred_histories[m].append((current_time_sec, method_bpms_now[m]))

                        if est_bpm is not None:
                            self.pred_history.append((current_time_sec, est_bpm))
                            if current_method == "all":
                                p_str = f"P:{method_bpms_now['pos']:.1f}" if method_bpms_now['pos'] else "P:--"
                                c_str = f"C:{method_bpms_now['chrom']:.1f}" if method_bpms_now['chrom'] else "C:--"
                                g_str = f"G:{method_bpms_now['green']:.1f}" if method_bpms_now['green'] else "G:--"
                                self.status_text = f"Tracking: {p_str} | {c_str} | {g_str} BPM"
                            else:
                                self.status_text = f"Tracking ({current_method.upper()}): {est_bpm:.1f} BPM"
                        else:
                            self.status_text = "Buffering skin pulse..."
                    else:
                        self.status_text = "Searching for face..."

                    # Live metrics against ground truth for all methods
                    if self.gt_times and self.gt_hr:
                        for m in ["pos", "chrom", "green"]:
                            if len(self.pred_histories[m]) > 0:
                                self.all_metrics[m] = compute_live_metrics(
                                    self.pred_histories[m],
                                    self.gt_times,
                                    self.gt_hr,
                                    skip_seconds=15.0
                                )
                        self.live_metrics = self.all_metrics.get(active_key, self.all_metrics["pos"])

                # Draw minimal HUD on video frame
                self._draw_overlay_badge(display_frame, est_bpm, sqi_val, has_face, method=current_method, method_bpms=method_bpms_now)

                # Encode frame to JPEG
                _, buffer = cv.imencode(".jpg", display_frame, [cv.IMWRITE_JPEG_QUALITY, 80])
                jpeg_bytes = buffer.tobytes()

                with self.lock:
                    self.latest_jpeg = jpeg_bytes

                # Throttle playback to match native video framerate (skipped during fast forward)
                if not VideoProcessing.is_fast_forward():
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

    def _draw_overlay_badge(
        self,
        frame: np.ndarray,
        bpm: Optional[float],
        sqi: int,
        has_face: bool,
        method: str = "pos",
        method_bpms: Optional[Dict[str, Optional[float]]] = None
    ):
        """Draws clean HUD status overlay in the corner of the video frame."""
        if method == "all" and has_face and method_bpms:
            # Wide badge for multi-method display
            cv.rectangle(frame, (10, 10), (320, 52), (15, 23, 42), -1)
            cv.rectangle(frame, (10, 10), (320, 52), (51, 65, 85), 1)
            cv.circle(frame, (25, 31), 6, (0, 230, 115), -1)

            p_val = f"{method_bpms.get('pos'):.0f}" if method_bpms.get('pos') else "--"
            c_val = f"{method_bpms.get('chrom'):.0f}" if method_bpms.get('chrom') else "--"
            g_val = f"{method_bpms.get('green'):.0f}" if method_bpms.get('green') else "--"

            txt = f"POS:{p_val}  CHM:{c_val}  GRN:{g_val}"
            cv.putText(frame, txt, (40, 36), cv.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv.LINE_AA)
        else:
            # Single method badge
            cv.rectangle(frame, (10, 10), (235, 52), (15, 23, 42), -1)
            cv.rectangle(frame, (10, 10), (235, 52), (51, 65, 85), 1)

            if has_face:
                status_color = (0, 230, 115)  # neon green
                m_label = method.upper() if method != "all" else "POS"
                status_str = f"{m_label}: {bpm:.1f} BPM" if bpm else f"{m_label}: BUFFERING..."
            else:
                status_color = (80, 80, 240)  # red/amber
                status_str = "SEARCHING FACE"

            cv.circle(frame, (25, 31), 6, status_color, -1)
            cv.putText(frame, status_str, (40, 37), cv.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv.LINE_AA)

