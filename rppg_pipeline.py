"""
Pipeline orchestrating color buffering, POS algorithm, filtering, and heart rate calculation.
"""

from collections import deque
from typing import Tuple, Optional, Sequence
import numpy as np

from config import (
    POS_WINDOW_TIME_SEC,
    PULSE_BUFFER_MAX_LEN,
    MIN_SAMPLES_FOR_BPM,
    DEFAULT_FPS
)
from signal_processing import (
    execute_pos_algorithm,
    butter_bandpass_filter,
    calculate_bpm
)


class RPPGPipeline:
    """
    Manages sliding RGB window buffer, applies the POS projection,
    accumulates pulse wave points, filters noise, and computes heart rate (BPM).
    """

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        window_time_sec: float = POS_WINDOW_TIME_SEC,
        pulse_buffer_max_len: int = PULSE_BUFFER_MAX_LEN,
        min_samples_for_bpm: int = MIN_SAMPLES_FOR_BPM
    ):
        self.fps = fps if fps > 0 else DEFAULT_FPS
        self.window_size = max(int(window_time_sec * self.fps), 10)
        self.pulse_buffer_max_len = pulse_buffer_max_len
        self.min_samples_for_bpm = min_samples_for_bpm

        # Buffers
        self.rgb_buffer = deque(maxlen=self.window_size)
        self.pulse_buffer = deque(maxlen=self.pulse_buffer_max_len)

        # State
        self.current_bpm: float = 0.0
        self.filtered_signal: np.ndarray = np.array([], dtype=np.float32)

    def set_fps(self, fps: float):
        """Update FPS and resize RGB buffer window accordingly."""
        if fps > 0:
            self.fps = fps
            self.window_size = max(int(POS_WINDOW_TIME_SEC * self.fps), 10)
            # Recreate buffer with new maxlen preserving recent items
            old_items = list(self.rgb_buffer)
            self.rgb_buffer = deque(old_items, maxlen=self.window_size)

    def update(self, mean_rgb: Sequence[float]) -> Tuple[np.ndarray, Optional[float], bool]:
        """
        Ingests a new frame's mean RGB color vector and updates pulse estimates.

        Args:
            mean_rgb: Sequence of length 3 containing (R, G, B) values.

        Returns:
            Tuple of:
                - filtered_signal: 1D numpy array of current bandpass filtered pulse
                - current_bpm: Estimated BPM (float) or None if still warming up
                - is_warming_up: Boolean indicating if pipeline needs more samples
        """
        self.rgb_buffer.append(mean_rgb)

        # Compute POS when the sliding window is full
        if len(self.rgb_buffer) == self.window_size:
            temporal_matrix = np.array(self.rgb_buffer, dtype=np.float32).T
            pulse_window = execute_pos_algorithm(temporal_matrix)
            newest_pulse_point = float(pulse_window[-1])
            self.pulse_buffer.append(newest_pulse_point)

        # Filter the accumulated master pulse signal
        raw_array = np.asarray(self.pulse_buffer, dtype=np.float32)
        self.filtered_signal = butter_bandpass_filter(raw_array, self.fps)

        is_warming_up = len(self.filtered_signal) < self.min_samples_for_bpm

        if not is_warming_up:
            self.current_bpm = calculate_bpm(self.filtered_signal, self.fps)
            return self.filtered_signal, self.current_bpm, False
        else:
            return self.filtered_signal, None, True

    def reset(self):
        """Clears all internal buffers."""
        self.rgb_buffer.clear()
        self.pulse_buffer.clear()
        self.current_bpm = 0.0
        self.filtered_signal = np.array([], dtype=np.float32)
