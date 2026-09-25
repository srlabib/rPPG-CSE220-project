"""
Pipeline orchestrating color buffering, POS algorithm, filtering, and heart rate calculation.
"""

from collections import deque
from typing import Tuple, Optional, Sequence, List
import numpy as np

from config import (
    POS_WINDOW_TIME_SEC,
    PULSE_BUFFER_MAX_LEN,
    MIN_SAMPLES_FOR_BPM,
    DEFAULT_FPS,
    NUM_PATCHES,
    TOP_K_PATCHES
)
from signal_processing import (
    execute_pos_algorithm,
    execute_pos_algorithm_batch,
    execute_chrom_algorithm_batch,
    execute_green_algorithm_batch,
    extract_pulse_batch,
    butter_bandpass_filter,
    calculate_bpm,
    calculate_snr,
    calculate_snr_batch,
    OverlapAddProcessor
)


class RPPGPipeline:
    """
    Manages sliding RGB window buffer, applies rPPG extraction (POS, CHROM, or GREEN),
    accumulates pulse wave points, filters noise, and computes heart rate (BPM).
    """

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        method: str = "pos",
        window_time_sec: float = POS_WINDOW_TIME_SEC,
        pulse_buffer_max_len: int = PULSE_BUFFER_MAX_LEN,
        min_samples_for_bpm: int = MIN_SAMPLES_FOR_BPM
    ):
        self.fps = fps if fps > 0 else DEFAULT_FPS
        self.method = (method or "pos").lower().strip()
        self.window_size = max(int(window_time_sec * self.fps), 10)
        self.pulse_buffer_max_len = pulse_buffer_max_len
        self.min_samples_for_bpm = min_samples_for_bpm

        # Buffers
        self.patch_buffers = [deque(maxlen=self.window_size) for _ in range(NUM_PATCHES)]
        self.pulse_buffer = deque(maxlen=self.pulse_buffer_max_len)
        self.ola_processor = OverlapAddProcessor(self.window_size)

        # Temporal BPM smoothing state (moving median + exponential moving average)
        self.bpm_history_len = max(int(2.0 * self.fps), 30)
        self.bpm_history = deque(maxlen=self.bpm_history_len)
        self.smoothed_bpm: float = 0.0

        # State
        self.current_bpm: float = 0.0
        self.filtered_signal: np.ndarray = np.array([], dtype=np.float32)
        self.active_patch_indices: List[int] = []

    def set_fps(self, fps: float):
        """Update FPS and resize sliding buffers only when window size changes."""
        if fps <= 0:
            return
        self.fps = fps
        new_window_size = max(int(POS_WINDOW_TIME_SEC * self.fps), 10)
        if new_window_size != self.window_size:
            self.window_size = new_window_size
            for i in range(NUM_PATCHES):
                old_items = list(self.patch_buffers[i])
                self.patch_buffers[i] = deque(old_items[-self.window_size:], maxlen=self.window_size)
            self.ola_processor = OverlapAddProcessor(self.window_size)

        new_history_len = max(int(2.0 * self.fps), 30)
        if new_history_len != self.bpm_history_len:
            self.bpm_history_len = new_history_len
            self.bpm_history = deque(self.bpm_history, maxlen=self.bpm_history_len)

    def update(self, mean_rgbs: Sequence[Sequence[float]]) -> Tuple[np.ndarray, Optional[float], bool]:
        """
        Ingests a new frame's 12 patch mean RGB vectors and updates pulse estimates.

        Args:
            mean_rgbs: List of 12 Sequences of (R, G, B) values.

        Returns:
            Tuple of:
                - filtered_signal: 1D numpy array of current bandpass filtered pulse
                - current_bpm: Estimated BPM (float) or None if still warming up
                - is_warming_up: Boolean indicating if pipeline needs more samples
        """
        if len(mean_rgbs) != NUM_PATCHES:
            return self.filtered_signal, self.current_bpm, True

        for i in range(NUM_PATCHES):
            self.patch_buffers[i].append(mean_rgbs[i])

        # Compute rPPG pulse when the sliding window is full
        if len(self.patch_buffers[0]) == self.window_size:
            # 1. Stack all patches: shape (12, window_size, 3) -> transpose to (12, 3, window_size)
            temporal_tensor = np.array(self.patch_buffers, dtype=np.float32).transpose(0, 2, 1)

            # 2. Vectorized rPPG pulse extraction across all 12 patches using active algorithm
            candidate_pulses = extract_pulse_batch(temporal_tensor, method=self.method)

            # 3. Vectorized SNR across all 12 patches simultaneously
            candidate_snrs = calculate_snr_batch(candidate_pulses, self.fps)

            # 4. Identify the Top K performing patches
            top_k_indices = np.argsort(candidate_snrs)[-TOP_K_PATCHES:]
            self.active_patch_indices = top_k_indices.tolist()

            # 5. Fuse ONLY the top patches
            top_pulses = candidate_pulses[top_k_indices]
            top_snrs = candidate_snrs[top_k_indices]
            weights = (top_snrs / (np.sum(top_snrs) + 1e-8))[:, None]
            fused_window = np.sum(top_pulses * weights, axis=0)

            # 6. Pass through Overlap-Add accumulator
            final_point = self.ola_processor.process_window(fused_window)
            self.pulse_buffer.append(final_point)

        # Filter the accumulated master pulse signal
        raw_array = np.asarray(self.pulse_buffer, dtype=np.float32)
        self.filtered_signal = butter_bandpass_filter(raw_array, self.fps)

        is_warming_up = len(self.filtered_signal) < self.min_samples_for_bpm

        if not is_warming_up:
            raw_bpm = calculate_bpm(self.filtered_signal, self.fps)
            if raw_bpm > 0:
                self.bpm_history.append(raw_bpm)
                # 1. Moving median over recent window rejects isolated spikes/glitches
                median_bpm = float(np.median(self.bpm_history))

                # 2. Exponential Moving Average (EMA) provides smooth, flicker-free transitions
                if self.smoothed_bpm <= 0:
                    self.smoothed_bpm = median_bpm
                else:
                    alpha = 0.15  # Responsive yet steady tracking
                    self.smoothed_bpm = alpha * median_bpm + (1.0 - alpha) * self.smoothed_bpm

                self.current_bpm = round(self.smoothed_bpm, 1)

            return self.filtered_signal, self.current_bpm, False
        else:
            return self.filtered_signal, None, True

    def set_method(self, method: str):
        """Switches active rPPG algorithm ('pos', 'chrom', or 'green') and resets state."""
        new_m = (method or "pos").lower().strip()
        if new_m != self.method:
            self.method = new_m
            self.reset()

    def reset(self):
        """Clears all internal buffers."""
        for buffer in self.patch_buffers:
            buffer.clear()
        self.pulse_buffer.clear()
        self.ola_processor = OverlapAddProcessor(self.window_size)
        self.bpm_history.clear()
        self.smoothed_bpm = 0.0
        self.current_bpm = 0.0
        self.filtered_signal = np.array([], dtype=np.float32)
        self.active_patch_indices = []


