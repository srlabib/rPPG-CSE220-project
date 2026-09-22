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
    DEFAULT_FPS,
    NUM_PATCHES,
    TOP_K_PATCHES
)
from signal_processing import (
    execute_pos_algorithm,
    butter_bandpass_filter,
    calculate_bpm,
    calculate_snr,
    OverlapAddProcessor
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
        self.patch_buffers = [deque(maxlen=self.window_size) for _ in range(NUM_PATCHES)]
        self.pulse_buffer = deque(maxlen=self.pulse_buffer_max_len)
        self.ola_processor = OverlapAddProcessor(self.window_size)

        # State
        self.current_bpm: float = 0.0
        self.filtered_signal: np.ndarray = np.array([], dtype=np.float32)
        self.active_patch_indices: List[int] = []

    def set_fps(self, fps: float):
        """Update FPS and resize buffers accordingly."""
        if fps > 0:
            self.fps = fps
            self.window_size = max(int(POS_WINDOW_TIME_SEC * self.fps), 10)
            
            # Recreate buffers with new maxlen
            for i in range(NUM_PATCHES):
                old_items = list(self.patch_buffers[i])
                self.patch_buffers[i] = deque(old_items, maxlen=self.window_size)
            
            self.ola_processor = OverlapAddProcessor(self.window_size)

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

        # Compute POS when the sliding window is full
        if len(self.patch_buffers[0]) == self.window_size:
            candidate_pulses = []
            candidate_snrs = []

            # 1. Process all patches independently
            for i in range(NUM_PATCHES):
                temporal_matrix = np.array(self.patch_buffers[i], dtype=np.float32).T
                pulse = execute_pos_algorithm(temporal_matrix)
                snr = calculate_snr(pulse, self.fps)
                candidate_pulses.append(pulse)
                candidate_snrs.append(snr)

            # 2. Identify the Top K performing patches
            candidate_snrs = np.array(candidate_snrs)
            # Get indices of the patches with the highest SNR
            top_k_indices = np.argsort(candidate_snrs)[-TOP_K_PATCHES:]
            self.active_patch_indices = top_k_indices.tolist()

            # 3. Fuse ONLY the top patches
            fused_window = np.zeros(self.window_size, dtype=np.float32)
            total_top_snr = np.sum(candidate_snrs[top_k_indices])

            for idx in top_k_indices:
                weight = candidate_snrs[idx] / (total_top_snr + 1e-8)
                fused_window += candidate_pulses[idx] * weight

            # 4. Pass through Overlap-Add accumulator
            final_point = self.ola_processor.process_window(fused_window)
            self.pulse_buffer.append(final_point)

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
        for buffer in self.patch_buffers:
            buffer.clear()
        self.pulse_buffer.clear()
        self.ola_processor = OverlapAddProcessor(self.window_size)
        self.current_bpm = 0.0
        self.filtered_signal = np.array([], dtype=np.float32)
        self.active_patch_indices = []
