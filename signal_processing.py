"""
Signal processing algorithms for rPPG pulse extraction and heart rate estimation.
"""

from typing import Union, Sequence
import numpy as np
from scipy.signal import butter, filtfilt, welch

from config import (
    MIN_FREQ_HZ,
    MAX_FREQ_HZ,
    FILTER_ORDER,
    POS_EPSILON
)


def execute_pos_algorithm(
    temporal_matrix: np.ndarray,
    eps: float = POS_EPSILON
) -> np.ndarray:
    """
    Executes the Plane-Orthogonal-to-Skin (POS) algorithm on a sliding window of RGB data.
    
    Reference:
        Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
        Algorithmic principles of remote PPG. IEEE TBME.

    Args:
        temporal_matrix: 2D array of shape (3, N) -> [R, G, B] over N frames.
        eps: Small epsilon to prevent division by zero in dark frames.

    Returns:
        h_centered: 1D extracted pulse wave (length N), mean-centered for this window.
    """
    # 1. Extract individual color channels
    r = temporal_matrix[0, :]
    g = temporal_matrix[1, :]
    b = temporal_matrix[2, :]

    # 2. Temporal Normalization (Divide each channel by its mean)
    r_n = r / (np.mean(r) + eps)
    g_n = g / (np.mean(g) + eps)
    b_n = b / (np.mean(b) + eps)

    # 3. Projection onto the orthogonal plane
    s1 = g_n - b_n
    s2 = -2.0 * r_n + g_n + b_n

    # 4. Alpha Tuning based on standard deviations
    sigma_s1 = np.std(s1)
    sigma_s2 = np.std(s2)
    alpha = sigma_s1 / (sigma_s2 + eps)

    # 5. Extract raw pulse signal
    h = s1 + alpha * s2

    # 6. Mean-centering (required for stitching overlapping windows)
    h_centered = h - np.mean(h)

    return h_centered


def butter_bandpass_filter(
    data: Union[np.ndarray, Sequence[float]],
    fps: float,
    lowcut: float = MIN_FREQ_HZ,
    highcut: float = MAX_FREQ_HZ,
    order: int = FILTER_ORDER
) -> np.ndarray:
    """
    Filters the raw POS signal using a zero-phase Butterworth bandpass filter.
    Restricts frequencies to human pulse range (default 0.7 Hz to 3.0 Hz / 42 to 180 BPM).

    Args:
        data: 1D pulse signal.
        fps: Sampling rate / frame rate of the video.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered 1D numpy array, or unmodified data array if insufficient samples.
    """
    data_arr = np.asarray(data, dtype=np.float32)

    # Safety check: Nyquist frequency must exceed highcut
    if fps <= 2.0 * highcut or len(data_arr) == 0:
        return data_arr

    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist

    # Ensure normalized frequencies stay within (0, 1)
    if not (0 < low < high < 1):
        return data_arr

    b, a = butter(order, [low, high], btype="band")

    # filtfilt requires sufficient samples for edge padding
    min_len = 3 * max(len(a), len(b))
    if len(data_arr) <= min_len:
        return data_arr

    # Zero-phase forward-backward digital filtering
    filtered_data = filtfilt(b, a, data_arr)
    return filtered_data


def calculate_bpm(
    filtered_signal: Union[np.ndarray, Sequence[float]],
    fps: float,
    min_freq: float = MIN_FREQ_HZ,
    max_freq: float = MAX_FREQ_HZ
) -> float:
    """
    Calculates the heart rate (BPM) from a filtered rPPG signal using Welch's method.

    Args:
        filtered_signal: 1D filtered rPPG signal.
        fps: Video frame rate.
        min_freq: Minimum physiological heart rate frequency in Hz (default 0.7 Hz).
        max_freq: Maximum physiological heart rate frequency in Hz (default 3.0 Hz).

    Returns:
        Heart rate in beats per minute (BPM), or 0.0 if not detectable.
    """
    sig = np.asarray(filtered_signal, dtype=np.float32)
    if len(sig) == 0 or fps <= 0:
        return 0.0

    # Power Spectral Density (PSD) via Welch's periodogram
    freqs, psd = welch(sig, fs=fps, nperseg=len(sig))

    # Restrict search to valid heart rate frequencies
    valid_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    if len(valid_indices) == 0:
        return 0.0

    valid_freqs = freqs[valid_indices]
    valid_psd = psd[valid_indices]

    # Dominant frequency has highest power
    peak_idx = np.argmax(valid_psd)
    heart_rate_hz = float(valid_freqs[peak_idx])

    # Convert Hz to BPM
    bpm = heart_rate_hz * 60.0
    return bpm
