"""
Signal processing algorithms for rPPG pulse extraction and heart rate estimation.
"""

from typing import Union, Sequence
import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks

from config import (
    MIN_FREQ_HZ,
    MAX_FREQ_HZ,
    FILTER_ORDER,
    POS_EPSILON,
    FFT_NFFT
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


def execute_pos_algorithm_batch(
    temporal_tensor: np.ndarray,
    eps: float = POS_EPSILON
) -> np.ndarray:
    """
    Executes the Plane-Orthogonal-to-Skin (POS) algorithm across multiple patches simultaneously.

    Args:
        temporal_tensor: 3D array of shape (K, 3, N) -> K patches, 3 channels (R,G,B), N frames.
        eps: Numerical stability constant.

    Returns:
        h_centered: 2D array of shape (K, N) of extracted, mean-centered pulse signals.
    """
    r = temporal_tensor[:, 0, :]
    g = temporal_tensor[:, 1, :]
    b = temporal_tensor[:, 2, :]

    r_n = r / (np.mean(r, axis=1, keepdims=True) + eps)
    g_n = g / (np.mean(g, axis=1, keepdims=True) + eps)
    b_n = b / (np.mean(b, axis=1, keepdims=True) + eps)

    s1 = g_n - b_n
    s2 = -2.0 * r_n + g_n + b_n

    sigma_s1 = np.std(s1, axis=1, keepdims=True)
    sigma_s2 = np.std(s2, axis=1, keepdims=True)
    alpha = sigma_s1 / (sigma_s2 + eps)

    h = s1 + alpha * s2
    h_centered = h - np.mean(h, axis=1, keepdims=True)
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
    max_freq: float = MAX_FREQ_HZ,
    nfft: int = FFT_NFFT
) -> float:
    """
    Calculates the heart rate (BPM) from a filtered rPPG signal using zero-padded FFT.
    Applies linear detrending, Hann windowing, zero-padding, and peak picking.
    """
    sig = np.asarray(filtered_signal, dtype=np.float32)
    n = len(sig)
    if n == 0 or fps <= 0:
        return 0.0

    # 1. Fast linear detrending
    t = np.arange(n, dtype=np.float32)
    t_centered = t - 0.5 * (n - 1)
    denom = np.sum(t_centered ** 2) + 1e-8
    sig_mean = np.mean(sig)
    sig_centered = sig - sig_mean
    slope = np.sum(sig_centered * t_centered) / denom
    sig_clean = sig_centered - slope * t_centered

    # 2. Windowed FFT
    window = np.hanning(n).astype(np.float32)
    actual_nfft = max(nfft, n)
    freqs = np.fft.rfftfreq(actual_nfft, d=1.0 / fps)
    fft_vals = np.fft.rfft(sig_clean * window, n=actual_nfft)
    psd = np.abs(fft_vals) ** 2

    # 3. Search restricted to valid heart rate frequencies
    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(valid_mask):
        return 0.0

    valid_freqs = freqs[valid_mask]
    valid_psd = psd[valid_mask]

    # 4. Find dominant peak
    peaks, _ = find_peaks(valid_psd)
    if len(peaks) > 0:
        best_peak = peaks[np.argmax(valid_psd[peaks])]
        heart_rate_hz = float(valid_freqs[best_peak])
        peak_power = float(valid_psd[best_peak])
    else:
        peak_idx = np.argmax(valid_psd)
        heart_rate_hz = float(valid_freqs[peak_idx])
        peak_power = float(valid_psd[peak_idx])

    # 5. Sub-harmonic verification (prevents 2x harmonic doubling, e.g. 68 BPM -> 136-140 BPM)
    # If dominant peak is high (>= 1.6 Hz / 96 BPM), check if true fundamental exists at f/2
    if heart_rate_hz >= 1.6:
        sub_freq = heart_rate_hz / 2.0
        if sub_freq >= min_freq:
            sub_mask = np.abs(valid_freqs - sub_freq) <= 0.15
            if np.any(sub_mask):
                sub_max_psd = float(np.max(valid_psd[sub_mask]))
                # If significant spectral power exists at f/2 (>= 35% of the harmonic peak),
                # the lower frequency is the true physiological fundamental heart rate
                if sub_max_psd >= 0.35 * peak_power:
                    sub_idx = np.where(sub_mask)[0][np.argmax(valid_psd[sub_mask])]
                    heart_rate_hz = float(valid_freqs[sub_idx])

    return heart_rate_hz * 60.0


def calculate_snr_batch(
    pulses: np.ndarray,
    fps: float,
    min_freq: float = MIN_FREQ_HZ,
    max_freq: float = MAX_FREQ_HZ,
    nfft: int = FFT_NFFT
) -> np.ndarray:
    """
    Computes SNR across multiple pulse waves (shape: K patches x N frames)
    simultaneously in a single vectorized FFT operation.
    """
    k, n = pulses.shape
    if n == 0 or fps <= 0:
        return np.zeros(k, dtype=np.float32)

    # 1. Vectorized linear detrend
    t = np.arange(n, dtype=np.float32)
    t_centered = t - 0.5 * (n - 1)
    denom = np.sum(t_centered ** 2) + 1e-8

    p_mean = np.mean(pulses, axis=1, keepdims=True)
    p_centered = pulses - p_mean
    slopes = np.sum(p_centered * t_centered, axis=1, keepdims=True) / denom
    clean = p_centered - slopes * t_centered

    # 2. Windowed FFT across all patches at once
    window = np.hanning(n).astype(np.float32)
    windowed = clean * window[None, :]

    actual_nfft = max(nfft, n)
    freqs = np.fft.rfftfreq(actual_nfft, d=1.0 / fps)
    fft_vals = np.fft.rfft(windowed, n=actual_nfft, axis=1)
    psd = np.abs(fft_vals) ** 2

    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(valid_mask):
        return np.zeros(k, dtype=np.float32)

    valid_freqs = freqs[valid_mask]
    valid_psd = psd[:, valid_mask]

    # 3. Find dominant peak for each patch
    peak_indices = np.argmax(valid_psd, axis=1)
    peak_freqs = valid_freqs[peak_indices]

    # 4. Signal band: peak +/- 0.1 Hz
    freq_diffs = np.abs(valid_freqs[None, :] - peak_freqs[:, None])
    signal_band = freq_diffs <= 0.1
    noise_band = ~signal_band

    signal_power = np.sum(valid_psd * signal_band, axis=1)
    noise_power = np.sum(valid_psd * noise_band, axis=1)

    snrs = signal_power / (noise_power + 1e-8)
    return snrs.astype(np.float32)


def calculate_snr(
    pulse_signal: Union[np.ndarray, Sequence[float]],
    fps: float,
    min_freq: float = MIN_FREQ_HZ,
    max_freq: float = MAX_FREQ_HZ,
    nfft: int = FFT_NFFT
) -> float:
    """
    Calculates SNR for a single pulse wave.
    """
    sig = np.asarray(pulse_signal, dtype=np.float32)
    if len(sig) == 0:
        return 0.0
    res = calculate_snr_batch(sig[None, :], fps, min_freq, max_freq, nfft)
    return float(res[0])




class OverlapAddProcessor:
    """
    Maintains an overlapping window accumulator to reconstruct a continuous 
    waveform from independently processed sliding windows.
    """
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer = np.zeros(window_size, dtype=np.float32)
        # Hanning window reduces boundary artifacts during overlap-add
        self.hanning_window = np.hanning(window_size).astype(np.float32)

    def process_window(self, fused_window: np.ndarray) -> float:
        """
        Takes a new fused window, applies a tapering function, accumulates it,
        and extracts the oldest (fully accumulated) point as the final output.
        """
        if len(fused_window) != self.window_size:
            return 0.0

        # Apply tapering to reduce edge discontinuities
        weighted_window = fused_window * self.hanning_window
        
        # Accumulate
        self.buffer += weighted_window
        
        # The oldest point (index 0) has received all possible overlap additions
        final_point = self.buffer[0]
        
        # Shift buffer left by 1 and zero out the newest slot
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = 0.0
        
        return float(final_point)

