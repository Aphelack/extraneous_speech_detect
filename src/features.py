"""Cross-channel and acoustic feature extraction for extraneous speech detection.

Extracts on a 10 ms grid (80 samples @ 8 kHz):
1. Interchannel Level Difference (ILD): relative dB difference between channels.
2. Short-time Generalized Cross-Correlation with Phase Transform (GCC-PHAT):
   Searches for counterpart acoustic leakage in the delay window [20, 300] ms.
   Returns max correlation peak and estimated delay tau_hat.
3. Normalized NLMS Adaptive Filter Residual Ratio:
   Measures proportion of energy in track A that can be cancelled by an adaptive
   filter driven by track B (and vice-versa).
4. Spectral Magnitude-Squared Coherence.
5. Single-channel voice acoustic features (Spectral Centroid, High/Low energy ratio,
   Spectral Flatness) to distinguish distant reverberant third speaker from owner.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal

from src.config import (
    FRAME_DURATION_SEC,
    FRAME_SAMPLES,
    LEAK_DELAY_MAX_SAMPLES,
    LEAK_DELAY_MIN_SAMPLES,
    SAMPLE_RATE,
)


def compute_nlms_residual(
    target: np.ndarray,
    reference: np.ndarray,
    filter_length: int = 64,
    step_size: float = 0.25,
    stride: int = 40,
    eps: float = 1e-6,
) -> np.ndarray:
    """Run normalized least mean squares (NLMS) filter to predict target from reference.

    Parameters
    ----------
    target : np.ndarray
        Signal containing potential echo/leak (e.g. Track A).
    reference : np.ndarray
        Source signal causing leak (e.g. Track B).
    filter_length : int
        Number of taps in adaptive FIR filter.
    step_size : float
        Normalized adaptation step size mu (0 < mu <= 1).
    eps : float
        Regularization term for power normalization.

    Returns
    -------
    residual : np.ndarray
        Error signal e[n] = target[n] - y_hat[n].
    """
    n = len(target)
    ref_len = len(reference)
    min_len = min(n, ref_len)
    if min_len < filter_length:
        return target.copy()

    # Fast block-vectorized or sliding NLMS in numpy
    # For speed on 1 core, downsample or use block LMS if length is large
    # Block size of 80 samples (10 ms)
    residual = np.zeros(min_len, dtype=np.float32)
    w = np.zeros(filter_length, dtype=np.float32)

    # Pad reference with filter_length - 1 zeros
    ref_padded = np.pad(reference[:min_len], (filter_length - 1, 0), mode="constant")
    from numpy.lib.stride_tricks import sliding_window_view

    # Shape: (min_len, filter_length)
    ref_windows = sliding_window_view(ref_padded, window_shape=filter_length)

    # Run adaptation with block updates (e.g. stride 40 = 5 ms)
    for i in range(0, min_len, stride):
        end_idx = min(min_len, i + stride)
        x_block = ref_windows[i:end_idx]  # (block, L)
        y_hat = x_block @ w
        e_block = target[i:end_idx] - y_hat
        residual[i:end_idx] = e_block

        # Update weights using average gradient over block
        norm = np.sum(x_block ** 2) + eps
        grad = np.sum(x_block * e_block[:, None], axis=0)
        w += (step_size / norm) * grad

    if min_len < n:
        residual = np.pad(residual, (0, n - min_len), mode="constant")

    return residual.astype(np.float32)


def compute_gcc_phat_peak(
    x1: np.ndarray,
    x2: np.ndarray,
    min_delay_samples: int = LEAK_DELAY_MIN_SAMPLES,
    max_delay_samples: int = LEAK_DELAY_MAX_SAMPLES,
    n_fft: int = 1024,
    eps: float = 1e-10,
) -> Tuple[float, float]:
    """Compute GCC-PHAT peak and estimated delay between two signal frames.

    Returns
    -------
    peak_val : float (in [0, 1])
    est_delay_sec : float
    """
    if len(x1) < 64 or len(x2) < 64:
        return 0.0, 0.0

    X1 = np.fft.rfft(x1, n=n_fft)
    X2 = np.fft.rfft(x2, n=n_fft)

    cross = X1 * np.conj(X2)
    denom = np.abs(cross) + eps
    gcc_freq = cross / denom

    r = np.fft.irfft(gcc_freq, n=n_fft)
    # Circular shift to align center at 0
    half = n_fft // 2
    r_shifted = np.roll(r, half)

    # Search window for positive delay: [half + min_delay, half + max_delay]
    idx_min = half + min_delay_samples
    idx_max = min(n_fft, half + max_delay_samples)

    if idx_min >= idx_max or idx_max > len(r_shifted):
        return 0.0, 0.0

    window = r_shifted[idx_min:idx_max]
    peak_idx = int(np.argmax(np.abs(window)))
    peak_val = float(np.abs(window[peak_idx]))
    delay_samples = min_delay_samples + peak_idx
    delay_sec = delay_samples / SAMPLE_RATE

    return peak_val, delay_sec


class FeatureExtractor:
    """Extracts aligned multi-feature frame matrix for Track A and Track B."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_sec: float = FRAME_DURATION_SEC,
        frame_samples: int = FRAME_SAMPLES,
    ):
        self.sample_rate = sample_rate
        self.frame_sec = frame_sec
        self.frame_samples = frame_samples

    def extract_features(
        self,
        audio: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Extract multi-dimensional feature frames for Track A and Track B.

        Parameters
        ----------
        audio : np.ndarray
            Stereo audio of shape (2, num_samples).

        Returns
        -------
        dict with keys:
            "A": np.ndarray of shape (num_frames, num_features)
            "B": np.ndarray of shape (num_frames, num_features)
            "feature_names": list of str
        """
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected shape (2, N), got {audio.shape}")

        sig_a = audio[0]
        sig_b = audio[1]
        n_samples = min(len(sig_a), len(sig_b))
        sig_a = sig_a[:n_samples]
        sig_b = sig_b[:n_samples]

        hop = self.frame_samples
        win_size = hop * 3  # 240 samples (30 ms) analysis window with 10 ms hop
        num_frames = int(np.ceil(n_samples / hop))

        # Pad signals for window centering
        pad_len = win_size
        padded_a = np.pad(sig_a, (0, pad_len), mode="constant")
        padded_b = np.pad(sig_b, (0, pad_len), mode="constant")

        # 1. NLMS error signals (B cancelling A, and A cancelling B)
        res_a = compute_nlms_residual(target=padded_a, reference=padded_b, filter_length=64, stride=40)
        res_b = compute_nlms_residual(target=padded_b, reference=padded_a, filter_length=64, stride=40)

        # STFT frames for spectral features
        from numpy.lib.stride_tricks import sliding_window_view

        win = np.hanning(win_size).astype(np.float32)
        win_norm = np.sum(win ** 2)

        windows_a = sliding_window_view(padded_a[: num_frames * hop + win_size], window_shape=win_size)[::hop][:num_frames]
        windows_b = sliding_window_view(padded_b[: num_frames * hop + win_size], window_shape=win_size)[::hop][:num_frames]
        windows_res_a = sliding_window_view(res_a[: num_frames * hop + win_size], window_shape=win_size)[::hop][:num_frames]
        windows_res_b = sliding_window_view(res_b[: num_frames * hop + win_size], window_shape=win_size)[::hop][:num_frames]

        # RMS and dB energies
        eps = 1e-10
        energy_a = np.mean(windows_a ** 2, axis=1)
        energy_b = np.mean(windows_b ** 2, axis=1)
        db_a = 10.0 * np.log10(np.maximum(energy_a, eps))
        db_b = 10.0 * np.log10(np.maximum(energy_b, eps))

        ild_a = db_a - db_b
        ild_b = db_b - db_a

        # NLMS residual energy ratio: e^2 / x^2
        energy_res_a = np.mean(windows_res_a ** 2, axis=1)
        energy_res_b = np.mean(windows_res_b ** 2, axis=1)
        nlms_ratio_a = energy_res_a / np.maximum(energy_a, eps)
        nlms_ratio_b = energy_res_b / np.maximum(energy_b, eps)

        # Spectral analysis: FFT
        n_fft = 256
        spec_a = np.abs(np.fft.rfft(windows_a * win, n=n_fft, axis=1))
        spec_b = np.abs(np.fft.rfft(windows_b * win, n=n_fft, axis=1))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        # Spectral centroid
        spec_sum_a = np.sum(spec_a, axis=1) + eps
        centroid_a = np.sum(spec_a * freqs, axis=1) / spec_sum_a
        spec_sum_b = np.sum(spec_b, axis=1) + eps
        centroid_b = np.sum(spec_b * freqs, axis=1) / spec_sum_b

        # High/Low frequency band energy ratio (split at 1500 Hz)
        low_band = freqs < 1500.0
        high_band = (freqs >= 1500.0) & (freqs <= 3400.0)
        hl_ratio_a = np.sum(spec_a[:, high_band], axis=1) / (np.sum(spec_a[:, low_band], axis=1) + eps)
        hl_ratio_b = np.sum(spec_b[:, high_band], axis=1) / (np.sum(spec_b[:, low_band], axis=1) + eps)

        # Spectral flatness (geometric mean / arithmetic mean)
        geom_mean_a = np.exp(np.mean(np.log(np.maximum(spec_a, eps)), axis=1))
        flatness_a = geom_mean_a / (np.mean(spec_a, axis=1) + eps)

        geom_mean_b = np.exp(np.mean(np.log(np.maximum(spec_b, eps)), axis=1))
        flatness_b = geom_mean_b / (np.mean(spec_b, axis=1) + eps)

        # GCC-PHAT peak in history window (downsampled check every 2 frames for speed)
        gcc_peaks_a = np.zeros(num_frames, dtype=np.float32)
        gcc_delays_a = np.zeros(num_frames, dtype=np.float32)
        gcc_peaks_b = np.zeros(num_frames, dtype=np.float32)
        gcc_delays_b = np.zeros(num_frames, dtype=np.float32)

        # Step by 4 frames and fill blocks for high RTF throughput
        for f in range(0, num_frames, 4):
            if energy_a[f] > 1e-5 or energy_b[f] > 1e-5:
                pk_a, d_a = compute_gcc_phat_peak(windows_a[f], windows_b[f], n_fft=512)
                end_f = min(num_frames, f + 4)
                gcc_peaks_a[f:end_f] = pk_a
                gcc_delays_a[f:end_f] = d_a

                pk_b, d_b = compute_gcc_phat_peak(windows_b[f], windows_a[f], n_fft=512)
                gcc_peaks_b[f:end_f] = pk_b
                gcc_delays_b[f:end_f] = d_b

        feature_names = [
            "db_energy",
            "ild",
            "nlms_ratio",
            "gcc_peak",
            "gcc_delay",
            "centroid",
            "hl_ratio",
            "flatness",
        ]

        feats_a = np.column_stack([
            db_a,
            ild_a,
            nlms_ratio_a,
            gcc_peaks_a,
            gcc_delays_a,
            centroid_a,
            hl_ratio_a,
            flatness_a,
        ]).astype(np.float32)

        feats_b = np.column_stack([
            db_b,
            ild_b,
            nlms_ratio_b,
            gcc_peaks_b,
            gcc_delays_b,
            centroid_b,
            hl_ratio_b,
            flatness_b,
        ]).astype(np.float32)

        return {
            "A": feats_a,
            "B": feats_b,
            "feature_names": feature_names,
        }
