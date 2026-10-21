"""Cross-channel and acoustic feature extraction for extraneous speech detection.

Extracts on a 10 ms grid (80 samples @ 8 kHz):

1. Interchannel Level Difference (ILD): relative dB difference between channels.
2. Short-time Generalized Cross-Correlation with Phase Transform (GCC-PHAT):
   the *current* 100 ms window of the local channel is correlated against the
   *history* of the counterpart channel covering the last 400 ms, and the peak
   is searched strictly on lags d in [20, 300] ms (160 .. 2400 samples).
   This is the only way to observe delays longer than the analysis window.
3. Delay-aligned NLMS adaptive filter residual ratio:
   the counterpart signal is first shifted by the per-frame delay estimate
   d_hat(t), then a 256-tap (32 ms) NLMS filter models the residual room
   response around the delay peak.
4. Spectral magnitude-squared coherence between the local spectrum and the
   delay-aligned counterpart spectrum.
5. Leak audibility features: local SNR above the noise floor, coherence and
   GCC peak combined into an explicit "audible leak" indicator.
6. Single-channel voice acoustic features (Spectral Centroid, High/Low energy
   ratio, Spectral Flatness) to distinguish a distant reverberant third
   speaker from the owner.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.fft import next_fast_len

from src.config import (
    FRAME_DURATION_SEC,
    FRAME_SAMPLES,
    LEAK_DELAY_MAX_SAMPLES,
    LEAK_DELAY_MIN_SAMPLES,
    SAMPLE_RATE,
)

# Analysis window used for the GCC-PHAT current frame (100 ms).
GCC_CURRENT_WINDOW_SAMPLES: int = 800
# NLMS filter length: 256 taps = 32 ms of room response around the delay peak.
NLMS_FILTER_LENGTH: int = 256
# Minimum local SNR (dB above the noise floor) for a leak to be audible at all.
LEAK_AUDIBILITY_SNR_DB: float = 3.0

FEATURE_NAMES: List[str] = [
    "db_energy",
    "ild",
    "nlms_ratio",
    "gcc_peak",
    "gcc_delay",
    "gcc_ncc",
    "centroid",
    "hl_ratio",
    "flatness",
    "counterpart_energy",
    "snr_local_db",
    "snr_counterpart_db",
    "coherence",
    "leak_audible",
    "third_score",
] + [f"local_mel_{i}" for i in range(12)] + [
    f"aligned_other_mel_{i}" for i in range(12)
] + [
    "anchor_cos_dist",
    "anchor_l1_low",
    "anchor_l1_high",
]

FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _shift_by_delay(x: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Return y[n] = x[n - delays[n]] using linear interpolation.

    ``delays`` is a per-sample array (float). Samples whose source index falls
    outside the signal are set to zero.
    """
    n = len(x)
    if n == 0:
        return x.copy()
    m = np.arange(n, dtype=np.float64) - np.asarray(delays, dtype=np.float64)
    i = np.floor(m).astype(np.int64)
    frac = m - i
    valid = (i >= 0) & (i < n - 1)
    out = np.zeros(n, dtype=np.float32)
    idx = np.where(valid)[0]
    if len(idx):
        i_val = i[idx]
        f_val = frac[idx].astype(np.float32)
        out[idx] = (1.0 - f_val) * x[i_val] + f_val * x[i_val + 1]
    return out


def compute_nlms_residual(
    target: np.ndarray,
    reference: np.ndarray,
    filter_length: int = NLMS_FILTER_LENGTH,
    step_size: float = 1.0,
    stride: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """Run normalized least mean squares (NLMS) filter to predict target from reference.

    The reference must already be delay-aligned with the target (see
    :func:`_shift_by_delay`); the adaptive filter then only has to model the
    short room response around the delay peak.

    Parameters
    ----------
    target : np.ndarray
        Signal containing potential echo/leak (e.g. Track A).
    reference : np.ndarray
        Delay-aligned source signal causing leak (e.g. Track B).
    filter_length : int
        Number of taps in adaptive FIR filter (256 taps = 32 ms @ 8 kHz).
    step_size : float
        Normalized adaptation step size mu (0 < mu <= 1).
    stride : int
        Block size in samples for weight updates (20 = 2.5 ms). With a 256-tap
        filter and step size 1.0 this converges within ~200 ms while keeping
        the single-core RTF budget.
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

    residual = np.zeros(min_len, dtype=np.float32)
    w = np.zeros(filter_length, dtype=np.float32)

    ref_padded = np.pad(reference[:min_len], (filter_length - 1, 0), mode="constant")
    from numpy.lib.stride_tricks import sliding_window_view

    ref_windows = sliding_window_view(ref_padded, window_shape=filter_length)

    for i in range(0, min_len, stride):
        end_idx = min(min_len, i + stride)
        x_block = ref_windows[i:end_idx]  # (block, L)
        y_hat = x_block @ w
        e_block = target[i:end_idx] - y_hat
        residual[i:end_idx] = e_block

        # Per-sample NLMS normalization: each tap update is normalized by the
        # energy of the input vector at that sample, not by the whole block.
        # Using the block energy would shrink the effective step by ~stride.
        row_energy = np.einsum("ij,ij->i", x_block, x_block) + eps
        grad = (x_block * (e_block / row_energy)[:, None]).sum(axis=0)
        w += (step_size / max(1, x_block.shape[0])) * grad

    if min_len < n:
        residual = np.pad(residual, (0, n - min_len), mode="constant")

    return residual.astype(np.float32)


def compute_gcc_phat_peak(
    x_current: np.ndarray,
    x_history: np.ndarray,
    min_delay_samples: int = LEAK_DELAY_MIN_SAMPLES,
    max_delay_samples: int = LEAK_DELAY_MAX_SAMPLES,
    n_fft: Optional[int] = None,
    eps: float = 1e-10,
) -> Tuple[float, float, float]:
    """GCC-PHAT peak, delay and normalized cross-correlation between two windows.

    ``x_current`` is the local channel window of length ``W`` starting at time
    ``t0``. ``x_history`` must be the counterpart channel slice covering
    ``[t0 + min_delay, t0 + W + max_delay)``, i.e. length
    ``max_delay - min_delay + W``.

    If the local channel is a delayed copy of the counterpart,
    ``local[n] = counterpart[n - d]``, then ``counterpart[t0 + d + n]`` aligns
    with ``local[t0 + n]``. The correlation is therefore evaluated for lags
    ``d`` in ``[min_delay_samples, max_delay_samples]`` only, so delays longer
    than the analysis window are observable.

    Returns
    -------
    peak_val : float
        GCC-PHAT peak magnitude (approximately in [0, 1]).
    est_delay_sec : float
        Estimated delay in seconds (0.0 if the search range is invalid).
    ncc : float
        Normalized cross-correlation coefficient at the estimated lag, in [-1, 1].
    """
    w_len = len(x_current)
    h_len = len(x_history)
    if w_len < 64 or h_len < w_len:
        return 0.0, 0.0, 0.0

    span = max_delay_samples - min_delay_samples
    if span <= 0:
        return 0.0, 0.0, 0.0

    # r[k] = sum_n x_current[n] * x_history[n + k], k in [0, span].
    # With x_history[j] = counterpart[t0 - max_delay + j] and
    # local[n] = counterpart[n - d], the match sits at k = max_delay - d,
    # so k=0 -> d=max_delay and k=span -> d=min_delay.
    if n_fft is None:
        n_fft = next_fast_len(w_len + h_len - 1)

    a = np.asarray(x_current, dtype=np.float64)
    h = np.asarray(x_history, dtype=np.float64)

    A = np.fft.rfft(a, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    # Correlation theorem: irfft(conj(A) * H)[k] = sum_n a[n] * h[n + k].
    cross = np.conj(A) * H

    # GCC-PHAT: whiten the cross-spectrum magnitude.
    gcc = np.fft.irfft(cross / (np.abs(cross) + eps), n=n_fft)

    # Plain (unnormalized) cross-correlation for the NCC value.
    raw = np.fft.irfft(cross, n=n_fft)

    k_max = min(span, len(gcc) - 1)
    if k_max < 0:
        return 0.0, 0.0, 0.0

    window = gcc[: k_max + 1]
    peak_idx = int(np.argmax(np.abs(window)))
    peak_val = float(np.abs(window[peak_idx]))

    delay_samples = max_delay_samples - peak_idx
    delay_sec = delay_samples / SAMPLE_RATE

    # Normalized cross-correlation at the estimated lag.
    raw_peak = float(raw[peak_idx])
    denom = float(np.linalg.norm(a) * np.linalg.norm(h)) + eps
    ncc = float(np.clip(raw_peak / denom, -1.0, 1.0))

    return peak_val, delay_sec, ncc


def get_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = 256,
    n_mels: int = 12,
    fmin: float = 300.0,
    fmax: float = 3400.0,
) -> np.ndarray:
    """Construct triangular Mel filterbank matrix in telephone band (300 - 3400 Hz)."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for b in range(left, center):
            fb[i, b] = (b - left) / max(1, center - left)
        for b in range(center, right):
            fb[i, b] = (right - b) / max(1, right - center)
    return fb


def extract_speaker_anchor_features(
    mel_matrix: np.ndarray,
    ild: np.ndarray,
    gcc_peak: np.ndarray,
    db_energy: np.ndarray,
) -> np.ndarray:
    """Extract speaker anchor similarity features to discriminate distant third speaker.

    1. Finds confident owner frames where ILD > 10 dB and GCC peak < 0.25 (min 20 frames,
       else falls back to top 15% loudest frames with positive ILD).
    2. Computes mean log-mel vector mu_own.
    3. Returns:
       - cosine distance: 1 - cos_sim(mel(t), mu_own)
       - L1 deviation in low-frequency bands (0..6)
       - L1 deviation in high-frequency bands (6..12)
    """
    clean_mask = (ild > 10.0) & (gcc_peak < 0.25) & (db_energy > -45.0)
    if np.sum(clean_mask) < 20:
        pos_mask = ild > 0.0
        if np.sum(pos_mask) >= 10:
            threshold = np.percentile(db_energy[pos_mask], 85)
            clean_mask = pos_mask & (db_energy >= threshold)
        else:
            clean_mask = np.ones(len(mel_matrix), dtype=bool)

    if np.sum(clean_mask) > 0:
        mu_own = np.mean(mel_matrix[clean_mask], axis=0)
    else:
        mu_own = np.mean(mel_matrix, axis=0)

    norm_mel = np.linalg.norm(mel_matrix, axis=1, keepdims=True) + 1e-8
    norm_mu = np.linalg.norm(mu_own) + 1e-8
    cos_sim = (mel_matrix @ mu_own) / (norm_mel.squeeze() * norm_mu)
    cos_dist = np.clip(1.0 - cos_sim, 0.0, 2.0)[:, None]

    l1_diff = np.abs(mel_matrix - mu_own)
    l1_low = np.mean(l1_diff[:, :6], axis=1, keepdims=True)
    l1_high = np.mean(l1_diff[:, 6:], axis=1, keepdims=True)

    return np.hstack([cos_dist, l1_low, l1_high]).astype(np.float32)


class FeatureExtractor:
    """Extracts aligned multi-feature frame matrix for Track A and Track B."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_sec: float = FRAME_DURATION_SEC,
        frame_samples: int = FRAME_SAMPLES,
        gcc_stride_frames: int = 4,
    ):
        self.sample_rate = sample_rate
        self.frame_sec = frame_sec
        self.frame_samples = frame_samples
        self.gcc_stride_frames = max(1, int(gcc_stride_frames))
        self.mel_fb = get_mel_filterbank(sr=sample_rate, n_fft=256, n_mels=12)
        self.gcc_window = GCC_CURRENT_WINDOW_SAMPLES
        self.min_delay = LEAK_DELAY_MIN_SAMPLES
        self.max_delay = LEAK_DELAY_MAX_SAMPLES

    # ------------------------------------------------------------------
    # Delay estimation
    # ------------------------------------------------------------------
    def _history_slice(self, sig: np.ndarray, t0: int) -> np.ndarray:
        """Counterpart history covering [t0 - max_delay, t0 + W - min_delay).

        A leak delayed by ``d`` samples satisfies ``local[n] = counterpart[n-d]``,
        so the counterpart content matching the local window at ``t0`` lies in
        the *past* of the counterpart signal.
        """
        hist_len = self.max_delay - self.min_delay + self.gcc_window
        hist = np.zeros(hist_len, dtype=np.float32)
        src_start = t0 - self.max_delay
        src_end = t0 + self.gcc_window - self.min_delay
        valid_start = max(0, src_start)
        valid_end = min(len(sig), src_end)
        if valid_end > valid_start:
            dst_start = valid_start - src_start
            count = valid_end - valid_start
            hist[dst_start : dst_start + count] = sig[valid_start:valid_end]
        return hist

    def _estimate_delay_profile(
        self,
        sig_local: np.ndarray,
        sig_counterpart: np.ndarray,
        num_frames: int,
        hop: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-frame delay (samples), GCC peak and NCC against the counterpart history."""
        if num_frames <= 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty

        sampled_frames: List[int] = []
        sampled_delays: List[float] = []
        sampled_peaks: List[float] = []
        sampled_nccs: List[float] = []

        for f in range(0, num_frames, self.gcc_stride_frames):
            t0 = f * hop
            current = sig_local[t0 : t0 + self.gcc_window]
            if len(current) < self.gcc_window:
                current = np.pad(current, (0, self.gcc_window - len(current)))
            history = self._history_slice(sig_counterpart, t0)

            peak, delay_sec, ncc = compute_gcc_phat_peak(
                current,
                history,
                min_delay_samples=self.min_delay,
                max_delay_samples=self.max_delay,
            )
            sampled_frames.append(f)
            sampled_delays.append(delay_sec * self.sample_rate)
            sampled_peaks.append(peak)
            sampled_nccs.append(ncc)

        frame_axis = np.arange(num_frames, dtype=np.float64)
        delays = np.interp(frame_axis, sampled_frames, sampled_delays).astype(np.float32)
        peaks = np.interp(frame_axis, sampled_frames, sampled_peaks).astype(np.float32)
        nccs = np.interp(frame_axis, sampled_frames, sampled_nccs).astype(np.float32)

        # Smooth the delay track to avoid per-frame jitter (5 frames = 50 ms).
        if num_frames >= 5:
            delays = signal.medfilt(delays, kernel_size=5).astype(np.float32)

        return delays, peaks, nccs

    def _expand_delay_to_samples(self, delays_frames: np.ndarray, num_samples: int, hop: int) -> np.ndarray:
        if len(delays_frames) == 0:
            return np.zeros(num_samples, dtype=np.float32)
        expanded = np.repeat(delays_frames, hop)[:num_samples]
        if len(expanded) < num_samples:
            expanded = np.pad(expanded, (0, num_samples - len(expanded)), mode="edge")
        return expanded.astype(np.float32)

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------
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

        # 1. Per-frame delay estimates (A vs B history, B vs A history).
        delay_a_frames, gcc_peaks_a, gcc_ncc_a = self._estimate_delay_profile(
            sig_a, sig_b, num_frames, hop
        )
        delay_b_frames, gcc_peaks_b, gcc_ncc_b = self._estimate_delay_profile(
            sig_b, sig_a, num_frames, hop
        )

        # 2. Delay-aligned counterpart signals (per-sample delay track).
        delay_a_samples = self._expand_delay_to_samples(delay_a_frames, n_samples, hop)
        delay_b_samples = self._expand_delay_to_samples(delay_b_frames, n_samples, hop)
        aligned_b = _shift_by_delay(sig_b, delay_a_samples)
        aligned_a = _shift_by_delay(sig_a, delay_b_samples)

        # 3. NLMS residual after delay alignment (256 taps = 32 ms).
        pad_len = win_size
        padded_a = np.pad(sig_a, (0, pad_len), mode="constant")
        padded_b = np.pad(sig_b, (0, pad_len), mode="constant")
        padded_aligned_b = np.pad(aligned_b, (0, pad_len), mode="constant")
        padded_aligned_a = np.pad(aligned_a, (0, pad_len), mode="constant")

        res_a = compute_nlms_residual(
            target=padded_a, reference=padded_aligned_b, filter_length=NLMS_FILTER_LENGTH
        )
        res_b = compute_nlms_residual(
            target=padded_b, reference=padded_aligned_a, filter_length=NLMS_FILTER_LENGTH
        )

        # STFT frames for spectral features
        from numpy.lib.stride_tricks import sliding_window_view

        win = np.hanning(win_size).astype(np.float32)
        frame_span = num_frames * hop + win_size

        def frames_of(x: np.ndarray) -> np.ndarray:
            padded = np.pad(x, (0, max(0, frame_span - len(x))), mode="constant")
            return sliding_window_view(padded[:frame_span], window_shape=win_size)[::hop][:num_frames]

        windows_a = frames_of(padded_a)
        windows_b = frames_of(padded_b)
        windows_aligned_b = frames_of(padded_aligned_b)
        windows_aligned_a = frames_of(padded_aligned_a)
        windows_res_a = frames_of(res_a)
        windows_res_b = frames_of(res_b)

        # RMS and dB energies
        eps = 1e-10
        energy_a = np.mean(windows_a ** 2, axis=1)
        energy_b = np.mean(windows_b ** 2, axis=1)
        db_a = 10.0 * np.log10(np.maximum(energy_a, eps))
        db_b = 10.0 * np.log10(np.maximum(energy_b, eps))

        ild_a = db_a - db_b
        ild_b = db_b - db_a

        # Noise floor and local SNR (audibility of anything in this channel).
        # If a channel has no quiet frames at all (dynamic range < 6 dB) it is
        # continuously occupied, so the percentile is not a noise floor and the
        # SNR must not be used to gate the leak label.
        snr_a_db, noise_floor_a_db = self._local_snr(db_a)
        snr_b_db, noise_floor_b_db = self._local_snr(db_b)

        # NLMS residual energy ratio: e^2 / x^2
        energy_res_a = np.mean(windows_res_a ** 2, axis=1)
        energy_res_b = np.mean(windows_res_b ** 2, axis=1)
        nlms_ratio_a = energy_res_a / np.maximum(energy_a, eps)
        nlms_ratio_b = energy_res_b / np.maximum(energy_b, eps)

        # Spectral analysis: FFT
        n_fft = 256
        Za = np.fft.rfft(windows_a * win, n=n_fft, axis=1)
        Zb = np.fft.rfft(windows_b * win, n=n_fft, axis=1)
        Zab = np.fft.rfft(windows_aligned_b * win, n=n_fft, axis=1)
        Zba = np.fft.rfft(windows_aligned_a * win, n=n_fft, axis=1)
        spec_a = np.abs(Za)
        spec_b = np.abs(Zb)
        spec_aligned_b = np.abs(Zab)
        spec_aligned_a = np.abs(Zba)
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

        # Magnitude-squared coherence with the delay-aligned counterpart.
        coherence_a = self._magnitude_squared_coherence(Za, Zab, freqs)
        coherence_b = self._magnitude_squared_coherence(Zb, Zba, freqs)

        # 12 Log-Mel band energies for the local channel and the aligned counterpart.
        mel_a = np.log(np.maximum(spec_a @ self.mel_fb.T, 1e-6))
        mel_b = np.log(np.maximum(spec_b @ self.mel_fb.T, 1e-6))
        mel_aligned_b = np.log(np.maximum(spec_aligned_b @ self.mel_fb.T, 1e-6))
        mel_aligned_a = np.log(np.maximum(spec_aligned_a @ self.mel_fb.T, 1e-6))

        # Weight the aligned counterpart mel bands by coherence and local activity.
        weight_a = np.clip(coherence_a, 0.0, 1.0) * (snr_a_db > LEAK_AUDIBILITY_SNR_DB)
        weight_b = np.clip(coherence_b, 0.0, 1.0) * (snr_b_db > LEAK_AUDIBILITY_SNR_DB)
        mel_aligned_b_weighted = mel_aligned_b * weight_a[:, None]
        mel_aligned_a_weighted = mel_aligned_a * weight_b[:, None]

        # Explicit audibility indicator: local speech present AND counterpart
        # coherently visible AND a strong GCC peak at a plausible delay.
        leak_audible_a = (
            (snr_a_db > LEAK_AUDIBILITY_SNR_DB)
            & (coherence_a > 0.25)
            & (gcc_peaks_a > 0.15)
        ).astype(np.float32)
        leak_audible_b = (
            (snr_b_db > LEAK_AUDIBILITY_SNR_DB)
            & (coherence_b > 0.25)
            & (gcc_peaks_b > 0.15)
        ).astype(np.float32)

        # Speaker anchor similarity features for third speaker discrimination
        anchor_a = extract_speaker_anchor_features(mel_a, ild_a, gcc_peaks_a, db_a)
        anchor_b = extract_speaker_anchor_features(mel_b, ild_b, gcc_peaks_b, db_b)

        # Explicit third-speaker score: a local talker that is NOT the owner
        # (far from the owner anchor), NOT a coherent delayed copy of the
        # counterpart (low coherence / weak GCC), and spectrally brighter and
        # flatter than the close-talk owner (distant, reverberant source).
        third_score_a = self._third_speaker_score(
            anchor_a[:, 0], coherence_a, gcc_peaks_a, hl_ratio_a, flatness_a, snr_a_db, db_a
        )
        third_score_b = self._third_speaker_score(
            anchor_b[:, 0], coherence_b, gcc_peaks_b, hl_ratio_b, flatness_b, snr_b_db, db_b
        )

        feats_a = np.column_stack([
            db_a,
            ild_a,
            nlms_ratio_a,
            gcc_peaks_a,
            delay_a_frames / self.sample_rate,
            gcc_ncc_a,
            centroid_a,
            hl_ratio_a,
            flatness_a,
            db_b,
            snr_a_db,
            snr_b_db,
            coherence_a,
            leak_audible_a,
            third_score_a,
            mel_a,
            mel_aligned_b_weighted,
            anchor_a,
        ]).astype(np.float32)

        feats_b = np.column_stack([
            db_b,
            ild_b,
            nlms_ratio_b,
            gcc_peaks_b,
            delay_b_frames / self.sample_rate,
            gcc_ncc_b,
            centroid_b,
            hl_ratio_b,
            flatness_b,
            db_a,
            snr_b_db,
            snr_a_db,
            coherence_b,
            leak_audible_b,
            third_score_b,
            mel_b,
            mel_aligned_a_weighted,
            anchor_b,
        ]).astype(np.float32)

        return {
            "A": feats_a,
            "B": feats_b,
            "feature_names": list(FEATURE_NAMES),
        }

    @staticmethod
    def _third_speaker_score(
        anchor_cos_dist: np.ndarray,
        coherence: np.ndarray,
        gcc_peak: np.ndarray,
        hl_ratio: np.ndarray,
        flatness: np.ndarray,
        snr_local_db: np.ndarray,
        db_energy: np.ndarray,
    ) -> np.ndarray:
        """Heuristic score in [0, 1] for a local, non-owner, non-leak talker.

        A third speaker is:
        - actually present in this channel (energy above the absolute floor),
        - spectrally far from the owner anchor (anchor_cos_dist high),
        - not a coherent delayed copy of the counterpart (coherence low),
        - not explained by a strong GCC peak at a leak delay,
        - brighter and flatter than close-talk owner speech (distant source),
        - audible above the local noise floor.

        The score is multiplicative in the activity gate so that silent frames
        can never be labelled as a third speaker.
        """
        anchor_term = np.clip((anchor_cos_dist - 0.45) / 0.40, 0.0, 1.0)
        coherence_term = np.clip(1.0 - coherence / 0.40, 0.0, 1.0)
        gcc_term = np.clip(1.0 - gcc_peak / 0.30, 0.0, 1.0)
        brightness_term = np.clip((hl_ratio - 0.60) / 0.40, 0.0, 1.0)
        flatness_term = np.clip((flatness - 0.25) / 0.25, 0.0, 1.0)

        # Hard activity gate: the channel must contain real energy.
        active = (db_energy > -52.0) & (snr_local_db > LEAK_AUDIBILITY_SNR_DB)

        score = (
            0.30 * anchor_term
            + 0.25 * coherence_term
            + 0.15 * gcc_term
            + 0.15 * brightness_term
            + 0.15 * flatness_term
        )
        return (score * active.astype(np.float32)).astype(np.float32)

    @staticmethod
    def _local_snr(db_energy: np.ndarray) -> Tuple[np.ndarray, float]:
        """Local SNR above the estimated noise floor.

        The noise floor is the 10th percentile of frame energies. When the
        dynamic range is small (< 6 dB) the channel is continuously occupied
        and the percentile is not a noise floor; in that case the SNR is
        reported as a large value so it does not gate the leak label.
        """
        if len(db_energy) == 0:
            return np.zeros(0, dtype=np.float32), -90.0

        floor = float(np.percentile(db_energy, 10))
        dynamic_range = float(np.percentile(db_energy, 95)) - floor
        if dynamic_range < 6.0:
            return np.full(len(db_energy), 60.0, dtype=np.float32), floor
        return (db_energy - floor).astype(np.float32), floor

    def _magnitude_squared_coherence(
        self,
        Z_local: np.ndarray,
        Z_counterpart: np.ndarray,
        freqs: np.ndarray,
        smooth_frames: int = 11,
    ) -> np.ndarray:
        """Time-smoothed magnitude-squared coherence averaged over the phone band."""
        cross = Z_local * np.conj(Z_counterpart)
        sxx = np.abs(Z_local) ** 2
        syy = np.abs(Z_counterpart) ** 2

        kernel = np.ones(smooth_frames, dtype=np.float32) / float(smooth_frames)
        cross_s = signal.fftconvolve(cross, kernel[:, None], mode="same", axes=0)
        sxx_s = signal.fftconvolve(sxx, kernel[:, None], mode="same", axes=0)
        syy_s = signal.fftconvolve(syy, kernel[:, None], mode="same", axes=0)

        coherence = np.abs(cross_s) ** 2 / (sxx_s * syy_s + 1e-12)
        band = (freqs >= 300.0) & (freqs <= 3400.0)
        if not np.any(band):
            return np.zeros(len(Z_local), dtype=np.float32)
        return np.clip(np.mean(coherence[:, band], axis=1), 0.0, 1.0).astype(np.float32)
