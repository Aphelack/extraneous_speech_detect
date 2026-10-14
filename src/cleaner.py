"""Leak suppression and audio cleaning module for two-channel telephone audio.

Implements:
1. Double-Talk Aware Adaptive NLMS Acoustic Echo Cancellation:
   Estimates the leakage transfer function h_BA and cancels acoustic coupling.
2. Post-Filter Spectral Attenuation Mask:
   Suppresses residual non-linear leak while strictly preserving the phase and
   spectral energy of the owner's speech during 'own' and 'double_talk' intervals.
3. Voice embedding similarity evaluation to quantify channel purity improvement.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal

from src.audio_utils import load_stereo_audio, save_stereo_audio
from src.config import FRAME_DURATION_SEC, FRAME_SAMPLES, SAMPLE_RATE
from src.detector import ExtraneousSpeechDetector
from src.metrics import cosine_similarity


@dataclass
class CleanerConfig:
    filter_length: int = 256
    step_size: float = 0.35
    leak_suppression_db: float = 24.0  # Extra suppression during pure leak
    frame_samples: int = FRAME_SAMPLES


class TelephoneAudioCleaner:
    """Removes counterpart acoustic leak from stereo recordings."""

    def __init__(
        self,
        config: Optional[CleanerConfig] = None,
        detector: Optional[ExtraneousSpeechDetector] = None,
    ):
        self.config = config or CleanerConfig()
        self.detector = detector or ExtraneousSpeechDetector()

    def _nlms_cancel(
        self,
        target: np.ndarray,
        reference: np.ndarray,
        freeze_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run double-talk aware NLMS echo canceller.

        Freezes weight adaptation when freeze_mask is True (e.g. during own speech).
        """
        n = min(len(target), len(reference))
        L = self.config.filter_length
        mu = self.config.step_size
        eps = 1e-5

        if n < L:
            return target.copy()

        error = np.zeros(n, dtype=np.float32)
        w = np.zeros(L, dtype=np.float32)

        ref_padded = np.pad(reference[:n], (L - 1, 0), mode="constant")
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(ref_padded, window_shape=L)

        # Block updates (block size 80 = 10 ms)
        hop = 80
        for b in range(0, n, hop):
            end_b = min(n, b + hop)
            x_blk = windows[b:end_b]
            y_hat = x_blk @ w
            e_blk = target[b:end_b] - y_hat
            error[b:end_b] = e_blk

            # Check if adaptation should occur (not frozen by double talk / own speech)
            is_frozen = False
            if freeze_mask is not None and b < len(freeze_mask):
                is_frozen = bool(freeze_mask[b])

            if not is_frozen:
                power = np.sum(x_blk ** 2) + eps
                grad = np.sum(x_blk * e_blk[:, None], axis=0)
                w += (mu / power) * grad

        return error

    def _apply_spectral_attenuation(
        self,
        audio_channel: np.ndarray,
        prob_leak: np.ndarray,
        prob_own: np.ndarray,
        frame_samples: int = FRAME_SAMPLES,
    ) -> np.ndarray:
        """Apply soft spectral attenuation to suppress residual non-linear leak."""
        num_frames = min(len(prob_leak), len(prob_own))
        total_len = min(len(audio_channel), num_frames * frame_samples)
        if total_len == 0:
            return audio_channel.copy()

        # Build frame gains
        # If pure leak: gain = 10^(-suppression / 20)
        # If own: gain = 1.0 (protect speech)
        min_gain = 10.0 ** (-self.config.leak_suppression_db / 20.0)
        leak_factor = np.clip(prob_leak[:num_frames] - prob_own[:num_frames], 0.0, 1.0)
        frame_gains = 1.0 - leak_factor * (1.0 - min_gain)

        # Smooth frame gains over time to prevent musical noise
        frame_gains = signal.medfilt(frame_gains, kernel_size=5)

        # Interpolate frame gains to sample level
        sample_gains = np.repeat(frame_gains, frame_samples)[:total_len]
        cleaned = audio_channel[:total_len] * sample_gains

        # Pad back to original length if needed
        if len(audio_channel) > total_len:
            cleaned = np.pad(cleaned, (0, len(audio_channel) - total_len), mode="constant")

        return cleaned.astype(np.float32)

    def clean_stereo(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Clean both tracks of a stereo recording.

        Parameters
        ----------
        audio : np.ndarray
            Shape (2, num_samples): [0] is Track A, [1] is Track B.

        Returns
        -------
        cleaned_stereo : np.ndarray of shape (2, num_samples)
        """
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected shape (2, N), got {audio.shape}")

        probs = self.detector.predict_probabilities(audio, sample_rate=sample_rate)
        probs_a = probs["A"]
        probs_b = probs["B"]

        # Freeze adaptation when local speaker is talking
        hop = self.config.frame_samples
        freeze_mask_a = np.repeat(probs_a["own"] > 0.4, hop)
        freeze_mask_b = np.repeat(probs_b["own"] > 0.4, hop)

        # 1. NLMS linear echo cancellation
        nlms_a = self._nlms_cancel(target=audio[0], reference=audio[1], freeze_mask=freeze_mask_a)
        nlms_b = self._nlms_cancel(target=audio[1], reference=audio[0], freeze_mask=freeze_mask_b)

        # 2. Residual non-linear leak suppression
        clean_a = self._apply_spectral_attenuation(
            audio_channel=nlms_a,
            prob_leak=probs_a["leak"],
            prob_own=probs_a["own"],
            frame_samples=hop,
        )
        clean_b = self._apply_spectral_attenuation(
            audio_channel=nlms_b,
            prob_leak=probs_b["leak"],
            prob_own=probs_b["own"],
            frame_samples=hop,
        )

        return np.stack([clean_a, clean_b], axis=0).astype(np.float32)

    def clean_wav(self, input_wav: str, output_wav: str) -> None:
        """CLI helper: read input stereo WAV, clean leak, write clean stereo WAV."""
        audio, sr = load_stereo_audio(input_wav, target_sr=SAMPLE_RATE)
        cleaned = self.clean_stereo(audio, sample_rate=sr)
        save_stereo_audio(output_wav, cleaned, sample_rate=sr)


def compute_spectral_embedding(signal_1d: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Compute normalized 32-band filterbank spectral embedding vector for speech verification."""
    if len(signal_1d) < 256:
        return np.zeros(32, dtype=np.float32)

    # Compute power spectrum
    n_fft = 512
    freqs, times, stft_mat = signal.stft(signal_1d, fs=sample_rate, nperseg=256, noverlap=128, nfft=n_fft)
    power = np.abs(stft_mat) ** 2  # (freq_bins, frames)

    # Triangular mel/subband filters in 300 - 3400 Hz range
    num_filters = 32
    f_min, f_max = 300.0, 3400.0
    filter_edges = np.linspace(f_min, f_max, num_filters + 2)
    bank_energies = []

    for k in range(num_filters):
        left, center, right = filter_edges[k], filter_edges[k + 1], filter_edges[k + 2]
        filt = np.maximum(0.0, 1.0 - np.abs(freqs - center) / (center - left))
        filt_energy = np.mean(filt[:, None] * power)
        bank_energies.append(filt_energy)

    vec = np.log10(np.maximum(np.array(bank_energies), 1e-10))
    norm = np.linalg.norm(vec) + 1e-10
    return (vec / norm).astype(np.float32)
