"""Delay-aligned neural spectral cleaner for two-channel telephone audio."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter1d, median_filter
import torch
import torch.nn as nn

from src.audio_utils import load_stereo_audio, resample_audio, save_stereo_audio
from src.config import FRAME_SAMPLES, SAMPLE_RATE
from src.detector import ExtraneousSpeechDetector


def create_triangular_filterbank(
    n_bins: int = 129,
    num_bands: int = 32,
    sample_rate: int = SAMPLE_RATE,
    f_min: float = 100.0,
    f_max: float = 3800.0,
) -> np.ndarray:
    freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    edges = np.linspace(f_min, f_max, num_bands + 2)
    bank = np.zeros((num_bands, n_bins), dtype=np.float32)
    for band in range(num_bands):
        left, center, right = edges[band:band + 3]
        bank[band] = np.maximum(
            0.0,
            np.minimum((freqs - left) / (center - left), (right - freqs) / (right - center)),
        )
    bank[0, freqs < f_min] = 1.0
    bank[-1, freqs > f_max] = 1.0
    return (bank / (np.sum(bank, axis=0, keepdims=True) + 1e-8)).astype(np.float32)


class _CleanerNet(nn.Module):
    """Small TCN producing a 32-band IRM estimate."""

    def __init__(self, in_channels: int = 164, hidden_dim: int = 48, num_bands: int = 32):
        super().__init__()
        self.in_channels = in_channels
        # Feature standardization statistics are part of the state_dict so the
        # checkpoint can be loaded with torch.load(..., weights_only=True).
        self.register_buffer("feat_mean", torch.zeros(1, in_channels, 1))
        self.register_buffer("feat_std", torch.ones(1, in_channels, 1))
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=d, dilation=d),
                nn.GroupNorm(4, hidden_dim),
                nn.GELU(),
            )
            for d in (1, 2, 4, 8)
        ])
        self.head = nn.Sequential(nn.Conv1d(hidden_dim, num_bands, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.ndim == 2
        if squeezed:
            x = x.unsqueeze(0)
        x = (x - self.feat_mean) / self.feat_std
        hidden = self.in_proj(x)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        output = self.head(hidden)
        return output.squeeze(0) if squeezed else output


@dataclass
class CleanerConfig:
    weights_path: str = "models/cleaner_weights.pt"
    frame_samples: int = FRAME_SAMPLES
    n_fft: int = 256
    num_bands: int = 32
    delay_min_frames: int = 2
    delay_max_frames: int = 30
    delay_smoothing_frames: int = 51
    noise_percentile: float = 5.0


class TelephoneAudioCleaner:
    """Remove counterpart leakage with delay-aligned spectral masking."""

    def __init__(self, config: Optional[CleanerConfig] = None, detector: Optional[ExtraneousSpeechDetector] = None):
        self.config = config or CleanerConfig()
        self.detector = detector or ExtraneousSpeechDetector()
        self.filterbank = create_triangular_filterbank(
            self.config.n_fft // 2 + 1, self.config.num_bands, SAMPLE_RATE
        )
        self._load_model()

    def _load_model(self) -> None:
        self.model = None
        path = Path(self.config.weights_path)
        if not path.exists():
            path = Path(__file__).resolve().parent.parent / self.config.weights_path
        if not path.exists():
            return
        try:
            # weights_only=True: only tensors / primitives are stored.
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            state = checkpoint["model_state_dict"]
            in_channels = int(state["in_proj.0.weight"].shape[1])
            num_bands = int(state["head.0.weight"].shape[0])
            model = _CleanerNet(in_channels=in_channels, hidden_dim=48, num_bands=num_bands)
            model.load_state_dict(state)
            model.eval()
            self.model = model
        except Exception as exc:
            print(f"Notice: Cleaner weights could not be loaded ({exc}); analytic mask enabled.")

    def estimate_delay_frames(self, mag_a: np.ndarray, mag_b: np.ndarray) -> np.ndarray:
        """Estimate local envelope delay and smooth it over 50 frames."""
        env_a = np.log(np.mean(mag_a ** 2, axis=0) + 1e-8)
        env_b = np.log(np.mean(mag_b ** 2, axis=0) + 1e-8)
        frames = len(env_a)
        half_window = 10

        def rolling_mean(values: np.ndarray) -> np.ndarray:
            return signal.convolve(values, np.ones(2 * half_window + 1) / (2 * half_window + 1), mode="same")

        mean_a = rolling_mean(env_a)
        centered_a = env_a - mean_a
        scores = np.full((self.config.delay_max_frames - self.config.delay_min_frames + 1, frames), -1.0, dtype=np.float32)
        for row, delay in enumerate(range(self.config.delay_min_frames, self.config.delay_max_frames + 1)):
            shifted_b = np.zeros(frames, dtype=np.float32)
            if delay < frames:
                shifted_b[delay:] = env_b[:-delay]
            mean_b = rolling_mean(shifted_b)
            centered_b = shifted_b - mean_b
            numerator = rolling_mean(centered_a * centered_b)
            denom = np.sqrt(rolling_mean(centered_a ** 2) * rolling_mean(centered_b ** 2)) + 1e-8
            scores[row] = numerator / denom
        delays = np.argmax(scores, axis=0) + self.config.delay_min_frames
        return np.clip(
            median_filter(delays, size=self.config.delay_smoothing_frames, mode="nearest"),
            self.config.delay_min_frames,
            self.config.delay_max_frames,
        ).astype(np.int32)

    def align_spectrum(self, spectrum_b: np.ndarray, delays: np.ndarray) -> np.ndarray:
        frames = spectrum_b.shape[1]
        indices = np.maximum(np.arange(frames) - delays.astype(np.int32), 0)
        return spectrum_b[:, indices]

    def _estimate_leakage(self, mag_a: np.ndarray, aligned_b: np.ndarray, p_own: np.ndarray, p_leak: np.ndarray) -> np.ndarray:
        bins, frames = mag_a.shape
        aligned_energy = np.mean(aligned_b ** 2, axis=0)
        noise_b_floor = float(np.percentile(aligned_energy, 5))
        active_reference = aligned_energy > max(noise_b_floor * 6.0, 1e-10)
        if np.any(active_reference):
            ratios = mag_a[:, active_reference] / (aligned_b[:, active_reference] + 1e-6)
            coefficient = np.percentile(ratios, 25, axis=1).astype(np.float32)
        else:
            coefficient = np.full(bins, 0.2, dtype=np.float32)
        coefficient = np.clip(coefficient, 0.08, 0.85)
        estimate = np.zeros_like(mag_a, dtype=np.float32)
        for t in range(frames):
            if p_leak[t] > 0.5 and p_own[t] < 0.2:
                ratio = mag_a[:, t] / (aligned_b[:, t] + 1e-6)
                coefficient = 0.95 * coefficient + 0.05 * np.clip(ratio, 0.08, 0.85)
            estimate[:, t] = coefficient * aligned_b[:, t]
        return estimate

    def extract_features(
        self,
        Za: np.ndarray,
        Zb: np.ndarray,
        prob_own: np.ndarray,
        prob_leak: np.ndarray,
        prob_third: np.ndarray,
        aligned_b: Optional[np.ndarray] = None,
        leak_estimate: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mag_a = np.abs(Za)
        if aligned_b is None:
            aligned_b = Zb
        mag_b = np.abs(aligned_b)
        if leak_estimate is None:
            leak_estimate = np.zeros_like(mag_a)
        W = self.filterbank
        log_a = np.log(W @ mag_a + 1e-6)
        log_b = np.log(W @ mag_b + 1e-6)
        log_leak = np.log(W @ leak_estimate + 1e-6)
        log_ratio = np.log((W @ mag_a + 1e-6) / (W @ leak_estimate + 1e-6))
        prev_b = np.concatenate([log_b[:, :1], log_b[:, :-1]], axis=1)
        next_b = np.concatenate([log_b[:, 1:], log_b[:, -1:]], axis=1)
        power_a = np.sum(mag_a ** 2, axis=0) + 1e-8
        power_b = np.sum(mag_b ** 2, axis=0) + 1e-8
        ild = (10.0 * np.log10(power_a / power_b) / 20.0).clip(-2.0, 2.0)
        frames = mag_a.shape[1]
        probs = np.zeros((3, frames), dtype=np.float32)
        for row, values in enumerate((prob_own, prob_leak, prob_third)):
            length = min(frames, len(values))
            if length:
                probs[row, :length] = values[:length]
                probs[row, length:] = values[length - 1]
        return np.vstack([log_a, log_b, prev_b, next_b, log_leak, log_ratio, probs, ild[None, :]]).astype(np.float32)

    def _apply_cleaner_mask(self, target_audio: np.ndarray, counterpart_audio: np.ndarray, prob_own: np.ndarray, prob_leak: np.ndarray, prob_third: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = min(len(target_audio), len(counterpart_audio))
        n_fft = self.config.n_fft
        hop = self.config.frame_samples
        window = signal.windows.hann(n_fft)
        _, _, Za = signal.stft(target_audio[:n_samples], fs=sample_rate, window=window, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, Zb = signal.stft(counterpart_audio[:n_samples], fs=sample_rate, window=window, nperseg=n_fft, noverlap=n_fft - hop)
        mag_a, mag_b = np.abs(Za), np.abs(Zb)
        delays = self.estimate_delay_frames(mag_a, mag_b)
        aligned_b = self.align_spectrum(Zb, delays)
        frames = mag_a.shape[1]
        p_own = np.resize(np.asarray(prob_own, dtype=np.float32), frames)
        p_leak = np.resize(np.asarray(prob_leak, dtype=np.float32), frames)
        p_third = np.resize(np.asarray(prob_third, dtype=np.float32), frames)
        leak_estimate = self._estimate_leakage(mag_a, np.abs(aligned_b), p_own, p_leak)
        features = self.extract_features(Za, Zb, p_own, p_leak, p_third, aligned_b, leak_estimate)
        if self.model is not None and self.model.in_channels == features.shape[0]:
            # Standardization is applied inside the network via buffers.
            with torch.no_grad():
                bands = self.model(torch.tensor(features, dtype=torch.float32)).numpy()
        else:
            band_a = self.filterbank @ mag_a
            band_leak = self.filterbank @ leak_estimate
            bands = 1.0 / np.sqrt(1.0 + band_leak / (band_a + 1e-6))

        neural_gain = np.clip(self.filterbank.T @ bands, 0.0, 1.0)
        analytic_gain = np.sqrt(
            np.maximum(mag_a ** 2 - 2.0 * leak_estimate ** 2, 0.0)
            / (mag_a ** 2 + 1e-8)
        )
        analytic_gain = np.clip(analytic_gain, 0.0, 1.0)
        double_talk_gain = np.clip(0.5 * neural_gain + 0.5 * analytic_gain, 0.04, 1.0)
        noise_floor = np.percentile(mag_a, self.config.noise_percentile, axis=1, keepdims=True)
        leak_ext = maximum_filter1d(p_leak, size=20, mode="nearest", origin=-9)
        frame_energy = np.mean(mag_a ** 2, axis=0)
        aligned_energy = np.mean(np.abs(aligned_b) ** 2, axis=0)
        floor_energy = float(np.mean(noise_floor ** 2))
        own_active = (p_own >= 0.35) & (frame_energy > floor_energy)
        pure_own = leak_ext < 0.20
        pure_leak = (~own_active) & (leak_ext >= 0.25)
        tail_pure_leak = (leak_ext >= 0.60) & (frame_energy < 0.75 * aligned_energy)
        pure_leak |= tail_pure_leak
        double_talk = own_active & (leak_ext >= 0.20)

        leak_dominates = leak_estimate > 0.7 * mag_a
        owner_dominates = mag_a > 1.8 * leak_estimate
        double_talk_gain[leak_dominates] = np.minimum(double_talk_gain[leak_dominates], 0.04)
        double_talk_gain[owner_dominates] = np.maximum(double_talk_gain[owner_dominates], 0.85)
        double_talk_gain = np.clip(double_talk_gain, 0.04, 1.0)

        gain = np.ones_like(mag_a, dtype=np.float32)
        gain[:, double_talk] = double_talk_gain[:, double_talk]
        gain[:, pure_leak] = np.clip(
            noise_floor / (mag_a[:, pure_leak] + 1e-8),
            0.015,
            0.02,
        )
        gain = np.clip(gain, 0.0, 1.0)
        cleaned_mag = np.maximum(mag_a * gain, noise_floor)
        cleaned_spec = Za * (cleaned_mag / (mag_a + 1e-8))
        if np.any(pure_leak):
            rng = np.random.RandomState(42)
            phase = rng.uniform(-np.pi, np.pi, size=Za.shape)
            comfort = 0.15 * noise_floor * np.exp(1j * phase)
            cleaned_spec[:, pure_leak] += comfort[:, :1]
        _, cleaned = signal.istft(cleaned_spec, fs=sample_rate, window=window, nperseg=n_fft, noverlap=n_fft - hop)
        return cleaned[:n_samples].astype(np.float32), gain.astype(np.float32)

    def clean_stereo(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        return_masks: bool = False,
        probabilities: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Clean both tracks.

        ``probabilities`` may be supplied when the caller has already run the
        detector (e.g. ``evaluate.py``), which avoids computing the feature
        matrix twice and keeps the pipeline within the RTF budget.
        """
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected shape (2, N), got {audio.shape}")
        probs = probabilities if probabilities is not None else self.detector.predict_probabilities(
            audio, sample_rate=sample_rate
        )
        clean_a, mask_a = self._apply_cleaner_mask(audio[0], audio[1], probs["A"]["own"], probs["A"]["leak"], probs["A"]["third"], sample_rate)
        clean_b, mask_b = self._apply_cleaner_mask(audio[1], audio[0], probs["B"]["own"], probs["B"]["leak"], probs["B"]["third"], sample_rate)
        cleaned = np.stack([clean_a, clean_b]).astype(np.float32)
        return (cleaned, {"A": mask_a, "B": mask_b}) if return_masks else cleaned

    def clean_wav(self, input_wav: str, output_wav: str, preview_wav: Optional[str] = None, preview_sr: int = 16000) -> None:
        audio, sr = load_stereo_audio(input_wav, target_sr=SAMPLE_RATE)
        cleaned = self.clean_stereo(audio, sample_rate=sr)
        save_stereo_audio(output_wav, cleaned, sample_rate=sr)
        if preview_wav:
            save_stereo_audio(preview_wav, resample_audio(cleaned, orig_sr=sr, target_sr=preview_sr), sample_rate=preview_sr)

    def clean(
        self,
        audio: np.ndarray,
        predictions: Optional[Any] = None,
        sample_rate: int = SAMPLE_RATE,
        probabilities: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> np.ndarray:
        return self.clean_stereo(
            audio, sample_rate=sample_rate, probabilities=probabilities
        )


def compute_spectral_embedding(signal_1d: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if len(signal_1d) < 256:
        return np.zeros(32, dtype=np.float32)
    freqs, _, stft_mat = signal.stft(signal_1d, fs=sample_rate, nperseg=256, noverlap=128, nfft=512)
    power = np.abs(stft_mat) ** 2
    edges = np.linspace(300.0, 3400.0, 34)
    energies = []
    for left, center, right in zip(edges[:-2], edges[1:-1], edges[2:]):
        filt = np.maximum(0.0, np.minimum((freqs - left) / (center - left), (right - freqs) / (right - center)))
        energies.append(np.mean(filt[:, None] * power))
    vector = np.log10(np.maximum(energies, 1e-10))
    return (vector / (np.linalg.norm(vector) + 1e-10)).astype(np.float32)
