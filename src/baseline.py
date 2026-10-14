"""Baseline rule detector for extraneous speech in two-channel telephone recordings.

Rule definition from project specification:
"Отрезок речи на дорожке считается own, если громкость дорожки на нём хотя бы
на 6 дБ выше громкости второй дорожки, иначе leak. Третьего говорящего правило
не различает."
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.audio_utils import compute_frame_energies_db, load_stereo_audio
from src.config import (
    BASELINE_THRESHOLD_DB,
    FRAME_DURATION_SEC,
    FRAME_SAMPLES,
    LABEL_LEAK,
    LABEL_OWN,
    LABEL_SILENCE,
    SAMPLE_RATE,
    BaselineConfig,
)
from src.metrics import frame_mask_to_intervals


class EnergyBaselineDetector:
    """Implements the baseline 6 dB energy comparison rule."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()

    def estimate_noise_floor_db(self, energies_db: np.ndarray) -> float:
        """Estimate stationary background noise floor using lower 15th percentile."""
        if len(energies_db) == 0:
            return -60.0
        return float(np.percentile(energies_db, 15))

    def detect_speech_vad(
        self,
        energies_db: np.ndarray,
        noise_margin_db: float = 10.0,
        min_threshold_db: Optional[float] = None,
    ) -> np.ndarray:
        """Energy-based voice activity detector with adaptive thresholding."""
        if len(energies_db) == 0:
            return np.array([], dtype=bool)

        min_thresh = min_threshold_db if min_threshold_db is not None else self.config.vad_energy_threshold_db

        p95 = float(np.percentile(energies_db, 95))
        p15 = float(np.percentile(energies_db, 15))
        dynamic_range = p95 - p15

        if dynamic_range > 12.0:
            # Clear presence of both speech bursts and quiet background
            noise_floor = min(p15, min_thresh)
            threshold = max(noise_floor + noise_margin_db, min_thresh)
        else:
            # Signal has almost uniform energy across frames (e.g. continuous speech or test signal)
            threshold = min_thresh

        return energies_db > threshold

    def predict_frames(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute frame-level label predictions and confidence scores for both channels.

        Parameters
        ----------
        audio : np.ndarray
            Shape (2, num_samples), where audio[0] is Track A, audio[1] is Track B.
        sample_rate : int
            Audio sample rate (default 8000 Hz).

    Returns
        -------
        labels_by_ch : dict {"A": np.ndarray(str), "B": np.ndarray(str)}
        scores_by_ch : dict {"A": np.ndarray(float), "B": np.ndarray(float)}
        """
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected audio shape (2, N), got {audio.shape}")

        frame_samples = int(sample_rate * self.config.frame_sec)

        # Compute frame dB energies for Track A and Track B
        e_a = compute_frame_energies_db(audio[0], frame_samples=frame_samples, hop_samples=frame_samples)
        e_b = compute_frame_energies_db(audio[1], frame_samples=frame_samples, hop_samples=frame_samples)

        num_frames = min(len(e_a), len(e_b))
        e_a = e_a[:num_frames]
        e_b = e_b[:num_frames]

        # VAD speech presence
        vad_a = self.detect_speech_vad(e_a)
        vad_b = self.detect_speech_vad(e_b)

        delta_db = self.config.threshold_db  # 6.0 dB

        # Channel A decision
        # diff_a = E_A - E_B
        # If diff_a >= 6 dB -> own, else -> leak
        diff_a = e_a - e_b
        labels_a = np.full(num_frames, LABEL_SILENCE, dtype=object)
        scores_a = np.zeros(num_frames, dtype=np.float32)

        speech_mask_a = vad_a
        own_mask_a = speech_mask_a & (diff_a >= delta_db)
        leak_mask_a = speech_mask_a & (diff_a < delta_db)

        labels_a[own_mask_a] = LABEL_OWN
        labels_a[leak_mask_a] = LABEL_LEAK

        # Confidence scores using logistic sigmoid over margin
        # For own: margin = diff_a - delta_db; for leak: margin = delta_db - diff_a
        margin_own_a = (diff_a - delta_db) / 6.0
        margin_leak_a = (delta_db - diff_a) / 6.0
        scores_a[own_mask_a] = 1.0 / (1.0 + np.exp(-np.clip(margin_own_a[own_mask_a], -10, 10)))
        scores_a[leak_mask_a] = 1.0 / (1.0 + np.exp(-np.clip(margin_leak_a[leak_mask_a], -10, 10)))

        # Channel B decision
        # diff_b = E_B - E_A
        # If diff_b >= 6 dB -> own, else -> leak
        diff_b = e_b - e_a
        labels_b = np.full(num_frames, LABEL_SILENCE, dtype=object)
        scores_b = np.zeros(num_frames, dtype=np.float32)

        speech_mask_b = vad_b
        own_mask_b = speech_mask_b & (diff_b >= delta_db)
        leak_mask_b = speech_mask_b & (diff_b < delta_db)

        labels_b[own_mask_b] = LABEL_OWN
        labels_b[leak_mask_b] = LABEL_LEAK

        margin_own_b = (diff_b - delta_db) / 6.0
        margin_leak_b = (delta_db - diff_b) / 6.0
        scores_b[own_mask_b] = 1.0 / (1.0 + np.exp(-np.clip(margin_own_b[own_mask_b], -10, 10)))
        scores_b[leak_mask_b] = 1.0 / (1.0 + np.exp(-np.clip(margin_leak_b[leak_mask_b], -10, 10)))

        return {"A": labels_a, "B": labels_b}, {"A": scores_a, "B": scores_b}

    def predict_intervals(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Produce interval predictions compliant with project JSON specification.

        Returns
        -------
        dict with structure:
        {
            "A": [{"start": float, "end": float, "label": str, "score": float}, ...],
            "B": [{"start": float, "end": float, "label": str, "score": float}, ...]
        }
        """
        labels_dict, scores_dict = self.predict_frames(audio, sample_rate=sample_rate)

        results: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}

        for ch in ("A", "B"):
            ch_labels = labels_dict[ch]
            ch_scores = scores_dict[ch]

            # Generate intervals for 'own' and 'leak'
            own_mask = ch_labels == LABEL_OWN
            leak_mask = ch_labels == LABEL_LEAK

            own_intervals = frame_mask_to_intervals(
                own_mask,
                label=LABEL_OWN,
                frame_sec=self.config.frame_sec,
                scores=ch_scores,
                min_duration_sec=self.config.min_speech_duration_sec,
            )
            leak_intervals = frame_mask_to_intervals(
                leak_mask,
                label=LABEL_LEAK,
                frame_sec=self.config.frame_sec,
                scores=ch_scores,
                min_duration_sec=self.config.min_speech_duration_sec,
            )

            # Combine and sort by start time
            combined = own_intervals + leak_intervals
            combined.sort(key=lambda x: x["start"])
            results[ch] = combined

        return results

    def predict_wav(self, wav_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load WAV and produce baseline interval predictions."""
        audio, sr = load_stereo_audio(wav_path, target_sr=SAMPLE_RATE)
        return self.predict_intervals(audio, sample_rate=sr)


def predict_baseline_rule(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience helper to run baseline detector on 2-channel audio array."""
    detector = EnergyBaselineDetector(config=config)
    return detector.predict_intervals(audio, sample_rate=sample_rate)
