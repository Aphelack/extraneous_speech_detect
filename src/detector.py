"""Detector for extraneous speech (own, leak, third) in two-channel telephone audio.

Combines cross-channel physical cues:
- GCC-PHAT peak & delay in history window [20, 300] ms
- NLMS adaptive filter cancellation residual ratio
- Interchannel Level Difference (ILD)
- Spectral brightness and high-to-low frequency ratios for distant third speaker
- Multi-label neural detector trained on realistic telephone acoustic scenarios
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_closing, median_filter

from src.audio_utils import load_stereo_audio
from src.config import (
    FRAME_DURATION_SEC,
    FRAME_SAMPLES,
    LABEL_LEAK,
    LABEL_OWN,
    LABEL_THIRD,
    SAMPLE_RATE,
)
from src.features import FeatureExtractor
from src.metrics import frame_mask_to_intervals


class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, channels: int = 48, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        pad = (dilation * (kernel_size - 1)) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(4, channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm2 = nn.GroupNorm(4, channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act2(y + res)


class _DetectorNet(nn.Module):
    """Contextual 1D-TCN (Dilated Conv1d) multi-label detector.

    4 residual blocks with dilations [1, 2, 4, 8], receptive field ~60 frames (600 ms).
    Strictly satisfies single-core CPU RTF <= 0.001 and lookahead <= 1 s.
    """

    def __init__(self, in_features: int = 36, hidden_dim: int = 48, num_classes: int = 3):
        super().__init__()
        self.in_features = in_features
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            ResidualDilatedConvBlock(hidden_dim, kernel_size=5, dilation=d)
            for d in [1, 2, 4, 8]
        ])
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts (B, C, T) or (T, C)
        squeeze_batch = False
        if x.ndim == 2:
            # (T, C) -> (1, C, T)
            x = x.transpose(0, 1).unsqueeze(0)
            squeeze_batch = True
        elif x.ndim == 3 and x.shape[1] != self.in_features and x.shape[2] == self.in_features:
            # (B, T, C) -> (B, C, T)
            x = x.transpose(1, 2)

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(h)  # (B, num_classes, T)

        if squeeze_batch:
            # (1, num_classes, T) -> (T, num_classes)
            return logits.squeeze(0).transpose(0, 1)
        return logits


@dataclass
class DetectorConfig:
    weights_path: str = "models/detector_weights.pt"
    frame_sec: float = FRAME_DURATION_SEC
    threshold_own: float = 0.50
    threshold_leak: float = 0.50
    threshold_third: float = 0.38
    smoothing_window_frames: int = 5  # 50 ms temporal smoothing
    min_segment_sec: float = 0.08     # 80 ms minimum duration
    min_segment_third_sec: float = 0.15  # 150 ms minimum duration for third
    vad_energy_threshold_db: float = -48.0


class ExtraneousSpeechDetector:
    """Multi-label detector for own speech, counterpart leak, and third speaker."""

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        self.config = config or DetectorConfig()
        self.fe = feature_extractor or FeatureExtractor()
        self._load_model()

    def _load_model(self) -> None:
        self.model = None
        self.mean = None
        self.std = None

        weights_file = Path(self.config.weights_path)
        if not weights_file.exists():
            repo_root = Path(__file__).resolve().parent.parent
            if (repo_root / self.config.weights_path).exists():
                weights_file = repo_root / self.config.weights_path
            else:
                curr = Path.cwd().resolve()
                for parent in [curr] + list(curr.parents):
                    candidate = parent / self.config.weights_path
                    if candidate.exists():
                        weights_file = candidate
                        break

        if weights_file.exists():
            try:
                ckpt = torch.load(weights_file, map_location="cpu", weights_only=False)
                in_dim = len(ckpt["mean"].flatten())
                model = _DetectorNet(in_features=in_dim, hidden_dim=48, num_classes=3)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self.model = model
                self.mean = ckpt["mean"]
                self.std = ckpt["std"]
            except Exception as e:
                print(f"Warning: could not load {weights_file} ({e}), falling back to heuristic detector.")

    def _estimate_vad_mask(self, db_energy: np.ndarray) -> np.ndarray:
        """Estimate speech presence mask."""
        if len(db_energy) == 0:
            return np.array([], dtype=bool)
        p15 = float(np.percentile(db_energy, 15))
        p95 = float(np.percentile(db_energy, 95))
        dynamic_range = p95 - p15

        if dynamic_range > 10.0:
            threshold = max(p15 + 8.0, self.config.vad_energy_threshold_db)
        else:
            threshold = self.config.vad_energy_threshold_db

        return db_energy > threshold

    def predict_probabilities(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute frame-level probability curves for each class on Track A and Track B."""
        feats = self.fe.extract_features(audio)
        feats_a = feats["A"]
        feats_b = feats["B"]

        probs: Dict[str, Dict[str, np.ndarray]] = {"A": {}, "B": {}}

        for ch in ("A", "B"):
            ch_feats = feats_a if ch == "A" else feats_b
            other_feats = feats_b if ch == "A" else feats_a

            num_frames = len(ch_feats)
            if num_frames == 0:
                probs[ch] = {LABEL_OWN: np.array([]), LABEL_LEAK: np.array([]), LABEL_THIRD: np.array([])}
                continue

            db_energy = ch_feats[:, 0]
            ild = ch_feats[:, 1]
            gcc_peak = ch_feats[:, 3]
            anchor_cos_dist = ch_feats[:, 33]
            vad_mask = self._estimate_vad_mask(db_energy)
            x_full = ch_feats

            if self.model is not None:
                # Standardize
                x_norm = (x_full - self.mean) / self.std
                with torch.no_grad():
                    t_x = torch.tensor(x_norm, dtype=torch.float32)
                    logits = self.model(t_x)
                    preds = torch.sigmoid(logits).cpu().numpy()

                p_own = preds[:, 0]
                p_leak = preds[:, 1]
                p_third = preds[:, 2]

                # Absolute silence is the only post-network physical constraint.
                silence_floor = db_energy < -60.0
                p_own = np.where(silence_floor, 0.0, p_own)
                p_leak = np.where(silence_floor, 0.0, p_leak)
                p_third = np.where(silence_floor, 0.0, p_third)
            else:
                # Calibrated physical heuristic fallback
                other_vad = self._estimate_vad_mask(other_feats[:, 0])

                p_leak = np.where(vad_mask & other_vad & (ild < 7.0), 0.85, 0.0)
                p_own = np.where(vad_mask & (ild > -7.0), 0.90, 0.0)
                p_third = np.where(
                    vad_mask & (~other_vad) & (ch_feats[:, 6] < 0.65) & (anchor_cos_dist > 0.35),
                    0.85,
                    0.0,
                )

            # A third speaker is local to this channel: cross-channel evidence
            # or a close owner-anchor match is incompatible with that label.
            third_is_remote = (p_leak > 0.65) & (gcc_peak > 0.45)
            third_is_owner = (ild > 8.0) & (anchor_cos_dist < 0.25)
            p_third = np.where(third_is_remote | third_is_owner, 0.0, p_third)

            # Temporal median filtering (50 ms)
            k = self.config.smoothing_window_frames
            if k > 1 and len(p_own) >= k:
                p_own = median_filter(p_own, size=k)
                p_leak = median_filter(p_leak, size=k)
                p_third = median_filter(p_third, size=k)

            probs[ch] = {
                LABEL_OWN: p_own.astype(np.float32),
                LABEL_LEAK: p_leak.astype(np.float32),
                LABEL_THIRD: p_third.astype(np.float32),
            }

        return probs

    def predict_intervals(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Produce interval predictions compliant with project JSON schema."""
        probs = self.predict_probabilities(audio, sample_rate=sample_rate)
        results: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}

        for ch in ("A", "B"):
            ch_probs = probs[ch]
            intervals_ch: List[Dict[str, Any]] = []

            # 1. Own intervals
            mask_own = ch_probs[LABEL_OWN] >= self.config.threshold_own
            intervals_ch.extend(
                frame_mask_to_intervals(
                    mask_own,
                    label=LABEL_OWN,
                    frame_sec=self.config.frame_sec,
                    scores=ch_probs[LABEL_OWN],
                    min_duration_sec=self.config.min_segment_sec,
                )
            )

            # 2. Leak intervals
            mask_leak = ch_probs[LABEL_LEAK] >= self.config.threshold_leak
            intervals_ch.extend(
                frame_mask_to_intervals(
                    mask_leak,
                    label=LABEL_LEAK,
                    frame_sec=self.config.frame_sec,
                    scores=ch_probs[LABEL_LEAK],
                    min_duration_sec=self.config.min_segment_sec,
                )
            )

            # 3. Third speaker intervals
            mask_third = ch_probs[LABEL_THIRD] >= self.config.threshold_third
            if len(mask_third) > 7:
                mask_third = binary_closing(mask_third, structure=np.ones(7, dtype=bool))
            min_dur_third = getattr(self.config, "min_segment_third_sec", 0.22)
            third_intervals = frame_mask_to_intervals(
                mask_third,
                label=LABEL_THIRD,
                frame_sec=self.config.frame_sec,
                scores=ch_probs[LABEL_THIRD],
                min_duration_sec=min_dur_third,
            )
            intervals_ch.extend(third_intervals)

            intervals_ch.sort(key=lambda x: x["start"])
            results[ch] = intervals_ch

        return results

    def predict_wav(self, wav_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load stereo WAV file and produce predictions."""
        audio, sr = load_stereo_audio(wav_path, target_sr=SAMPLE_RATE)
        return self.predict_intervals(audio, sample_rate=sr)

    # Standard alias matching cleaner/baseline interface
    detect = predict_intervals
