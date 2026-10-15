"""Data loader and speaker partitioner for real speech audio datasets.

Guarantees strict speaker separation between train / val / test splits
so that speakers in validation and testing never leak into training.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

from src.config import SAMPLE_RATE


class RealSpeechLoader:
    """Loads and caches real speech utterances grouped by speaker ID."""

    def __init__(self, data_root: Union[str, Path] = "data/cmuarctic", target_sr: int = SAMPLE_RATE):
        path = Path(data_root)
        if not path.exists():
            # If called from a subdirectory (e.g. notebooks/ or scripts/), resolve relative to repo root
            repo_root = Path(__file__).resolve().parent.parent
            if (repo_root / data_root).exists():
                path = repo_root / data_root
            else:
                curr = Path.cwd().resolve()
                for parent in [curr] + list(curr.parents):
                    candidate = parent / data_root
                    if candidate.exists():
                        path = candidate
                        break
        self.data_root = path
        self.target_sr = target_sr

    def discover_speakers(self) -> Dict[str, List[Path]]:
        """Find all available speakers and their WAV files."""
        speakers: Dict[str, List[Path]] = {}

        # Look in data_root (e.g. ARCTIC/cmu_us_aew_arctic/wav/*.wav)
        for wav_path in self.data_root.rglob("*.wav"):
            # Identify speaker from parent directory or name
            parts = wav_path.parts
            speaker_id = "unknown"
            for p in parts:
                if "cmu_us_" in p:
                    speaker_id = p.replace("cmu_us_", "").replace("_arctic", "")
                    break
                elif p in ("aew", "slt", "bdl", "clb", "rms", "fem", "jmk", "ksp"):
                    speaker_id = p
                    break

            if speaker_id not in speakers:
                speakers[speaker_id] = []
            speakers[speaker_id].append(wav_path)

        return speakers

    def load_speaker_utterances(
        self,
        speaker_id: str,
        max_utterances: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Load and normalize real utterances for a specific speaker at target_sr."""
        speakers = self.discover_speakers()
        if speaker_id not in speakers:
            raise KeyError(f"Speaker '{speaker_id}' not found. Available: {list(speakers.keys())}")

        file_paths = speakers[speaker_id]
        if max_utterances is not None:
            file_paths = file_paths[:max_utterances]

        utterances: List[np.ndarray] = []
        for p in file_paths:
            try:
                sr, data = wavfile.read(p)
                # Convert to float32 in [-1.0, 1.0]
                if data.dtype == np.int16:
                    audio = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    audio = data.astype(np.float32) / 2147483648.0
                else:
                    audio = data.astype(np.float32)

                # Resample to target_sr (8000 Hz)
                if sr != self.target_sr:
                    num_target = int(round(len(audio) * self.target_sr / sr))
                    audio = signal.resample(audio, num_target).astype(np.float32)

                # Peak normalize
                peak = np.max(np.abs(audio))
                if peak > 1e-4:
                    audio = audio / peak * 0.7

                utterances.append(audio)
            except Exception as e:
                print(f"Warning: failed reading {p}: {e}")

        return utterances

    def build_utterance_pool(
        self,
        speaker_a: str = "aew",
        speaker_b: str = "slt",
        speaker_third: Optional[str] = "bdl",
        max_per_speaker: int = 50,
    ) -> Dict[str, List[np.ndarray]]:
        """Construct an utterance pool dictionary for dialogue synthesis."""
        pool: Dict[str, List[np.ndarray]] = {
            "A": self.load_speaker_utterances(speaker_a, max_utterances=max_per_speaker),
            "B": self.load_speaker_utterances(speaker_b, max_utterances=max_per_speaker),
        }
        if speaker_third:
            try:
                pool["third"] = self.load_speaker_utterances(speaker_third, max_utterances=max_per_speaker)
            except KeyError:
                pass
        return pool

