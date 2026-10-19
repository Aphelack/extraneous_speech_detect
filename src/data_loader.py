"""Data loader and speaker partitioner for Russian speech datasets.

Supports:
- Mozilla Common Voice ru (with client_id speaker clustering)
- SberDevices Golos (general Russian speech segments)
- VoxCeleb1 test split (for embedding verification)
- Legacy CMU Arctic fallback

Guarantees strict speaker separation between train / val / test splits
so that speakers in validation and testing never leak into training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

from src.config import SAMPLE_RATE


def _resample_to_target(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    num_target = int(round(len(audio) * target_sr / orig_sr))
    return signal.resample(audio, num_target).astype(np.float32)


def _normalize_audio(audio: np.ndarray, peak_level: float = 0.7) -> np.ndarray:
    """Peak-normalize audio to [-peak_level, peak_level]."""
    # Convert integer formats to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        audio = audio / peak * peak_level
    return audio


class CommonVoiceRuLoader:
    """Load Russian speech utterances from Common Voice with speaker ID grouping."""

    def __init__(
        self,
        data_root: Union[str, Path] = "data/common_voice_ru",
        target_sr: int = SAMPLE_RATE,
    ):
        self.data_root = self._resolve_path(data_root)
        self.target_sr = target_sr
        self._dataset = None
        self._speaker_index: Optional[Dict[str, List[int]]] = None

    @staticmethod
    def _resolve_path(data_root: Union[str, Path]) -> Path:
        """Resolve data root, checking multiple candidate locations."""
        path = Path(data_root)
        if path.exists():
            return path
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / data_root
        if candidate.exists():
            return candidate
        curr = Path.cwd().resolve()
        for parent in [curr] + list(curr.parents):
            candidate = parent / data_root
            if candidate.exists():
                return candidate
        return path

    def _load_dataset(self):
        """Load the HuggingFace dataset from disk."""
        if self._dataset is not None:
            return

        dataset_path = self.data_root / "dataset"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Common Voice ru dataset not found at {dataset_path}. "
                "Run: python scripts/download_data.py --dataset common_voice_ru"
            )

        from datasets import load_from_disk
        self._dataset = load_from_disk(str(dataset_path))
        print(f"Loaded Common Voice ru: {len(self._dataset)} samples")

    def _build_speaker_index(self):
        """Build an index mapping speaker IDs to sample indices."""
        if self._speaker_index is not None:
            return

        self._load_dataset()
        self._speaker_index = {}
        for i in range(len(self._dataset)):
            cid = self._dataset[i].get("client_id", "unknown")
            if cid not in self._speaker_index:
                self._speaker_index[cid] = []
            self._speaker_index[cid].append(i)

        print(f"Indexed {len(self._speaker_index)} speakers")

    def discover_speakers(self) -> Dict[str, int]:
        """Return dict of {speaker_id: utterance_count}."""
        self._build_speaker_index()
        return {spk: len(indices) for spk, indices in self._speaker_index.items()}

    def get_speakers_with_min_utterances(self, min_count: int = 15) -> List[str]:
        """Return list of speaker IDs with at least min_count utterances."""
        speakers = self.discover_speakers()
        return [spk for spk, cnt in speakers.items() if cnt >= min_count]

    def load_speaker_utterances(
        self,
        speaker_id: str,
        max_utterances: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Load and normalize utterances for a specific speaker at target_sr."""
        self._build_speaker_index()
        if speaker_id not in self._speaker_index:
            raise KeyError(
                f"Speaker '{speaker_id}' not found. "
                f"Available: {len(self._speaker_index)} speakers"
            )

        indices = self._speaker_index[speaker_id]
        if max_utterances is not None:
            indices = indices[:max_utterances]

        utterances: List[np.ndarray] = []
        for idx in indices:
            try:
                sample = self._dataset[idx]
                audio_data = sample["audio"]
                audio_array = np.array(audio_data["array"], dtype=np.float32)
                orig_sr = audio_data["sampling_rate"]

                # Resample to target
                audio_array = _resample_to_target(audio_array, orig_sr, self.target_sr)
                audio_array = _normalize_audio(audio_array)

                if len(audio_array) > 0:
                    utterances.append(audio_array)
            except Exception as e:
                print(f"Warning: failed reading sample {idx}: {e}")

        return utterances

    def build_utterance_pool(
        self,
        speaker_a: Optional[str] = None,
        speaker_b: Optional[str] = None,
        speaker_third: Optional[str] = None,
        max_per_speaker: int = 50,
        min_utterances: int = 15,
    ) -> Dict[str, List[np.ndarray]]:
        """Construct an utterance pool for dialogue synthesis.

        If speaker IDs are not specified, automatically selects speakers
        with the most utterances.
        """
        self._build_speaker_index()

        # Auto-select speakers if not specified
        candidates = self.get_speakers_with_min_utterances(min_utterances)
        if not candidates:
            raise RuntimeError(
                f"No speakers with >= {min_utterances} utterances found. "
                "Check that Common Voice ru data is downloaded correctly."
            )

        # Sort by utterance count (most utterances first)
        candidates.sort(key=lambda s: len(self._speaker_index[s]), reverse=True)

        if speaker_a is None and len(candidates) >= 1:
            speaker_a = candidates[0]
        if speaker_b is None and len(candidates) >= 2:
            speaker_b = candidates[1]
        if speaker_third is None and len(candidates) >= 3:
            speaker_third = candidates[2]

        print(f"Selected speakers: A={speaker_a}, B={speaker_b}, third={speaker_third}")

        pool: Dict[str, List[np.ndarray]] = {}
        if speaker_a:
            pool["A"] = self.load_speaker_utterances(speaker_a, max_utterances=max_per_speaker)
            print(f"  Speaker A ({speaker_a}): {len(pool['A'])} utterances")
        if speaker_b:
            pool["B"] = self.load_speaker_utterances(speaker_b, max_utterances=max_per_speaker)
            print(f"  Speaker B ({speaker_b}): {len(pool['B'])} utterances")
        if speaker_third:
            pool["third"] = self.load_speaker_utterances(speaker_third, max_utterances=max_per_speaker)
            print(f"  Speaker third ({speaker_third}): {len(pool['third'])} utterances")

        return pool

    def get_disjoint_speaker_sets(
        self,
        n_train: int = 10,
        n_val: int = 5,
        n_test: int = 5,
        min_utterances: int = 15,
    ) -> Dict[str, List[str]]:
        """Split speakers into strictly non-overlapping train/val/test sets."""
        candidates = self.get_speakers_with_min_utterances(min_utterances)
        candidates.sort(key=lambda s: len(self._speaker_index[s]), reverse=True)

        total_needed = n_train + n_val + n_test
        if len(candidates) < total_needed:
            print(f"Warning: only {len(candidates)} speakers available, need {total_needed}")
            # Reduce proportionally
            ratio = len(candidates) / total_needed
            n_train = max(1, int(n_train * ratio))
            n_val = max(1, int(n_val * ratio))
            n_test = max(1, int(n_test * ratio))

        return {
            "train": candidates[:n_train],
            "val": candidates[n_train : n_train + n_val],
            "test": candidates[n_train + n_val : n_train + n_val + n_test],
        }


class GolosLoader:
    """Load Russian speech from SberDevices Golos dataset."""

    def __init__(
        self,
        data_root: Union[str, Path] = "data/golos_test",
        target_sr: int = SAMPLE_RATE,
    ):
        self.data_root = CommonVoiceRuLoader._resolve_path(data_root)
        self.target_sr = target_sr
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is not None:
            return
        dataset_path = self.data_root / "dataset"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Golos dataset not found at {dataset_path}. "
                "Run: python scripts/download_data.py --dataset golos_test"
            )
        from datasets import load_from_disk
        self._dataset = load_from_disk(str(dataset_path))
        print(f"Loaded Golos: {len(self._dataset)} samples")

    def load_random_utterances(
        self,
        count: int = 50,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[np.ndarray]:
        """Load random utterances (no speaker grouping — used for noise/filler)."""
        self._load_dataset()
        if rng is None:
            rng = np.random.RandomState()

        indices = rng.choice(len(self._dataset), size=min(count, len(self._dataset)), replace=False)
        utterances = []
        for idx in indices:
            try:
                sample = self._dataset[int(idx)]
                audio_data = sample["audio"]
                audio_array = np.array(audio_data["array"], dtype=np.float32)
                orig_sr = audio_data["sampling_rate"]
                audio_array = _resample_to_target(audio_array, orig_sr, self.target_sr)
                audio_array = _normalize_audio(audio_array)
                if len(audio_array) > 0:
                    utterances.append(audio_array)
            except Exception as e:
                print(f"Warning: failed reading Golos sample {idx}: {e}")

        return utterances


class FolderSpeechLoader:
    """Loads utterances from directory organized as data_root/<speaker_id>/*.wav."""

    def __init__(self, data_root: Union[str, Path] = "data/russian_speech", target_sr: int = SAMPLE_RATE):
        self.data_root = CommonVoiceRuLoader._resolve_path(data_root)
        self.target_sr = target_sr

    def discover_speakers(self) -> Dict[str, int]:
        speakers = {}
        if not self.data_root.exists():
            return speakers
        for spk_dir in sorted(self.data_root.iterdir()):
            if spk_dir.is_dir() and not spk_dir.name.startswith("."):
                wavs = list(spk_dir.glob("*.wav"))
                if wavs:
                    speakers[spk_dir.name] = len(wavs)
        return speakers

    def load_speaker_utterances(
        self, speaker_id: str, max_utterances: Optional[int] = None
    ) -> List[np.ndarray]:
        spk_dir = self.data_root / speaker_id
        if not spk_dir.exists():
            raise KeyError(f"Speaker '{speaker_id}' not found in {self.data_root}")

        file_paths = sorted(spk_dir.glob("*.wav"))
        if max_utterances is not None:
            file_paths = file_paths[:max_utterances]

        utterances = []
        for p in file_paths:
            try:
                sr, data = wavfile.read(p)
                audio = _normalize_audio(data)
                audio = _resample_to_target(audio, sr, self.target_sr)
                utterances.append(audio)
            except Exception as e:
                print(f"Warning: failed reading {p}: {e}")
        return utterances

    def build_utterance_pool(
        self,
        speaker_a: Optional[str] = None,
        speaker_b: Optional[str] = None,
        speaker_third: Optional[str] = None,
        max_per_speaker: int = 50,
    ) -> Dict[str, List[np.ndarray]]:
        speakers = sorted(self.discover_speakers().keys())
        if not speakers:
            raise RuntimeError(f"No speakers found in {self.data_root}")

        sa = speaker_a or speakers[0]
        sb = speaker_b or (speakers[1] if len(speakers) > 1 else speakers[0])
        st = speaker_third or (speakers[2] if len(speakers) > 2 else None)

        pool = {
            "A": self.load_speaker_utterances(sa, max_utterances=max_per_speaker),
            "B": self.load_speaker_utterances(sb, max_utterances=max_per_speaker),
        }
        if st:
            pool["third"] = self.load_speaker_utterances(st, max_utterances=max_per_speaker)
        return pool


class VoxCeleb1ArrowLoader:
    """Loads utterances from VoxCeleb1 Arrow dataset without requiring HuggingFace datasets package."""

    def __init__(
        self,
        data_root: Union[str, Path] = "data/voxceleb1_test",
        target_sr: int = SAMPLE_RATE,
    ):
        self.data_root = CommonVoiceRuLoader._resolve_path(data_root)
        self.target_sr = target_sr
        self._tables: Optional[List[Any]] = None
        self._speaker_index: Optional[Dict[str, List[Tuple[int, int]]]] = None

    def _ensure_loaded(self):
        if self._speaker_index is not None:
            return
        import io
        import pyarrow.ipc as ipc

        dataset_path = self.data_root / "dataset"
        arrow_files = sorted(dataset_path.glob("*.arrow"))
        if not arrow_files:
            arrow_files = sorted(self.data_root.glob("*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"No arrow files found in {self.data_root}")

        self._tables = [ipc.open_stream(f).read_all() for f in arrow_files]
        self._speaker_index = {}
        for t_idx, table in enumerate(self._tables):
            spks = table["speaker_id"].to_pylist()
            for r_idx, s in enumerate(spks):
                if s not in self._speaker_index:
                    self._speaker_index[s] = []
                self._speaker_index[s].append((t_idx, r_idx))

    def discover_speakers(self) -> Dict[str, int]:
        self._ensure_loaded()
        return {s: len(idx_list) for s, idx_list in self._speaker_index.items()}

    def load_speaker_utterances(
        self, speaker_id: str, max_utterances: Optional[int] = None
    ) -> List[np.ndarray]:
        self._ensure_loaded()
        if speaker_id not in self._speaker_index:
            raise KeyError(f"Speaker '{speaker_id}' not found in VoxCeleb1")

        import io
        import soundfile as sf

        indices = self._speaker_index[speaker_id]
        if max_utterances is not None:
            indices = indices[:max_utterances]

        utterances = []
        for t_idx, r_idx in indices:
            try:
                table = self._tables[t_idx]
                raw_bytes = table["audio"][r_idx]["bytes"].as_py()
                data, orig_sr = sf.read(io.BytesIO(raw_bytes))
                audio = _normalize_audio(data)
                audio = _resample_to_target(audio, orig_sr, self.target_sr)
                if len(audio) > 0:
                    utterances.append(audio)
            except Exception as e:
                print(f"Warning: failed reading VoxCeleb sample ({t_idx}, {r_idx}): {e}")

        return utterances

    def build_utterance_pool(
        self,
        speaker_a: Optional[str] = None,
        speaker_b: Optional[str] = None,
        speaker_third: Optional[str] = None,
        max_per_speaker: int = 50,
    ) -> Dict[str, List[np.ndarray]]:
        speakers = sorted(self.discover_speakers().keys())
        if not speakers:
            raise RuntimeError(f"No speakers found in {self.data_root}")
        sa = speaker_a or speakers[0]
        sb = speaker_b or (speakers[1] if len(speakers) > 1 else speakers[0])
        st = speaker_third or (speakers[2] if len(speakers) > 2 else None)

        pool = {
            "A": self.load_speaker_utterances(sa, max_utterances=max_per_speaker),
            "B": self.load_speaker_utterances(sb, max_utterances=max_per_speaker),
        }
        if st:
            pool["third"] = self.load_speaker_utterances(st, max_utterances=max_per_speaker)
        return pool


class CombinedSpeechLoader:
    """Provide Russian speakers for dialogue synthesis.

    VoxCeleb remains available through its own loader for embedding checks,
    but is never selected as a dialogue source.
    """

    def __init__(
        self,
        russian_root: Union[str, Path] = "data/russian_speech",
        voxceleb_root: Union[str, Path] = "data/voxceleb1_test",
        target_sr: int = SAMPLE_RATE,
    ):
        self.target_sr = target_sr
        self.loaders: Dict[str, Any] = {}

        # 1. Russian speech
        ru_path = CommonVoiceRuLoader._resolve_path(russian_root)
        if ru_path.exists():
            self.loaders["ru"] = FolderSpeechLoader(ru_path, target_sr)

        # 2. VoxCeleb1
        vox_path = CommonVoiceRuLoader._resolve_path(voxceleb_root)
        if (vox_path / "dataset").exists() or (vox_path / "data-00000-of-00003.arrow").exists():
            self.loaders["vox"] = VoxCeleb1ArrowLoader(vox_path, target_sr)

    def discover_speakers(self) -> Dict[str, int]:
        all_spks = {}
        if "ru" in self.loaders:
            for k, v in self.loaders["ru"].discover_speakers().items():
                all_spks[f"ru_{k}"] = v
        if "vox" in self.loaders:
            for k, v in self.loaders["vox"].discover_speakers().items():
                all_spks[f"vox_{k}"] = v
        return all_spks

    def load_speaker_utterances(
        self, speaker_id: str, max_utterances: Optional[int] = None
    ) -> List[np.ndarray]:
        if speaker_id.startswith("ru_"):
            raw_id = speaker_id[3:]
            return self.loaders["ru"].load_speaker_utterances(raw_id, max_utterances)
        elif speaker_id.startswith("vox_"):
            raw_id = speaker_id[4:]
            return self.loaders["vox"].load_speaker_utterances(raw_id, max_utterances)
        elif "ru" in self.loaders and speaker_id in self.loaders["ru"].discover_speakers():
            return self.loaders["ru"].load_speaker_utterances(speaker_id, max_utterances)
        elif "vox" in self.loaders and speaker_id in self.loaders["vox"].discover_speakers():
            return self.loaders["vox"].load_speaker_utterances(speaker_id, max_utterances)
        else:
            raise KeyError(f"Unknown speaker: {speaker_id}")

    def get_disjoint_splits(self) -> Dict[str, List[str]]:
        """Return strictly disjoint Russian train and test speaker sets."""
        ru_spks = sorted([k for k in self.discover_speakers().keys() if k.startswith("ru_")])
        ru_train = ru_spks[:4]
        ru_test = ru_spks[4:]

        return {
            "train": ru_train,
            "test": ru_test,
        }

    def sample_dialogue_speakers(
        self,
        split: str = "train",
        russian_prob: float = 1.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[str, str, Optional[str]]:
        """Sample three Russian speakers from the requested split."""
        if rng is None:
            rng = np.random.RandomState()

        splits = self.get_disjoint_splits()
        ru_spks = [s for s in splits[split] if s.startswith("ru_")]
        if len(ru_spks) < 2:
            raise RuntimeError(f"Need at least two Russian speakers in {split}, found {ru_spks}")
        chosen = rng.choice(ru_spks, size=min(3, len(ru_spks)), replace=False)
        sa = str(chosen[0])
        sb = str(chosen[1])
        remaining = [s for s in ru_spks if s != sa and s != sb]
        st = str(rng.choice(remaining)) if remaining else None

        return sa, sb, st

    def build_utterance_pool(
        self,
        speaker_a: Optional[str] = None,
        speaker_b: Optional[str] = None,
        speaker_third: Optional[str] = None,
        max_per_speaker: int = 50,
    ) -> Dict[str, List[np.ndarray]]:
        spks = sorted(s for s in self.discover_speakers().keys() if s.startswith("ru_"))
        if len(spks) < 2:
            raise RuntimeError("Russian dialogue pool requires at least two Russian speakers")
        sa = speaker_a or spks[0]
        sb = speaker_b or (spks[1] if len(spks) > 1 else spks[0])
        st = speaker_third or (spks[2] if len(spks) > 2 else None)

        pool = {
            "A": self.load_speaker_utterances(sa, max_utterances=max_per_speaker),
            "B": self.load_speaker_utterances(sb, max_utterances=max_per_speaker),
        }
        if st:
            pool["third"] = self.load_speaker_utterances(st, max_utterances=max_per_speaker)
        return pool


class RealSpeechLoader:
    """Unified loader: prefers Russian speech + VoxCeleb1 combined loader, then Common Voice, then CMU Arctic."""

    def __init__(
        self,
        data_root: Union[str, Path] = "data/russian_speech",
        target_sr: int = SAMPLE_RATE,
    ):
        self.target_sr = target_sr
        self._loader = None

        requested_root = CommonVoiceRuLoader._resolve_path(data_root)
        default_root = CommonVoiceRuLoader._resolve_path("data/russian_speech")
        if requested_root != default_root:
            if not requested_root.exists():
                raise FileNotFoundError(f"Dataset directory not found: {requested_root}")
            self._loader = FolderSpeechLoader(requested_root, target_sr)
            print(f"Using dataset from: {requested_root}")
            return

        # 1. Try Combined Russian + VoxCeleb1
        ru_path = CommonVoiceRuLoader._resolve_path("data/russian_speech")
        vox_path = CommonVoiceRuLoader._resolve_path("data/voxceleb1_test")
        if ru_path.exists() and vox_path.exists():
            self._loader = CombinedSpeechLoader(ru_path, vox_path, target_sr)
            print(f"Using Combined Speech Loader: Russian ({ru_path}) + VoxCeleb1 ({vox_path})")
            return

        # 2. Try local folder-based Russian speech (e.g. data/russian_speech)
        path = CommonVoiceRuLoader._resolve_path(data_root)
        if path.exists() and any(p.is_dir() for p in path.iterdir()):
            self._loader = FolderSpeechLoader(path, target_sr)
            print(f"Using Russian speech data from: {path}")
            return

        # 3. Try Common Voice ru
        cv_root = CommonVoiceRuLoader._resolve_path("data/common_voice_ru")
        if (cv_root / "dataset").exists():
            self._loader = CommonVoiceRuLoader(cv_root, target_sr)
            print("Using Common Voice ru (Russian speech)")
            return

        # 4. Fallback to CMU Arctic
        arctic_root = CommonVoiceRuLoader._resolve_path("data/cmuarctic")
        if arctic_root.exists():
            print("WARNING: Falling back to CMU Arctic (English).")
            self._loader = _LegacyArcticLoader(str(arctic_root), target_sr)
            return

        # Default fallback
        self._loader = FolderSpeechLoader(path, target_sr)

    def discover_speakers(self) -> Dict[str, int]:
        return self._loader.discover_speakers() if self._loader else {}

    def get_disjoint_splits(self) -> Dict[str, List[str]]:
        if hasattr(self._loader, "get_disjoint_splits"):
            return self._loader.get_disjoint_splits()
        all_spks = sorted(self.discover_speakers().keys())
        half = len(all_spks) // 2
        return {"train": all_spks[:half], "test": all_spks[half:]}

    def sample_dialogue_speakers(
        self,
        split: str = "train",
        russian_prob: float = 0.80,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[str, str, Optional[str]]:
        if hasattr(self._loader, "sample_dialogue_speakers"):
            return self._loader.sample_dialogue_speakers(split, russian_prob, rng)
        splits = self.get_disjoint_splits()
        spk_set = splits[split]
        if rng is None:
            rng = np.random.RandomState()
        chosen = rng.choice(spk_set, size=min(3, len(spk_set)), replace=False)
        sa = str(chosen[0])
        sb = str(chosen[1]) if len(chosen) > 1 else str(chosen[0])
        st = str(chosen[2]) if len(chosen) > 2 else None
        return sa, sb, st

    def load_speaker_utterances(
        self, speaker_id: str, max_utterances: Optional[int] = None
    ) -> List[np.ndarray]:
        return self._loader.load_speaker_utterances(speaker_id, max_utterances) if self._loader else []

    def build_utterance_pool(
        self,
        speaker_a: Optional[str] = None,
        speaker_b: Optional[str] = None,
        speaker_third: Optional[str] = None,
        max_per_speaker: int = 50,
    ) -> Dict[str, List[np.ndarray]]:
        return self._loader.build_utterance_pool(speaker_a, speaker_b, speaker_third, max_per_speaker) if self._loader else {}




class _LegacyArcticLoader:
    """Legacy CMU Arctic loader (English). For backward compatibility only."""

    def __init__(self, data_root: str, target_sr: int = SAMPLE_RATE):
        self.data_root = Path(data_root)
        self.target_sr = target_sr

    def discover_speakers(self) -> Dict[str, int]:
        speakers: Dict[str, List[Path]] = {}
        for wav_path in self.data_root.rglob("*.wav"):
            parts = wav_path.parts
            speaker_id = "unknown"
            for p in parts:
                if "cmu_us_" in p:
                    speaker_id = p.replace("cmu_us_", "").replace("_arctic", "")
                    break
            if speaker_id not in speakers:
                speakers[speaker_id] = []
            speakers[speaker_id].append(wav_path)
        return {spk: len(files) for spk, files in speakers.items()}

    def load_speaker_utterances(
        self, speaker_id: str, max_utterances: Optional[int] = None,
    ) -> List[np.ndarray]:
        speakers: Dict[str, List[Path]] = {}
        for wav_path in self.data_root.rglob("*.wav"):
            parts = wav_path.parts
            sid = "unknown"
            for p in parts:
                if "cmu_us_" in p:
                    sid = p.replace("cmu_us_", "").replace("_arctic", "")
                    break
            if sid not in speakers:
                speakers[sid] = []
            speakers[sid].append(wav_path)

        if speaker_id not in speakers:
            raise KeyError(f"Speaker '{speaker_id}' not found. Available: {list(speakers.keys())}")

        file_paths = speakers[speaker_id]
        if max_utterances is not None:
            file_paths = file_paths[:max_utterances]

        utterances = []
        for p in file_paths:
            try:
                sr, data = wavfile.read(p)
                audio = _normalize_audio(data)
                audio = _resample_to_target(audio, sr, self.target_sr)
                utterances.append(audio)
            except Exception as e:
                print(f"Warning: failed reading {p}: {e}")
        return utterances

    def build_utterance_pool(
        self, speaker_a: str, speaker_b: str, speaker_third: str, max_per_speaker: int,
    ) -> Dict[str, List[np.ndarray]]:
        pool = {
            "A": self.load_speaker_utterances(speaker_a, max_utterances=max_per_speaker),
            "B": self.load_speaker_utterances(speaker_b, max_utterances=max_per_speaker),
        }
        try:
            pool["third"] = self.load_speaker_utterances(speaker_third, max_utterances=max_per_speaker)
        except KeyError:
            pass
        return pool
