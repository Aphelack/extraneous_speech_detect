"""Speaker embedding extraction using pre-trained ECAPA-TDNN (SpeechBrain).

Provides a fast, language-agnostic 192-dimensional speaker embedding
suitable for evaluating voice purity on Russian telephone audio.

Usage:
    embedder = SpeakerEmbedder()
    emb = embedder.embed(audio_np, sample_rate=8000)  # -> np.ndarray (192,)
    sim = embedder.cosine_similarity(emb_a, emb_b)    # -> float in [-1, 1]
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio


class SpeakerEmbedder:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN.

    The model expects 16 kHz mono audio. Input at other sample rates
    is automatically resampled.

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of the output embedding (192 for ECAPA-TDNN).
    """

    embedding_dim: int = 192

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_source : str
            HuggingFace Hub model identifier or local path.
        savedir : str, optional
            Local directory to cache model weights. Defaults to models/spkrec-ecapa-voxceleb.
        device : str, optional
            "cpu" or "cuda". Auto-detects if not provided.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if savedir is None:
            # Default: project_root/models/spkrec-ecapa-voxceleb
            project_root = Path(__file__).resolve().parent.parent
            savedir = str(project_root / "models" / "spkrec-ecapa-voxceleb")

        from speechbrain.inference.speaker import EncoderClassifier

        self.classifier = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": device},
        )
        self._resamplers: dict = {}

    def _get_resampler(self, orig_sr: int, target_sr: int = 16000) -> torchaudio.transforms.Resample:
        """Get or create a cached resampler."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            )
        return self._resamplers[orig_sr]

    def embed(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Extract a 192-dim speaker embedding from an audio segment.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor
            Mono audio waveform (1D or 2D with shape [1, N]).
        sample_rate : int
            Sample rate of the input audio.

        Returns
        -------
        np.ndarray of shape (192,)
            L2-normalized speaker embedding vector.
        """
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        # Ensure 2D: (1, N)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim == 2 and audio_tensor.shape[0] > 1:
            # Take first channel (mono)
            audio_tensor = audio_tensor[0:1]

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)

        # Extract embedding
        with torch.no_grad():
            emb = self.classifier.encode_batch(audio_tensor)

        # Shape: (1, 1, 192) -> (192,)
        emb_np = emb.squeeze().cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(emb_np)
        if norm > 1e-8:
            emb_np = emb_np / norm

        return emb_np

    def embed_segments(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        intervals: list,
        sample_rate: int = 16000,
        min_duration_sec: float = 0.3,
    ) -> np.ndarray:
        """Extract averaged embedding over multiple speech segments.

        Parameters
        ----------
        audio : array
            Full mono audio waveform.
        intervals : list of dict
            Each dict has 'start' and 'end' keys (seconds).
        sample_rate : int
            Sample rate.
        min_duration_sec : float
            Minimum segment duration to include.

        Returns
        -------
        np.ndarray of shape (192,)
            Mean embedding (L2-normalized) across valid segments.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        embeddings = []
        for seg in intervals:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            duration = seg["end"] - seg["start"]
            if duration < min_duration_sec:
                continue
            if end_sample > len(audio):
                end_sample = len(audio)
            if start_sample >= end_sample:
                continue

            chunk = audio[start_sample:end_sample]
            emb = self.embed(chunk, sample_rate=sample_rate)
            embeddings.append(emb)

        if not embeddings:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb = mean_emb / norm
        return mean_emb.astype(np.float32)

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors.

        Parameters
        ----------
        emb_a, emb_b : np.ndarray
            Speaker embedding vectors (should be L2-normalized).

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        a = np.asarray(emb_a, dtype=np.float64).flatten()
        b = np.asarray(emb_b, dtype=np.float64).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

