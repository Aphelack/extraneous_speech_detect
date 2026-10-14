"""Audio input/output and processing utilities."""

import os
from typing import Tuple, Union
import numpy as np
from scipy.io import wavfile

from src.config import FRAME_SAMPLES, SAMPLE_RATE


def load_stereo_audio(
    wav_path: str,
    target_sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Load stereo WAV file and return float32 array of shape (2, num_samples).

    Channels:
        channel 0 = Track A (Left)
        channel 1 = Track B (Right)
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    try:
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        # soundfile returns (num_samples, channels)
        if audio.ndim == 1:
            # Mono to stereo
            audio = np.stack([audio, audio], axis=1)
        audio = audio.T  # -> (channels, num_samples)
    except Exception:
        # Fallback to scipy.io.wavfile
        sr, data = wavfile.read(wav_path)
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
        else:
            audio = data.astype(np.float32)

        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        audio = audio.T  # -> (channels, num_samples)

    if sr != target_sr:
        # Resample if needed
        import scipy.signal as signal
        num_target_samples = int(round(audio.shape[1] * target_sr / sr))
        audio = signal.resample(audio, num_target_samples, axis=1).astype(np.float32)
        sr = target_sr

    # Ensure exactly 2 channels
    if audio.shape[0] < 2:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    return audio.astype(np.float32), sr


def save_stereo_audio(
    wav_path: str,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Save 2-channel float32 audio of shape (2, num_samples) to 16-bit PCM WAV."""
    if audio.ndim != 2:
        raise ValueError(f"Expected 2D audio of shape (2, N), got shape {audio.shape}")

    # Transpose to (num_samples, 2)
    if audio.shape[0] == 2 and audio.shape[1] != 2:
        data = audio.T
    else:
        data = audio

    # Clip to prevent overflow
    clipped = np.clip(data, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)

    try:
        import soundfile as sf
        sf.write(wav_path, int16_data, sample_rate, subtype="PCM_16")
    except Exception:
        wavfile.write(wav_path, sample_rate, int16_data)


def compute_frame_energies_db(
    signal: np.ndarray,
    frame_samples: int = FRAME_SAMPLES,
    hop_samples: int = FRAME_SAMPLES,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute frame-level RMS energy in decibels (dB) for a 1D audio signal.

    Parameters
    ----------
    signal : np.ndarray
        1D float audio signal.
    frame_samples : int
        Number of samples per frame.
    hop_samples : int
        Hop size in samples.
    eps : float
        Small epsilon to prevent log of zero.

    Returns
    -------
    db_energies : np.ndarray of shape [num_frames]
    """
    n_samples = len(signal)
    if n_samples == 0:
        return np.array([], dtype=np.float32)

    # Calculate number of full frames
    num_frames = max(1, int(math_num_frames(n_samples, hop_samples)))
    pad_len = (num_frames - 1) * hop_samples + frame_samples
    if pad_len > n_samples:
        padded = np.pad(signal, (0, pad_len - n_samples), mode="constant")
    else:
        padded = signal[:pad_len]

    # Vectorized frame slicing using stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    # Get windows of length frame_samples, stepped by hop_samples
    windows = sliding_window_view(padded, window_shape=frame_samples)[::hop_samples]
    if len(windows) > num_frames:
        windows = windows[:num_frames]

    # Mean squared energy
    mse = np.mean(windows ** 2, axis=-1)
    db = 10.0 * np.log10(np.maximum(mse, eps))
    return db.astype(np.float32)


def math_num_frames(num_samples: int, hop_samples: int) -> int:
    return int(np.ceil(num_samples / hop_samples))

