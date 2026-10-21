"""Unit tests for GCC-PHAT delay estimation and delay-aligned NLMS.

The reviewer requirement is explicit: the delay estimator must recover delays
across the whole physical range 20-300 ms, which is longer than the analysis
window. These tests use both noise and speech-like signals.
"""

import numpy as np
import pytest
from scipy import signal

from src.config import LEAK_DELAY_MAX_SAMPLES, LEAK_DELAY_MIN_SAMPLES, SAMPLE_RATE
from src.features import (
    GCC_CURRENT_WINDOW_SAMPLES,
    NLMS_FILTER_LENGTH,
    FeatureExtractor,
    compute_gcc_phat_peak,
    compute_nlms_residual,
)

DELAYS_MS = [20, 50, 150, 280, 300]


def _speech_like_signal(num_samples: int, seed: int = 0) -> np.ndarray:
    """Band-limited noise with speech-like spectral envelope."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(num_samples).astype(np.float32)
    sos = signal.butter(4, [300.0, 3400.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    shaped = signal.sosfilt(sos, noise).astype(np.float32)
    # Amplitude modulation to mimic syllables.
    t = np.arange(num_samples) / SAMPLE_RATE
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2
    return (shaped * envelope).astype(np.float32)


def _delayed_copy(x: np.ndarray, delay_samples: int) -> np.ndarray:
    """Return y with y[n] = x[n - delay_samples]."""
    y = np.zeros_like(x)
    if delay_samples < len(x):
        y[delay_samples:] = x[: len(x) - delay_samples]
    return y


@pytest.mark.parametrize("delay_ms", DELAYS_MS)
def test_gcc_phat_recovers_delay_on_noise(delay_ms: int):
    """GCC-PHAT must find the exact delay for a delayed noise copy."""
    fe = FeatureExtractor()
    x = _speech_like_signal(24000, seed=1)
    delay_samples = int(round(delay_ms * SAMPLE_RATE / 1000.0))
    local = _delayed_copy(x, delay_samples)

    t0 = 8000
    current = local[t0 : t0 + GCC_CURRENT_WINDOW_SAMPLES]
    history = fe._history_slice(x, t0)

    peak, delay_sec, ncc = compute_gcc_phat_peak(current, history)

    estimated_ms = delay_sec * 1000.0
    assert abs(estimated_ms - delay_ms) <= 2.0, (
        f"Expected ~{delay_ms} ms, got {estimated_ms:.1f} ms"
    )
    assert peak > 0.1, f"GCC-PHAT peak too weak: {peak}"
    assert ncc > 0.2, f"Normalized cross-correlation too weak: {ncc}"


@pytest.mark.parametrize("delay_ms", DELAYS_MS)
def test_gcc_phat_recovers_delay_on_speech(delay_ms: int):
    """GCC-PHAT must find the delay for a speech-like signal too."""
    fe = FeatureExtractor()
    x = _speech_like_signal(24000, seed=2)
    delay_samples = int(round(delay_ms * SAMPLE_RATE / 1000.0))
    local = _delayed_copy(x, delay_samples)

    t0 = 9000
    current = local[t0 : t0 + GCC_CURRENT_WINDOW_SAMPLES]
    history = fe._history_slice(x, t0)

    _, delay_sec, _ = compute_gcc_phat_peak(current, history)
    estimated_ms = delay_sec * 1000.0
    assert abs(estimated_ms - delay_ms) <= 3.0, (
        f"Expected ~{delay_ms} ms, got {estimated_ms:.1f} ms"
    )


def test_gcc_phat_search_range_is_physical():
    """The estimator must never return a delay outside [20, 300] ms."""
    fe = FeatureExtractor()
    x = _speech_like_signal(24000, seed=3)
    for delay_ms in DELAYS_MS:
        delay_samples = int(round(delay_ms * SAMPLE_RATE / 1000.0))
        local = _delayed_copy(x, delay_samples)
        t0 = 8000
        _, delay_sec, _ = compute_gcc_phat_peak(
            local[t0 : t0 + GCC_CURRENT_WINDOW_SAMPLES], fe._history_slice(x, t0)
        )
        delay_samples_est = delay_sec * SAMPLE_RATE
        assert LEAK_DELAY_MIN_SAMPLES - 1 <= delay_samples_est <= LEAK_DELAY_MAX_SAMPLES + 1


def test_nlms_filter_length_covers_room_response():
    """NLMS must use 256 taps (32 ms), not the old 64 taps (8 ms)."""
    assert NLMS_FILTER_LENGTH == 256
    assert NLMS_FILTER_LENGTH / SAMPLE_RATE >= 0.030


def test_nlms_cancels_aligned_echo():
    """After delay alignment, NLMS must strongly cancel a scaled echo."""
    rng = np.random.RandomState(4)
    reference = rng.randn(16000).astype(np.float32) * 0.1
    # Echo = delayed + attenuated + short room tail of the reference.
    rir = np.zeros(200, dtype=np.float32)
    rir[0] = 0.5
    rir[40] = 0.25
    rir[120] = 0.1
    echo = signal.fftconvolve(reference, rir, mode="full")[: len(reference)].astype(np.float32)
    target = echo + rng.randn(len(reference)).astype(np.float32) * 1e-4

    residual = compute_nlms_residual(target, reference, filter_length=NLMS_FILTER_LENGTH)

    # Measure steady state after the filter has converged (first 1 s excluded).
    start = SAMPLE_RATE
    echo_power = float(np.mean(echo[start:] ** 2))
    residual_power = float(np.mean(residual[start:] ** 2))
    assert residual_power < 0.05 * echo_power, (
        f"NLMS failed to cancel aligned echo: {residual_power / echo_power:.4f}"
    )


def test_feature_matrix_contains_leak_audibility_features():
    """The feature matrix must expose SNR, coherence and audibility columns."""
    fe = FeatureExtractor()
    audio = np.random.randn(2, 16000).astype(np.float32) * 0.05
    feats = fe.extract_features(audio)
    names = feats["feature_names"]

    for required in ("snr_local_db", "coherence", "leak_audible", "gcc_delay"):
        assert required in names, f"Missing feature: {required}"

    assert feats["A"].shape[1] == len(names)
    assert not np.any(np.isnan(feats["A"]))
    assert not np.any(np.isnan(feats["B"]))
