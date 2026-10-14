"""Unit tests for physical telephone signal and dialogue synthesizer."""

import numpy as np
import pytest

from src.config import LABEL_LEAK, LABEL_OWN, LABEL_THIRD, SAMPLE_RATE, SynthesisConfig
from src.synthesizer import (
    TelephoneDialogueSynthesizer,
    apply_alaw_compression,
    apply_bandpass_filter,
    apply_packet_loss,
    apply_speaker_nonlinearity,
    apply_time_varying_delay,
    generate_synthetic_rir,
)


def test_alaw_compression_bounds_and_stability():
    """Verify A-law companding operates stably without NaNs or Infs."""
    x = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    y = apply_alaw_compression(x)
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))
    assert np.all(y >= -1.05) and np.all(y <= 1.05)
    # Origin mapping
    assert abs(apply_alaw_compression(np.array([0.0]))[0]) < 1e-4


def test_bandpass_attenuates_low_and_high_frequencies():
    """Verify bandpass filter suppresses frequencies outside 300 - 3400 Hz."""
    sr = 8000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 50 Hz (below 300) and 3800 Hz (above 3400) vs 1000 Hz (in band)
    low_tone = np.sin(2 * np.pi * 50 * t).astype(np.float32)
    mid_tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    high_tone = np.sin(2 * np.pi * 3800 * t).astype(np.float32)

    filtered_low = apply_bandpass_filter(low_tone, sample_rate=sr)
    filtered_mid = apply_bandpass_filter(mid_tone, sample_rate=sr)
    filtered_high = apply_bandpass_filter(high_tone, sample_rate=sr)

    # In-band energy must be much higher than out-of-band energy
    energy_low = np.mean(filtered_low ** 2)
    energy_mid = np.mean(filtered_mid ** 2)
    energy_high = np.mean(filtered_high ** 2)

    assert energy_mid > energy_low * 20.0, "Low frequencies (<300 Hz) must be strongly attenuated"
    assert energy_mid > energy_high * 10.0, "High frequencies (>3400 Hz) must be strongly attenuated"


def test_time_varying_delay_drift():
    """Verify time-varying delay handles drifting delay gracefully."""
    x = np.sin(2 * np.pi * 400 * np.linspace(0, 1.0, 8000)).astype(np.float32)
    # Drift delay from 160 to 320 samples
    delays = np.linspace(160, 320, len(x), dtype=np.float32)
    delayed = apply_time_varying_delay(x, delays)

    assert len(delayed) == len(x)
    assert not np.any(np.isnan(delayed))
    # Early samples before delay should be zero
    assert np.all(delayed[:100] == 0.0)


def test_synthesizer_generates_dialogue_and_valid_ground_truth():
    """Verify synthesizer produces 2-channel audio and valid JSON intervals."""
    synth = TelephoneDialogueSynthesizer(seed=42)
    res = synth.synthesize_dialogue(duration_sec=6.0, scenario="normal")

    assert res.audio.shape[0] == 2
    assert res.audio.shape[1] == int(6.0 * SAMPLE_RATE)
    assert not np.any(np.isnan(res.audio))
    assert res.sample_rate == SAMPLE_RATE

    gt = res.ground_truth_intervals
    assert "A" in gt and "B" in gt
    assert len(gt["A"]) > 0
    assert len(gt["B"]) > 0

    valid_labels = {LABEL_OWN, LABEL_LEAK, LABEL_THIRD}
    for ch in ("A", "B"):
        for item in gt[ch]:
            assert item["label"] in valid_labels
            assert 0.0 <= item["start"] <= item["end"] <= 6.0


def test_synthesizer_all_hard_scenarios():
    """Verify all designated hard scenarios synthesize without errors."""
    synth = TelephoneDialogueSynthesizer(seed=123)
    scenarios = ["normal", "double_talk", "high_leak_lt_10db", "path_drift", "third_speaker"]

    for sc in scenarios:
        res = synth.synthesize_dialogue(duration_sec=5.0, scenario=sc)
        assert res.scenario == sc
        assert res.audio.shape == (2, int(5.0 * SAMPLE_RATE))
        if sc == "third_speaker":
            labels_a = [s["label"] for s in res.ground_truth_intervals["A"]]
            assert LABEL_THIRD in labels_a
