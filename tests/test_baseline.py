"""Unit tests for baseline +6 dB energy comparison detector."""

import numpy as np
import pytest

from src.baseline import EnergyBaselineDetector, predict_baseline_rule
from src.config import LABEL_LEAK, LABEL_OWN, LABEL_SILENCE, LABEL_THIRD, SAMPLE_RATE


def test_baseline_speaker_a_dominant():
    """When Track A is significantly louder (> 6 dB), A is own and B is leak."""
    duration_sec = 2.0
    num_samples = int(SAMPLE_RATE * duration_sec)

    # 440 Hz tone as speech proxy on Track A
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    sig_a = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Track B gets attenuated version (15 dB attenuation = factor ~0.177)
    sig_b = sig_a * 0.177

    audio = np.stack([sig_a, sig_b], axis=0).astype(np.float32)

    detector = EnergyBaselineDetector()
    labels, scores = detector.predict_frames(audio, sample_rate=SAMPLE_RATE)

    # Track A must be detected as 'own'
    assert np.all(labels["A"] == LABEL_OWN)
    # Track B must be detected as 'leak' (energy is ~15 dB below A, so E_B - E_A = -15 dB < 6 dB)
    assert np.all(labels["B"] == LABEL_LEAK)


def test_baseline_speaker_b_dominant():
    """When Track B is significantly louder (> 6 dB), B is own and A is leak."""
    duration_sec = 1.0
    num_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    sig_b = 0.5 * np.sin(2 * np.pi * 500 * t)
    sig_a = sig_b * 0.1  # 20 dB attenuation

    audio = np.stack([sig_a, sig_b], axis=0).astype(np.float32)

    detector = EnergyBaselineDetector()
    labels, _ = detector.predict_frames(audio, sample_rate=SAMPLE_RATE)

    assert np.all(labels["B"] == LABEL_OWN)
    assert np.all(labels["A"] == LABEL_LEAK)


def test_baseline_output_json_format():
    """Verify that predict_intervals returns format required by specification."""
    duration_sec = 2.0
    num_samples = int(SAMPLE_RATE * duration_sec)
    audio = np.zeros((2, num_samples), dtype=np.float32)

    # Track A has sound from 0.5s to 1.5s
    start_idx = int(0.5 * SAMPLE_RATE)
    end_idx = int(1.5 * SAMPLE_RATE)
    t = np.arange(end_idx - start_idx) / SAMPLE_RATE
    audio[0, start_idx:end_idx] = 0.4 * np.sin(2 * np.pi * 300 * t)
    audio[1, start_idx:end_idx] = 0.05 * np.sin(2 * np.pi * 300 * t)

    intervals = predict_baseline_rule(audio, sample_rate=SAMPLE_RATE)

    assert "A" in intervals and "B" in intervals
    assert isinstance(intervals["A"], list)
    assert isinstance(intervals["B"], list)

    for ch in ("A", "B"):
        for item in intervals[ch]:
            assert "start" in item and "end" in item and "label" in item
            assert item["start"] <= item["end"]
            assert item["label"] in (LABEL_OWN, LABEL_LEAK)
def test_baseline_fails_on_double_talk():
    """Document baseline failure: during double-talk with balanced volumes (|A - B| < 6 dB),
    baseline classifies speech as leak even though both speakers are talking simultaneously."""
    duration_sec = 1.0
    num_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Both speakers talking at similar amplitude (diff < 6 dB, e.g. 2 dB)
    sig_a = 0.4 * np.sin(2 * np.pi * 300 * t)
    sig_b = 0.35 * np.sin(2 * np.pi * 500 * t)

    audio = np.stack([sig_a, sig_b], axis=0).astype(np.float32)
    detector = EnergyBaselineDetector()
    labels, _ = detector.predict_frames(audio, sample_rate=SAMPLE_RATE)

    # In double talk with balanced amplitude:
    # E_A - E_B < 6 dB, so Track A owner speech is erroneously misclassified as LEAK!
    assert np.all(labels["A"] == LABEL_LEAK)
    assert np.all(labels["B"] == LABEL_LEAK)


def test_baseline_fails_on_third_speaker():
    """Document baseline failure: baseline cannot detect 'third' speaker at all."""
    duration_sec = 1.0
    num_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Third speaker stands near mic A, louder than mic B by 10 dB
    sig_a = 0.3 * np.sin(2 * np.pi * 350 * t)
    sig_b = 0.05 * np.sin(2 * np.pi * 350 * t)

    audio = np.stack([sig_a, sig_b], axis=0).astype(np.float32)
    intervals = predict_baseline_rule(audio, sample_rate=SAMPLE_RATE)

    labels_in_a = {item["label"] for item in intervals["A"]}
    # Baseline only predicts 'own' or 'leak', never 'third'
    assert LABEL_THIRD not in labels_in_a
    assert LABEL_OWN in labels_in_a  # Erroneously thinks third speaker is owner!

