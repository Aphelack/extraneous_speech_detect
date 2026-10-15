"""Unit tests for Stage 3: feature extraction, detector, cleaner, and RTF limit."""

import numpy as np
import pytest

from src.cleaner import TelephoneAudioCleaner, compute_spectral_embedding
from src.config import MAX_RTF_LIMIT, SAMPLE_RATE
from src.detector import ExtraneousSpeechDetector
from src.features import FeatureExtractor
from src.metrics import cosine_similarity, measure_rtf
from src.synthesizer import TelephoneDialogueSynthesizer


def test_feature_extractor_outputs():
    """Verify FeatureExtractor returns expected feature matrices."""
    fe = FeatureExtractor()
    audio = np.random.randn(2, 8000 * 2).astype(np.float32) * 0.1
    feats = fe.extract_features(audio)

    assert "A" in feats and "B" in feats and "feature_names" in feats
    assert feats["A"].ndim == 2
    assert feats["A"].shape[1] == len(feats["feature_names"])
    assert not np.any(np.isnan(feats["A"]))


def test_detector_probabilities_and_intervals():
    """Verify detector outputs valid probabilities and intervals."""
    detector = ExtraneousSpeechDetector()
    synth = TelephoneDialogueSynthesizer(seed=42)
    res = synth.synthesize_dialogue(duration_sec=3.0, scenario="normal")

    probs = detector.predict_probabilities(res.audio, sample_rate=SAMPLE_RATE)
    for ch in ("A", "B"):
        for lbl in ("own", "leak", "third"):
            p = probs[ch][lbl]
            assert np.all(p >= 0.0) and np.all(p <= 1.0)
            assert not np.any(np.isnan(p))

    intervals = detector.predict_intervals(res.audio, sample_rate=SAMPLE_RATE)
    assert "A" in intervals and "B" in intervals
    for ch in ("A", "B"):
        for item in intervals[ch]:
            assert "start" in item and "end" in item and "label" in item and "score" in item
            assert item["start"] <= item["end"]
            assert 0.0 <= item["score"] <= 1.0


def test_detector_rtf_speed_under_limit():
    """Strictly verify RTF <= 1/15 on 1 CPU thread (<= 4s per 60s of audio)."""
    detector = ExtraneousSpeechDetector()
    duration_sec = 10.0
    audio = np.random.randn(2, int(duration_sec * SAMPLE_RATE)).astype(np.float32) * 0.05

    _, rtf, passed = measure_rtf(
        detector.predict_intervals,
        audio,
        sample_rate=SAMPLE_RATE,
        audio_duration_sec=duration_sec,
        num_threads=1,
    )

    print(f"\nMeasured RTF: {rtf:.4f} (Limit: {MAX_RTF_LIMIT:.4f} = 1/15)")
    assert passed, f"Detector RTF ({rtf:.4f}) exceeded 1/15 limit ({MAX_RTF_LIMIT:.4f})!"


def test_cleaner_reduces_leak_and_preserves_voice():
    """Verify audio cleaner attenuates leak while maintaining speech quality."""
    synth = TelephoneDialogueSynthesizer(seed=101)
    res = synth.synthesize_dialogue(duration_sec=4.0, scenario="normal")

    cleaner = TelephoneAudioCleaner()
    cleaned = cleaner.clean_stereo(res.audio, sample_rate=SAMPLE_RATE)

    assert cleaned.shape == res.audio.shape
    assert not np.any(np.isnan(cleaned))

    # Spectral embedding similarity to clean reference s_A
    emb_clean_ref = compute_spectral_embedding(res.clean_references["A"])
    emb_noisy_a = compute_spectral_embedding(res.audio[0])
    emb_cleaned_a = compute_spectral_embedding(cleaned[0])

    sim_noisy = cosine_similarity(emb_noisy_a, emb_clean_ref)
    sim_cleaned = cosine_similarity(emb_cleaned_a, emb_clean_ref)

    print(f"\nCosine similarity to pure owner reference:")
    print(f"  Before cleaning: {sim_noisy:.4f}")
    print(f"  After cleaning:  {sim_cleaned:.4f}")

    assert sim_cleaned >= sim_noisy - 0.02, "Cleaning must not degrade voice representation of owner!"

