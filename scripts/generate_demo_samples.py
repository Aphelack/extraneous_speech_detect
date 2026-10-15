"""Generate real audio demonstration samples and run full evaluation.

Creates realistic telephone dialogue recordings using real human speech,
runs baseline rule vs detector, cleans acoustic leak, and calculates
comprehensive metrics.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.audio_utils import save_stereo_audio
from src.baseline import EnergyBaselineDetector
from src.cleaner import TelephoneAudioCleaner, compute_spectral_embedding
from src.config import FRAME_DURATION_SEC, SAMPLE_RATE
from src.data_loader import RealSpeechLoader
from src.detector import ExtraneousSpeechDetector
from src.metrics import cosine_similarity, evaluate_channels
from src.synthesizer import TelephoneDialogueSynthesizer


def main():
    out_dir = Path("samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Real Audio Telephone Samples & Benchmarking")
    print("=" * 70)

    # 1. Load real speech utterances
    loader = RealSpeechLoader(data_root="data/cmuarctic")
    speakers = loader.discover_speakers()
    print(f"Available real speakers in dataset: {list(speakers.keys())}")

    speaker_a = "aew" if "aew" in speakers else list(speakers.keys())[0]
    speaker_b = "slt" if "slt" in speakers else (list(speakers.keys())[1] if len(speakers) > 1 else speaker_a)
    speaker_3 = "bdl" if "bdl" in speakers else None

    print(f"Assigning speakers -> Track A (Owner): {speaker_a}, Track B (Counterpart): {speaker_b}, Third: {speaker_3}")
    pool = loader.build_utterance_pool(speaker_a=speaker_a, speaker_b=speaker_b, speaker_third=speaker_3, max_per_speaker=30)

    synth = TelephoneDialogueSynthesizer(seed=2026)
    detector = ExtraneousSpeechDetector()
    baseline = EnergyBaselineDetector()
    cleaner = TelephoneAudioCleaner(detector=detector)

    scenarios = [
        ("normal", 10.0, "Normal turn-taking with moderate leak"),
        ("double_talk", 12.0, "Simultaneous double talk between speakers A and B"),
        ("high_leak_lt_10db", 10.0, "Severe leak (< 10 dB attenuation)"),
        ("third_speaker", 12.0, "Third speaker present near Track A microphone"),
    ]

    summary_rows = []

    for sc_name, dur, sc_desc in scenarios:
        print(f"\n--- Scenario: {sc_name} ({sc_desc}) ---")
        sc_folder = out_dir / sc_name
        sc_folder.mkdir(parents=True, exist_ok=True)

        res = synth.synthesize_dialogue(
            utterance_pool=pool,
            duration_sec=dur,
            scenario=sc_name,
            third_speaker_prob=1.0 if sc_name == "third_speaker" else 0.0,
        )

        # 1. Save mixed input stereo WAV
        mixed_path = sc_folder / "input_telephone_mixed.wav"
        save_stereo_audio(str(mixed_path), res.audio, sample_rate=res.sample_rate)

        # 2. Save pure references for Track A and Track B
        clean_ref_a_stereo = np.stack([res.clean_references["A"], np.zeros_like(res.clean_references["A"])], axis=0)
        clean_ref_b_stereo = np.stack([np.zeros_like(res.clean_references["B"]), res.clean_references["B"]], axis=0)
        save_stereo_audio(str(sc_folder / "reference_clean_A.wav"), clean_ref_a_stereo, sample_rate=res.sample_rate)
        save_stereo_audio(str(sc_folder / "reference_clean_B.wav"), clean_ref_b_stereo, sample_rate=res.sample_rate)

        # 3. Save Ground Truth JSON
        gt_path = sc_folder / "ground_truth.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(res.ground_truth_intervals, f, indent=2, ensure_ascii=False)

        # 4. Predict Baseline
        base_preds = baseline.predict_intervals(res.audio, sample_rate=res.sample_rate)
        with open(sc_folder / "predictions_baseline.json", "w", encoding="utf-8") as f:
            json.dump(base_preds, f, indent=2, ensure_ascii=False)

        # 5. Predict Detector
        det_preds = detector.predict_intervals(res.audio, sample_rate=res.sample_rate)
        with open(sc_folder / "predictions_detector.json", "w", encoding="utf-8") as f:
            json.dump(det_preds, f, indent=2, ensure_ascii=False)

        # 6. Clean audio with cleaner
        cleaned_audio = cleaner.clean_stereo(res.audio, sample_rate=res.sample_rate)
        save_stereo_audio(str(sc_folder / "cleaned_output.wav"), cleaned_audio, sample_rate=res.sample_rate)

        # 7. Evaluate metrics with 50 ms collar
        m_base = evaluate_channels(
            ground_truth=res.ground_truth_intervals,
            predictions=base_preds,
            num_frames=res.num_frames,
            frame_sec=FRAME_DURATION_SEC,
            collar_frames=5,
        )
        m_det = evaluate_channels(
            ground_truth=res.ground_truth_intervals,
            predictions=det_preds,
            num_frames=res.num_frames,
            frame_sec=FRAME_DURATION_SEC,
            collar_frames=5,
        )

        # Embedding cosine similarities to pure reference
        emb_ref_a = compute_spectral_embedding(res.clean_references["A"])
        emb_mixed_a = compute_spectral_embedding(res.audio[0])
        emb_clean_a = compute_spectral_embedding(cleaned_audio[0])

        sim_before = cosine_similarity(emb_mixed_a, emb_ref_a)
        sim_after = cosine_similarity(emb_clean_a, emb_ref_a)

        row = {
            "scenario": sc_name,
            "base_macro_f1": m_base["macro_f1"],
            "det_macro_f1": m_det["macro_f1"],
            "base_own_f1": m_base["by_label"]["own"]["f1"],
            "det_own_f1": m_det["by_label"]["own"]["f1"],
            "base_leak_f1": m_base["by_label"]["leak"]["f1"],
            "det_leak_f1": m_det["by_label"]["leak"]["f1"],
            "base_third_f1": m_base["by_label"]["third"]["f1"],
            "det_third_f1": m_det["by_label"]["third"]["f1"],
            "sim_before": sim_before,
            "sim_after": sim_after,
        }
        summary_rows.append(row)

        print(f"  Baseline Macro F1: {m_base['macro_f1']:.4f}  |  Detector Macro F1: {m_det['macro_f1']:.4f}")
        print(f"    - Own   F1: Base = {row['base_own_f1']:.3f} -> Det = {row['det_own_f1']:.3f}")
        print(f"    - Leak  F1: Base = {row['base_leak_f1']:.3f} -> Det = {row['det_leak_f1']:.3f}")
        print(f"    - Third F1: Base = {row['base_third_f1']:.3f} -> Det = {row['det_third_f1']:.3f}")
        print(f"  Voice Embedding Cosine Sim: {sim_before:.4f} (noisy) -> {sim_after:.4f} (cleaned)")
        print(f"  Files saved in: {sc_folder}/")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE (Real Speech Evaluation with 50 ms Collar)")
    print("=" * 70)
    header = f"{'Scenario':<18} | {'Base F1':<8} | {'Det F1':<8} | {'Own F1 (B/D)':<14} | {'Leak F1 (B/D)':<14} | {'Third F1 (B/D)':<14}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{r['scenario']:<18} | {r['base_macro_f1']:<8.3f} | {r['det_macro_f1']:<8.3f} | "
            f"{r['base_own_f1']:.2f} / {r['det_own_f1']:.2f}     | "
            f"{r['base_leak_f1']:.2f} / {r['det_leak_f1']:.2f}     | "
            f"{r['base_third_f1']:.2f} / {r['det_third_f1']:.2f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()

