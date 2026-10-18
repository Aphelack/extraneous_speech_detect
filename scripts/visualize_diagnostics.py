"""Diagnostic script: Generates 5-panel spectrograms and mono WAV files.

For each scenario ('double_talk', 'high_leak_lt_10db', 'normal'):
1. Generates spectrogram_<scenario>.png with:
   - Panel 1: STFT log-spectrogram of Input Channel A (own + leak)
   - Panel 2: STFT log-spectrogram of Input Channel B (leak source)
   - Panel 3: Detector probabilities P(own), P(leak), P(third) + Ground Truth intervals
   - Panel 4: STFT log-spectrogram of Cleaned Channel A + Suppression mask heatmap G(f, t)
   - Panel 5: STFT log-spectrogram of Target without leak (target_a = input_a - leak_a)
2. Saves individual mono audio files for direct ear evaluation:
   - chA_1_noisy.wav
   - chA_2_cleaned.wav
   - chA_3_reference.wav
"""

import os
import sys
from pathlib import Path
from typing import Union

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

from src.cleaner import TelephoneAudioCleaner
from src.config import FRAME_SAMPLES, SAMPLE_RATE
from src.data_loader import RealSpeechLoader
from src.detector import ExtraneousSpeechDetector
from src.synthesizer import TelephoneDialogueSynthesizer


def compute_stft_log_spectrogram(
    audio_1d: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 256,
    hop: int = FRAME_SAMPLES,
):
    win = signal.windows.hann(n_fft)
    freqs, times, Z = signal.stft(
        audio_1d,
        fs=sample_rate,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop,
    )
    log_spec = 20.0 * np.log10(np.maximum(np.abs(Z), 1e-5))
    return freqs, times, log_spec, Z


def write_mono_wav(path: Union[str, Path], audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    norm = np.clip(audio, -1.0, 1.0)
    int16_data = (norm * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, int16_data)


def run_diagnostics(
    output_dir: str = "samples/diagnostics",
    duration_sec: float = 7.0,
    seed: int = 42,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Visual & Instrumental Audio Cleaning Diagnostics")
    print("=" * 68)

    loader = RealSpeechLoader()
    synth = TelephoneDialogueSynthesizer(seed=seed)
    detector = ExtraneousSpeechDetector()
    cleaner = TelephoneAudioCleaner(detector=detector)

    scenarios = ["double_talk", "high_leak_lt_10db", "normal", "third_speaker"]
    rng = np.random.RandomState(seed)

    for sc in scenarios:
        print(f"\nProcessing diagnostic scenario: {sc}")
        sc_dir = out_dir / sc
        sc_dir.mkdir(parents=True, exist_ok=True)

        sa, sb, st = loader.sample_dialogue_speakers("test", russian_prob=0.80, rng=rng)
        pool = loader.build_utterance_pool(sa, sb, st, max_per_speaker=8)

        # Set audible leak (8-12 dB attenuation) so leak is clearly audible on input_a and eliminated on cleaned_a
        leak_atten = 10.0 if sc in ("double_talk", "high_leak_lt_10db") else 12.0

        res = synth.synthesize_dialogue(
            utterance_pool=pool,
            duration_sec=duration_sec,
            scenario=sc,
            leak_attenuation_db=leak_atten,
            third_speaker_prob=1.0 if sc == "third_speaker" else 0.0,
        )

        input_a = res.audio[0]
        input_b = res.audio[1]
        target_a = res.target_audio["A"] if res.target_audio else res.clean_references["A"]

        # Run detector
        probs = detector.predict_probabilities(res.audio, sample_rate=res.sample_rate)
        probs_a = probs["A"]

        # Run cleaner with mask return
        cleaned_stereo, masks = cleaner.clean_stereo(res.audio, sample_rate=res.sample_rate, return_masks=True)
        cleaned_a = cleaned_stereo[0]
        mask_a = masks["A"]  # (129, T)

        # Save individual mono files
        # 1. Inside subfolder
        write_mono_wav(sc_dir / "chA_1_noisy.wav", input_a, res.sample_rate)
        write_mono_wav(sc_dir / "chA_2_cleaned.wav", cleaned_a, res.sample_rate)
        write_mono_wav(sc_dir / "chA_3_reference.wav", target_a, res.sample_rate)

        # 2. In root of samples/diagnostics/ with prefix for quick access
        write_mono_wav(out_dir / f"{sc}_chA_1_noisy.wav", input_a, res.sample_rate)
        write_mono_wav(out_dir / f"{sc}_chA_2_cleaned.wav", cleaned_a, res.sample_rate)
        write_mono_wav(out_dir / f"{sc}_chA_3_reference.wav", target_a, res.sample_rate)

        # Compute STFTs
        f_a, t_a, spec_in_a, _ = compute_stft_log_spectrogram(input_a, res.sample_rate)
        f_b, t_b, spec_in_b, _ = compute_stft_log_spectrogram(input_b, res.sample_rate)
        f_c, t_c, spec_clean_a, _ = compute_stft_log_spectrogram(cleaned_a, res.sample_rate)
        f_tgt, t_tgt, spec_tgt_a, _ = compute_stft_log_spectrogram(target_a, res.sample_rate)

        # Build 6-panel Figure (5 logical panels with both cleaned spectrogram and mask heatmap)
        fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1.0, 1.2, 1.0, 1.2]})
        plt.subplots_adjust(hspace=0.28)

        vmin, vmax = -60.0, 0.0

        # Panel 1: Input A
        im1 = axes[0].pcolormesh(t_a, f_a, spec_in_a, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"[{sc}] Panel 1: Input Channel A (Target: Own Speech + Acoustic Leak)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Freq (Hz)")
        plt.colorbar(im1, ax=axes[0], pad=0.01).set_label("dB")

        # Panel 2: Input B
        im2 = axes[1].pcolormesh(t_b, f_b, spec_in_b, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
        axes[1].set_title("Panel 2: Input Channel B (Counterpart Speech — Leak Source)", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Freq (Hz)")
        plt.colorbar(im2, ax=axes[1], pad=0.01).set_label("dB")

        # Panel 3: Probabilities and Ground Truth
        ax3 = axes[2]
        time_probs = np.arange(len(probs_a["own"])) * 0.010  # 10 ms hop

        # Ground Truth spans
        gt_intervals = res.ground_truth_intervals["A"]
        for seg in gt_intervals:
            lbl = seg["label"]
            st_s, ed_s = seg["start"], seg["end"]
            if lbl == "own":
                ax3.axvspan(st_s, ed_s, color="#2ca02c", alpha=0.18)
            elif lbl == "leak":
                ax3.axvspan(st_s, ed_s, color="#d62728", alpha=0.18)
            elif lbl == "third":
                ax3.axvspan(st_s, ed_s, color="#ff7f0e", alpha=0.18)

        ax3.plot(time_probs, probs_a["own"], label="P(own)", color="#2ca02c", lw=1.8)
        ax3.plot(time_probs, probs_a["leak"], label="P(leak)", color="#d62728", lw=1.8)
        ax3.plot(time_probs, probs_a["third"], label="P(third)", color="#ff7f0e", lw=1.8)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_ylabel("Probability")
        ax3.set_title("Panel 3: Extraneous Speech Detector Probabilities & Ground Truth Segments", fontsize=11, fontweight="bold")
        ax3.legend(loc="upper right", ncol=3, framealpha=0.9, fontsize=9)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Panel 4a: Cleaned Channel A
        im4a = axes[3].pcolormesh(t_c, f_c, spec_clean_a, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
        axes[3].set_title("Panel 4a: Cleaned Channel A (After Neural Spectral Masking)", fontsize=11, fontweight="bold")
        axes[3].set_ylabel("Freq (Hz)")
        plt.colorbar(im4a, ax=axes[3], pad=0.01).set_label("dB")

        # Panel 4b: Suppression Mask Heatmap G(f, t)
        t_mask = np.linspace(0, duration_sec, mask_a.shape[1])
        im4b = axes[4].pcolormesh(t_mask, f_c, mask_a, shading="gouraud", cmap="viridis", vmin=0.0, vmax=1.0)
        axes[4].set_title("Panel 4b: Neural Spectral Suppression Mask G(f, t) Heatmap (1.0 = Keep, 0.0 = Suppress)", fontsize=11, fontweight="bold")
        axes[4].set_ylabel("Freq (Hz)")
        plt.colorbar(im4b, ax=axes[4], pad=0.01).set_label("Gain")

        # Panel 5: Clean Reference / Target A without leak
        im5 = axes[5].pcolormesh(t_tgt, f_tgt, spec_tgt_a, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
        axes[5].set_title("Panel 5: Clean Target Channel A without Acoustic Leak (target_a = input_a - leak_a)", fontsize=11, fontweight="bold")
        axes[5].set_ylabel("Freq (Hz)")
        axes[5].set_xlabel("Time (seconds)", fontsize=11)
        plt.colorbar(im5, ax=axes[5], pad=0.01).set_label("dB")

        axes[5].set_xlim(0, duration_sec)

        plot_path = out_dir / f"spectrogram_{sc}.png"
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved diagnostic plot: {plot_path}")
        print(f"  -> Saved mono audio: {sc_dir / 'chA_1_noisy.wav'}")
        print(f"  -> Saved mono audio: {sc_dir / 'chA_2_cleaned.wav'}")
        print(f"  -> Saved mono audio: {sc_dir / 'chA_3_reference.wav'}")

    print(f"\nAll diagnostics completed successfully in: {output_dir}")


if __name__ == "__main__":
    run_diagnostics()

