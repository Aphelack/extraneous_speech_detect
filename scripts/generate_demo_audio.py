#!/usr/bin/env python3
"""Generate a listening demo of the detector and cleaner after the review fixes.

For every scenario and language the script writes:

    samples/demo_post_review/<scenario>/<lang>/
        input_stereo.wav      original two-channel recording
        cleaned_stereo.wav    output of clean.py
        chA_1_noisy.wav       track A before cleaning (mono, 8 kHz)
        chA_2_cleaned.wav     track A after cleaning (mono, 8 kHz)
        chA_3_clean_ref.wav   track A without leakage (mono, 8 kHz)
        detection_plot.png    spectrograms + P(own)/P(leak)/P(third) vs GT
        metrics.json          per-record metrics

Scenarios:
    1_normal_echo        turn taking, leak attenuated by 10-14 dB
    2_high_leak_lt_10db  severe leak, 6-9 dB (counterpart almost as loud)
    3_double_talk        simultaneous speech and interruptions
    4_no_leak            no leakage at all (cleaner must not damage the audio)

Languages:
    ru  speakers from data/russian_speech/spk_5..7 (test split)
    en  speakers from data/voxceleb1_test (test split)

Usage:
    python scripts/generate_demo_audio.py
    python scripts/generate_demo_audio.py --duration 9.0 --seed 7
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal  # noqa: E402

from src.audio_utils import save_stereo_audio  # noqa: E402
from src.cleaner import TelephoneAudioCleaner  # noqa: E402
from src.config import COLLAR_FRAMES, FRAME_DURATION_SEC, LABEL_LEAK, SAMPLE_RATE  # noqa: E402
from src.data_loader import RealSpeechLoader  # noqa: E402
from src.detector import ExtraneousSpeechDetector  # noqa: E402
from src.embedder import SpeakerEmbedder  # noqa: E402
from src.metrics import (  # noqa: E402
    FrameCountAccumulator,
    cosine_similarity,
    intervals_to_frame_matrix,
)
from src.synthesizer import TelephoneDialogueSynthesizer  # noqa: E402

DEMO_SCENARIOS: List[Dict[str, Any]] = [
    {
        "key": "1_normal_echo",
        "scenario": "normal",
        "leak_attenuation_db": (10.0, 14.0),
        "description": "Turn taking with a clearly audible echo (10-14 dB).",
    },
    {
        "key": "2_high_leak_lt_10db",
        "scenario": "high_leak_lt_10db",
        "leak_attenuation_db": (6.0, 9.0),
        "description": "Severe leak: the counterpart is almost as loud as the owner.",
    },
    {
        "key": "3_double_talk",
        "scenario": "double_talk",
        "leak_attenuation_db": None,
        "description": "Simultaneous speech and interruptions.",
    },
    {
        "key": "4_no_leak",
        "scenario": "no_leak",
        "leak_attenuation_db": None,
        "description": "No leakage at all: the cleaner must leave the audio intact.",
    },
]

LANGUAGES = ("ru", "en")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-10) -> float:
    """Scale-invariant signal-to-distortion ratio in dB."""
    n = min(len(reference), len(estimate))
    ref = reference[:n].astype(np.float64)
    est = estimate[:n].astype(np.float64)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.dot(ref, ref) + eps
    alpha = np.dot(est, ref) / ref_energy
    target = alpha * ref
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def leak_attenuation_db(
    raw: np.ndarray,
    cleaned: np.ndarray,
    gt_intervals: List[Dict[str, Any]],
    sr: int,
) -> Optional[float]:
    """Energy reduction on pure-leak frames (owner silent)."""
    speech = np.zeros(len(raw), dtype=bool)
    leak = np.zeros(len(raw), dtype=bool)
    for seg in gt_intervals:
        s = max(0, int(seg["start"] * sr))
        e = min(len(raw), int(seg["end"] * sr))
        if seg["label"] in ("own", "third"):
            speech[s:e] = True
        elif seg["label"] == "leak":
            leak[s:e] = True
    pure_leak = leak & ~speech
    if np.sum(pure_leak) < 800:
        return None
    raw_power = float(np.mean(raw[pure_leak] ** 2)) + 1e-10
    clean_power = float(np.mean(cleaned[pure_leak] ** 2)) + 1e-10
    return float(10.0 * np.log10(raw_power / clean_power))


def track_purity(
    signal_audio: np.ndarray,
    reference_audio: np.ndarray,
    gt_intervals: List[Dict[str, Any]],
    sr: int,
    embedder: Optional[SpeakerEmbedder],
) -> Optional[float]:
    """ECAPA cosine similarity between the track and the owner reference."""
    if embedder is None:
        return None
    slices_sig, slices_ref = [], []
    for seg in gt_intervals:
        if seg["label"] in ("own", "leak"):
            s = max(0, int(seg["start"] * sr))
            e = min(len(signal_audio), int(seg["end"] * sr))
            if e - s > 400:
                slices_sig.append(signal_audio[s:e])
                slices_ref.append(reference_audio[s:e])
    if not slices_sig:
        return None
    cat_sig = np.concatenate(slices_sig)
    cat_ref = np.concatenate(slices_ref)
    if len(cat_sig) < 2400 or len(cat_ref) < 2400:
        return None
    emb_sig = embedder.embed(cat_sig, sample_rate=sr)
    emb_ref = embedder.embed(cat_ref, sample_rate=sr)
    return float(cosine_similarity(emb_sig, emb_ref))


def record_f1(
    gt: Dict[str, List[Dict[str, Any]]],
    pred: Dict[str, List[Dict[str, Any]]],
    num_frames: int,
) -> Dict[str, float]:
    """Global micro-F1 for a single record (both channels summed)."""
    acc = FrameCountAccumulator()
    acc.add(gt, pred, num_frames=num_frames, collar_frames=COLLAR_FRAMES)
    result = acc.finalize()
    return {
        "own": result["by_label"]["own"]["f1"],
        "leak": result["by_label"]["leak"]["f1"],
        "third": result["by_label"]["third"]["f1"],
        "macro": result["macro_f1"],
    }


def leak_fpr(
    gt: Dict[str, List[Dict[str, Any]]],
    pred: Dict[str, List[Dict[str, Any]]],
    num_frames: int,
) -> Optional[float]:
    """Leak false-positive rate on frames where the counterpart speaks."""
    fp = 0
    negatives = 0
    for ch, other in (("A", "B"), ("B", "A")):
        gt_leak = intervals_to_frame_matrix(gt.get(ch, []), num_frames, labels=(LABEL_LEAK,))[LABEL_LEAK]
        pred_leak = intervals_to_frame_matrix(pred.get(ch, []), num_frames, labels=(LABEL_LEAK,))[LABEL_LEAK]
        counterpart = intervals_to_frame_matrix(gt.get(other, []), num_frames, labels=("own",))["own"]
        mask = counterpart & ~gt_leak
        fp += int(np.sum(pred_leak & mask))
        negatives += int(np.sum(mask))
    if negatives == 0:
        return None
    return float(fp / negatives)


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_detection(
    noisy: np.ndarray,
    cleaned: np.ndarray,
    reference: np.ndarray,
    probs: Dict[str, np.ndarray],
    gt_intervals: List[Dict[str, Any]],
    num_frames: int,
    sr: int,
    title: str,
    output_path: Path,
) -> None:
    """Spectrograms of the three signals plus probability curves vs ground truth."""
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)

    signals = [
        (noisy, "chA_1_noisy (owner + echo)"),
        (cleaned, "chA_2_cleaned (echo suppressed)"),
        (reference, "chA_3_clean_ref (no leakage)"),
    ]
    for ax, (sig, label) in zip(axes[:3], signals):
        freqs, times, spec = signal.spectrogram(
            sig, fs=sr, nperseg=256, noverlap=192, window="hann"
        )
        ax.pcolormesh(
            times,
            freqs,
            10.0 * np.log10(spec + 1e-12),
            shading="auto",
            cmap="magma",
            vmin=-110,
            vmax=-20,
        )
        ax.set_ylabel("Hz")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_ylim(0, 3800)

    # Ground-truth masks
    gt_masks = intervals_to_frame_matrix(
        gt_intervals, num_frames, frame_sec=FRAME_DURATION_SEC, labels=("own", "leak", "third")
    )
    frame_times = np.arange(num_frames) * FRAME_DURATION_SEC

    ax = axes[3]
    ax.pcolormesh(
        frame_times,
        np.arange(3),
        np.vstack([gt_masks["own"], gt_masks["leak"], gt_masks["third"]]).astype(float),
        shading="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["own", "leak", "third"])
    ax.set_title("Ground truth (track A)", loc="left", fontsize=10)

    ax = axes[4]
    for label, color in (("own", "tab:green"), ("leak", "tab:red"), ("third", "tab:blue")):
        curve = probs[label]
        ax.plot(frame_times[: len(curve)], curve, label=f"P({label})", color=color, linewidth=1.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_title("Detector probabilities (track A)", loc="left", fontsize=10)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------
# Speaker pools
# ----------------------------------------------------------------------
def build_pools(loader: RealSpeechLoader) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Build one utterance pool per language using test-split speakers only."""
    splits = loader.get_disjoint_splits()
    test_speakers = list(splits["test"])
    ru_speakers = [s for s in test_speakers if s.startswith("ru_")]
    vox_speakers = [s for s in test_speakers if s.startswith("vox_")]

    if len(ru_speakers) < 3:
        raise RuntimeError(f"Need 3 Russian test speakers, found {ru_speakers}")
    if len(vox_speakers) < 3:
        raise RuntimeError(f"Need 3 VoxCeleb test speakers, found {vox_speakers}")

    pools: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for lang, speakers in (("ru", ru_speakers[:3]), ("en", vox_speakers[:3])):
        pools[lang] = {
            "A": loader.load_speaker_utterances(speakers[0], max_utterances=12),
            "B": loader.load_speaker_utterances(speakers[1], max_utterances=12),
            "third": loader.load_speaker_utterances(speakers[2], max_utterances=12),
        }
        print(
            f"  {lang}: A={speakers[0]} ({len(pools[lang]['A'])} utt), "
            f"B={speakers[1]} ({len(pools[lang]['B'])} utt), "
            f"third={speakers[2]} ({len(pools[lang]['third'])} utt)"
        )
    return pools


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def generate_demo(
    output_root: str = "samples/demo_post_review",
    duration_sec: float = 9.0,
    seed: int = 42,
) -> Dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DEMO GENERATION: detector + cleaner after the review fixes")
    print("=" * 78)

    loader = RealSpeechLoader()
    print("Building test-split speaker pools:")
    pools = build_pools(loader)

    detector = ExtraneousSpeechDetector()
    cleaner = TelephoneAudioCleaner(detector=detector)
    try:
        embedder: Optional[SpeakerEmbedder] = SpeakerEmbedder()
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Notice: SpeakerEmbedder unavailable ({exc}); purity will be skipped.")
        embedder = None

    summary: Dict[str, Any] = {}
    commands: List[str] = []

    for scenario_index, spec in enumerate(DEMO_SCENARIOS):
        for lang_index, lang in enumerate(LANGUAGES):
            record_seed = seed + scenario_index * 100 + lang_index
            synth = TelephoneDialogueSynthesizer(seed=record_seed)

            leak_attenuation = None
            if spec["leak_attenuation_db"] is not None:
                low, high = spec["leak_attenuation_db"]
                leak_attenuation = float(synth.rng.uniform(low, high))

            result = synth.synthesize_dialogue(
                utterance_pool=pools[lang],
                duration_sec=duration_sec,
                scenario=spec["scenario"],
                leak_attenuation_db=leak_attenuation,
                third_speaker_prob=0.0,
                # Guarantee the demo matches its description: scenarios 1-3
                # must contain leakage, scenario 4 must not.
                force_leak=spec["scenario"] != "no_leak",
            )

            stereo = result.audio
            sr = result.sample_rate
            gt = result.ground_truth_intervals
            num_frames = result.num_frames

            # --- Detector + cleaner ---
            probs = detector.predict_probabilities(stereo, sample_rate=sr)
            preds = detector.predict_intervals(stereo, sample_rate=sr, probabilities=probs)
            cleaned = cleaner.clean(stereo, preds, sr, probabilities=probs)

            # --- Reference signals for track A ---
            noisy_a = stereo[0]
            cleaned_a = cleaned[0]
            # `target_audio["A"]` is the track with the leak component removed:
            # exactly what the cleaner is supposed to produce.
            reference_a = result.target_audio["A"] if result.target_audio else result.clean_references["A"]

            # --- Metrics ---
            f1 = record_f1(gt, preds, num_frames)
            fpr = leak_fpr(gt, preds, num_frames)
            attenuation = leak_attenuation_db(noisy_a, cleaned_a, gt["A"], sr)

            # SI-SDR is only meaningful when the reference actually differs from
            # the input. In `no_leak` records the target equals the input, so
            # the ratio is dominated by numerical noise and must not be reported.
            has_leak = bool(result.metadata["leak_enabled_a"])
            if has_leak:
                sdr_before: Optional[float] = si_sdr(reference_a, noisy_a)
                sdr_after: Optional[float] = si_sdr(reference_a, cleaned_a)
                sdr_improvement: Optional[float] = sdr_after - sdr_before
            else:
                sdr_before = sdr_after = sdr_improvement = None

            purity_before = track_purity(noisy_a, reference_a, gt["A"], sr, embedder)
            purity_after = track_purity(cleaned_a, reference_a, gt["A"], sr, embedder)

            # --- Write files ---
            out_dir = root / spec["key"] / lang
            out_dir.mkdir(parents=True, exist_ok=True)

            save_stereo_audio(str(out_dir / "input_stereo.wav"), stereo, sample_rate=sr)
            save_stereo_audio(str(out_dir / "cleaned_stereo.wav"), cleaned, sample_rate=sr)

            for name, sig in (
                ("chA_1_noisy.wav", noisy_a),
                ("chA_2_cleaned.wav", cleaned_a),
                ("chA_3_clean_ref.wav", reference_a),
            ):
                sf.write(out_dir / name, np.clip(sig, -1.0, 1.0), sr, subtype="PCM_16")

            plot_detection(
                noisy=noisy_a,
                cleaned=cleaned_a,
                reference=reference_a,
                probs=probs["A"],
                gt_intervals=gt["A"],
                num_frames=num_frames,
                sr=sr,
                title=(
                    f"{spec['key']} / {lang} — {spec['description']}\n"
                    f"leak attenuation {result.metadata['leak_attenuation_db']:.1f} dB, "
                    f"delay {result.metadata['leak_delay_sec'] * 1000:.0f} ms"
                ),
                output_path=out_dir / "detection_plot.png",
            )

            metrics = {
                "scenario": spec["key"],
                "language": lang,
                "seed": record_seed,
                "duration_sec": duration_sec,
                "leak_attenuation_db": result.metadata["leak_attenuation_db"],
                "leak_delay_ms": result.metadata["leak_delay_sec"] * 1000.0,
                "leak_enabled_a": result.metadata["leak_enabled_a"],
                "leak_enabled_b": result.metadata["leak_enabled_b"],
                "f1": f1,
                "leak_fpr": fpr,
                "leak_attenuation_measured_db": attenuation,
                "si_sdr_before_db": sdr_before,
                "si_sdr_after_db": sdr_after,
                "si_sdr_improvement_db": sdr_improvement,
                "track_purity_before": purity_before,
                "track_purity_after": purity_after,
            }
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary[f"{spec['key']}/{lang}"] = metrics

            commands.extend(
                [
                    f"ffplay -autoexit -nodisp {out_dir / 'chA_1_noisy.wav'}",
                    f"ffplay -autoexit -nodisp {out_dir / 'chA_2_cleaned.wav'}",
                    f"ffplay -autoexit -nodisp {out_dir / 'chA_3_clean_ref.wav'}",
                ]
            )

            sdr_text = (
                f"SI-SDR {sdr_before:+.1f} -> {sdr_after:+.1f} dB"
                if sdr_improvement is not None
                else "SI-SDR n/a (no leak)"
            )
            print(
                f"  [{spec['key']}/{lang}] own={f1['own']:.3f} leak={f1['leak']:.3f} "
                f"third={f1['third']:.3f} | atten={attenuation if attenuation is None else round(attenuation, 1)} dB "
                f"| {sdr_text}"
            )

    # --- Console report ---
    print("\n" + "=" * 78)
    print("PER-FILE METRICS")
    print("=" * 78)
    header = (
        f"{'scenario/lang':<26} | {'Own':>5} | {'Leak':>5} | {'Third':>5} | "
        f"{'LeakAtt':>8} | {'dSI-SDR':>8} | {'Purity':>7}"
    )
    print(header)
    print("-" * len(header))
    for key, m in summary.items():
        atten = m["leak_attenuation_measured_db"]
        atten_str = "n/a" if atten is None else f"{atten:.1f}"
        purity = m["track_purity_after"]
        purity_str = "n/a" if purity is None else f"{purity:.3f}"
        sdr = m["si_sdr_improvement_db"]
        sdr_str = "n/a" if sdr is None else f"{sdr:+.2f}"
        print(
            f"{key:<26} | {m['f1']['own']:>5.3f} | {m['f1']['leak']:>5.3f} | "
            f"{m['f1']['third']:>5.3f} | {atten_str:>8} | "
            f"{sdr_str:>8} | {purity_str:>7}"
        )
    print("-" * len(header))

    print("\nListening commands (mono track A, 8 kHz):")
    for cmd in commands:
        print(f"  {cmd}")
    print("\nAlternative player: aplay <file.wav>")

    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDemo written to: {root}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="samples/demo_post_review")
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_demo(
        output_root=args.output_root,
        duration_sec=args.duration,
        seed=args.seed,
    )
