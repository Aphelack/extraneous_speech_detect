#!/usr/bin/env python3
"""Cross-generator generalisation test.

This script deliberately does **not** import ``TelephoneDialogueSynthesizer``.
It re-implements the physical model from scratch with different choices, so a
good score here is evidence that the detector learned transferable acoustic
cues rather than the artefacts of ``src/synthesizer.py``.

Differences from the training generator
---------------------------------------
* Nonlinearity: asymmetric soft/hard clipping (different positive/negative
  thresholds) followed by a simple AGC envelope compressor, instead of a
  symmetric ``tanh``.
* Transmission filter: 6th-order elliptic band-pass 280-3450 Hz instead of a
  4th-order Butterworth 300-3400 Hz.
* Quantisation: 8-bit mu-law instead of G.711 A-law.
* Noise: synthetic pink noise instead of OpenSLR point-source recordings.
* RIR: independent exponentially-decaying noise model with random DRR/T60.
* Conversation structure: long monologues up to 6 s, short 0.4 s inserts,
  random missing leak on one of the tracks.

Ground truth is derived from the audible energy of each generated component on
the 10 ms grid (same principle as the training generator, independent code).

Usage
-----
    python scripts/test_cross_generator.py
    python scripts/test_cross_generator.py --records-per-scenario 4 --seed 7
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.audio_utils import save_stereo_audio  # noqa: E402
from src.baseline import EnergyBaselineDetector  # noqa: E402
from src.cleaner import TelephoneAudioCleaner  # noqa: E402
from src.config import COLLAR_FRAMES, FRAME_DURATION_SEC, LABEL_LEAK, SAMPLE_RATE  # noqa: E402
from src.data_loader import RealSpeechLoader  # noqa: E402
from src.detector import ExtraneousSpeechDetector  # noqa: E402
from src.embedder import SpeakerEmbedder  # noqa: E402
from src.metrics import (  # noqa: E402
    FrameCountAccumulator,
    cosine_similarity,
    frame_mask_to_intervals,
    intervals_to_frame_matrix,
)

SR = SAMPLE_RATE
HOP = int(SR * FRAME_DURATION_SEC)  # 80 samples

SCENARIOS = [
    "normal",
    "double_talk",
    "high_leak_lt_10db",
    "path_drift",
    "third_speaker",
    "no_leak",
]

# Audibility thresholds (independent implementation, same physical principle).
AUDIBLE_NOISE_RATIO = 1.5
AUDIBLE_ABS_DBFS = -52.0


# ----------------------------------------------------------------------
# Independent DSP building blocks
# ----------------------------------------------------------------------
def elliptic_bandpass(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """6th-order elliptic band-pass 280-3450 Hz (different from training)."""
    sos = signal.ellip(
        6,
        rp=0.5,
        rs=45.0,
        Wn=[280.0, 3450.0],
        btype="bandpass",
        fs=sr,
        output="sos",
    )
    return signal.sosfilt(sos, x).astype(np.float32)


def mulaw_quantize(x: np.ndarray, mu: float = 255.0, bits: int = 8) -> np.ndarray:
    """8-bit mu-law companding and quantisation (instead of A-law)."""
    x = np.clip(x, -1.0, 1.0)
    companded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    levels = float(2 ** (bits - 1))
    quantized = np.round(companded * levels) / levels
    expanded = np.sign(quantized) * (np.expm1(np.abs(quantized) * np.log1p(mu)) / mu)
    return expanded.astype(np.float32)


def asymmetric_clip(x: np.ndarray, pos: float = 0.85, neg: float = 0.62) -> np.ndarray:
    """Asymmetric soft/hard clipping: different headroom for positive/negative."""
    out = x.copy()
    positive = out >= 0.0
    # Soft knee above the threshold, hard ceiling at 1.0.
    out[positive] = np.tanh(out[positive] / max(pos, 1e-6)) * pos
    out[~positive] = np.tanh(out[~positive] / max(neg, 1e-6)) * neg
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def agc(x: np.ndarray, sr: int = SR, target_rms: float = 0.12) -> np.ndarray:
    """Simple envelope-follower AGC (attack/release), independent of training."""
    attack = np.exp(-1.0 / (0.005 * sr))
    release = np.exp(-1.0 / (0.200 * sr))
    envelope = np.zeros(len(x), dtype=np.float32)
    level = 0.0
    for i, sample in enumerate(np.abs(x)):
        coeff = attack if sample > level else release
        level = coeff * level + (1.0 - coeff) * sample
        envelope[i] = level
    gain = target_rms / (envelope + 1e-4)
    gain = np.clip(gain, 0.25, 4.0)
    return (x * gain).astype(np.float32)


def make_rir(
    rng: np.random.RandomState,
    t60: float,
    drr_db: float,
    length_sec: float = 0.35,
) -> np.ndarray:
    """Independent exponentially-decaying noise RIR."""
    n = int(SR * length_sec)
    t = np.arange(n) / SR
    decay = np.exp(-6.9078 * t / max(t60, 0.02))
    tail = rng.randn(n).astype(np.float32) * decay
    tail /= np.linalg.norm(tail) + 1e-10
    direct = np.zeros(n, dtype=np.float32)
    direct[0] = 1.0
    drr = 10.0 ** (drr_db / 20.0)
    rir = direct + tail / drr
    return (rir / (np.max(np.abs(rir)) + 1e-10)).astype(np.float32)


def causal_conv(x: np.ndarray, rir: np.ndarray, n: int) -> np.ndarray:
    out = signal.fftconvolve(x, rir, mode="full")[:n]
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out.astype(np.float32)


def fractional_delay(x: np.ndarray, delays: np.ndarray) -> np.ndarray:
    n = len(x)
    m = np.arange(n, dtype=np.float64) - delays
    i = np.floor(m).astype(np.int64)
    frac = m - i
    valid = (i >= 0) & (i < n - 1)
    out = np.zeros(n, dtype=np.float32)
    idx = np.where(valid)[0]
    if len(idx):
        out[idx] = (1.0 - frac[idx]) * x[i[idx]] + frac[idx] * x[i[idx] + 1]
    return out


def pink_noise(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Synthetic pink noise via 1/f spectral shaping."""
    white = rng.randn(n).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    spectrum[1:] /= np.sqrt(np.maximum(freqs[1:], 1.0))
    spectrum[0] = 0.0
    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
    return pink / (np.std(pink) + 1e-10)


def frame_db(x: np.ndarray, n_frames: int) -> np.ndarray:
    needed = n_frames * HOP
    if len(x) < needed:
        x = np.pad(x, (0, needed - len(x)))
    frames = x[:needed].reshape(n_frames, HOP).astype(np.float64)
    energy = np.mean(frames ** 2, axis=1)
    return 10.0 * np.log10(np.maximum(energy, 1e-12))


def audible_mask(component: np.ndarray, noise: np.ndarray, n_frames: int) -> np.ndarray:
    comp_db = frame_db(component, n_frames)
    noise_db = frame_db(noise, n_frames)
    comp_energy = 10.0 ** (comp_db / 10.0)
    noise_energy = 10.0 ** (noise_db / 10.0)
    return (comp_energy > noise_energy * AUDIBLE_NOISE_RATIO) & (comp_db > AUDIBLE_ABS_DBFS)


# ----------------------------------------------------------------------
# Independent dialogue generator
# ----------------------------------------------------------------------
def build_schedule(rng: np.random.RandomState, duration: float) -> List[Tuple[str, float, float]]:
    """Free conversation: long monologues up to 6 s, short 0.4 s inserts."""
    events: List[Tuple[str, float, float]] = []
    speaker = "A" if rng.rand() < 0.5 else "B"
    t = float(rng.uniform(0.1, 0.5))
    limit = duration - 0.2

    while t < limit:
        if rng.rand() < 0.55:
            dur = float(rng.uniform(2.5, 6.0))  # long monologue
        else:
            dur = float(rng.uniform(0.4, 1.0))  # short insert
        dur = min(dur, limit - t)
        if dur < 0.3:
            break
        events.append((speaker, t, dur))
        end = t + dur

        other = "B" if speaker == "A" else "A"
        if rng.rand() < 0.35:
            overlap = float(rng.uniform(0.2, min(1.5, dur * 0.6)))
            speaker = other
            t = max(t + 0.2, end - overlap)
        else:
            speaker = other
            t = end + float(rng.uniform(0.05, 0.5))
    return events


def pick_utterance(
    pool: Dict[str, List[np.ndarray]],
    speaker: str,
    duration: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    target_len = max(1, int(SR * duration))
    candidates = pool.get(speaker) or []
    if candidates:
        utt = candidates[rng.randint(len(candidates))].copy()
        if rng.rand() < 0.6:
            factor = float(rng.uniform(0.9, 1.1))
            utt = signal.resample(utt, max(160, int(len(utt) / factor))).astype(np.float32)
        if len(utt) >= target_len:
            return utt[:target_len].copy()
        return np.pad(utt, (0, target_len - len(utt)))
    # Fallback: synthetic vowel.
    t = np.arange(target_len) / SR
    pitch = 130.0 if speaker == "A" else 200.0
    return (0.4 * np.sin(2 * np.pi * pitch * t)).astype(np.float32)


def synthesize_record(
    pool: Dict[str, List[np.ndarray]],
    scenario: str,
    duration: float,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """Build one two-channel record with independent physics and ground truth."""
    n = int(SR * duration)
    n_frames = int(round(duration / FRAME_DURATION_SEC))

    timeline_a = np.zeros(n, dtype=np.float32)
    timeline_b = np.zeros(n, dtype=np.float32)
    timeline_3 = np.zeros(n, dtype=np.float32)

    for speaker, start, dur in build_schedule(rng, duration):
        s = int(start * SR)
        e = min(n, s + int(dur * SR))
        if e <= s:
            continue
        utt = pick_utterance(pool, speaker, (e - s) / SR, rng)
        seg = min(len(utt), e - s)
        (timeline_a if speaker == "A" else timeline_b)[s : s + seg] += utt[:seg]

    # Third speaker on a random channel.
    third_channel: Optional[str] = None
    if scenario == "third_speaker" and duration > 3.0:
        third_channel = str(rng.choice(["A", "B"]))
        start = float(rng.uniform(0.8, max(1.0, duration - 2.0)))
        dur = float(rng.uniform(0.8, 2.5))
        s = int(start * SR)
        e = min(n, s + int(dur * SR))
        if e > s:
            utt = pick_utterance(pool, "third", (e - s) / SR, rng)
            seg = min(len(utt), e - s)
            timeline_3[s : s + seg] += utt[:seg]

    # Room responses (independent RIR model).
    rec_a = causal_conv(timeline_a, make_rir(rng, rng.uniform(0.15, 0.30), rng.uniform(10, 18)), n)
    rec_b = causal_conv(timeline_b, make_rir(rng, rng.uniform(0.15, 0.30), rng.uniform(10, 18)), n)

    rec_third = np.zeros(n, dtype=np.float32)
    if third_channel is not None:
        rec_third = causal_conv(timeline_3, make_rir(rng, rng.uniform(0.35, 0.60), rng.uniform(2, 6)), n)
        rec_third *= 10.0 ** (-rng.uniform(3.0, 15.0) / 20.0)

    # Leakage configuration.
    if scenario == "no_leak":
        leak_a = leak_b = False
    else:
        leak_a = leak_b = True
        if scenario != "high_leak_lt_10db" and rng.rand() < 0.25:
            mode = str(rng.choice(["A", "B"]))
            if mode == "A":
                leak_a = False
            else:
                leak_b = False

    if scenario == "high_leak_lt_10db":
        atten_a = float(rng.uniform(6.0, 9.0))
        atten_b = float(rng.uniform(6.0, 9.0))
    elif scenario == "double_talk":
        atten_a = float(rng.uniform(10.0, 28.0))
        atten_b = float(rng.uniform(10.0, 28.0))
    elif scenario == "path_drift":
        atten_a = float(rng.uniform(8.0, 32.0))
        atten_b = float(rng.uniform(8.0, 32.0))
    else:
        atten_a = float(rng.uniform(10.0, 30.0))
        atten_b = float(rng.uniform(10.0, 30.0))

    drift_prob = 1.0 if scenario == "path_drift" else 0.3

    def leak_path(source: np.ndarray, atten_db: float) -> np.ndarray:
        shaped = asymmetric_clip(source, pos=rng.uniform(0.7, 0.95), neg=rng.uniform(0.5, 0.8))
        shaped = agc(shaped, target_rms=0.12)
        coupled = causal_conv(shaped, make_rir(rng, rng.uniform(0.15, 0.35), rng.uniform(4, 10)), n)
        base = rng.uniform(0.020, 0.300) * SR
        if rng.rand() < drift_prob:
            rate = rng.uniform(0.5, 4.0) * (1.0 if rng.rand() < 0.5 else -1.0) / 1000.0
            profile = base + np.arange(n) * rate
        else:
            profile = np.full(n, base, dtype=np.float64)
        profile = np.clip(profile, 0.020 * SR, 0.300 * SR)
        delayed = fractional_delay(coupled, profile)
        return (delayed * 10.0 ** (-atten_db / 20.0)).astype(np.float32)

    leak_b_to_a = leak_path(rec_b, atten_a) if leak_a else np.zeros(n, dtype=np.float32)
    leak_a_to_b = leak_path(rec_a, atten_b) if leak_b else np.zeros(n, dtype=np.float32)

    mic_a = rec_a + leak_b_to_a + (rec_third if third_channel == "A" else 0.0)
    mic_b = rec_b + leak_a_to_b + (rec_third if third_channel == "B" else 0.0)

    # Transmission chain: elliptic filter -> mu-law -> AGC -> pink noise.
    def transmit(x: np.ndarray) -> np.ndarray:
        y = elliptic_bandpass(x)
        y = mulaw_quantize(y)
        y = agc(y, target_rms=0.12)
        return y

    tx_a = transmit(mic_a)
    tx_b = transmit(mic_b)

    def add_noise(x: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray]:
        noise = pink_noise(len(x), rng)
        noise = elliptic_bandpass(noise)
        ref_power = float(np.mean(x ** 2)) + 1e-12
        target_power = ref_power / (10.0 ** (snr_db / 10.0))
        noise *= np.sqrt(target_power / (float(np.mean(noise ** 2)) + 1e-12))
        return (x + noise).astype(np.float32), noise.astype(np.float32)

    track_a, noise_a = add_noise(tx_a, float(rng.uniform(15.0, 30.0)))
    track_b, noise_b = add_noise(tx_b, float(rng.uniform(15.0, 30.0)))

    # Ground truth from audible component energy.
    comp_own_a = elliptic_bandpass(rec_a)
    comp_own_b = elliptic_bandpass(rec_b)
    comp_leak_ba = elliptic_bandpass(leak_b_to_a)
    comp_leak_ab = elliptic_bandpass(leak_a_to_b)
    comp_third = elliptic_bandpass(rec_third)

    masks_a = {
        "own": audible_mask(comp_own_a, noise_a, n_frames),
        "leak": audible_mask(comp_leak_ba, noise_a, n_frames),
        "third": audible_mask(comp_third, noise_a, n_frames) if third_channel == "A" else np.zeros(n_frames, dtype=bool),
    }
    masks_b = {
        "own": audible_mask(comp_own_b, noise_b, n_frames),
        "leak": audible_mask(comp_leak_ab, noise_b, n_frames),
        "third": audible_mask(comp_third, noise_b, n_frames) if third_channel == "B" else np.zeros(n_frames, dtype=bool),
    }

    gt: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
    for ch, masks in (("A", masks_a), ("B", masks_b)):
        for label, mask in masks.items():
            gt[ch].extend(frame_mask_to_intervals(mask, label=label, frame_sec=FRAME_DURATION_SEC))
        gt[ch].sort(key=lambda item: (item["start"], item["label"]))

    return {
        "audio": np.stack([track_a, track_b]).astype(np.float32),
        "ground_truth": gt,
        "num_frames": n_frames,
        "duration": duration,
        # Pure owner speech (no leak, no noise) — the correct reference for
        # speaker purity. Using `track - leak` would trivially equal the input.
        "reference_a": comp_own_a.astype(np.float32),
        # Track A with the leak component removed — the SI-SDR target.
        "target_a": (track_a - comp_leak_ba).astype(np.float32),
        "metadata": {
            "scenario": scenario,
            "leak_enabled_a": leak_a,
            "leak_enabled_b": leak_b,
            "leak_attenuation_a_db": atten_a,
            "leak_attenuation_b_db": atten_b,
            "third_channel": third_channel,
        },
    }


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def leak_fpr(
    gt: Dict[str, List[Dict[str, Any]]],
    pred: Dict[str, List[Dict[str, Any]]],
    n_frames: int,
) -> Tuple[int, int]:
    fp = 0
    negatives = 0
    for ch, other in (("A", "B"), ("B", "A")):
        gt_leak = intervals_to_frame_matrix(gt.get(ch, []), n_frames, labels=(LABEL_LEAK,))[LABEL_LEAK]
        pred_leak = intervals_to_frame_matrix(pred.get(ch, []), n_frames, labels=(LABEL_LEAK,))[LABEL_LEAK]
        counterpart = intervals_to_frame_matrix(gt.get(other, []), n_frames, labels=("own",))["own"]
        mask = counterpart & ~gt_leak
        fp += int(np.sum(pred_leak & mask))
        negatives += int(np.sum(mask))
    return fp, negatives


def purity(
    signal_audio: np.ndarray,
    reference: np.ndarray,
    gt_intervals: List[Dict[str, Any]],
    embedder: Optional[SpeakerEmbedder],
) -> Optional[float]:
    if embedder is None:
        return None
    sig_slices, ref_slices = [], []
    for seg in gt_intervals:
        if seg["label"] in ("own", "leak"):
            s = max(0, int(seg["start"] * SR))
            e = min(len(signal_audio), int(seg["end"] * SR))
            if e - s > 400:
                sig_slices.append(signal_audio[s:e])
                ref_slices.append(reference[s:e])
    if not sig_slices:
        return None
    cat_sig = np.concatenate(sig_slices)
    cat_ref = np.concatenate(ref_slices)
    if len(cat_sig) < 2400:
        return None
    return float(
        cosine_similarity(
            embedder.embed(cat_sig, sample_rate=SR),
            embedder.embed(cat_ref, sample_rate=SR),
        )
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_cross_generator_test(
    records_per_scenario: int = 4,
    duration_sec: float = 8.0,
    seed: int = 2024,
    output_dir: str = "samples/cross_generator",
    save_audio: bool = True,
) -> Dict[str, Any]:
    print("=" * 78)
    print("CROSS-GENERATOR GENERALISATION TEST (independent physics)")
    print("=" * 78)
    print("Generator: elliptic 280-3450 Hz, mu-law, asymmetric clip + AGC, pink noise")
    print("           long monologues up to 6 s, short 0.4 s inserts")
    print("-" * 78)

    loader = RealSpeechLoader()
    splits = loader.get_disjoint_splits()
    test_speakers = list(splits["test"])
    ru = [s for s in test_speakers if s.startswith("ru_")]
    vox = [s for s in test_speakers if s.startswith("vox_")]
    if len(ru) < 3 or len(vox) < 3:
        raise RuntimeError(f"Need 3 ru + 3 vox test speakers, got {ru} / {vox}")

    pools: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for lang, speakers in (("ru", ru[:3]), ("en", vox[:3])):
        pools[lang] = {
            "A": loader.load_speaker_utterances(speakers[0], max_utterances=12),
            "B": loader.load_speaker_utterances(speakers[1], max_utterances=12),
            "third": loader.load_speaker_utterances(speakers[2], max_utterances=12),
        }
        print(f"  {lang}: A={speakers[0]}, B={speakers[1]}, third={speakers[2]}")

    detector = ExtraneousSpeechDetector()
    cleaner = TelephoneAudioCleaner(detector=detector)
    baseline = EnergyBaselineDetector()
    try:
        embedder: Optional[SpeakerEmbedder] = SpeakerEmbedder()
    except Exception as exc:  # pragma: no cover
        print(f"Notice: SpeakerEmbedder unavailable ({exc}); purity skipped.")
        embedder = None

    out_root = Path(output_dir)
    if save_audio:
        out_root.mkdir(parents=True, exist_ok=True)

    base_acc = FrameCountAccumulator()
    det_acc = FrameCountAccumulator()
    scenario_acc: Dict[str, Dict[str, FrameCountAccumulator]] = {
        sc: {"baseline": FrameCountAccumulator(), "detector": FrameCountAccumulator()}
        for sc in SCENARIOS
    }
    no_leak_fp = {"baseline": 0, "detector": 0}
    no_leak_neg = {"baseline": 0, "detector": 0}
    purity_raw: List[float] = []
    purity_clean: List[float] = []

    rng = np.random.RandomState(seed)
    total = len(SCENARIOS) * records_per_scenario
    index = 0

    for scenario in SCENARIOS:
        for record_index in range(records_per_scenario):
            index += 1
            lang = "ru" if record_index % 2 == 0 else "en"
            record = synthesize_record(pools[lang], scenario, duration_sec, rng)
            audio = record["audio"]
            gt = record["ground_truth"]
            n_frames = record["num_frames"]

            base_pred = baseline.detect(audio, SR)
            probs = detector.predict_probabilities(audio, SR)
            det_pred = detector.predict_intervals(audio, SR, probabilities=probs)
            cleaned = cleaner.clean(audio, det_pred, SR, probabilities=probs)

            base_acc.add(gt, base_pred, num_frames=n_frames, collar_frames=COLLAR_FRAMES)
            det_acc.add(gt, det_pred, num_frames=n_frames, collar_frames=COLLAR_FRAMES)
            scenario_acc[scenario]["baseline"].add(gt, base_pred, num_frames=n_frames, collar_frames=COLLAR_FRAMES)
            scenario_acc[scenario]["detector"].add(gt, det_pred, num_frames=n_frames, collar_frames=COLLAR_FRAMES)

            if scenario == "no_leak":
                for name, pred in (("baseline", base_pred), ("detector", det_pred)):
                    fp, neg = leak_fpr(gt, pred, n_frames)
                    no_leak_fp[name] += fp
                    no_leak_neg[name] += neg

            p_raw = purity(audio[0], record["reference_a"], gt["A"], embedder)
            p_clean = purity(cleaned[0], record["reference_a"], gt["A"], embedder)
            if p_raw is not None and p_clean is not None:
                purity_raw.append(p_raw)
                purity_clean.append(p_clean)

            if save_audio:
                rec_dir = out_root / scenario / f"record_{record_index + 1:02d}_{lang}"
                rec_dir.mkdir(parents=True, exist_ok=True)
                save_stereo_audio(str(rec_dir / "input_stereo.wav"), audio, sample_rate=SR)
                save_stereo_audio(str(rec_dir / "cleaned_stereo.wav"), cleaned, sample_rate=SR)
                (rec_dir / "ground_truth.json").write_text(
                    json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            sys.stdout.write(f"\rProcessed {index}/{total} records [{scenario}/{lang}]...")
            sys.stdout.flush()

    print("\n" + "=" * 78)
    print("CROSS-GENERATOR RESULTS (global micro-F1, collar 50 ms excluded)")
    print("=" * 78)

    base_global = base_acc.finalize()
    det_global = det_acc.finalize()

    b_fpr = no_leak_fp["baseline"] / no_leak_neg["baseline"] if no_leak_neg["baseline"] else 0.0
    d_fpr = no_leak_fp["detector"] / no_leak_neg["detector"] if no_leak_neg["detector"] else 0.0

    header = f"{'Metric':<34} | {'Baseline':>10} | {'Detector':>10} | {'Delta':>9}"
    print(header)
    print("-" * len(header))
    for label in ("own", "leak", "third"):
        b = base_global["by_label"][label]["f1"]
        d = det_global["by_label"][label]["f1"]
        print(f"{label.capitalize() + ' F1':<34} | {b:>10.4f} | {d:>10.4f} | {d - b:>+9.4f}")
    print(
        f"{'Macro F1':<34} | {base_global['macro_f1']:>10.4f} | "
        f"{det_global['macro_f1']:>10.4f} | {det_global['macro_f1'] - base_global['macro_f1']:>+9.4f}"
    )
    print(f"{'Leak FPR on no_leak':<34} | {b_fpr:>10.4f} | {d_fpr:>10.4f} | {'':>9}")
    if purity_raw:
        print(
            f"{'Track purity (raw -> cleaned)':<34} | "
            f"{np.mean(purity_raw):>10.4f} | {np.mean(purity_clean):>10.4f} | "
            f"{np.mean(purity_clean) - np.mean(purity_raw):>+9.4f}"
        )
    print("-" * len(header))

    print("\nPer-scenario Macro F1:")
    print(f"{'Scenario':<22} | {'Baseline':>10} | {'Detector':>10} | {'Delta':>9}")
    print("-" * 58)
    scenario_summary: Dict[str, Dict[str, float]] = {}
    for scenario in SCENARIOS:
        b = scenario_acc[scenario]["baseline"].finalize()["macro_f1"]
        d = scenario_acc[scenario]["detector"].finalize()["macro_f1"]
        scenario_summary[scenario] = {"baseline": b, "detector": d}
        print(f"{scenario:<22} | {b:>10.4f} | {d:>10.4f} | {d - b:>+9.4f}")
    print("-" * 58)

    result = {
        "records": total,
        "duration_sec": duration_sec,
        "seed": seed,
        "baseline": base_global,
        "detector": det_global,
        "leak_fpr_no_leak": {"baseline": b_fpr, "detector": d_fpr},
        "purity": {
            "raw": float(np.mean(purity_raw)) if purity_raw else None,
            "cleaned": float(np.mean(purity_clean)) if purity_clean else None,
        },
        "scenarios": scenario_summary,
    }
    if save_audio:
        (out_root / "cross_generator_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nResults and audio written to: {out_root}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-per-scenario", type=int, default=4)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", default="samples/cross_generator")
    parser.add_argument("--no-save-audio", action="store_true")
    args = parser.parse_args()
    run_cross_generator_test(
        records_per_scenario=args.records_per_scenario,
        duration_sec=args.duration,
        seed=args.seed,
        output_dir=args.output_dir,
        save_audio=not args.no_save_audio,
    )
