"""Generate Russian listening samples grouped by measured cleaner quality."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.audio_utils import save_stereo_audio
from src.cleaner import TelephoneAudioCleaner, compute_spectral_embedding
from src.config import SAMPLE_RATE
from src.data_loader import RealSpeechLoader
from src.detector import ExtraneousSpeechDetector
from src.metrics import cosine_similarity
from src.synthesizer import TelephoneDialogueSynthesizer


def save_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    data = np.clip(audio, -1.0, 1.0)
    wavfile.write(str(path), sample_rate, (data * 32767.0).astype(np.int16))


def leak_attenuation(raw: np.ndarray, clean: np.ndarray, intervals: list[dict], sr: int) -> float | None:
    speech = np.zeros(len(raw), dtype=bool)
    leak = np.zeros(len(raw), dtype=bool)
    for item in intervals:
        start = int(item["start"] * sr)
        end = min(len(raw), int(item["end"] * sr))
        if item["label"] in ("own", "third"):
            speech[start:end] = True
        elif item["label"] == "leak":
            leak[start:end] = True
    pure = leak & ~speech
    if pure.sum() < sr // 10:
        return None
    raw_power = np.mean(raw[pure] ** 2) + 1e-10
    clean_power = np.mean(clean[pure] ** 2) + 1e-10
    return float(10.0 * np.log10(raw_power / clean_power))


def main() -> None:
    output_root = Path("samples/quality_listening")
    output_root.mkdir(parents=True, exist_ok=True)
    loader = RealSpeechLoader(data_root="data/russian_speech")
    detector = ExtraneousSpeechDetector()
    cleaner = TelephoneAudioCleaner(detector=detector)
    scenarios = ["normal", "double_talk", "high_leak_lt_10db", "path_drift", "third_speaker"]
    candidates: list[dict] = []

    for seed in range(10, 25):
        rng = np.random.RandomState(seed)
        speakers = loader.sample_dialogue_speakers("test", rng=rng)
        pool = loader.build_utterance_pool(*speakers, max_per_speaker=24)
        scenario = scenarios[(seed - 10) % len(scenarios)]
        synth = TelephoneDialogueSynthesizer(seed=seed)
        result = synth.synthesize_dialogue(
            utterance_pool=pool,
            duration_sec=8.0,
            scenario=scenario,
            third_speaker_prob=1.0 if scenario == "third_speaker" else 0.0,
        )
        cleaned = cleaner.clean_stereo(result.audio, sample_rate=result.sample_rate)
        attenuation = leak_attenuation(result.audio[0], cleaned[0], result.ground_truth_intervals["A"], result.sample_rate)
        reference_embedding = compute_spectral_embedding(result.clean_references["A"])
        noisy_purity = cosine_similarity(compute_spectral_embedding(result.audio[0]), reference_embedding)
        clean_purity = cosine_similarity(compute_spectral_embedding(cleaned[0]), reference_embedding)
        candidates.append({
            "seed": seed,
            "scenario": scenario,
            "speakers": {"A": speakers[0], "B": speakers[1], "third": speakers[2]},
            "attenuation_db": attenuation,
            "noisy_purity": float(noisy_purity),
            "clean_purity": float(clean_purity),
            "purity_delta": float(clean_purity - noisy_purity),
            "input": result.audio,
            "cleaned": cleaned,
            "reference_a": result.clean_references["A"],
        })
        print(
            f"seed={seed} scenario={scenario} attenuation={attenuation} dB "
            f"purity_delta={clean_purity - noisy_purity:+.4f}"
        )

    def quality_score(item: dict) -> float:
        attenuation = item["attenuation_db"] if item["attenuation_db"] is not None else -20.0
        return attenuation + 20.0 * item["purity_delta"]

    ranked = sorted(candidates, key=quality_score)
    groups = {
        "bad": ranked[:5],
        "medium": ranked[5:10],
        "good": ranked[10:15],
    }
    manifest: dict[str, list[dict]] = {}
    for group, items in groups.items():
        manifest[group] = []
        for index, item in enumerate(items, start=1):
            folder = output_root / group / f"{index:02d}_{item['scenario']}_seed{item['seed']}"
            folder.mkdir(parents=True, exist_ok=True)
            save_stereo_audio(str(folder / "input.wav"), item["input"], SAMPLE_RATE)
            save_stereo_audio(str(folder / "cleaned.wav"), item["cleaned"], SAMPLE_RATE)
            save_mono(folder / "reference_A.wav", item["reference_a"], SAMPLE_RATE)
            record = {
                key: value for key, value in item.items()
                if key not in {"input", "cleaned", "reference_a"}
            }
            record["quality_score"] = quality_score(item)
            record["files"] = {"input": "input.wav", "cleaned": "cleaned.wav", "reference": "reference_A.wav"}
            with (folder / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False)
            manifest[group].append(record)
            print(f"{group:6s}: {folder} | score={record['quality_score']:.3f}")

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"\nListening samples written to {output_root}")


if __name__ == "__main__":
    main()
