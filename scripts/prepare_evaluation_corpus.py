#!/usr/bin/env python3
"""Create a prepared WAV/JSON evaluation corpus from validation speakers."""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.audio_utils import save_stereo_audio
from src.data_loader import FolderSpeechLoader
from src.synthesizer import TelephoneDialogueSynthesizer

SCENARIOS = [
    "normal",
    "double_talk",
    "high_leak_lt_10db",
    "path_drift",
    "third_speaker",
    "no_leak",
]


def prepare_corpus(
    validation_path: str,
    output_path: str,
    records_per_scenario: int = 10,
    duration_sec: float = 10.0,
    seed: int = 42,
) -> None:
    loader = FolderSpeechLoader(validation_path)
    speakers = sorted(loader.discover_speakers())
    if len(speakers) < 3:
        raise RuntimeError("The validation dataset must contain at least three speaker folders")
    pool = loader.build_utterance_pool(*speakers[:3], max_per_speaker=None)
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    for scenario_index, scenario in enumerate(SCENARIOS):
        for record_index in range(1, records_per_scenario + 1):
            record_seed = seed + scenario_index * 1000 + record_index
            result = TelephoneDialogueSynthesizer(seed=record_seed).synthesize_dialogue(
                utterance_pool=pool,
                duration_sec=duration_sec,
                scenario=scenario,
                third_speaker_prob=1.0 if scenario == "third_speaker" else 0.2,
            )
            record_dir = output_root / scenario / f"record_{record_index:03d}"
            record_dir.mkdir(parents=True, exist_ok=True)
            save_stereo_audio(str(record_dir / "input.wav"), result.audio, result.sample_rate)
            sf.write(
                record_dir / "reference_A.wav",
                result.clean_references["A"],
                result.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                record_dir / "reference_B.wav",
                result.clean_references["B"],
                result.sample_rate,
                subtype="PCM_16",
            )
            (record_dir / "ground_truth.json").write_text(
                json.dumps(result.ground_truth_intervals, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (record_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "scenario": scenario,
                        "seed": record_seed,
                        "duration_sec": duration_sec,
                        "sample_rate": result.sample_rate,
                        "source_validation_path": str(Path(validation_path)),
                        "speakers": speakers[:3],
                        **result.metadata,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(record_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-path", default="data/russian_speech_validation")
    parser.add_argument("--output-path", default="data/evaluation_corpus")
    parser.add_argument("--records-per-scenario", type=int, default=10)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_corpus(
        args.validation_path,
        args.output_path,
        args.records_per_scenario,
        args.duration_sec,
        args.seed,
    )
