#!/usr/bin/env python3
"""Resample 8 kHz WAV files to 16 kHz for playback in VS Code and web browsers.

Usage:
    python scripts/resample_wav.py input_8k.wav [output_16k.wav]
    python scripts/resample_wav.py samples/
"""

import argparse
from pathlib import Path
import soundfile as sf
import scipy.signal as signal
import math


def resample_file(input_path: Path, output_path: Path, target_sr: int = 16000) -> None:
    data, orig_sr = sf.read(str(input_path), dtype="float32")
    if orig_sr == target_sr:
        print(f"Skipping {input_path} (already {target_sr} Hz)")
        return

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    resampled = signal.resample_poly(data, up, down, axis=0)
    sf.write(str(output_path), resampled, target_sr, subtype="PCM_16")
    print(f"Resampled: {input_path} ({orig_sr} Hz) -> {output_path} ({target_sr} Hz)")


def main():
    parser = argparse.ArgumentParser(
        description="Resample 8 kHz WAV files to 16 kHz for VS Code / browser compatibility."
    )
    parser.add_argument("path", help="WAV file or directory of WAV files to resample")
    parser.add_argument("output", nargs="?", default=None, help="Optional output WAV path (for single file mode)")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate (default: 16000)")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: {target} does not exist")
        return

    if target.is_file():
        out = Path(args.output) if args.output else target.with_name(f"{target.stem}_16k{target.suffix}")
        resample_file(target, out, target_sr=args.target_sr)
    elif target.is_dir():
        wav_files = list(target.rglob("*.wav"))
        count = 0
        for wav_path in wav_files:
            if "_16k" in wav_path.stem or "_preview" in wav_path.stem:
                continue
            out = wav_path.with_name(f"{wav_path.stem}_16k{wav_path.suffix}")
            resample_file(wav_path, out, target_sr=args.target_sr)
            count += 1
        print(f"\nDone! Generated {count} 16 kHz preview WAV files in {target}")


if __name__ == "__main__":
    main()

