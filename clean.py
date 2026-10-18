#!/usr/bin/env python3
"""CLI interface for acoustic leak removal.

Usage:
    python clean.py input.wav clean.wav
"""

import argparse
import sys
from pathlib import Path

from src.cleaner import TelephoneAudioCleaner


def main():
    parser = argparse.ArgumentParser(
        description="Clean acoustic leak from two-channel telephone audio."
    )
    parser.add_argument("input_wav", help="Path to input stereo 8 kHz WAV")
    parser.add_argument("output_wav", help="Path to output cleaned stereo 8 kHz WAV")
    parser.add_argument(
        "--preview",
        "-p",
        default=None,
        help="Optional path to save an upsampled (16 kHz) preview WAV for playback in VS Code",
    )
    args = parser.parse_args()

    if not Path(args.input_wav).exists():
        print(f"Error: input file '{args.input_wav}' does not exist.")
        sys.exit(1)

    cleaner = TelephoneAudioCleaner()
    cleaner.clean_wav(args.input_wav, args.output_wav, preview_wav=args.preview)
    print(f"Cleaned stereo audio successfully written to: {args.output_wav}")
    if args.preview:
        print(f"Upsampled preview WAV written to: {args.preview} (compatible with VS Code playback)")


if __name__ == "__main__":
    main()

