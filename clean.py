#!/usr/bin/env python3
"""CLI interface for acoustic leak removal.

Usage:
    python clean.py input.wav clean.wav
"""

import sys
from pathlib import Path

from src.cleaner import TelephoneAudioCleaner


def main():
    if len(sys.argv) < 3:
        print("Usage: python clean.py input.wav clean.wav")
        sys.exit(1)

    input_wav = sys.argv[1]
    output_wav = sys.argv[2]

    if not Path(input_wav).exists():
        print(f"Error: input file '{input_wav}' does not exist.")
        sys.exit(1)

    cleaner = TelephoneAudioCleaner()
    cleaner.clean_wav(input_wav, output_wav)
    print(f"Cleaned stereo audio successfully written to: {output_wav}")


if __name__ == "__main__":
    main()

