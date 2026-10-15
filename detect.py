#!/usr/bin/env python3
"""CLI interface for extraneous speech detection.

Usage:
    python detect.py input.wav result.json
"""

import json
import sys
from pathlib import Path

from src.detector import ExtraneousSpeechDetector


def main():
    if len(sys.argv) < 3:
        print("Usage: python detect.py input.wav result.json")
        sys.exit(1)

    input_wav = sys.argv[1]
    output_json = sys.argv[2]

    if not Path(input_wav).exists():
        print(f"Error: input file '{input_wav}' does not exist.")
        sys.exit(1)

    detector = ExtraneousSpeechDetector()
    results = detector.predict_wav(input_wav)

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Predictions successfully written to: {output_json}")
    num_a = len(results.get("A", []))
    num_b = len(results.get("B", []))
    print(f"Detected segments - Track A: {num_a}, Track B: {num_b}")


if __name__ == "__main__":
    main()

