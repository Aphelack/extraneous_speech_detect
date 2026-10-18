"""Download and prepare datasets for extraneous speech detection.

Downloads the correct datasets per data_description.csv and ТЗ:
1. Mozilla Common Voice 17.0 ru (validated split) — Russian speech with client_id per speaker
2. SberDevices Golos test split — general Russian speech
3. VoxCeleb1 test split (asahi417/voxceleb1-test-split) — 40 speakers with verified speaker_id
4. OpenSLR 28 RIRS_NOISES — room impulse responses and point noises
5. SpeechBrain ECAPA-TDNN speaker embedding model — for voice cosine similarity evaluation
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Dataset source definitions
# ---------------------------------------------------------------------------

DATASET_SOURCES = {
    "rirs_noises": {
        "url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "filename": "rirs_noises.zip",
        "size_mb": 1300,
        "description": "Room impulse responses and point noises from OpenSLR 28 (~1.3 GB)",
        "source_type": "direct",
        "archive_type": "zip",
    },
    "voxceleb1_test": {
        "repo_id": "asahi417/voxceleb1-test-split",
        "description": "Compact VoxCeleb1 test split — 40 speakers with verified speaker_id (1.29 GB)",
        "source_type": "huggingface",
    },
    "common_voice_ru": {
        "repo_id": "fsicoli/common_voice_17_0",
        "lang": "ru",
        "description": "Mozilla Common Voice 17.0 ru — Russian speech with client_id speaker clustering",
        "source_type": "common_voice",
        "min_utterances_per_speaker": 15,
    },
    "golos_test": {
        "repo_id": "bond005/sberdevices_golos_10h_crowd",
        "description": "SberDevices Golos compact crowd split (2.27 GB) — Russian speech",
        "source_type": "huggingface",
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file_with_progress(url: str, output_path: Path) -> None:
    """Download a file with console progress indicator."""
    print(f"Downloading from: {url}")
    print(f"Destination: {output_path}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent}% [{mb:.1f} MB / {total_mb:.1f} MB]")
        else:
            mb = (count * block_size) / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {mb:.1f} MB downloaded")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(output_path), reporthook=reporthook)
    print("\nDownload complete.")


def unpack_archive(archive_path: Path, extract_dir: Path, archive_type: str) -> None:
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)
    elif archive_type == "tar":
        with tarfile.open(archive_path, "r") as t:
            t.extractall(extract_dir)
    print("Extraction complete.")


# ---------------------------------------------------------------------------
# Dataset-specific downloaders
# ---------------------------------------------------------------------------

def download_direct(info: dict, target_dir: Path) -> None:
    """Download and unpack a direct URL archive."""
    archive_path = target_dir / info["filename"]
    if not archive_path.exists():
        download_file_with_progress(info["url"], archive_path)
    else:
        print(f"Archive already exists at: {archive_path}")

    archive_type = info.get("archive_type", "zip")
    unpack_archive(archive_path, target_dir, archive_type)


def download_huggingface(info: dict, target_dir: Path) -> None:
    """Download a dataset from Hugging Face Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not installed. Run: pip install datasets")
        return

    repo_id = info["repo_id"]
    print(f"Downloading from Hugging Face: {repo_id}...")

    # Use streaming=False to download to disk
    ds = load_dataset(repo_id, split="train", cache_dir=str(target_dir / ".cache"))
    print(f"Downloaded {len(ds)} samples")

    # Save to disk for fast reloading
    save_path = target_dir / "dataset"
    ds.save_to_disk(str(save_path))
    print(f"Saved to: {save_path}")


def download_common_voice_ru(info: dict, target_dir: Path, max_speakers: int = 60) -> None:
    """Download Common Voice 17.0 ru with speaker filtering.

    Two-pass approach:
    1. Stream to count utterances per client_id
    2. Download only speakers with >= min_utterances
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not installed. Run: pip install datasets")
        return

    repo_id = info["repo_id"]
    lang = info.get("lang", "ru")
    min_utt = info.get("min_utterances_per_speaker", 15)

    print(f"Loading Common Voice ru from {repo_id}...")
    print(f"Filter: speakers with >= {min_utt} utterances")

    # --- Pass 1: Count utterances per speaker (streaming) ---
    print("Pass 1: Counting utterances per speaker (streaming)...")
    try:
        ds_stream = load_dataset(repo_id, lang, split="validated", streaming=True, trust_remote_code=True)
    except Exception:
        print("Falling back to 'test' split...")
        ds_stream = load_dataset(repo_id, lang, split="test", streaming=True, trust_remote_code=True)

    speaker_counts = Counter()
    total_scanned = 0
    for item in ds_stream:
        cid = item.get("client_id", "")
        if cid:
            speaker_counts[cid] += 1
        total_scanned += 1
        if total_scanned % 5000 == 0:
            print(f"  Scanned {total_scanned} samples, {len(speaker_counts)} unique speakers...")

    print(f"Total scanned: {total_scanned}, unique speakers: {len(speaker_counts)}")

    # Filter speakers
    valid_speakers = {spk for spk, cnt in speaker_counts.items() if cnt >= min_utt}
    print(f"Speakers with >= {min_utt} utterances: {len(valid_speakers)}")

    # Limit to max_speakers for manageable size
    if len(valid_speakers) > max_speakers:
        # Take top speakers by utterance count
        sorted_speakers = sorted(valid_speakers, key=lambda s: speaker_counts[s], reverse=True)
        valid_speakers = set(sorted_speakers[:max_speakers])
        print(f"Limited to top {max_speakers} speakers")

    # --- Pass 2: Download filtered data ---
    print("Pass 2: Downloading filtered data...")
    try:
        ds_full = load_dataset(repo_id, lang, split="validated", trust_remote_code=True,
                               cache_dir=str(target_dir / ".cache"))
    except Exception:
        ds_full = load_dataset(repo_id, lang, split="test", trust_remote_code=True,
                               cache_dir=str(target_dir / ".cache"))

    ds_filtered = ds_full.filter(lambda x: x["client_id"] in valid_speakers)
    print(f"Filtered dataset: {len(ds_filtered)} samples from {len(valid_speakers)} speakers")

    # Save
    save_path = target_dir / "dataset"
    ds_filtered.save_to_disk(str(save_path))
    print(f"Saved to: {save_path}")

    # Save speaker map for reference
    import json
    speaker_info = {spk: speaker_counts[spk] for spk in valid_speakers}
    with open(target_dir / "speaker_info.json", "w") as f:
        json.dump(speaker_info, f, indent=2)
    print(f"Speaker info saved: {len(speaker_info)} speakers")


def download_voxceleb1_test(info: dict, target_dir: Path) -> None:
    """Download VoxCeleb1 test split from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not installed. Run: pip install datasets")
        return

    repo_id = info["repo_id"]
    print(f"Downloading VoxCeleb1 test from {repo_id}...")

    ds = load_dataset(repo_id, split="test", cache_dir=str(target_dir / ".cache"),
                      trust_remote_code=True)
    print(f"Downloaded {len(ds)} samples")

    save_path = target_dir / "dataset"
    ds.save_to_disk(str(save_path))
    print(f"Saved to: {save_path}")


def download_embedder_model(target_dir: Path) -> None:
    """Pre-download the ECAPA-TDNN speaker embedding model from SpeechBrain."""
    print("\n=== Downloading Speaker Embedding Model (ECAPA-TDNN) ===")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = str(target_dir / "spkrec-ecapa-voxceleb")
        print(f"Downloading speechbrain/spkrec-ecapa-voxceleb to {savedir}...")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"},
        )
        print("ECAPA-TDNN model downloaded and cached successfully.")
        del classifier
    except ImportError:
        print("WARNING: speechbrain not installed. Run: pip install speechbrain")
    except Exception as e:
        print(f"WARNING: Could not download ECAPA-TDNN model: {e}")
        print("The model will be downloaded on first use.")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def download_dataset(dataset_key: str, data_dir: Path, dry_run: bool = False) -> None:
    """Download and prepare a specific dataset."""
    info = DATASET_SOURCES.get(dataset_key)
    if not info:
        print(f"Unknown dataset key: {dataset_key}")
        return

    target_dir = data_dir / dataset_key
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}")
    print(f"Description: {info['description']}")
    print(f"Target: {target_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("[Dry Run] Would download to:", target_dir)
        return

    source_type = info.get("source_type", "direct")
    if source_type == "direct":
        download_direct(info, target_dir)
    elif source_type == "common_voice":
        download_common_voice_ru(info, target_dir)
    elif source_type == "huggingface" and dataset_key == "voxceleb1_test":
        download_voxceleb1_test(info, target_dir)
    elif source_type == "huggingface":
        download_huggingface(info, target_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for extraneous speech detection (Russian)."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "rirs_noises", "voxceleb1_test", "common_voice_ru", "golos_test", "embedder"],
        default="all",
        help="Which dataset to download (default: all).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for downloaded datasets.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory for downloaded models.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned downloads without actually downloading.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        targets = list(DATASET_SOURCES.keys())
    elif args.dataset == "embedder":
        targets = []
    else:
        targets = [args.dataset]

    print(f"Selected targets: {targets}")
    print(f"Data directory: {data_dir.resolve()}")

    for t in targets:
        download_dataset(t, data_dir, dry_run=args.dry_run)

    # Always download embedder model when --dataset=all or --dataset=embedder
    if args.dataset in ("all", "embedder") and not args.dry_run:
        download_embedder_model(models_dir)


if __name__ == "__main__":
    main()
