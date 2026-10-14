"""Script to download and prepare minimal, clean dataset subsets (~5-6 GB total).

Target subsets (as specified in data_description.csv):
1. OpenSLR 28 RIRS_NOISES (~1.3 GB): Real & simulated RIRs and point noises.
2. VoxCeleb1 test split (~1.29 GB, asahi417/voxceleb1-test-split): 40 speakers with verified IDs.
3. Mozilla Common Voice ru (ru test/validated split): Filtered speakers with >= 15-20 utterances.
4. SberDevices Golos test split (~1.3 GB): General Russian speech utterances.
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path


DATASET_SOURCES = {
    "rirs_noises": {
        "url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "filename": "rirs_noises.zip",
        "size_mb": 1300,
        "description": "Room impulse responses and point noises from OpenSLR 28",
        "archive_type": "zip",
    },
    "voxceleb1_test": {
        "repo_id": "asahi417/voxceleb1-test-split",
        "description": "Compact VoxCeleb1 test split with 40 speakers with verified IDs (1.29 GB)",
        "source_type": "huggingface",
    },
    "common_voice_ru": {
        "repo_id": "mozilla-foundation/common_voice_17_0",
        "description": "Mozilla Common Voice ru with client_id speaker clustering",
        "source_type": "huggingface",
    },
    "golos_test": {
        "url": "https://sc.link/1V3",  # SberDevices Golos test.tar mirror
        "filename": "test.tar",
        "size_mb": 1300,
        "description": "SberDevices Golos test split (1.3 GB)",
        "archive_type": "tar",
    },
}


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


def download_dataset(dataset_key: str, data_dir: Path, dry_run: bool = False) -> None:
    """Download and unpack a specific dataset."""
    info = DATASET_SOURCES.get(dataset_key)
    if not info:
        print(f"Unknown dataset key: {dataset_key}")
        return

    target_dir = data_dir / dataset_key
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Dataset: {dataset_key} ===")
    print(f"Description: {info['description']}")

    if dry_run:
        print("[Dry Run] Would download to:", target_dir)
        return

    source_type = info.get("source_type", "direct")
    if source_type == "direct":
        archive_path = target_dir / info["filename"]
        if not archive_path.exists():
            download_file_with_progress(info["url"], archive_path)
        else:
            print(f"Archive already exists at: {archive_path}")

        archive_type = info.get("archive_type", "zip")
        unpack_archive(archive_path, target_dir, archive_type)

    elif source_type == "huggingface":
        try:
            from huggingface_hub import snapshot_download
            repo_id = info["repo_id"]
            print(f"Downloading from Hugging Face Hub: {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=str(target_dir), repo_type="dataset")
            print("Hugging Face download complete.")
        except ImportError:
            print("huggingface_hub is not installed. Please run `pip install huggingface_hub`.")
            print(f"Or visit https://huggingface.co/datasets/{info['repo_id']}")


def main():
    parser = argparse.ArgumentParser(description="Download minimal clean dataset subsets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "rirs_noises", "voxceleb1_test", "common_voice_ru", "golos_test"],
        default="all",
        help="Which dataset subset to download.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for downloaded datasets.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned downloads without actually downloading.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    targets = list(DATASET_SOURCES.keys()) if args.dataset == "all" else [args.dataset]
    print(f"Selected targets: {targets}")
    print(f"Target data directory: {data_dir.resolve()}")

    for t in targets:
        download_dataset(t, data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
