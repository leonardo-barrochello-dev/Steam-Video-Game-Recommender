import os
import subprocess
import sys
from pathlib import Path

DATASET = "antonkozyriev/game-recommendations-on-steam"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REQUIRED_FILES = ["games.csv", "recommendations.csv", "users.csv", "games_metadata.json"]


KAGGLE_DIR = Path.home() / ".kaggle"


def check_credentials():
    token_file = KAGGLE_DIR / "access_token"

    if token_file.exists():
        print("Kaggle credentials found")
        return

    print(
        f"""
Kaggle access_token not found at: {token_file}

Steps:
1. Go to https://www.kaggle.com/settings/api
2. Click "Generate New Token" or "Create New API Token"
3. Save the generated token in:
   {token_file}
"""
    )
    sys.exit(1)


def download_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for file in REQUIRED_FILES:
        file_path = DATA_DIR / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1_000_000
            print(f"  {file} already exists ({size_mb:.1f} MB), skipping")
            continue

        print(f"Downloading {file}...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                DATASET,
                "-p",
                str(DATA_DIR),
                "-f",
                file,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error downloading {file}: {result.stderr}")
            sys.exit(1)

        size_mb = file_path.stat().st_size / 1_000_000
        print(f"  Downloaded {file} ({size_mb:.1f} MB)")

    print("\nAll files downloaded successfully!")


def validate_files():
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]

    if missing:
        print(f"Missing files after download: {missing}")
        sys.exit(1)

    print("Files in data/:")
    for f in REQUIRED_FILES:
        size_mb = (DATA_DIR / f).stat().st_size / 1_000_000
        print(f"  {f} ({size_mb:.1f} MB)")


def main():
    print("\n=== Steam Dataset Download ===\n")
    check_credentials()
    download_files()
    validate_files()
    print("\nDone! Run 'python ml/train.py' to train the model.\n")


if __name__ == "__main__":
    main()
