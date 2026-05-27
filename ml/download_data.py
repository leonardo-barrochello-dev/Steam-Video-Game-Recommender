import os
import subprocess
import sys
from pathlib import Path

DATASET = "antonkozyriev/game-recommendations-on-steam"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REQUIRED_FILES = ["games.csv", "recommendations.csv", "users.csv", "games_metadata.json"]


def check_credentials():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print(
            """
╔══════════════════════════════════════════════════════════╗
║ Kaggle API token not found                              ║
╠══════════════════════════════════════════════════════════╣
║ 1. Go to https://www.kaggle.com/settings/api            ║
║ 2. Click "Generate New Token" or "Create New API Token" ║
║ 3. Set the environment variable:                        ║
║                                                         ║
║    Windows (PowerShell):                                ║
║      $env:KAGGLE_API_TOKEN = "seu_token_aqui"           ║
║                                                         ║
║    Windows (CMD):                                       ║
║      set KAGGLE_API_TOKEN=seu_token_aqui                ║
║                                                         ║
║    Linux/Mac:                                           ║
║      export KAGGLE_API_TOKEN="seu_token_aqui"           ║
╚══════════════════════════════════════════════════════════╝
"""
        )
        sys.exit(1)
    print("Kaggle API token found")


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
