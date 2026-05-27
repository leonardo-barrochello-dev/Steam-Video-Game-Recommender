import zipfile
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


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all((DATA_DIR / f).exists() for f in REQUIRED_FILES):
        print("Dataset already downloaded")
        return

    print("Downloading dataset from Kaggle...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(DATA_DIR),
        ],
        check=True,
    )

    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        print("No zip file found, assuming files already extracted")
        return

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(DATA_DIR)
        zip_file.unlink()
        print(f"Extracted {zip_file.name}")

    print("Dataset downloaded and extracted successfully")


def validate_files():
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]

    if missing:
        print(f"Missing files: {missing}")
        sys.exit(1)

    print("Files in data/:")
    for f in REQUIRED_FILES:
        size_mb = (DATA_DIR / f).stat().st_size / 1_000_000
        print(f"  {f} ({size_mb:.1f} MB)")


def main():
    print("\n=== Steam Dataset Download ===\n")
    check_credentials()
    download_dataset()
    validate_files()
    print("\nDone! Run 'python ml/train.py' to train the model.\n")


if __name__ == "__main__":
    main()
