from pathlib import Path

# Base project directory (assumes this file is in src/utils)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OCR_DATA_DIR = DATA_DIR / "ocr"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Data dir:", DATA_DIR)
