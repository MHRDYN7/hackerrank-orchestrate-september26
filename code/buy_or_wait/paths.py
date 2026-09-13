from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
DATA_DIR = CODE_DIR / "data"
DB_PATH = DATA_DIR / "state.db"
IMAGE_AMOUNTS_PATH = DATA_DIR / "image_amounts.json"
OUTPUT_PATH = REPO_ROOT / "output.csv"
USAGE_REPORT_PATH = CODE_DIR / "evaluation" / "usage_report.md"
ENV_PATH = REPO_ROOT / ".env"
ENV_PATHS = [REPO_ROOT / ".env", CODE_DIR / ".env"]
MEDIA_DIR = DATASET_DIR / "media" / "images"
