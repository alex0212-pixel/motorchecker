import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_FILE = DATA_DIR / "results.json"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

if not RESULTS_FILE.exists():
    import json
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": 0,
            "groups": [],
            "results": []
        }, f, ensure_ascii=False, indent=2)
