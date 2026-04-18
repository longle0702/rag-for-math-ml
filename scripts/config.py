from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR /"pdf" /"mml-book_clean.pdf"
PDF_IMAGES_DIR = DATA_DIR / "images"
MARKDOWN_DIR = DATA_DIR / "markdown"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
CORPUS_JSON = DATA_DIR / "corpus.json"
SAMPLE_JSON = DATA_DIR / "sample.json"

# ── LM Studio API ───────────────────────────────────────────────────────────
# LM Studio runs on Windows; from WSL use the vEthernet (WSL) adapter IP.
# Make sure LM Studio server is set to listen on 0.0.0.0 (not 127.0.0.1).
API_BASE_URL = "http://100.76.121.29:1234/v1"
API_MODEL = "allenai/olmocr-2-7b"

# ── Workers ──────────────────────────────────────────────────────────────────
# Number of concurrent API requests (keep at 1 for single-GPU setups).
NUM_WORKERS = 1

# ── Retry settings ───────────────────────────────────────────────────────────
# Retries per page when the model crashes or returns an error.
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds to wait before retrying (gives LM Studio time to reload)

# ── Image settings ───────────────────────────────────────────────────────────
TARGET_LONGEST_IMAGE_DIM = 1024

# ── Generation settings ─────────────────────────────────────────────────────
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.8

# ── Sample size ──────────────────────────────────────────────────────────────
SAMPLE_PAGES = 10

# ── Output formatting ─────────────────────────────────────────────────────────
# If True, extract all embedded images/diagrams from each PDF page.
EXTRACT_EMBEDDED_IMAGES = True