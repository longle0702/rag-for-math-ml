"""Configuration and environment settings."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
APP_DIR = PROJECT_ROOT / "app"

CORPUS_PATH = DATA_DIR / "corpus.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
FAISS_INDEX_PATH = APP_DIR / "index" / "faiss_index.bin"
CHUNKS_PATH = APP_DIR / "index" / "chunks.json"
BM25_INDEX_PATH = APP_DIR / "index" / "bm25_index.pkl"
INDEX_METADATA_PATH = APP_DIR / "index" / "metadata.json"
INDEX_DIR = APP_DIR / "index"
VECTOR_DB_PATH = APP_DIR / "vector_db"
VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "math_ml_chunks")
VECTOR_DB_METRIC = os.getenv("VECTOR_DB_METRIC", "cosine")

INDEX_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

# ── Shared defaults ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
N_RETRIES = 3

# ── Internal Agent (RAG) ────────────────────────────────────────────────────
INTERNAL_AGENT_MODEL = "gpt-4o-mini"
INTERNAL_AGENT_TEMPERATURE = 0.2
INTERNAL_AGENT_CONTEXT_WINDOW = 128000
TOP_K_RETRIEVAL = 8
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_BACKEND = "bi_encoder"
BI_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
CROSS_ENCODER_ENABLED = True
BM25_WEIGHT = 0.1
FAISS_WEIGHT = 0.7

# ── External Agent (Web) ────────────────────────────────────────────────────
EXTERNAL_AGENT_MODEL = "gpt-4o-mini"
EXTERNAL_AGENT_TEMPERATURE = 0.2
EXTERNAL_AGENT_CONTEXT_WINDOW = 128000
# Web search settings
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
WEB_SEARCH_MODEL = "gpt-4o-mini"
WEB_SEARCH_MAX_RESULTS = 5
SCRAPE_MAX_CHARS_PER_PAGE = 10000
WEB_CHUNK_TOP_K = 5

# ── Synthesizer Agent ───────────────────────────────────────────────────────
SYNTHESIZER_AGENT_MODEL = "gpt-4o-mini"
SYNTHESIZER_AGENT_TEMPERATURE = 0.2
SYNTHESIZER_AGENT_CONTEXT_WINDOW = 128000

# ── Cost tracking ────────────────────────────────────────────────────────────
EMBEDDING_COST_PER_1K = 0.02 / 1_000_000   # $0.02 per 1M tokens
CHAT_INPUT_COST_PER_1K = 0.15 / 1_000_000  # $0.15 per 1M input tokens
CHAT_OUTPUT_COST_PER_1K = 0.60 / 1_000_000 # $0.60 per 1M output tokens

# ── Backward-compatible aliases (used by rag_pipeline) ───────────────────────
TEMPERATURE = INTERNAL_AGENT_TEMPERATURE
CONTEXT_WINDOW = INTERNAL_AGENT_CONTEXT_WINDOW
