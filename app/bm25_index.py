"""BM25 index: build, persist, and load helpers."""
import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25(texts: List[str]) -> BM25Okapi:
    """Build a BM25Okapi index from a list of document strings."""
    tokenized = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


def save_bm25(bm25: BM25Okapi, path: Path) -> None:
    """Pickle the BM25 index to disk."""
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25(path: Path) -> BM25Okapi:
    """Load a pickled BM25 index from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
