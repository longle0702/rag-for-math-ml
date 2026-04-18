"""Bi-encoder and cross-encoder utilities for retrieval and reranking."""
from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np

from app.config import BI_ENCODER_MODEL, CROSS_ENCODER_MODEL

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None
    CrossEncoder = None


class EncoderManager:
    """Lazy loader for bi-encoder and cross-encoder models."""

    def __init__(self):
        self._bi_encoder = None
        self._cross_encoder = None

    @property
    def bi_encoder_available(self) -> bool:
        return SentenceTransformer is not None

    @property
    def cross_encoder_available(self) -> bool:
        return CrossEncoder is not None

    def get_bi_encoder(self):
        if self._bi_encoder is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install dependencies to use bi-encoder retrieval."
                )
            self._bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
        return self._bi_encoder

    def get_cross_encoder(self):
        if self._cross_encoder is None:
            if CrossEncoder is None:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install dependencies to use cross-encoder reranking."
                )
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._cross_encoder

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        model = self.get_bi_encoder()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        model = self.get_bi_encoder()
        embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return embedding.astype("float32")

    def rerank(self, query: str, chunks: List[Dict]) -> Optional[np.ndarray]:
        if not chunks:
            return np.array([], dtype="float32")

        model = self.get_cross_encoder()
        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = model.predict(pairs, convert_to_numpy=True)
        return scores.astype("float32")


encoder_manager = EncoderManager()
