"""Vector database wrapper for persistent chunk storage and retrieval.

Uses ChromaDB as the underlying engine. Chunks are stored with their
pre-computed embeddings and metadata (source, page, char offsets).

ID convention: each chunk is assigned a string ID equal to its integer
position in the original corpus list (e.g. "0", "1", ..., N-1).
This makes it trivial to correlate VectorDB hits with BM25 hits during
hybrid-score fusion — both operate on the same integer indices.
"""
from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np

import chromadb
from chromadb.config import Settings


class VectorDB:
    """ChromaDB-backed persistent vector store for corpus chunks.

    Args:
        persist_dir:      Path to the directory where ChromaDB stores its data.
        collection_name:  Name of the ChromaDB collection to use.
        metric:           Distance metric — ``"cosine"`` (default), ``"l2"``,
                          or ``"ip"`` (inner-product).
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "math_ml_chunks",
        metric: str = "cosine",
    ) -> None:
        self._persist_dir = str(persist_dir)
        self._collection_name = collection_name
        self._metric = metric

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._metric},
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict],
        start_id: int = 0,
    ) -> None:
        """Insert chunks with pre-computed embeddings.

        Args:
            embeddings:  Float32 array of shape ``(N, dim)``.
            documents:   Raw text for each chunk.
            metadatas:   Per-chunk metadata dicts (source, page, char_start, char_end).
            start_id:    Integer offset for ID generation; defaults to 0.
                         Pass ``self.count()`` when appending to an existing collection.
        """
        if len(documents) == 0:
            return

        ids = [str(start_id + i) for i in range(len(documents))]

        # ChromaDB expects plain Python lists, not numpy arrays.
        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def clear(self) -> None:
        """Remove all items from the collection and recreate it."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._metric},
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(self, embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Return the *top_k* nearest chunks to *embedding*.

        Each result dict contains:
            ``_idx``         – original integer chunk index (for BM25 fusion)
            ``text``         – chunk text
            ``source``       – source document name
            ``page``         – page number
            ``char_start``   – character start offset
            ``char_end``     – character end offset
            ``vector_score`` – relevance in ``[0, 1]`` (1 = most similar)

        Distance-to-relevance conversion:
            - cosine : ``relevance = 1 - dist / 2``  (dist ∈ [0, 2])
            - l2     : ``relevance = 1 / (1 + dist)``
            - ip     : ``relevance = 1 / (1 + max(0, -dist))``
        """
        n = self.count()
        if n == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, n),
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[Dict] = []
        for id_, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            relevance = self._dist_to_relevance(float(dist))
            chunks.append(
                {
                    "_idx": int(id_),
                    "text": doc,
                    "source": meta.get("source", ""),
                    "page": int(meta.get("page", 0)),
                    "char_start": int(meta.get("char_start", 0)),
                    "char_end": int(meta.get("char_end", 0)),
                    "vector_score": relevance,
                }
            )

        return chunks

    def get_all(self) -> List[Dict]:
        """Return every stored chunk **in insertion order**.

        Used to rebuild the BM25 index so that its integer positions stay
        aligned with the VectorDB IDs.
        """
        total = self.count()
        if total == 0:
            return []

        # Fetch by explicit ordered IDs to guarantee insertion-order output.
        results = self._collection.get(
            ids=[str(i) for i in range(total)],
            include=["documents", "metadatas"],
        )

        chunks: List[Dict] = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", ""),
                    "page": int(meta.get("page", 0)),
                    "char_start": int(meta.get("char_start", 0)),
                    "char_end": int(meta.get("char_end", 0)),
                }
            )
        return chunks

    def count(self) -> int:
        """Return the total number of chunks stored in the collection."""
        return self._collection.count()

    def exists(self) -> bool:
        """Return ``True`` if the collection already contains data."""
        return self.count() > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dist_to_relevance(self, dist: float) -> float:
        """Convert a ChromaDB distance value to a relevance score in ``[0, 1]``."""
        if self._metric == "cosine":
            # ChromaDB cosine distance = 1 - cosine_similarity ∈ [0, 2]
            return max(0.0, 1.0 - dist / 2.0)
        elif self._metric == "l2":
            return 1.0 / (1.0 + dist)
        else:  # "ip"
            # ChromaDB ip distance = 1 - inner_product; map back to [0, 1]
            return max(0.0, 1.0 - dist / 2.0)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VectorDB(collection={self._collection_name!r}, "
            f"metric={self._metric!r}, count={self.count()})"
        )
