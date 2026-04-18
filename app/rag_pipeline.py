"""RAG pipeline for chunking, indexing, retrieval, and generation."""
import json
import numpy as np
import faiss
import time
from typing import List, Dict, Tuple, Optional
from openai import OpenAI, RateLimitError, APIError
import tiktoken


from app.config import (
    OPENAI_API_KEY, CORPUS_PATH, FAISS_INDEX_PATH, CHUNKS_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, CHAT_MODEL, TOP_K_RETRIEVAL,
    TEMPERATURE, EMBEDDING_MODEL, INDEX_METADATA_PATH, RETRIEVAL_BACKEND,
    CHAT_INPUT_COST_PER_1K, CHAT_OUTPUT_COST_PER_1K, N_RETRIES,
    BM25_INDEX_PATH, BM25_WEIGHT, FAISS_WEIGHT, RETRIEVAL_CANDIDATE_MULTIPLIER,
    CONTEXT_WINDOW, CROSS_ENCODER_ENABLED, BI_ENCODER_MODEL, CROSS_ENCODER_MODEL,
    VECTOR_DB_PATH, VECTOR_DB_COLLECTION, VECTOR_DB_METRIC,
)
from app.embeddings import embed_texts, embed_query
from app.bm25_index import build_bm25, save_bm25, load_bm25, tokenize as bm25_tokenize
from app.encoders import encoder_manager
from app.vector_db import VectorDB

# Alias so the rest of the file reads cleanly
VECTOR_WEIGHT = FAISS_WEIGHT

client = OpenAI(api_key=OPENAI_API_KEY)
chat_encoding = tiktoken.encoding_for_model(CHAT_MODEL)

# Global cost tracking
total_chat_input_tokens = 0
total_chat_output_tokens = 0
total_chat_cost = 0.0

def get_tokens_num(api_messages : list):
        try:
            # Count the total number of tokens for the input
            gpt_encoder = tiktoken.encoding_for_model(CHAT_MODEL)
            tokens_num = 0

            for messages in api_messages:
                tokens_num += 3
                for _, value in messages.items():
                    tokens_num += len(gpt_encoder.encode(value))

                tokens_num += 3

            return tokens_num
        except Exception:
            # Error counting - Ignore tokens number
            return -1

class RAGPipeline:
    """Complete RAG pipeline."""

    def __init__(self):
        self.vector_db = VectorDB(
            persist_dir=str(VECTOR_DB_PATH),
            collection_name=VECTOR_DB_COLLECTION,
            metric=VECTOR_DB_METRIC,
        )
        self.index = None          # FAISS index (loaded if index file exists)
        self.index_metric = "l2"    # FAISS distance metric used at build time
        self.chunks: List[Dict] = []  # in-memory cache used by BM25
        self.bm25 = None
        self.loaded = False
        self.index_backend = "openai"
    
    def load_corpus(self) -> List[Dict]:
        """Load corpus from JSON file."""
        with open(CORPUS_PATH, 'r') as f:
            corpus = json.load(f)
        return corpus
    
    def chunk_text(self, text: str, source: str, page: int) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            source: Source document name
            page: Page number
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        text_len = len(text)
        
        for start in range(0, text_len, CHUNK_SIZE - CHUNK_OVERLAP):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            
            if len(chunk_text.strip()) > 50:  # Skip very short chunks
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "char_start": start,
                    "char_end": end,
                })
        
        return chunks
    
    def _save_index_metadata(self):
        metadata = {
            "retrieval_backend": self.index_backend,
            "vector_db_metric": VECTOR_DB_METRIC,
            "bi_encoder_model": BI_ENCODER_MODEL if self.index_backend == "bi_encoder" else None,
            "cross_encoder_model": CROSS_ENCODER_MODEL if CROSS_ENCODER_ENABLED else None,
        }
        with open(INDEX_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_index_metadata(self) -> Dict:
        if not INDEX_METADATA_PATH.exists():
            return {"retrieval_backend": "openai"}
        with open(INDEX_METADATA_PATH, "r") as f:
            return json.load(f)

    def build_index(self, force_rebuild: bool = False):
        """
        Build vector database index from corpus.

        Args:
            force_rebuild: Force rebuild even if index exists
        """
        # Check if index already exists
        if self.vector_db.exists() and not force_rebuild:
            print("✓ Loading existing vector database...")
            self.load_index()
            return

        print("📚 Building new index...")

        # Clear any stale data from a previous (incomplete) build
        if self.vector_db.exists():
            print("  Clearing existing vector database...")
            self.vector_db.clear()

        # Load corpus
        corpus = self.load_corpus()
        print(f"  Loaded {len(corpus)} documents")

        # Chunk all documents
        all_chunks = []
        for doc in corpus:
            chunks = self.chunk_text(doc["text"], doc["source"], doc["page"])
            all_chunks.extend(chunks)

        print(f"  Created {len(all_chunks)} chunks")

        # Extract texts and metadata for storage
        texts = [chunk["text"] for chunk in all_chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
            }
            for chunk in all_chunks
        ]

        use_bi_encoder = RETRIEVAL_BACKEND == "bi_encoder" and encoder_manager.bi_encoder_available

        # Embed all chunks
        if use_bi_encoder:
            print(f"  Using bi-encoder model: {BI_ENCODER_MODEL}")
            embeddings = encoder_manager.encode_texts(texts)
            self.index_backend = "bi_encoder"
        else:
            if RETRIEVAL_BACKEND == "bi_encoder" and not encoder_manager.bi_encoder_available:
                print("  sentence-transformers not available; falling back to OpenAI embeddings retrieval.")
            embeddings, _ = embed_texts(texts, show_progress=True)
            self.index_backend = "openai"

        # Store embeddings + chunks in the vector database
        print("  Storing chunks in vector database...")
        self.vector_db.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        self._save_index_metadata()

        self.chunks = all_chunks

        # Build and save BM25 index
        print("  Building BM25 index...")
        self.bm25 = build_bm25(texts)
        save_bm25(self.bm25, BM25_INDEX_PATH)

        self.loaded = True

        print(f"✓ Index built and saved ({len(all_chunks)} chunks)")
    
    def load_index(self):
        """Load pre-built index from the vector database."""
        if not self.vector_db.exists():
            raise ValueError(
                "Vector database is empty. Run build_index() first."
            )

        # Populate the in-memory chunk list from the persistent vector database.
        # This keeps BM25 indices aligned with vector database IDs.
        self.chunks = self.vector_db.get_all()

        metadata = self._load_index_metadata()
        self.index_backend = metadata.get("retrieval_backend", "openai")

        # Always rebuild BM25 from the currently-loaded chunks so it is
        # guaranteed to be in sync with self.chunks (a stale pickle from a
        # previous build_index run would have a different corpus size and
        # produce out-of-range indices).
        print("  Rebuilding BM25 index from loaded chunks...")
        self.bm25 = build_bm25([c["text"] for c in self.chunks])

        # Optionally load the legacy FAISS index if it still exists on disk.
        if FAISS_INDEX_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            self.index_metric = self._load_index_metadata().get("metric", "l2")

        self.loaded = True
        print(
            f"\u2713 Loaded vector database with {len(self.chunks)} chunks "
            f"(backend={self.index_backend})"
        )

    def _retrieve_vector_db(self, query_embedding: np.ndarray, candidate_k: int) -> List[Dict]:
        """Return top-candidate_k results from the vector database.

        Primary semantic retrieval path. Each result contains ``_idx`` and
        ``vector_score`` (relevance in [0, 1]).
        """
        return self.vector_db.query(query_embedding.flatten(), candidate_k)

    def _retrieve_faiss(self, query_embedding: np.ndarray, candidate_k: int) -> List[Dict]:
        """Return top-candidate_k results from the legacy FAISS index.

        Kept alongside ``_retrieve_vector_db`` for backward compatibility and
        offline/no-ChromaDB scenarios. Each result uses the same ``vector_score``
        key as ``_retrieve_vector_db`` so the caller's merge logic is identical.

        Raises:
            ValueError: if the FAISS index has not been loaded.
        """
        if self.index is None:
            raise ValueError(
                "FAISS index not loaded. Ensure faiss_index.bin exists before calling load_index()."
            )

        distances, indices = self.index.search(query_embedding.reshape(1, -1), candidate_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                raw_distance = float(distances[0][i])
                if self.index_metric == "ip":
                    relevance = (raw_distance + 1.0) / 2.0
                else:
                    relevance = 1.0 / (1.0 + raw_distance)
                results.append({
                    "_idx": int(idx),
                    **chunk,
                    "distance": raw_distance,
                    "vector_score": relevance,
                })
        return results

    def _retrieve_bm25(self, query: str, candidate_k: int) -> List[Dict]:
        """Return top-candidate_k BM25 results with raw BM25 score."""
        tokens = bm25_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:candidate_k]
        results = []
        for idx in top_indices:
            results.append({
                "_idx": int(idx),
                **self.chunks[idx],
                "bm25_score": float(scores[idx]),
            })
        return results

    @staticmethod
    def _is_vector_dim_mismatch_error(exc: Exception) -> bool:
        msg = str(exc)
        return "expecting embedding with dimension" in msg and "got" in msg

    def _embed_query_for_backend(self, query: str, backend: str) -> np.ndarray:
        """Embed query for a specific retrieval backend."""
        if backend == "bi_encoder" and encoder_manager.bi_encoder_available:
            return encoder_manager.encode_query(query)
        query_embedding, _ = embed_query(query)
        return query_embedding

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """
        Hybrid Vector DB + BM25 retrieval with weighted score fusion,
        followed by Cross-encoder reranking if enabled.
        """
        if not self.loaded:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")

        candidate_k = max(top_k, top_k * RETRIEVAL_CANDIDATE_MULTIPLIER)

        # Embed/query with preferred backend first, then fall back if vector DB
        # dimension does not match (e.g. stale metadata says bi_encoder while
        # collection was built with OpenAI vectors).
        preferred_backend = self.index_backend if self.index_backend in {"openai", "bi_encoder"} else "openai"
        backend_order = [preferred_backend, "openai" if preferred_backend == "bi_encoder" else "bi_encoder"]

        vector_results: List[Dict] | None = None
        last_vector_error: Exception | None = None

        for backend in backend_order:
            if backend == "bi_encoder" and not encoder_manager.bi_encoder_available:
                continue
            try:
                query_embedding = self._embed_query_for_backend(query, backend)
                vector_results = self._retrieve_vector_db(query_embedding, candidate_k)
                if backend != self.index_backend:
                    print(
                        "Vector DB embedding dimension mismatch detected; "
                        f"switching runtime retrieval backend from {self.index_backend} to {backend}."
                    )
                    self.index_backend = backend
                break
            except Exception as exc:
                last_vector_error = exc
                if self._is_vector_dim_mismatch_error(exc):
                    continue
                raise

        if vector_results is None:
            if last_vector_error is not None:
                raise last_vector_error
            raise RuntimeError("Vector retrieval failed unexpectedly.")

        # --- BM25 retrieval ---
        bm25_results = self._retrieve_bm25(query, candidate_k)

        # --- Merge unique chunks by chunk index ---
        merged: Dict[int, Dict] = {}
        for r in vector_results:
            merged[r["_idx"]] = {**r, "bm25_score": 0.0}
        for r in bm25_results:
            idx = r["_idx"]
            if idx in merged:
                merged[idx]["bm25_score"] = r["bm25_score"]
            else:
                merged[idx] = {**r, "vector_score": 0.0}

        candidates = list(merged.values())

        # --- Min-max normalise each score set independently ---
        vector_scores = np.array([c["vector_score"] for c in candidates])
        bm25_scores = np.array([c["bm25_score"] for c in candidates])

        def _minmax(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi == lo:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        vector_norm = _minmax(vector_scores)
        bm25_norm = _minmax(bm25_scores)

        # --- Compute hybrid score and attach to candidates ---
        for i, c in enumerate(candidates):
            c["vector_score_norm"] = float(vector_norm[i])
            c["bm25_score_norm"] = float(bm25_norm[i])
            c["relevance_score"] = float(
                VECTOR_WEIGHT * vector_norm[i] + BM25_WEIGHT * bm25_norm[i]
            )

        # --- Cross-encoder rerank ---
        if CROSS_ENCODER_ENABLED and encoder_manager.cross_encoder_available and candidates:
            try:
                ce_scores = encoder_manager.rerank(query, candidates)
                for chunk, ce_score in zip(candidates, ce_scores):
                    chunk["cross_encoder_score"] = float(ce_score)
                    # Use cross-encoder score as ultimate relevance, mapped to 0-1
                    chunk["relevance_score"] = float(1.0 / (1.0 + np.exp(-ce_score)))
            except Exception as exc:
                print(f"Cross-encoder reranking skipped: {exc}")

        # --- Sort and return top_k ---
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        # Remove internal key before returning
        for c in candidates:
            c.pop("_idx", None)

        return candidates[:top_k]

    def generate_answer(self, query: str, history: list, retrieved_chunks: List[Dict]) -> Tuple[str, float]:
        global total_chat_input_tokens, total_chat_output_tokens, total_chat_cost

        if not retrieved_chunks or all(chunk["relevance_score"] < 0.3 for chunk in retrieved_chunks):
            return (
                "I don't have information about this topic in my knowledge base. "
                "Please try asking a question related to the Mathematics for Machine Learning content.",
                0.0
            )

        # Build context from retrieved chunks
        context = "\n\n---\n\n".join([
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])

        # Build prompt
        system_prompt = """You are a helpful assistant that answers questions about Mathematics for Machine Learning.
You must:
Answer based only on the provided context, if the question is not very specific to the context, reply asking for clarification. Other wise:
2. Cite your sources (document name and page number).
3. If the question is out of scope, politely decline.
4. Be concise and accurate.
5. Format mathematical expressions using LaTeX ($...$ for inline, $$...$$ for blocks).
"""

        user_message = f"""Question: {query}

Context from the knowledge base:
{context}

Please answer the question using the provided context. Include source citations."""

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(history)
        api_messages.append({"role": "user", "content": user_message})

        tokens_num = get_tokens_num(api_messages)
        if tokens_num >= CONTEXT_WINDOW:
            return "ERNEWCONV", 0.0

        response = None

        # Call GPT-4o-mini
        for attempt in range(N_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=api_messages,
                    temperature=TEMPERATURE,
                    max_tokens=1000
                )
                break
            except RateLimitError:
                if attempt < N_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    return "ERRATE", 0.0
            except APIError as e:
                print(f"API error: {e}")
                time.sleep(2)

        if not response:
            return "ERNORES", 0.0

        answer = response.choices[0].message.content

        # Track costs
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        total_chat_input_tokens += input_tokens
        total_chat_output_tokens += output_tokens

        cost = (input_tokens * CHAT_INPUT_COST_PER_1K + 
                output_tokens * CHAT_OUTPUT_COST_PER_1K)
        total_chat_cost += cost

        return answer, cost

    def answer_question(
        self,
        query: str,
        history: Optional[list] = None,
        top_k: int = TOP_K_RETRIEVAL,
        return_sources: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate answer.

        Args:
            query: User question
            return_sources: Whether to include retrieved sources

        Returns:
            Dict with answer, sources, and cost
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        # Generate answer
        if history is None:
            history = []
        answer, cost = self.generate_answer(query, history, retrieved_chunks)

        result = {
            "question": query,
            "answer": answer,
            "cost": cost,
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "relevance": chunk["relevance_score"],
                    "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                }
                for chunk in retrieved_chunks
            ]
        
        return result
    
    def get_chat_stats(self) -> dict:
        """Get chat statistics."""
        return {
            "input_tokens": total_chat_input_tokens,
            "output_tokens": total_chat_output_tokens,
            "total_cost": total_chat_cost,
        }


# Global pipeline instance
pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or create global pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline
