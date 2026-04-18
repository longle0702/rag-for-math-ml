# Hybrid BM25 + FAISS Retrieval — Deep Dive

## Pipeline Overview

```
                         ┌─────────────────────────────────────────────────┐
                         │                  QUERY (string)                 │
                         └───────────────────┬─────────────────────────────┘
                                             │
                    ┌────────────────────────┴───────────────────────────┐
                    │                                                     │
                    ▼                                                     ▼
       ┌────────────────────────┐                        ┌───────────────────────────┐
       │   OpenAI Embeddings    │                        │      BM25 Tokenizer       │
       │  text-embedding-3-small│                        │  re.findall([a-z0-9]+)    │
       │  query → ℝ^1536        │                        │  "Cholesky matrix" →      │
       └────────────┬───────────┘                        │  ["cholesky", "matrix"]   │
                    │                                     └──────────────┬────────────┘
                    ▼                                                     ▼
       ┌────────────────────────┐                        ┌───────────────────────────┐
       │    FAISS IndexFlatL2   │                        │      BM25Okapi.get_scores  │
       │  L2 distance search    │                        │  IDF × TF saturation      │
       │  top 2k candidates     │                        │  top 2k candidates        │
       └────────────┬───────────┘                        └──────────────┬────────────┘
                    │                                                     │
                    │        faiss_score ∈ [0, 1]                        │  bm25_score ∈ ℝ≥0
                    │                                                     │
                    └────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   Merge unique chunks    │
                              │   by chunk index         │
                              │   missing score → 0.0    │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   Min-Max Normalisation  │
                              │   both scores → [0, 1]   │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   Weighted Fusion        │
                              │   0.7·faiss + 0.3·bm25   │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │  Sort ↓ hybrid_score     │
                              │  Return top_k chunks     │
                              └──────────────────────────┘
```

---

## 1 — Corpus Preparation

### 1.1 Chunking

The raw corpus (one entry per PDF page) is split into overlapping character windows so that long pages don't exceed the embedding context window and so that every relevant sentence has a chance of being the *start* of a chunk.

```python
for start in range(0, text_len, CHUNK_SIZE - CHUNK_OVERLAP):   # step = 400 chars
    end   = start + CHUNK_SIZE                                  # window = 500 chars
    chunk = text[start:end]
    if len(chunk.strip()) > 50:
        chunks.append({"text": chunk, "source": ..., "page": ...})
```

Overlap ensures that a sentence spanning the boundary between two non-overlapping windows is
fully captured in at least one chunk:

```
|<──── chunk i (500) ────>|
                 |<── overlap (100) ──>|<──── chunk i+1 (500) ────>|
```

### 1.2 FAISS Index Build

Every chunk is embedded once and stored in a flat L2 index:

```python
embeddings, _ = embed_texts(texts)          # shape: (N_chunks, 1536)
index = faiss.IndexFlatL2(embedding_dim)    # euclidean distance, exact search
index.add(embeddings)
faiss.write_index(index, "faiss_index.bin")
```

### 1.3 BM25 Index Build

```python
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

tokenized_corpus = [tokenize(chunk["text"]) for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)
pickle.dump(bm25, open("bm25_index.pkl", "wb"))
```

`BM25Okapi` precomputes:
- $\text{IDF}$ for every vocabulary token across the corpus
- Per-document token frequencies and length normalisation factors

This makes individual query scoring a fast dot-product at runtime.

---

## 2 — BM25 Scoring

### 2.1 The Formula

$$\text{BM25}(D, Q) = \sum_{i=1}^{|Q|} \underbrace{\log\!\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5 + 1}\right)}_{\text{IDF}(q_i)} \cdot \underbrace{\frac{f(q_i,D)\,(k_1+1)}{f(q_i,D) + k_1\!\left(1 - b + b\,\frac{|D|}{\text{avgdl}}\right)}}_{\text{TF saturation}}$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| $N$ | Total number of documents (chunks) | 2 335 |
| $n(q_i)$ | Chunks containing term $q_i$ | — |
| $f(q_i, D)$ | Raw count of $q_i$ in chunk $D$ | — |
| $\|D\|$ | Length of chunk $D$ in tokens | — |
| $\text{avgdl}$ | Average chunk length | — |
| $k_1$ | TF saturation parameter | 1.5 |
| $b$ | Length normalisation parameter | 0.75 |

### 2.2 What Each Term Does

**IDF — Inverse Document Frequency**

$$\text{IDF}(q_i) = \log\!\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}$$

- Rare terms (e.g. `cholesky`) → high IDF → high contribution
- Common terms (e.g. `the`, `is`) → IDF ≈ 0 → near-zero contribution

```
      IDF
  high │ ●  "cholesky"
       │    ●  "eigenvalue"
       │       ●  "matrix"
   low │              ●  "the"   ●  "is"
       └───────────────────────────────────> document frequency
```

**TF Saturation — diminishing returns**

Without saturation, a term appearing 100× would score 100× a term appearing once.
$k_1 = 1.5$ flattens the curve:

$$\text{tf\_sat}(f) = \frac{f \cdot (k_1+1)}{f + k_1}$$

```
  tf_sat
    2.5 │                        ─────────────
        │                  ────
    1.5 │           ────                        ← asymptote at k₁+1 = 2.5
        │      ───
    0.0 └──────────────────────────────────────> raw term frequency f
            1   2   3   5   10   20
```

**Length Normalisation** ($b = 0.75$) penalises long documents that naturally accumulate more term occurrences just by being longer.

### 2.3 Implementation

```python
def _retrieve_bm25(self, query: str, candidate_k: int) -> List[Dict]:
    tokens = bm25_tokenize(query)            # ["cholesky", "decomposition"]
    scores = self.bm25.get_scores(tokens)    # np.ndarray, shape (N_chunks,)
    top_indices = np.argsort(scores)[::-1][:candidate_k]
    return [
        {"_idx": int(idx), **self.chunks[idx], "bm25_score": float(scores[idx])}
        for idx in top_indices
    ]
```

---

## 3 — FAISS Semantic Scoring

### 3.1 Embeddings

OpenAI `text-embedding-3-small` maps text to a unit-ish vector in $\mathbb{R}^{1536}$.
Semantically similar texts are close in this space regardless of exact wording:

```
  ℝ^1536 (projected to 2D for illustration)

        ● "Cholesky factor L"
        ● "lower triangular matrix factorisation"      ← semantically close (small L2)
             ...
                   ● "gradient descent update"
                   ● "negative gradient step"          ← semantically close
```

### 3.2 L2 Distance → Similarity

FAISS returns squared L2 distances $d$. We convert to a similarity in $(0, 1]$:

$$s_{\text{faiss}} = \frac{1}{1 + d}$$

As $d \to 0$ (identical vectors), $s_{\text{faiss}} \to 1$.
As $d \to \infty$, $s_{\text{faiss}} \to 0$.

```python
def _retrieve_faiss(self, query_embedding: np.ndarray, candidate_k: int) -> List[Dict]:
    distances, indices = self.index.search(query_embedding, candidate_k)
    return [
        {
            "_idx": int(idx),
            **self.chunks[idx],
            "faiss_score": 1.0 / (1.0 + float(distances[0][i])),
        }
        for i, idx in enumerate(indices[0]) if idx < len(self.chunks)
    ]
```

---

## 4 — Hybrid Fusion

### 4.1 The Problem: Incompatible Score Ranges

| Retriever | Raw score range | Example values |
|-----------|----------------|----------------|
| FAISS | $(0, 1]$ | 0.12 – 0.55 |
| BM25 | $\mathbb{R}_{\geq 0}$ | 0.0 – 14.3 |

Combining these directly would make BM25 dominate purely due to scale.

### 4.2 Min-Max Normalisation

For each retriever, rescale the candidate scores to $[0, 1]$:

$$s_{\text{norm}} = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$

Special case: if $s_{\max} = s_{\min}$ (all scores identical), return $\mathbf{0}$.

```python
def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)

faiss_norm = _minmax(np.array([c["faiss_score"] for c in candidates]))
bm25_norm  = _minmax(np.array([c["bm25_score"]  for c in candidates]))
```

### 4.3 Candidate Pool Expansion

Each retriever independently fetches $2k$ candidates (where $k$ is the final `top_k`).
This ensures the hybrid has a wide diverse pool before pruning:

```
FAISS top-16:  [c3, c7, c1, c22, c8, ...]    (semantic top)
BM25  top-16:  [c7, c14, c3, c9, c2, ...]    (keyword top)

merged unique: [c1, c2, c3, c7, c8, c9, c14, c22, ...]  (up to 32 unique chunks)
                              ↓
           re-rank by hybrid_score, keep top-8
```

Chunks that appear in only one list receive score `0.0` in the other:

```python
merged: Dict[int, Dict] = {}
for r in faiss_results:
    merged[r["_idx"]] = {**r, "bm25_score": 0.0}   # default bm25 = 0
for r in bm25_results:
    if r["_idx"] in merged:
        merged[r["_idx"]]["bm25_score"] = r["bm25_score"]
    else:
        merged[r["_idx"]] = {**r, "faiss_score": 0.0}  # default faiss = 0
```

### 4.4 Weighted Score Fusion

$$\boxed{\text{hybrid\_score} = \underbrace{0.7}_{\alpha_{\text{faiss}}} \cdot s_{\text{faiss\_norm}} + \underbrace{0.3}_{\alpha_{\text{bm25}}} \cdot s_{\text{bm25\_norm}}}$$

These weights are configurable in `app/config.py`:

```python
FAISS_WEIGHT = 0.7   # semantic search dominates
BM25_WEIGHT  = 0.3   # keyword boost for exact terms
```

The 70/30 split reflects the nature of the corpus (math textbook):
most questions are conceptual (favouring semantics) but technical term queries
like "Cholesky" or "Taylor polynomial" benefit strongly from keyword boosting.

### 4.5 Score Contributions Illustrated

```
Query: "What is the Cholesky decomposition?"

Chunk  │ faiss_norm │ bm25_norm │ hybrid = 0.7f + 0.3b
───────┼────────────┼───────────┼─────────────────────
c_105a │   1.000    │   0.792   │  0.7×1.0  + 0.3×0.792 = 0.938  ← top
c_105b │   0.895    │   1.000   │  0.7×0.895 + 0.3×1.0  = 0.926
c_125  │   0.875    │   0.817   │  0.7×0.875 + 0.3×0.817 = 0.858
c_12   │   0.701    │   0.000   │  0.7×0.701 + 0.3×0.0   = 0.491  ← BM25 adds nothing
c_88   │   0.000    │   0.543   │  0.7×0.0   + 0.3×0.543 = 0.163  ← FAISS adds nothing
```

---

## 5 — Why This Beats Either Alone

| Query type | Only FAISS | Only BM25 | Hybrid |
|---|---|---|---|
| Exact term: `"Cholesky decomposition"` | May miss exact phrasing | ✅ Exact match | ✅ BM25 pulls it to top |
| Semantic: `"what captures spread of data"` | ✅ Finds covariance | ❌ No keyword overlap | ✅ FAISS carries it |
| Mixed: `"eigenvalue PCA variance"` | ✅ Good | ✅ Good | ✅ Both reinforce, rank rises |
| Out-of-scope: `"What is IoT?"` | Low score | Low score | Both low → threshold blocks it |

The `relevance_score < 0.3` guard in `generate_answer()` now operates on the hybrid score,
which is in $[0, 1]$ and more stable than raw FAISS distance.

---

## 6 — Configuration Reference

All parameters are in [`app/config.py`](app/config.py):

```python
TOP_K_RETRIEVAL              = 8    # final number of chunks returned
RETRIEVAL_CANDIDATE_MULTIPLIER = 2  # each retriever fetches 8×2 = 16 candidates
FAISS_WEIGHT                 = 0.7  # α for semantic score
BM25_WEIGHT                  = 0.3  # α for keyword score
                                    # constraint: FAISS_WEIGHT + BM25_WEIGHT = 1.0
```

### Tuning Guidance

| Scenario | Recommended adjustment |
|---|---|
| Users ask mostly exact definition queries | Increase `BM25_WEIGHT` → 0.4–0.5 |
| Users ask mostly conceptual questions | Increase `FAISS_WEIGHT` → 0.8–0.9 |
| Too many irrelevant results | Raise `relevance_score` threshold above 0.3 |
| Missing some relevant chunks | Increase `RETRIEVAL_CANDIDATE_MULTIPLIER` to 3 |

---

## 7 — Index Files

| File | Size | Contents |
|---|---|---|
| `app/index/faiss_index.bin` | ~35 MB | 2 335 × 1536 float32 vectors |
| `app/index/chunks.json` | ~2.5 MB | Chunk text + source + page metadata |
| `app/index/bm25_index.pkl` | ~5 MB | Pickled `BM25Okapi` (IDF table + TF arrays) |

Rebuild all three with a single command (requires OpenAI API key, costs ~$0.05):

```bash
python scripts/build_index.py
```
