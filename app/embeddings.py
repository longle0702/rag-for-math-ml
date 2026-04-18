"""Embeddings module for generating and managing vectors."""
import json
import numpy as np
from typing import List, Tuple
from openai import OpenAI
import tiktoken
from app.config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHUNKS_PATH, 
    EMBEDDING_COST_PER_1K
)

client = OpenAI(api_key=OPENAI_API_KEY)

# Token counter for cost tracking
encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
total_embedding_tokens = 0
total_embedding_cost = 0.0


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(encoding.encode(text))


def embed_texts(texts: List[str], show_progress: bool = True) -> Tuple[np.ndarray, float]:
    """
    Embed multiple texts using OpenAI API.
    
    Args:
        texts: List of texts to embed
        show_progress: Whether to show progress
    
    Returns:
        Tuple of (embeddings array, cost)
    """
    global total_embedding_tokens, total_embedding_cost
    
    if not texts:
        return np.array([]), 0.0
    
    if show_progress:
        from tqdm import tqdm
        texts_iter = tqdm(texts, desc="Embedding chunks")
    else:
        texts_iter = texts
    
    all_embeddings = []
    total_tokens = 0
    
    # Process in batches to be efficient
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        if show_progress and i > 0:
            texts_iter.update(min(batch_size, len(texts) - i))
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float"
        )
        
        for item in response.data:
            all_embeddings.append(item.embedding)
        
        total_tokens += response.usage.prompt_tokens
    
    # Update global tracking
    cost = total_tokens * EMBEDDING_COST_PER_1K
    total_embedding_tokens += total_tokens
    total_embedding_cost += cost
    
    if show_progress:
        texts_iter.close()
    
    embeddings = np.array(all_embeddings).astype('float32')
    
    print(f"  Embedded {len(texts)} chunks ({total_tokens} tokens, ${cost:.6f})")
    
    return embeddings, cost


def embed_query(query: str) -> Tuple[np.ndarray, float]:
    """
    Embed a single query.
    
    Args:
        query: Query text to embed
    
    Returns:
        Tuple of (embedding array, cost)
    """
    global total_embedding_tokens, total_embedding_cost
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
        encoding_format="float"
    )
    
    embedding = np.array(response.data[0].embedding).astype('float32')
    tokens = response.usage.prompt_tokens
    cost = tokens * EMBEDDING_COST_PER_1K
    
    total_embedding_tokens += tokens
    total_embedding_cost += cost
    
    return embedding, cost


def get_embedding_stats() -> dict:
    """Get embedding statistics."""
    return {
        "total_tokens": total_embedding_tokens,
        "total_cost": total_embedding_cost,
    }
