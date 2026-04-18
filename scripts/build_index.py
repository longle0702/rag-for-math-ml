#!/usr/bin/env python3
"""Script to build the vector database index from corpus."""
import sys
from pathlib import Path

# Add project root to path (scripts/ is one level below project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag_pipeline import get_pipeline
from app.embeddings import get_embedding_stats


def main():
    print("=" * 60)
    print("Vector Database Index Builder")
    print("=" * 60)
    print()

    pipeline = get_pipeline()

    print("Building vector database index...")
    print()

    # Build index (will check if it exists first)
    pipeline.build_index(force_rebuild=False)

    print()
    print("=" * 60)
    print("Build Complete!")
    print("=" * 60)

    # Show stats
    embed_stats = get_embedding_stats()
    print()
    print("📊 Embedding Statistics:")
    print(f"  Total tokens: {embed_stats['total_tokens']:,}")
    print(f"  Total cost: ${embed_stats['total_cost']:.6f}")
    print()
    print(f"✓ Vector DB stored at:  app/vector_db/")
    print(f"✓ BM25 index saved to:  app/index/bm25_index.pkl")
    print(f"✓ Total chunks: {len(pipeline.chunks)}")
    print()


if __name__ == "__main__":
    main()
