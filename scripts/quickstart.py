#!/usr/bin/env python3
"""Quick start guide and launcher for RAG system."""
import sys
from pathlib import Path

# Add project root to path (scripts/ is one level below project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_menu():
    """Print main menu."""
    print("\n" + "=" * 60)
    print("🤖 RAG System for Math ML - Quick Start")
    print("=" * 60)
    print("\nWhat would you like to do?\n")
    print("1. Build FAISS index from corpus")
    print("2. Run web interface (Gradio)")
    print("3. Run API server (FastAPI)")
    print("4. Check system status")
    print("5. View cost statistics")
    print("6. Evaluate on test questions")
    print("7. Exit")
    print()


def check_env():
    """Check if .env file exists."""
    env_path = project_root / ".env"
    if not env_path.exists():
        print("❌ .env file not found!")
        print("Create one with: echo 'OPENAI_API_KEY=sk-...' > .env")
        return False
    return True


def build_index():
    """Build FAISS index."""
    print("\n📚 Building FAISS Index...")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "build_index.py")])


def run_frontend():
    """Run Gradio frontend."""
    print("\n🌐 Starting Gradio Frontend...")
    print("Opening browser at http://localhost:7860\n")

    from app.frontend import create_interface
    demo = create_interface()
    demo.launch(server_port=7860, share=False)


def run_backend():
    """Run FastAPI backend."""
    print("\n🚀 Starting FastAPI Backend...")
    print("API available at http://localhost:8000\n")
    print("Endpoints:")
    print("  - GET  /health")
    print("  - POST /query")
    print("  - GET  /stats")
    print("  - GET  /evaluate")
    print()

    import subprocess
    subprocess.run([sys.executable, "-m", "uvicorn", "app.backend:app", "--reload"],
                  cwd=str(project_root))


def check_status():
    """Check system status."""
    from app.config import FAISS_INDEX_PATH, CHUNKS_PATH, CORPUS_PATH

    print("\n📊 System Status")
    print("=" * 60)

    # Check corpus
    if CORPUS_PATH.exists():
        import json
        with open(CORPUS_PATH) as f:
            corpus = json.load(f)
        print(f"✓ Corpus loaded: {len(corpus)} documents")
    else:
        print(f"❌ Corpus not found at {CORPUS_PATH}")

    # Check index
    if FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists():
        import json
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        print(f"✓ FAISS index ready: {len(chunks)} chunks")
    else:
        print(f"❌ Index not found. Run option 1 to build.")

    # Check API key
    import os
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
    if os.getenv("OPENAI_API_KEY"):
        print(f"✓ OpenAI API key configured")
    else:
        print(f"❌ OpenAI API key not found in .env")

    print()


def view_stats():
    """View cost statistics."""
    try:
        from app.rag_pipeline import get_pipeline
        from app.embeddings import get_embedding_stats

        pipeline = get_pipeline()
        if not pipeline.loaded:
            try:
                pipeline.load_index()
            except Exception:
                print("❌ Index not loaded. Run option 1 first.")
                return

        embed_stats = get_embedding_stats()
        chat_stats = pipeline.get_chat_stats()

        total_cost = embed_stats['total_cost'] + chat_stats['total_cost']

        print("\n💰 Cost Statistics")
        print("=" * 60)
        print(f"Embedding tokens: {embed_stats['total_tokens']:,}")
        print(f"Embedding cost: ${embed_stats['total_cost']:.6f}")
        print()
        print(f"Chat input tokens: {chat_stats['input_tokens']:,}")
        print(f"Chat output tokens: {chat_stats['output_tokens']:,}")
        print(f"Chat cost: ${chat_stats['total_cost']:.6f}")
        print()
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Remaining budget: ${5.00 - total_cost:.6f}")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")


def evaluate():
    """Evaluate on test questions and save results."""
    try:
        from app.rag_pipeline import get_pipeline
        from app.config import QUESTIONS_PATH
        import json
        import os

        pipeline = get_pipeline()
        if not pipeline.loaded:
            try:
                pipeline.load_index()
            except Exception:
                print("❌ Index not loaded. Run option 1 first.")
                return

        with open(QUESTIONS_PATH) as f:
            questions = json.load(f)

        print(f"\n🧪 Evaluating on {len(questions)} test questions...")
        print("=" * 60)

        results = []

        for i, q in enumerate(questions, 1):
            print(f"\n[{i}] {q['question']}")
            result = pipeline.answer_question(q['question'])

            print(f"    Answer: {result['answer']}")
            print(f"    Cost: ${result['cost']:.6f}")

            # 👇 save each result
            results.append({
                "question": q["question"],
                "predicted_answer": result["answer"],
                "cost": result["cost"]
            })

        output_path = "data/eval_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to {output_path}")
        print("✓ Evaluation complete!")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main menu loop."""
    if not check_env():
        sys.exit(1)

    while True:
        print_menu()
        choice = input("Enter your choice (1-7): ").strip()

        if choice == "1":
            build_index()
        elif choice == "2":
            run_frontend()
        elif choice == "3":
            run_backend()
        elif choice == "4":
            check_status()
        elif choice == "5":
            view_stats()
        elif choice == "6":
            evaluate()
        elif choice == "7":
            print("\nGoodbye! 👋")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
