#!/usr/bin/env python3
"""Verification script to ensure RAG system is properly configured."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_environment():
    """Check Python and virtual environment."""
    print("🔍 Checking environment...\n")
    
    python_version = sys.version_info
    print(f"  Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or python_version.minor < 12:
        print("  ⚠️  Warning: Python 3.12+ recommended\n")
    else:
        print("  ✓ Python version OK\n")


def check_files():
    """Check required files exist."""
    print("📂 Checking required files...\n")
    
    required_files = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "data/corpus.json",
        PROJECT_ROOT / "data/questions.json",
        PROJECT_ROOT / "app/config.py",
        PROJECT_ROOT / "app/embeddings.py",
        PROJECT_ROOT / "app/rag_pipeline.py",
        PROJECT_ROOT / "app/backend.py",
        PROJECT_ROOT / "app/frontend.py",
    ]
    
    all_exist = True
    for file in required_files:
        if file.exists():
            print(f"  ✓ {file.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  ✗ {file.relative_to(PROJECT_ROOT)} NOT FOUND")
            all_exist = False
    
    print()
    return all_exist


def check_dependencies():
    """Check if required packages are installed."""
    print("📦 Checking dependencies...\n")
    
    packages = [
        "openai",
        "faiss",
        "numpy",
        "tiktoken",
        "gradio",
        "fastapi",
        "uvicorn",
        "pydantic",
        "dotenv",
    ]
    
    missing = []
    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        print()
        return False
    
    print()
    return True


def check_env_file():
    """Check .env file and API key."""
    print("🔐 Checking API key...\n")
    
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("  ✗ .env file not found")
        print("  Create it with:")
        print("  echo 'OPENAI_API_KEY=sk-...' > .env")
        print()
        return False
    
    with open(env_file) as f:
        content = f.read()
    
    if "OPENAI_API_KEY" not in content:
        print("  ✗ OPENAI_API_KEY not found in .env")
        print()
        return False
    
    if "sk-" not in content:
        print("  ✗ OPENAI_API_KEY looks invalid (should start with 'sk-')")
        print()
        return False
    
    print("  ✓ API key configured")
    print()
    return True


def check_corpus():
    """Check corpus data."""
    print("📚 Checking corpus...\n")
    
    corpus_file = PROJECT_ROOT / "data/corpus.json"
    if not corpus_file.exists():
        print("  ✗ corpus.json not found")
        print()
        return False
    
    try:
        import json
        with open(corpus_file) as f:
            corpus = json.load(f)
        
        print(f"  ✓ Loaded {len(corpus)} documents")
        
        # Check structure
        if corpus and isinstance(corpus[0], dict):
            required_fields = ["text", "source", "page"]
            missing_fields = []
            for field in required_fields:
                if field not in corpus[0]:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"  ⚠️  Missing fields in documents: {missing_fields}")
            else:
                print(f"  ✓ Document structure looks good")
        
        print()
        return True
    
    except Exception as e:
        print(f"  ✗ Error reading corpus: {e}")
        print()
        return False


def check_index():
    """Check if FAISS index exists."""
    print("🔍 Checking FAISS index...\n")
    
    index_dir = PROJECT_ROOT / "app/index"
    index_file = index_dir / "faiss_index.bin"
    chunks_file = index_dir / "chunks.json"
    
    if index_file.exists() and chunks_file.exists():
        print(f"  ✓ FAISS index found")
        
        try:
            import json
            with open(chunks_file) as f:
                chunks = json.load(f)
            print(f"  ✓ {len(chunks)} chunks loaded")
        except Exception as e:
            print(f"  ⚠️  Error reading chunks: {e}")
        
        print()
        return True
    else:
        print("  ⚠️  FAISS index not found")
        print("    Run: python scripts/build_index.py")
        print()
        return False


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("RAG SYSTEM VERIFICATION")
    print("="*60 + "\n")
    
    checks = [
        ("Environment", check_environment),
        ("Files", check_files),
        ("Dependencies", check_dependencies),
        ("API Key", check_env_file),
        ("Corpus", check_corpus),
        ("FAISS Index", check_index),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error checking {name}: {e}\n")
            results.append((name, False))
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\n  {passed}/{total} checks passed\n")
    
    if passed == total:
        print("🎉 All checks passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. Run: python scripts/build_index.py (if not done)")
        print("  2. Run: python -m app.frontend")
        print("  3. Open: http://localhost:7860\n")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help:")
        print("  - See README.md")
        print("  - Verify .env and app/index files\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
