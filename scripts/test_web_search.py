#!/usr/bin/env python3
"""
Test the 3-step web search pipeline directly (no MCP server needed).

Step 1: generate_web_query  — LLM turns a question into a Google search query
Step 2: google_search       — SerpAPI returns URLs + snippets
Step 3: scrape_urls         — Crawl4AI scrapes the URLs into markdown

Outputs:
  data/web_sources.json  — URL list with title & snippet metadata
  data/web_context.md    — full scraped markdown (all sources combined)

Run:  .venv/bin/python -m scripts.test_web_search
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.mcp_server import generate_web_query, google_search, scrape_urls

QUESTION = "What is the kernel trick in support vector machines?"
TOP_N_URLS = 5

SEPARATOR = "=" * 70
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


async def main():
    print(SEPARATOR)
    print("STEP 1 — generate_web_query")
    print(SEPARATOR)
    web_query = generate_web_query(QUESTION)
    print(f"  Question : {QUESTION}")
    print(f"  Web query: {web_query}")

    if web_query.startswith("EREX"):
        print(f"  ❌ Failed: {web_query}")
        return

    print()
    print(SEPARATOR)
    print("STEP 2 — google_search")
    print(SEPARATOR)
    search_raw = google_search(web_query)
    print(f"  Raw (first 500 chars): {search_raw[:500]}")

    if search_raw.startswith("EREX"):
        print(f"  ❌ Failed: {search_raw}")
        return

    try:
        results = json.loads(search_raw)
    except json.JSONDecodeError:
        print(f"  ❌ Could not parse JSON: {search_raw[:200]}")
        return

    urls = [r["url"] for r in results if r.get("url")]
    print(f"  Found {len(urls)} URLs:")
    for i, url in enumerate(urls):
        print(f"    {i+1}. {url}")

    if not urls:
        print("  ❌ No URLs found")
        return

    # ── Save URL metadata as JSON ──
    sources_path = DATA_DIR / "web_sources.json"
    sources_data = {
        "question": QUESTION,
        "web_query": web_query,
        "sources": results,
    }
    sources_path.write_text(json.dumps(sources_data, indent=2, ensure_ascii=False))
    print(f"\n  📄 Saved URL metadata → {sources_path}")

    print()
    print(SEPARATOR)
    print(f"STEP 3 — scrape_urls (Crawl4AI) — top {TOP_N_URLS}")
    print(SEPARATOR)
    test_urls = urls[:TOP_N_URLS]
    print(f"  Scraping {len(test_urls)} URLs...")
    scraped = await scrape_urls(test_urls)
    print(f"  Total scraped length: {len(scraped)} chars")

    # ── Save scraped markdown ──
    context_path = DATA_DIR / "web_context.md"
    context_path.write_text(scraped, encoding="utf-8")
    print(f"  📄 Saved scraped content → {context_path}")

    # Preview
    print()
    print(scraped[:1000])
    print("... (truncated)" if len(scraped) > 1000 else "")

    print()
    print(SEPARATOR)
    print("✅ All 3 steps completed successfully!")
    print(f"   → {sources_path}")
    print(f"   → {context_path}")
    print(SEPARATOR)


if __name__ == "__main__":
    asyncio.run(main())
