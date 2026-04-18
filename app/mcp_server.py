import json
import asyncio
import fastmcp as fmcp
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from serpapi import GoogleSearch
from crawl4ai import AsyncWebCrawler
from app.rag_pipeline import get_pipeline, RAGPipeline
from app.config import (
    OPENAI_API_KEY,
    SERPAPI_KEY,
    WEB_SEARCH_MODEL,
    WEB_SEARCH_MAX_RESULTS,
    SCRAPE_MAX_CHARS_PER_PAGE,
)

_openai_client = OpenAI(api_key=OPENAI_API_KEY)


mcp = fmcp.FastMCP("Mathematics for ML MCP Server")

@mcp.tool
def dummy_tool(dummy_param):
    return "Success"

# Step 1 - Corpus and Db lookup
@mcp.tool
def search_internal_knowledge(query : str, top_k : int = 5) -> str:
    try:
        pipeline = get_pipeline()
        results = pipeline.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant information found."

        formatted_response = "\n\n---\n\n".join([
            f"[Source: {chunk['source']}, Page: {chunk['page']}, Relevance: {chunk.get('relevance_score', 0.0):.4f}]\n{chunk['text']}" for chunk in results
        ])

        return formatted_response
    except Exception as e:
        print(f"MCP Tool Error: {e}")
        return "EREX"

# Step 2 – Save an approved draft to disk
@mcp.tool
def save_report(content: str, filename: str, directory: str = "data/reports") -> str:
    """
    Persist an approved draft as a Markdown file.

    Args:
        content:   The text of the approved report.
        filename:  Target filename (without extension is fine; .md is appended if missing).
        directory: Path relative to the project root where the file is saved.
                   Defaults to ``data/reports``.

    Returns:
        The absolute path of the saved file.
    """
    project_root = Path(__file__).parent.parent
    target_dir = project_root / directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the file ends with .md
    if not filename.endswith(".md"):
        filename = filename + ".md"

    file_path = target_dir / filename
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)

# Step 3 - Assemble a markdown research report from crawl4ai output (no extra LLM needed)
@mcp.tool
def create_markdown_report(query: str, search_results: str, scraped_content: str) -> str:
    """Assemble a markdown research report from search results and crawl4ai scraped content."""
    try:
        results = json.loads(search_results)
    except Exception:
        results = []

    sources_md = "\n".join(
        f"- [{r.get('title') or r['url']}]({r['url']})"
        + (f" — {r['snippet']}" if r.get("snippet") else "")
        for r in results
    )

    return (
        f"# Research Report: {query}\n\n"
        f"## Sources\n\n{sources_md}\n\n"
        f"## Scraped Content\n\n{scraped_content}"
    )


# ── Web search tools ────────────────────────────────────────────────────────

@mcp.tool
def generate_web_query(question: str) -> str:
    """Ask gpt-4.1-nano to turn a user question into a concise Google search query."""
    try:
        response = _openai_client.chat.completions.create(
            model=WEB_SEARCH_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user's question into a concise Google search query. "
                        "Return only the query string, nothing else."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"EREX: {e}"


@mcp.tool
def google_search(query: str, num_results: int = WEB_SEARCH_MAX_RESULTS) -> str:
    """Search Google via SerpAPI. Returns a JSON list of {title, url, snippet}."""
    if not SERPAPI_KEY:
        return "EREX: SERPAPI_KEY not configured."
    try:
        search = GoogleSearch({"q": query, "num": num_results, "api_key": SERPAPI_KEY})
        data = search.get_dict()
        results = [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in data.get("organic_results", [])
        ]
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"EREX: {e}"


@mcp.tool
async def scrape_urls(urls: list) -> str:
    """Crawl a list of URLs with Crawl4AI and return their markdown content, truncated per page."""
    parts = []
    try:
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url=url)
                    md = (result.markdown or "")[:SCRAPE_MAX_CHARS_PER_PAGE]
                    parts.append(f"## {url}\n\n{md}")
                except Exception as e:
                    parts.append(f"## {url}\n\nERROR: {e}")
    except Exception as e:
        return f"EREX: {e}"
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Run "fastmcp run mcp_server.py"
    # Load the index into the MCP server's memory before starting
    print("Loading RAG pipeline indexes...")
    pipeline = get_pipeline()
    pipeline.load_index()

    mcp.run(transport="http", host="0.0.0.0", port=9000)
