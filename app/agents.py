import json
import re
from typing import Any
from fastmcp import Client
import openai as opai
import asyncio
import time

from app.config import (
    OPENAI_API_KEY, TOP_K_RETRIEVAL, N_RETRIES, CHAT_MODEL,
    CHAT_INPUT_COST_PER_1K, CHAT_OUTPUT_COST_PER_1K,
    WEB_CHUNK_TOP_K,
    INTERNAL_AGENT_MODEL, INTERNAL_AGENT_TEMPERATURE, INTERNAL_AGENT_CONTEXT_WINDOW,
    EXTERNAL_AGENT_MODEL, EXTERNAL_AGENT_TEMPERATURE, EXTERNAL_AGENT_CONTEXT_WINDOW,
    SYNTHESIZER_AGENT_MODEL, SYNTHESIZER_AGENT_TEMPERATURE, SYNTHESIZER_AGENT_CONTEXT_WINDOW,
)
from app.rag_pipeline import get_tokens_num

DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 9000

# Module-level token / cost tracking for all Agent GPT calls
_agent_chat_input_tokens = 0
_agent_chat_output_tokens = 0
_agent_chat_cost = 0.0


def get_agent_chat_stats() -> dict:
    """Return cumulative chat token / cost stats for Agent-based calls."""
    return {
        "input_tokens": _agent_chat_input_tokens,
        "output_tokens": _agent_chat_output_tokens,
        "total_cost": _agent_chat_cost,
    }



class Agent:
    def __init__(self, mcp_server_ip: str = DEFAULT_MCP_HOST, mcp_server_port: int = DEFAULT_MCP_PORT,
                 model: str = CHAT_MODEL, temperature: float = 0.2, context_window: int = 128000):
        self.mcp_client = Client(f"http://{mcp_server_ip}:{mcp_server_port}/mcp")
        self.gpt_client = opai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.temperature = temperature
        self.context_window = context_window

    async def connect_mcp(self):
        async with self.mcp_client:
            await self.mcp_client.ping()
            result = await self.mcp_client.call_tool("dummy_tool", {"dummy_param": "dummy_value"})
            print(result)

    def request_gpt(self, messages: list, max_tokens: int = 1000) -> str:
        global _agent_chat_input_tokens, _agent_chat_output_tokens, _agent_chat_cost
        response = None

        for attempt in range(N_RETRIES):
            try:
                response = self.gpt_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                break
            except opai.RateLimitError:
                if attempt < N_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    return "ERRATE"
            except opai.APIError as e:
                print(f"API error: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"Unexpected GPT error: {e}")
                time.sleep(1)

        if not response:
            return "ERNORES"

        # Track token usage
        if response.usage:
            inp = response.usage.prompt_tokens
            out = response.usage.completion_tokens
            _agent_chat_input_tokens += inp
            _agent_chat_output_tokens += out
            _agent_chat_cost += inp * CHAT_INPUT_COST_PER_1K + out * CHAT_OUTPUT_COST_PER_1K

        return response.choices[0].message.content

    def request_tool(self, tool_name : str, params : dict) -> Any:
        async def request_tool_async():
            async with self.mcp_client:
                return await self.mcp_client.call_tool(tool_name, params)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running — safe to use asyncio.run()
            return asyncio.run(request_tool_async())

        # Already inside a running loop (e.g. called from main.py's asyncio.run).
        # Run the coroutine in a separate thread to avoid the nested-loop error.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, request_tool_async()).result()

    async def request_tool_async(self, tool_name: str, params: dict) -> Any:
        """Async version of request_tool for use inside an already-running event loop."""
        async with self.mcp_client:
            return await self.mcp_client.call_tool(tool_name, params)

    @staticmethod
    def _tool_result_to_text(tool_result: Any) -> str:
        """Normalize FastMCP tool output into plain text."""
        if isinstance(tool_result, str):
            return tool_result
        if isinstance(tool_result, dict):
            content_items = tool_result.get("content") or []
            if content_items and isinstance(content_items[0], dict):
                text = content_items[0].get("text")
                if text:
                    return str(text)
            return json.dumps(tool_result)
        if hasattr(tool_result, "content"):
            for item in tool_result.content:
                text = getattr(item, "text", None)
                if text:
                    return str(text)
        return str(tool_result)


class InternalAgent(Agent):
    def __init__(self, mcp_server_ip: str = DEFAULT_MCP_HOST, mcp_server_port: int = DEFAULT_MCP_PORT):
        super().__init__(mcp_server_ip, mcp_server_port,
                         model=INTERNAL_AGENT_MODEL,
                         temperature=INTERNAL_AGENT_TEMPERATURE,
                         context_window=INTERNAL_AGENT_CONTEXT_WINDOW)

    def _extract_sources(self, context_text: str) -> list[dict]:
        sources: list[dict] = []
        seen: set[tuple[str, int]] = set()

        for block in context_text.split("\n\n---\n\n"):
            match = re.search(
                r"\[Source:\s*(?P<source>.*?),\s*Page:\s*(?P<page>\d+)"
                r"(?:,\s*Relevance:\s*(?P<relevance>[\d.]+))?"
                r"\]",
                block,
            )
            if not match:
                continue

            source = match.group("source").strip()
            page = int(match.group("page"))
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)

            relevance = float(match.group("relevance") or 0.0)
            text = re.sub(r"\[Source:.*?\]\n?", "", block).strip()
            sources.append(
                {
                    "source": source,
                    "page": page,
                    "relevance": relevance,
                    "preview": text[:200] + ("..." if len(text) > 200 else ""),
                }
            )

        return sources

    def run(self, user_query: str, history: list | None = None) -> tuple[str, list[dict]]:
        history = history or []
        raw_results = self.request_tool(
            "search_internal_knowledge",
            {"query": user_query, "top_k": TOP_K_RETRIEVAL},
        )
        results = self._tool_result_to_text(raw_results)

        if str(results) == "EREX":
            return "EREX", []

        system_prompt = """
You are a helpful assistant that answers questions about Mathematics for Machine Learning. 
You must:
Answer based only on the provided context, if the question is not very specific to the context, reply asking for clarification. Otherwise:
1. Write a short summary of the full answer.
2. Cite your sources (document name and page number).
3. Be concise and accurate.
4. Format mathematical expressions using LaTeX ($...$ for inline, $$...$$ for blocks).
        """

        user_message = f"""Question: {user_query}
Context from the knowledge base:
{results}

Please answer the question as a summary using the provided context. Include source citations.
"""

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(history)
        api_messages.append({"role": "user", "content": user_message})

        tokens_num = get_tokens_num(api_messages)
        if tokens_num >= self.context_window:
            return "ERNEWCONV", []

        response = self.request_gpt(messages=api_messages, max_tokens=1000)

        if str(response).startswith("ER"):
            return response, []

        sources = self._extract_sources(results)

        return response, sources

class ExternalAgent(Agent):
    """Chains: generate_web_query → google_search → scrape_urls → chunk & filter → GPT answer."""

    def __init__(self, mcp_server_ip: str = DEFAULT_MCP_HOST, mcp_server_port: int = DEFAULT_MCP_PORT):
        super().__init__(mcp_server_ip, mcp_server_port,
                         model=EXTERNAL_AGENT_MODEL,
                         temperature=EXTERNAL_AGENT_TEMPERATURE,
                         context_window=EXTERNAL_AGENT_CONTEXT_WINDOW)

    @staticmethod
    def _filter_relevant_chunks(text: str, query: str, top_k: int, chunk_size: int = 500) -> str:
        """Split text into chunks, embed them with OpenAI, and return the top-K most relevant."""
        from app.embeddings import embed_texts
        import numpy as np

        # Split into chunks by paragraphs, then by size
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: list[str] = []
        for para in paragraphs:
            if len(para) <= chunk_size:
                if para:
                    chunks.append(para)
            else:
                for i in range(0, len(para), chunk_size):
                    piece = para[i:i + chunk_size].strip()
                    if piece:
                        chunks.append(piece)

        if not chunks:
            return text

        if len(chunks) <= top_k:
            return "\n\n".join(chunks)

        # Embed query + chunks together (query is first)
        all_texts = [query] + chunks
        embeddings, _ = embed_texts(all_texts, show_progress=False)

        query_vec = embeddings[0]
        chunk_vecs = embeddings[1:]

        # Cosine similarity
        norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        chunk_vecs_normed = chunk_vecs / norms
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec_normed = query_vec / query_norm
        else:
            query_vec_normed = query_vec

        similarities = chunk_vecs_normed @ query_vec_normed
        top_indices = np.argsort(similarities)[::-1][:top_k]
        # Keep original order for readability
        top_indices = sorted(top_indices)

        selected = [chunks[i] for i in top_indices]
        return "\n\n".join(selected)

    def run(self, user_query: str) -> tuple[str, list[dict]]:
        # Step 1: LLM turns the question into a concise search query
        raw_query = self.request_tool("generate_web_query", {"question": user_query})
        web_query = self._tool_result_to_text(raw_query)
        print(f"[ExternalAgent] web query: {web_query}")

        if web_query.startswith("EREX"):
            return f"Web query generation failed: {web_query}", []

        # Step 2: Google search via SerpAPI
        raw_search = self.request_tool("google_search", {"query": web_query})
        search_text = self._tool_result_to_text(raw_search)
        print(f"[ExternalAgent] search results: {search_text[:300]}")

        if search_text.startswith("EREX"):
            return f"Google search failed: {search_text}", []

        # Extract URLs and structured sources from the JSON result
        try:
            results_list = json.loads(search_text)
            urls = [r["url"] for r in results_list if r.get("url")]
        except (json.JSONDecodeError, KeyError):
            return f"Could not parse search results: {search_text[:200]}", []

        if not urls:
            return "No URLs found from search.", []

        # Build structured web sources
        web_sources = [
            {
                "source": r.get("title", r.get("url", "Unknown")),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
                "origin": "external",
            }
            for r in results_list if r.get("url")
        ]

        # Step 3: Scrape the URLs with Crawl4AI
        try:
            raw_scraped = self.request_tool("scrape_urls", {"urls": urls[:5]})
            scraped_text = self._tool_result_to_text(raw_scraped)
        except Exception as e:
            scraped_text = f"EREX: {e}"

        # Fallback: if scraping is unavailable (e.g. missing Playwright browser),
        # build context from search snippets so the workflow can still answer.
        if scraped_text.startswith("EREX"):
            print(f"[ExternalAgent] scrape_urls failed, using snippets fallback: {scraped_text[:300]}")
            snippet_blocks = []
            for item in results_list[:5]:
                url = item.get("url", "")
                title = item.get("title", "Web result")
                snippet = item.get("snippet", "").strip()
                snippet_blocks.append(f"## {title}\nURL: {url}\n\n{snippet}")
            scraped_text = "\n\n---\n\n".join(snippet_blocks)
        else:
            print(f"[ExternalAgent] scraped {min(len(urls), 5)} URLs, {len(scraped_text)} chars")

        # Step 4: Filter to top-K relevant chunks using OpenAI embeddings
        filtered_text = self._filter_relevant_chunks(
            scraped_text, user_query, top_k=WEB_CHUNK_TOP_K,
        )
        print(f"[ExternalAgent] filtered to {len(filtered_text)} chars (top {WEB_CHUNK_TOP_K} chunks)")

        # Step 5: GPT generates an answer from the filtered web context only
        system_prompt = """You are a helpful assistant that answers questions using web search results.
You must:
1. Answer based only on the provided web content.
2. Be concise and accurate.
3. If the web content does not contain enough information to answer, say so clearly.
4. Format mathematical expressions using LaTeX ($...$ for inline, $$...$$ for blocks).
"""

        user_message = f"""Question: {user_query}

Web content (from search results):
{filtered_text}

Please answer the question using only the provided web content."""

        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tokens_num = get_tokens_num(api_messages)
        if tokens_num >= self.context_window:
            # Truncate filtered_text to fit
            max_chars = len(filtered_text) * self.context_window // (tokens_num + 1)
            user_message = f"""Question: {user_query}

Web content (from search results):
{filtered_text[:max_chars]}

Please answer the question using only the provided web content."""
            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        response = self.request_gpt(messages=api_messages, max_tokens=1000)

        if str(response).startswith("ER"):
            return response, web_sources

        return response, web_sources

class SynthesizerAgent(Agent):
    def __init__(self, mcp_server_ip: str = DEFAULT_MCP_HOST, mcp_server_port: int = DEFAULT_MCP_PORT):
        super().__init__(mcp_server_ip, mcp_server_port,
                         model=SYNTHESIZER_AGENT_MODEL,
                         temperature=SYNTHESIZER_AGENT_TEMPERATURE,
                         context_window=SYNTHESIZER_AGENT_CONTEXT_WINDOW)

    def run(self, user_query: str, in_agent_input: str | None, ex_agent_input: str | None, history: list | None = None) -> str:
        if history is None:
            history = []
        if not in_agent_input:
            in_agent_input = "No internal knowledge available."
        if not ex_agent_input:
            ex_agent_input = "No external fact-check information available."

        system_prompt = (
            "You are a synthesizer assistant that combines internal research and external fact-check results into a final answer.\n"
            "Use only the provided internal and external information.\n"
            "If the answer is uncertain or incomplete, say so clearly.\n"
            "Keep the response concise and math-aware."
        )

        user_prompt = f"""User question:
{user_query}

Internal research output:
{in_agent_input}

External fact-check output:
{ex_agent_input}

Please provide a final answer to the user and mention whether the internal and external checks are aligned."""

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(history)
        api_messages.append({"role": "user", "content": user_prompt})

        tokens_num = get_tokens_num(api_messages)
        if tokens_num >= self.context_window:
            return "ERNEWCONV"

        return self.request_gpt(messages=api_messages, max_tokens=1000)


class JudgeAgent(Agent):
    """LLM-as-Judge: evaluates answer quality on accuracy, hallucination, and relevance."""

    def __init__(self, mcp_server_ip: str = DEFAULT_MCP_HOST, mcp_server_port: int = DEFAULT_MCP_PORT):
        super(JudgeAgent, self).__init__(mcp_server_ip, mcp_server_port)

    def run(self, user_query: str, synthesized_answer: str, in_agent_input: str, ex_agent_input: str) -> dict:
        """Score the synthesized answer on accuracy, hallucination, and relevance.

        Returns:
            dict with scores (0-100) and overall confidence (0-100)
        """
        system_prompt = (
            "You are a strict quality judge for AI-generated answers about Mathematics for Machine Learning.\n"
            "You must be critical and rigorous. Perfect scores (90+) should be extremely rare.\n\n"
            "Evaluate the provided answer against ONLY the source excerpts on three dimensions:\n\n"
            "1. ACCURACY (0-100): Does every claim in the answer have a direct basis in the source text? "
            "Deduct points for any claim not explicitly supported, even if it seems plausible.\n"
            "2. HALLUCINATION (0-100): Is the answer strictly grounded in the provided sources? "
            "(100 = perfectly grounded, 0 = entirely fabricated). Deduct heavily for any statement "
            "that goes beyond or embellishes the source material.\n"
            "3. RELEVANCE (0-100): Does the answer directly and completely address the user's question? "
            "Deduct points for tangential information or if key parts of the question are unanswered.\n\n"
            "Be skeptical. A typical good answer scores 60-80. Only flawless answers score above 85.\n\n"
            "Return ONLY a JSON object with these exact keys: accuracy, hallucination, relevance, confidence.\n"
            "Confidence is the average of the three scores.\n"
            "Do not include any other text."
        )

        user_prompt = f"""User question:
{user_query}

Source excerpts (ground truth — judge the answer ONLY against these):
{in_agent_input[:1500]}

Generated answer to evaluate:
{synthesized_answer}

Provide scores in JSON format."""

        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.request_gpt(messages=api_messages, max_tokens=200)

            if str(response).startswith("ER"):
                return {"accuracy": 50, "hallucination": 50, "relevance": 50, "confidence": 50}

            response_text = response.strip()
            # Strip markdown code fences that GPT often wraps around JSON
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text)
            try:
                scores = json.loads(response_text)
                return {
                    "accuracy": scores.get("accuracy", 70),
                    "hallucination": scores.get("hallucination", 70),
                    "relevance": scores.get("relevance", 70),
                    "confidence": scores.get("confidence", 70),
                }
            except json.JSONDecodeError:
                return {
                    "accuracy": 60,
                    "hallucination": 60,
                    "relevance": 60,
                    "confidence": 60,
                }
        except Exception as e:
            return {
                "accuracy": 50,
                "hallucination": 50,
                "relevance": 50,
                "confidence": 50,
            }


def agent_workflow(user_query: str, history: list | None = None, mode: str = "synthesized") -> dict:
    """Run the agent pipeline.

    Args:
        user_query: The user's question.
        history: Conversation history.
        mode: One of "internal", "external", or "synthesized".
    """
    if history is None:
        history = []

    judge = JudgeAgent()

    # ── Internal only ──
    if mode == "internal":
        internal_agent = InternalAgent()
        internal_output, internal_sources = internal_agent.run(user_query, history)
        for s in internal_sources:
            s["origin"] = "internal"
        if str(internal_output).startswith("ER"):
            return {"answer": str(internal_output), "sources": internal_sources, "judge_scores": {}}
        judge_scores = judge.run(user_query, internal_output, internal_output, "")
        return {"answer": internal_output, "sources": internal_sources, "judge_scores": judge_scores}

    # ── External only ──
    if mode == "external":
        external_agent = ExternalAgent()
        external_output, external_sources = external_agent.run(user_query)
        if str(external_output).startswith("ER"):
            return {"answer": str(external_output), "sources": external_sources, "judge_scores": {}}
        judge_scores = judge.run(user_query, external_output, "", external_output)
        return {"answer": external_output, "sources": external_sources, "judge_scores": judge_scores}

    # ── Synthesized (default): both agents + synthesizer ──
    internal_agent = InternalAgent()
    external_agent = ExternalAgent()
    synthesizer = SynthesizerAgent()

    internal_output, internal_sources = internal_agent.run(user_query, history)
    if str(internal_output).startswith("ER"):
        return {
            "answer": str(internal_output),
            "sources": internal_sources,
            "judge_scores": {},
        }

    # Tag internal sources with origin
    for s in internal_sources:
        s["origin"] = "internal"

    external_output, external_sources = external_agent.run(user_query)

    # Merge sources with external first so web URLs are visible in Slack citations.
    all_sources = external_sources + internal_sources

    synthesized_answer = synthesizer.run(user_query, internal_output, external_output, history)
    if str(synthesized_answer).startswith("ER"):
        return {
            "answer": str(synthesized_answer),
            "sources": all_sources,
            "judge_scores": {},
        }
    judge_scores = judge.run(user_query, synthesized_answer, internal_output, external_output)


    return {
        "answer": synthesized_answer,
        "sources": all_sources,
        "judge_scores": judge_scores,
    }


if __name__ == "__main__":
    print("Agent start...")
    agent = Agent("127.0.0.1", 9000)
    asyncio.run(agent.connect_mcp())
