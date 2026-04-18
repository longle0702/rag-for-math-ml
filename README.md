# RAG for Mathematics in Machine Learning

A Retrieval-Augmented Generation (RAG) project focused on the *Mathematics for Machine Learning* content. The system retrieves relevant passages from a local corpus, then generates grounded answers with source citations.

## Features

- End-to-end RAG pipeline (chunking, embedding, indexing, retrieval, generation)
- FAISS vector index for semantic search
- Bi-encoder retrieval with optional cross-encoder reranking
- OpenAI embeddings + chat generation fallback
- Gradio web UI for interactive Q&A
- FastAPI backend for API usage
- Source citations with relevance scores
- Cost and token tracking
- LaTeX-aware answer rendering in the UI

## Project Structure

```text
rag-math-ml/
├── app/
│   ├── backend.py
│   ├── config.py
│   ├── embeddings.py
│   ├── frontend.py
│   ├── rag_pipeline.py
│   └── index/                  # auto-generated
├── data/
│   ├── corpus.json
│   └── questions.json
├── scripts/
│   ├── build_index.py
│   ├── extract.py
│   ├── quickstart.py
│   └── verify.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` in project root:

```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env`:

```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
```

3. Build index (first run, or after changing retrieval models):

```bash
python scripts/build_index.py
```

4. Start web app:

```bash
python -m app.frontend
```

5. Open: `http://localhost:7860`

## Which Files To Run

- `scripts/build_index.py`
  - Run this to build/rebuild FAISS + chunk artifacts.
  - Command: `python scripts/build_index.py`
- `app/frontend.py`
  - Run this for the Gradio chat UI.
  - Command: `python -m app.frontend`
- `app/backend.py`
  - Run this for the FastAPI server.
  - Command: `python -m app.backend`
- `scripts/quickstart.py`
  - Optional menu launcher for common actions.
  - Command: `python scripts/quickstart.py`
- `scripts/verify.py`
  - Optional environment/setup validation.
  - Command: `python scripts/verify.py`

## API Usage

Run backend:

```bash
python -m app.backend
```

Available endpoints:

- `GET /health`
- `POST /query`
- `GET /stats`
- `GET /evaluate`

Example query payload:

```json
{
  "question": "What is the trace of a matrix?",
  "top_k": 5
}
```

## Slack Integration

The project ships a Slack bot (`app/slack_bot.py` + `app/slack_server.py`) that
exposes the multi-agent RAG pipeline through Slack's Events API. Both Socket
Mode (recommended for local dev, no public URL required) and HTTP mode are
supported.

### 1. Create the Slack app

1. Go to https://api.slack.com/apps and click **Create New App → From scratch**.
2. Under **OAuth & Permissions**, add these Bot Token scopes:
   - `app_mentions:read`
   - `chat:write`
   - `im:history`
   - `im:read`
   - `im:write`
   - `commands` (only if you want to use the `/ragmath` slash command)
3. Install the app to your workspace and copy the **Bot User OAuth Token**
   (starts with `xoxb-`) into `SLACK_BOT_TOKEN` in `.env`.
4. Under **Event Subscriptions**, enable events and subscribe the bot to:
   - `app_mention`
   - `message.im`
5. (Socket Mode) Under **Socket Mode**, enable it and generate an
   **App-Level Token** with the `connections:write` scope. Copy it (starts with
   `xapp-`) into `SLACK_APP_TOKEN` in `.env`.
6. (HTTP mode) Under **Basic Information → App Credentials**, copy the
   **Signing Secret** into `SLACK_SIGNING_SECRET` in `.env`, and set the
   request URL to `https://<your-host>/slack/events`.
7. (Optional) Under **Slash Commands**, create `/ragmath` with request URL
   `https://<your-host>/slack/commands` (HTTP mode) — Socket Mode handles it
   automatically.

### 2. Run the bot

First make sure the MCP server is running (the agents need it for retrieval):

```bash
python -m app.mcp_server
```

Then start the Slack bot in another terminal:

```bash
# Auto-detects Socket Mode if SLACK_APP_TOKEN is set, otherwise HTTP.
python -m app.slack_server

# Or force a specific mode:
python -m app.slack_server --mode socket
python -m app.slack_server --mode http --port 3000
```

### 3. Talk to the bot

- @-mention the bot in any channel it's invited to.
- DM the bot directly.
- Prefix a question to pick an agent mode:
  - `!internal <question>` — RAG on the local corpus only
  - `!external <question>` — web search + scraping
  - `!synth    <question>` — synthesize both (default)
- Use `/ragmath <question>` for one-shot Q&A.

Conversation history is scoped per Slack thread (or per DM), and capped at
the last 8 turns to fit the model context window.

## Verification and Utilities

Run system verification:

```bash
python scripts/verify.py
```

Use interactive launcher:

```bash
python scripts/quickstart.py
```

## Configuration

Main settings are in `app/config.py`:

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K_RETRIEVAL`
- `TEMPERATURE`
- model names and cost constants

### Retrieval Modes

The project now supports two-stage retrieval:

1. Bi-encoder retrieves top semantic candidates.
2. Cross-encoder reranks those candidates for better final relevance.

Environment variables (optional):

```bash
RETRIEVAL_BACKEND=bi_encoder
BI_ENCODER_MODEL=BAAI/bge-small-en-v1.5
CROSS_ENCODER_ENABLED=true
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_CANDIDATE_MULTIPLIER=4
```

Notes:
- If `sentence-transformers` is unavailable, retrieval falls back to OpenAI embeddings.
- After changing retrieval backend/model, rebuild the index (`python scripts/build_index.py`).

## Notes on Math Extraction and Rendering

- The corpus extraction pipeline is OCR-first to preserve mathematical notation as LaTeX-friendly text.
- The Gradio answer panel is configured to render LaTeX delimiters automatically (`$...$`, `$$...$$`, `\\(...\\)`, `\\[...\\]`).
- The generation prompt explicitly asks the LLM to output math in LaTeX format.
