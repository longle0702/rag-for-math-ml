# RAG for Mathematics in Machine Learning

Retrieval-Augmented Generation (RAG) system for answering questions about *Mathematics for Machine Learning* using a local corpus plus optional web augmentation.

> This project was developed as part of the **Gen AI Week** course at **EPITA**, with my teammates Hamza El Hamdi, Kim Tan Truong, and Malo Fargeas.

## What this project does

- Retrieves relevant passages from indexed course content
- Generates grounded answers with citations
- Supports internal-only, external-web, or synthesized agent modes
- Exposes both a Gradio UI and API endpoints

## Setup

### 1. Requirements

- Python 3.11+
- An OpenAI API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env`

```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
```

Add Slack and web-search keys only if you plan to use those integrations.

### 4. Build the index (first run)

```bash
python scripts/build_index.py
```

Re-run this after changing retrieval models or corpus content.

## Run the app

### Recommended (single command)

This starts the MCP server first, then the frontend:

```bash
python run.py
```

Open: `http://localhost:7860`

### Manual (two terminals)

Terminal 1:

```bash
python -m app.mcp_server
```

Terminal 2:

```bash
python -m app.frontend
```

## Core commands

- Build/rebuild retrieval index: `python scripts/build_index.py`
- Start frontend only: `python -m app.frontend`
- Start backend API only: `python -m app.backend`
- Start MCP server only: `python -m app.mcp_server`

## API usage

Start backend:

```bash
python -m app.backend
```

Endpoints:

- `GET /health`
- `POST /query`
- `GET /stats`
- `GET /evaluate`

Example payload:

```json
{
  "question": "What is the trace of a matrix?",
  "top_k": 5
}
```

## Slack integration

The project includes a Slack bot (`app/slack_bot.py` + `app/slack_server.py`) supporting Socket Mode and HTTP mode.

### 1. Create the Slack app

1. Go to `https://api.slack.com/apps` and create a new app.
2. Add bot scopes: `app_mentions:read`, `chat:write`, `im:history`, `im:read`, `im:write` (and `commands` if using `/ragmath`).
3. Install the app and place the bot token in `.env` as `SLACK_BOT_TOKEN`.
4. Enable event subscriptions for `app_mention` and `message.im`.
5. For Socket Mode, add `SLACK_APP_TOKEN`.
6. For HTTP mode, add `SLACK_SIGNING_SECRET` and configure `/slack/events`.

### 2. Run the bot

```bash
# Start MCP server first
python -m app.mcp_server

# Then start Slack server (auto mode)
python -m app.slack_server
```

Optional forced modes:

```bash
python -m app.slack_server --mode socket
python -m app.slack_server --mode http --port 3000
```

## Configuration

Primary settings are in `app/config.py`, including chunking, retrieval, model selection, and generation parameters.

Optional retrieval-related environment variables:

```bash
RETRIEVAL_BACKEND=bi_encoder
BI_ENCODER_MODEL=BAAI/bge-small-en-v1.5
CROSS_ENCODER_ENABLED=true
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_CANDIDATE_MULTIPLIER=4
```
