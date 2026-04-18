"""Entry point for running the Slack RAG bot.

Two transport modes are supported:

* **Socket Mode** (default when ``SLACK_APP_TOKEN`` is present) — opens a
  WebSocket to Slack; no public URL is required. Ideal for local development.
* **HTTP mode** — runs an aiohttp web server that Slack posts events to.
  Requires a publicly reachable URL (e.g. via a reverse proxy or ngrok) and
  ``SLACK_SIGNING_SECRET``.

Required environment variables are loaded from ``.env`` via ``python-dotenv``
(already transitively imported through ``app.config``). The server will also
load ``.env`` directly as a belt-and-braces for the case where the caller
runs the module without touching ``app.config`` (e.g. ``python -m`` with a
stripped ``PYTHONPATH``).

Run:
    python -m app.slack_server                  # auto-detect mode
    python -m app.slack_server --mode socket    # force Socket Mode
    python -m app.slack_server --mode http --port 3000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env early so token lookups below succeed.
load_dotenv()

from app.slack_bot import build_app  # noqa: E402  (import after load_dotenv)

logger = logging.getLogger("app.slack_server")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _detect_mode(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("SLACK_APP_TOKEN"):
        return "socket"
    return "http"


async def _run_socket_mode() -> None:
    """Run the bot using Socket Mode (no public URL required)."""
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not bot_token or not app_token:
        logger.error(
            "Socket Mode requires both SLACK_BOT_TOKEN (xoxb-...) and "
            "SLACK_APP_TOKEN (xapp-...) to be set."
        )
        sys.exit(2)

    app = build_app(bot_token=bot_token)
    handler = AsyncSocketModeHandler(app, app_token)
    logger.info("Starting Slack bot in Socket Mode")
    await handler.start_async()


def _run_http_mode(host: str, port: int) -> None:
    """Run the bot as an HTTP server for Slack Events API callbacks."""
    from slack_bolt.adapter.aiohttp.async_handler import AsyncSlackRequestHandler
    from aiohttp import web

    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    signing_secret = os.environ.get("SLACK_SIGNING_SECRET")
    if not bot_token or not signing_secret:
        logger.error(
            "HTTP mode requires SLACK_BOT_TOKEN (xoxb-...) and "
            "SLACK_SIGNING_SECRET to be set."
        )
        sys.exit(2)

    app = build_app(bot_token=bot_token, signing_secret=signing_secret)
    handler = AsyncSlackRequestHandler(app)

    async def slack_events(req: web.Request) -> web.Response:
        return await handler.handle(req)

    async def health(_req: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    web_app = web.Application()
    web_app.router.add_post("/slack/events", slack_events)
    web_app.router.add_post("/slack/commands", slack_events)
    web_app.router.add_get("/health", health)

    logger.info("Starting Slack bot in HTTP mode on %s:%d", host, port)
    logger.info("  → Events URL: http://%s:%d/slack/events", host, port)
    logger.info("  → Commands URL: http://%s:%d/slack/commands", host, port)
    web.run_app(web_app, host=host, port=port, print=None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Slack RAG bot.")
    parser.add_argument(
        "--mode",
        choices=["auto", "socket", "http"],
        default="auto",
        help="Transport mode. 'auto' picks socket if SLACK_APP_TOKEN is set.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address.")
    parser.add_argument("--port", type=int, default=3000, help="HTTP bind port.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    mode = _detect_mode(args.mode)
    if mode == "socket":
        try:
            asyncio.run(_run_socket_mode())
        except KeyboardInterrupt:
            logger.info("Shutting down on Ctrl-C")
    else:
        _run_http_mode(args.host, args.port)


if __name__ == "__main__":
    main()
