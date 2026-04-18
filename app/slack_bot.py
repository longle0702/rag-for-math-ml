"""Slack bot integration for the RAG Math ML assistant.

Wires Slack events (app mentions + DMs) to the multi-agent RAG pipeline
defined in ``app.agents.agent_workflow``.

Two transports are supported by ``app.slack_server``:
  * Socket Mode — requires ``SLACK_BOT_TOKEN`` (xoxb-...) and
    ``SLACK_APP_TOKEN`` (xapp-...). No public URL needed.
  * HTTP mode   — requires ``SLACK_BOT_TOKEN`` and ``SLACK_SIGNING_SECRET``.
    Suitable for a production deploy behind a reverse proxy.

This module exposes the :class:`SlackRAGBot` factory plus a module-level
``build_app`` helper used by the server entry point. Message handling is
intentionally defensive: any exception raised by the pipeline is logged and
surfaced to the user as a short, friendly error, never as a crash.
"""
from __future__ import annotations
import json

import logging
import os
import re
import traceback
from typing import Any, Callable
from slack_bolt.async_app import AsyncApp

logger = logging.getLogger(__name__)


# ── Agent mode prefixes ──────────────────────────────────────────────────────
# Users can prefix a question with one of these tokens to select a mode.
# Example: "!internal What is the trace of a matrix?"
_MODE_PREFIXES: dict[str, str] = {
    "!default": "internal",
    "!internal": "internal",
    "!external": "external",
    "!web": "external",
    "!combined": "synthesized",
    "!synth": "synthesized",
    "!both": "synthesized",
}
_DEFAULT_MODE = "internal"

# ── In-memory conversation history ───────────────────────────────────────────
# Keyed by (channel_id, thread_ts). Slack threads provide a natural scope for
# multi-turn conversations; DMs without a thread fall back to channel scope.
_CONVERSATIONS: dict[tuple[str, str], list[dict[str, str]]] = {}
# Hard cap to avoid unbounded growth of agent context windows.
_MAX_HISTORY_TURNS = 8


def _conv_key(channel: str, thread_ts: str | None) -> tuple[str, str]:
    return (channel or "", thread_ts or channel or "")


def _get_history(channel: str, thread_ts: str | None) -> list[dict[str, str]]:
    return _CONVERSATIONS.setdefault(_conv_key(channel, thread_ts), [])


def _append_turn(channel: str, thread_ts: str | None, user: str, assistant: str) -> None:
    history = _get_history(channel, thread_ts)
    history.append({"role": "user", "content": user})
    history.append({"role": "assistant", "content": assistant})
    # Trim to last N turns (user + assistant counted separately).
    if len(history) > _MAX_HISTORY_TURNS * 2:
        del history[: len(history) - _MAX_HISTORY_TURNS * 2]


# ── Message parsing / formatting helpers ─────────────────────────────────────
_MENTION_RE = re.compile(r"<@[UW][A-Z0-9]+>")


def _clean_incoming_text(text: str) -> str:
    """Strip bot mentions and collapse whitespace from an incoming message."""
    if not text:
        return ""
    stripped = _MENTION_RE.sub("", text).strip()
    return re.sub(r"\s+", " ", stripped)


def _extract_mode(text: str) -> tuple[str, str]:
    """Return ``(mode, remaining_text)`` honouring any ``!<mode>`` prefix."""
    tokens = text.split(maxsplit=1)
    if tokens and tokens[0].lower() in _MODE_PREFIXES:
        mode = _MODE_PREFIXES[tokens[0].lower()]
        remaining = tokens[1] if len(tokens) > 1 else ""
        return mode, remaining.strip()
    return _DEFAULT_MODE, text


def _latex_to_slack(text: str) -> str:
    # 1. Handle Block Math -> Triple backticks (Slack's block code)
    text = re.sub(
        r"(\$\$(.+?)\$\$|\\\[(.+?)\\\])",
        lambda m: f"\n```\n{(m.group(2) or m.group(3)).strip()}\n```\n",
        text,
        flags=re.DOTALL
    )

    # 2. Parentheses-style Inline \( ... \) -> Slack Single Backticks
    text = re.sub(r"\\\((.+?)\\\)", r"`\1`", text)

    # 3. Dollar-style Inline $ ... $ -> Slack Single Backticks
    text = re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", r"`\1`", text)

    return text

def _markdown_to_slack(text: str) -> str:
    # 1. Convert Links: [text](url) -> <url|text>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

    # 2. Bold: **text** -> *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)

    # 3. Headings: # Heading -> *Heading*
    # (Slack doesn't have headers, so bolding is the best alternative)
    text = re.sub(r"^#{1,6}\s*(.+)$", r"*\1*", text, flags=re.MULTILINE)

    # 4. Italic: *text* -> _text_ 
    # (Be careful: this can conflict with bolding if not handled in order)
    # text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"_\1_", text)

    return text

def _format_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    external_sources = [s for s in sources if s.get("origin") == "external"]
    internal_sources = [s for s in sources if s.get("origin") != "external"]

    lines = [""]
    if external_sources:
        lines.append("*Web Sources:*")
        for idx, src in enumerate(external_sources[:5], start=1):
            title = src.get("source") or src.get("url") or "Web result"
            url = src.get("url", "")
            if url:
                lines.append(f"{idx}. <{url}|{title}>")
            else:
                lines.append(f"{idx}. {title}")

    if internal_sources:
        lines.append("*Internal Sources:*")
        for idx, src in enumerate(internal_sources[:5], start=1):
            name = src.get("source", "Internal")
            page = src.get("page")
            rel = src.get("relevance")
            rel_part = f" · {rel:.0%}" if isinstance(rel, (int, float)) and rel else ""
            page_part = f", p.{int(page)}" if page else ""
            lines.append(f"{idx}. {name}{page_part}{rel_part}")
    return "\n".join(lines)


def _format_confidence(scores: dict[str, Any] | None) -> str:
    if not scores:
        return ""
    conf = scores.get("confidence")
    if conf is None:
        return ""
    return f"\n_Confidence: {int(conf)}/100_"


_ERROR_MESSAGES: dict[str, str] = {
    "ERNEWCONV": "The conversation is too long for the model's context. Start a new thread to continue.",
    "ERNORES": "I couldn't generate an answer right now. Please try again.",
    "ERRATE": "Rate limit hit against the language model. Please wait a moment and retry.",
    "EREX": "An error occurred while searching the knowledge base. Please try again later.",
}


def _format_error(answer: str) -> str:
    return _ERROR_MESSAGES.get(answer, f"Error: {answer}")


# ── Pipeline invocation ──────────────────────────────────────────────────────
def _run_pipeline(
    question: str,
    history: list[dict[str, str]],
    mode: str,
    workflow: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Run the agent workflow, catching pipeline errors defensively."""
    try:
        return workflow(question, history=history, mode=mode)
    except Exception as exc:  # noqa: BLE001 — any error must not crash the bot
        logger.exception("Agent workflow failed")
        return {
            "answer": f"ERNORES: {exc}",
            "sources": [],
            "judge_scores": {},
            "_exception": traceback.format_exc(),
        }


def build_response_text(result: dict[str, Any]) -> str:
    """Render an agent result as a single Slack mrkdwn-formatted string."""
    answer_raw = str(result.get("answer", "")).strip()

    if answer_raw.startswith("ER"):
        return _format_error(answer_raw.split(":", 1)[0])

    body = _markdown_to_slack(_latex_to_slack(answer_raw))
    sources = _format_sources(result.get("sources") or [])
    confidence = _format_confidence(result.get("judge_scores") or {})
    return (body + sources + confidence).strip()


# ── Bolt app factory ─────────────────────────────────────────────────────────
def build_app(
    *,
    workflow: Callable[..., dict[str, Any]] | None = None,
    bot_token: str | None = None,
    signing_secret: str | None = None,
) -> AsyncApp:
    """Construct and wire an :class:`AsyncApp` with our event handlers.

    Injecting ``workflow`` makes the handlers trivial to unit test.
    """
    if workflow is None:
        # Imported lazily so that test environments without the full RAG stack
        # (FAISS, OpenAI, etc.) can still import this module.
        from app.agents import agent_workflow as workflow  # noqa: WPS433

    app = AsyncApp(
        token=bot_token or os.environ.get("SLACK_BOT_TOKEN"),
        signing_secret=signing_secret or os.environ.get("SLACK_SIGNING_SECRET"),
    )

    async def _handle_question(
        *,
        text: str,
        channel: str,
        thread_ts: str | None,
        user: str | None,
        say,
        client,
    ) -> None:
        cleaned = _clean_incoming_text(text)
        if not cleaned:
            await say(
                text=(
                    "Hi! Ask me a question about *Mathematics for Machine Learning*.\n"
                    "Prefix with `!default`, `!external`, or `!combined` to pick a mode."
                ),
                thread_ts=thread_ts,
            )
            return

        mode, question = _extract_mode(cleaned)
        if not question:
            await say(
                text="Please include a question after the mode prefix.",
                thread_ts=thread_ts,
            )
            return

        # Let the user know we're working on it.
        placeholder = await say(
            text=f":hourglass_flowing_sand: Thinking… _(mode: {mode})_",
            thread_ts=thread_ts,
        )

        history = list(_get_history(channel, thread_ts))
        result = _run_pipeline(question, history, mode, workflow)
        response_text = build_response_text(result)

        # Persist history only on successful (non-error) answers.
        answer = str(result.get("answer", ""))
        if not answer.startswith("ER"):
            _append_turn(channel, thread_ts, question, answer)

        # Replace the placeholder with the real answer (best effort).
        try:
            ts = placeholder.get("ts") if isinstance(placeholder, dict) else None
            if ts:
                await client.chat_update(
                    channel=channel,
                    ts=ts,
                    text=response_text,
                    unfurl_media=False,
                    unfurl_links=False,
                )
                return
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update placeholder message; sending new one")

        await say(text=response_text, thread_ts=thread_ts, unfurl_media=False, unfurl_links=False)

    @app.event("app_mention")
    async def _on_app_mention(event, say, client, logger=logger):  # noqa: D401
        """Answer when the bot is @-mentioned in a channel."""
        try:
            await _handle_question(
                text=event.get("text", ""),
                channel=event.get("channel", ""),
                thread_ts=event.get("thread_ts") or event.get("ts"),
                user=event.get("user"),
                say=say,
                client=client,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error in app_mention handler")
            await say(text="Internal bot error. Please try again.")

    @app.event("message")
    async def _on_message(event, say, client, logger=logger):  # noqa: D401
        """Answer direct messages to the bot. Ignore channel noise / edits."""
        # Ignore bot messages, edits, message_changed subtype noise.
        if event.get("bot_id") or event.get("subtype"):
            return
        # Only respond in DMs (channel type "im") to avoid being chatty in channels;
        # channel mentions are handled by app_mention above.
        if event.get("channel_type") != "im":
            return
        try:
            await _handle_question(
                text=event.get("text", ""),
                channel=event.get("channel", ""),
                thread_ts=event.get("thread_ts"),
                user=event.get("user"),
                say=say,
                client=client,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error in DM message handler")
            await say(text="Internal bot error. Please try again.")

    @app.command("/ragmath")
    async def _on_slash(ack, respond, command, client, logger=logger):  # noqa: D401
        """Slash command for one-shot questions or modal-based input."""
        await ack()
        text = (command.get("text") or "").strip()
        if text:
            mode, question = _extract_mode(text)
            if not question:
                await respond(text="Please include a question after the mode prefix.")
                return
            result = _run_pipeline(question, history=[], mode=mode, workflow=workflow)
            await respond(
                text=build_response_text(result),
                response_type="in_channel",
                unfurl_links=False,
                unfurl_media=False,
            )
            return

        trigger_id = command.get("trigger_id")
        channel_id = command.get("channel_id", "")
        user_id = command.get("user_id", "")
        if not trigger_id:
            await respond(text="Unable to open the form. Please use `/ragmath <question>`.")
            return

        try:
            await client.views_open(
                trigger_id=trigger_id,
                view={
                    "type": "modal",
                    "callback_id": "ragmath_modal_submit",
                    "private_metadata": json.dumps(
                        {"channel_id": channel_id, "user_id": user_id}
                    ),
                    "title": {"type": "plain_text", "text": "RAG Math"},
                    "submit": {"type": "plain_text", "text": "Ask Question"},
                    "close": {"type": "plain_text", "text": "Cancel"},
                    "blocks": [
                        {
                            "type": "input",
                            "block_id": "ragmath_question_block",
                            "label": {"type": "plain_text", "text": "Question"},
                            "element": {
                                "type": "plain_text_input",
                                "action_id": "ragmath_question_input",
                                "multiline": True,
                                "placeholder": {
                                    "type": "plain_text",
                                    "text": "Ask about Mathematics for Machine Learning",
                                },
                            },
                        },
                        {
                            "type": "input",
                            "optional": True,
                            "block_id": "ragmath_mode_block",
                            "label": {"type": "plain_text", "text": "Mode"},
                            "element": {
                                "type": "static_select",
                                "action_id": "ragmath_mode_select",
                                "initial_option": {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Default",
                                    },
                                    "value": "internal",
                                },
                                "options": [
                                    {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "Default",
                                        },
                                        "value": "internal",
                                    },
                                    {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "External Search",
                                        },
                                        "value": "external",
                                    },
                                    {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "Combined",
                                        },
                                        "value": "synthesized",
                                    },
                                ],
                            },
                        },
                    ],
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to open /ragmath modal")
            await respond(text="Couldn't open the form. Please use `/ragmath <question>`.")

    @app.view("ragmath_modal_submit")
    async def _on_ragmath_modal_submit(ack, body, view, client, logger=logger):  # noqa: D401
        """Handle interactive /ragmath modal submission."""
        values = ((view or {}).get("state") or {}).get("values") or {}
        question = (
            values.get("ragmath_question_block", {})
            .get("ragmath_question_input", {})
            .get("value", "")
            .strip()
        )
        if not question:
            await ack(
                response_action="errors",
                errors={"ragmath_question_block": "Please provide a question."},
            )
            return

        mode = (
            values.get("ragmath_mode_block", {})
            .get("ragmath_mode_select", {})
            .get("selected_option", {})
            .get("value")
            or _DEFAULT_MODE
        )
        await ack()

        metadata_raw = (view or {}).get("private_metadata") or "{}"
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
        channel_id = str(metadata.get("channel_id") or "")
        user_id = str(
            metadata.get("user_id") or ((body or {}).get("user") or {}).get("id") or ""
        )
        if not channel_id:
            logger.error("Missing channel_id metadata for /ragmath modal submission")
            return

        try:
            placeholder = await client.chat_postMessage(
                channel=channel_id,
                text=f":hourglass_flowing_sand: Thinking… _(mode: {mode})_",
                unfurl_links=False,
                unfurl_media=False,
            )
            result = _run_pipeline(question, history=[], mode=mode, workflow=workflow)
            response_text = build_response_text(result)

            ts = placeholder.get("ts") if isinstance(placeholder, dict) else None
            if ts:
                await client.chat_update(
                    channel=channel_id,
                    ts=ts,
                    text=response_text,
                    unfurl_links=False,
                    unfurl_media=False,
                )
                return

            await client.chat_postMessage(
                channel=channel_id,
                text=response_text,
                unfurl_links=False,
                unfurl_media=False,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error in /ragmath modal submission")
            if channel_id and user_id:
                try:
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text="Internal bot error. Please try again.",
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to send ephemeral modal error")

    return app


class SlackRAGBot:
    """Thin convenience wrapper — mirrors the shape of the old ``TeamsRAGBot``."""

    def __init__(self, workflow: Callable[..., dict[str, Any]] | None = None):
        self.app = build_app(workflow=workflow)

    def get_app(self) -> AsyncApp:
        return self.app
