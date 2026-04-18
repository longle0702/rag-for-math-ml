"""Human-in-the-Loop (HITL) orchestration for draft approval and saving via MCP."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Callable

from fastmcp import Client

from app.agents import agent_workflow, DEFAULT_MCP_HOST, DEFAULT_MCP_PORT


@dataclass
class DraftResult:
    """Container for generated draft data."""

    question: str
    draft: str


class DraftAgent:
    """Creates a first-pass draft using the existing multi-agent workflow."""

    def create_draft(self, question: str) -> DraftResult:
        result = agent_workflow(question, history=[])
        draft = str(result.get("answer", "")).strip()
        if not draft:
            raise RuntimeError("Draft agent returned an empty answer.")
        return DraftResult(question=question, draft=draft)


class HumanApprovalAgent:
    """Human gate: draft is saved only after explicit approval."""

    def __init__(self, input_fn: Callable[[str], str] = input):
        self._input_fn = input_fn

    def approve(self, draft: str) -> bool:
        print("\n===== Draft (HITL Review) =====\n")
        print(draft)
        print("\n===============================\n")

        while True:
            decision = self._input_fn("Approve this draft and save it? [y/n]: ").strip().lower()
            if decision in {"y", "yes"}:
                return True
            if decision in {"n", "no"}:
                return False
            print("Please answer with 'y' or 'n'.")


class ActionAgent:
    """Calls MCP tools to persist approved drafts."""

    def __init__(self, mcp_server_ip: str = DEFAULT_MCP_HOST, mcp_server_port: int = DEFAULT_MCP_PORT):
        self.mcp_client = Client(f"http://{mcp_server_ip}:{mcp_server_port}/mcp")

    async def save_approved_draft(self, content: str, filename: str, directory: str) -> str:
        async with self.mcp_client:
            response = await self.mcp_client.call_tool(
                "save_report",
                {
                    "content": content,
                    "filename": filename,
                    "directory": directory,
                },
            )

        # FastMCP may return a structured tool result. Prefer plain text path if present.
        if hasattr(response, "content"):
            for item in response.content:
                text = getattr(item, "text", None)
                if text:
                    return str(text)
        if isinstance(response, dict) and "content" in response:
            content_items = response.get("content") or []
            if content_items and isinstance(content_items[0], dict) and content_items[0].get("text"):
                return str(content_items[0]["text"])

        return str(response)


async def run_hitl_workflow(
    question: str,
    filename: str,
    directory: str,
    input_fn: Callable[[str], str] = input,
) -> dict:
    """Run the HITL flow: draft -> human approval -> MCP save."""
    draft_agent = DraftAgent()
    human_agent = HumanApprovalAgent(input_fn=input_fn)
    action_agent = ActionAgent()

    draft_result = draft_agent.create_draft(question)
    approved = human_agent.approve(draft_result.draft)

    if not approved:
        return {
            "status": "rejected",
            "message": "Draft rejected by human reviewer. Nothing was saved.",
            "draft": draft_result.draft,
        }

    saved_path = await action_agent.save_approved_draft(
        content=draft_result.draft,
        filename=filename,
        directory=directory,
    )

    return {
        "status": "saved",
        "saved_path": saved_path,
        "draft": draft_result.draft,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HITL approval and save flow via MCP.")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question or prompt to generate the draft from.",
    )
    parser.add_argument(
        "--filename",
        default="approved_draft.md",
        help="Output filename used by MCP save_report tool.",
    )
    parser.add_argument(
        "--directory",
        default="data/reports",
        help="Destination directory passed to MCP save_report tool.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    question = args.question
    if not question:
        question = input("Enter your question/prompt: ").strip()

    if not question:
        raise SystemExit("No question provided. Exiting.")

    result = asyncio.run(
        run_hitl_workflow(
            question=question,
            filename=args.filename,
            directory=args.directory,
        )
    )

    if result["status"] == "saved":
        print(f"Approved draft saved successfully: {result['saved_path']}")
    else:
        print(result["message"])


if __name__ == "__main__":
    main()
