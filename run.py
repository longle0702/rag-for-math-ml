#!/usr/bin/env python3
"""Launch MCP server and frontend with one command."""
from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path


def _terminate_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return

    print(f"\nStopping {name}...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    project_root = Path(__file__).resolve().parent

    print("Starting MCP server...")
    mcp_proc = subprocess.Popen(
        [sys.executable, "-m", "app.mcp_server"],
        cwd=project_root,
    )

    def _handle_shutdown(signum, _frame) -> None:
        print(f"\nReceived signal {signum}. Shutting down...")
        _terminate_process(mcp_proc, "MCP server")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # Give MCP a short head start and fail fast if it crashes immediately.
    time.sleep(1)
    if mcp_proc.poll() is not None:
        print("MCP server failed to start.")
        return mcp_proc.returncode or 1

    print("Starting frontend...")
    print("Open http://localhost:7860 once the app is ready.\n")

    frontend_code = 0
    try:
        frontend_code = subprocess.call(
            [sys.executable, "-m", "app.frontend"],
            cwd=project_root,
        )
    finally:
        _terminate_process(mcp_proc, "MCP server")

    return frontend_code


if __name__ == "__main__":
    raise SystemExit(main())
